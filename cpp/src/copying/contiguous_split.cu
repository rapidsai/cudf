static constexpr size_t split_align = 8;

template<typename T>
struct column_buf_size_functor_impl {
   void operator()(column_view const& c, size_t& running_data_size, size_t& running_validity_size)
   {
      running_data_size += cudf::util::div_rounding_up_safe(c.size() * sizeof(T), split_align) * split_align;      
      if(c.nullable()){
         running_validity_size += cudf::bitmask_allocation_size_bytes(c.size(), split_align);         
      }
   }
};

// TODO
template<>
struct column_buf_size_functor_impl<string_view> {
   void operator()(column_view const& c, size_t& running_data_size, size_t& running_validity_size){};
};

struct column_buf_size_functor {
   template<typename T>
   void operator()(column_view const& c, size_t& running_data_size, size_t& running_validity_size)
   {
      column_buf_size_functor_impl<T> sizer{};
      sizer(c, running_data_size, running_validity_size);
   }
};



template<typename T>
struct column_copy_functor_impl {
   void operator()(column_view const& in, char *&dst, std::vector<column_view>& out_cols
      // PERFORMANCE TEST 4
      /*, rmm::device_scalar<cudf::size_type> &valid_count*/)
   {      
      // there's some unnecessary recomputation of sizes happening here, but it really shouldn't affect much.
      size_t data_size = 0;
      size_t validity_size = 0;      
      column_buf_size_functor_impl<T>{}(in, data_size, validity_size);

      // outgoing pointers
      char* data = dst;
      bitmask_type* validity = validity_size == 0 ? nullptr : reinterpret_cast<bitmask_type*>(dst + data_size);

      // increment working buffer
      dst += (data_size + validity_size);      

      // custom copy kernel (which should probably just be an in-place copy() function in cudf.
      cudf::size_type num_els = cudf::util::round_up_safe(in.size(), cudf::experimental::detail::warp_size);
      constexpr int block_size = 256;
      cudf::experimental::detail::grid_1d grid{num_els, block_size, 1};      
      
      // so there's a significant performance issue that comes up. our incoming column_view objects
      // are the result of a slice.  because of this, they have an UNKNOWN_NULL_COUNT.  because of that,
      // calling column_device_view::create() will cause a recompute of the count, which ends up being
      // -extremely- slow because a.) the typical use case here will involve -huge- numbers of calls and
      // b.) the count recompute involves tons of device allocs and memcopies, which sort of nullifies
      // the entire point of contiguous_split.
      // so to get around this, I am manually constructing a fake-ish view here where the null
      // count is arbitrarily bashed to 0.
      //      
      // I ran 5 performance tests here, all on 6 gigs of input data, 512 columns split 256 ways, for a total of
      // 128k calls to this function.
      //
      // 1. 500 ms    : no valdity information.
      // 2. 10,000 ms  : validify information, leaving UNKNOWN_NULL_COUNTS in place. the time difference 
      //    here is in the null_count() recomputation that happens in column_device_view::create() and the time
      //    spent allocating/reading from the device scalar to get the resulting null count
      // 3. 3,600 ms  : validity information, faking 0 input null count,  allocating a device scalar on the spot, 
      //    recomputing null count in the copy_in_place_kernel and reading it back.
      // 4. 2,700 ms  : validity information, faking 0 input null count, keeping a global device scalar, 
      //    recomputing null count in the copy_in_place_kernel and reading it back.
      // 5. 500 ms    : validity information, faking 0 input null count, setting output null count to UNKNOWN_NULL_COUNT. the
      //    implication here of course is that someone else might end up paying this price later on down the road.
      //
      // Summary : nothing super surprising.  anything that causes memory allocation or copying between host and device
      //           is super slow and becomes extremely noticeable at scale. best bet here seems to be to go with case 5 and 
      //           let someone else pay the cost for lazily evaluating null counts down the road.
      //
      //

      // see performance note above about null counts.
      column_view          in_hacked{  in.type(), in.size(), in.head<T>(), 
                                       in.null_mask(), in.null_mask() == nullptr ? UNKNOWN_NULL_COUNT : 0,
                                       in.offset() };
      mutable_column_view  mcv{in.type(), in.size(), data, 
                               validity, validity == nullptr ? UNKNOWN_NULL_COUNT : 0 };      
      if(in.nullable()){         
         // PERFORMANCE TEST 2, 3
         // rmm::device_scalar<cudf::size_type> valid_count{0, 0, rmm::mr::get_default_resource()};
         copy_in_place_kernel<block_size, T, true><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_hacked), 
                           *mutable_column_device_view::create(mcv)
                           // PERFORMANCE TEST 2, 3, 4
                           //, valid_count.data()
                           );
         // PERFORMANCE TEST 2, 3, 4
         // mcv.set_null_count(in.size() - valid_count.value());                  
      } else
       {
         copy_in_place_kernel<block_size, T, false><<<grid.num_blocks, block_size, 0, 0>>>(
                           *column_device_view::create(in_hacked), 
                           *mutable_column_device_view::create(mcv)
                           // PERFORMANCE TEST 2, 3, 4
                           /*, nullptr*/);
      }
      mcv.set_null_count(cudf::UNKNOWN_NULL_COUNT);

      out_cols.push_back(mcv);
   }
};

// TODO
template<>
struct column_copy_functor_impl<string_view> {
   void operator()(column_view const& in, char *&dst, std::vector<column_view>& out_cols
                  // PERFORMANCE TEST 4
                  /*, rmm::device_scalar<cudf::size_type> &valid_count*/)
   {       
   };
};

struct column_copy_functor {
   template<typename T>   
   void operator()(column_view const& in, char *&dst, std::vector<column_view>& out_cols
                  // PERFORMANCE TEST 4
                  /*, rmm::device_scalar<cudf::size_type> &valid_count*/)
   {
      column_copy_functor_impl<T> fn{};      
      fn(in, dst, out_cols 
         // PERFORMANCE TEST 4
         /* , valid_count*/);
   }
};

struct contiguous_split_result {
   cudf::table_view                    table;
   std::unique_ptr<rmm::device_buffer> all_data;
};
typedef std::vector<cudf::column_view> subcolumns;

std::vector<contiguous_split_result> contiguous_split(cudf::table_view const& input,
                                                      std::vector<size_type> const& splits,
                                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                      cudaStream_t stream = 0)
{    
   // slice each column into a set of sub-columns   
   std::vector<subcolumns> split_table;
   split_table.reserve(input.num_columns());
   for(int idx=0; idx<input.num_columns(); idx++){
      split_table.push_back(cudf::experimental::slice(input.column(idx), splits));
   }
   size_t num_out_tables = split_table[0].size();   

   std::vector<contiguous_split_result> result;

   // DEBUG --------------------------
      float total_alloc_time = 0.0f;
      int total_allocs = 0;
      float total_copy_time = 0.0f;
      int total_copies = 0;
   // DEBUG --------------------------

   // PERFORMANCE TEST 4
   // rmm::device_scalar<cudf::size_type> valid_count{0, 0, rmm::mr::get_default_resource()};

   // output packing for a given table will be:
   // (C0)(V0)(C1)(V1)
   // where Cx = column x and Vx = validity mask x
   // padding to split_align boundaries between each buffer
   for(size_t t_idx=0; t_idx<num_out_tables; t_idx++){  
      size_t subtable_data_size = 0;
      size_t subtable_validity_size = 0;

      // compute sizes
      size_t sts = split_table.size();
      for(size_t c_idx=0; c_idx<split_table.size(); c_idx++){
         column_view const& subcol = split_table[c_idx][t_idx];         
         cudf::experimental::type_dispatcher(subcol.type(), column_buf_size_functor{}, subcol, subtable_data_size, subtable_validity_size);
      }
      
      // DEBUG --------------------------
         scope_timer_manual alloc_time;
         alloc_time.start();      
      // DEBUG --------------------------
      // allocate the blob
      auto device_buf = std::make_unique<rmm::device_buffer>(rmm::device_buffer{subtable_data_size + subtable_validity_size, stream, mr});
      char* buf = static_cast<char*>(device_buf->data());       
      // DEBUG --------------------------
         cudaDeviceSynchronize();
         alloc_time.end();
         total_alloc_time += alloc_time.total_time_ms();
         total_allocs++;
      // DEBUG --------------------------
            
      // DEBUG --------------------------
         scope_timer_manual copy_time;
         copy_time.start();
      // DEBUG --------------------------
      // create columns for the subtables
      std::vector<column_view> out_cols;
      out_cols.reserve(split_table.size());
      for(size_t c_idx=0; c_idx<split_table.size(); c_idx++){
         // copy
         column_view const& subcol = split_table[c_idx][t_idx];                  
         cudf::experimental::type_dispatcher(subcol.type(), column_copy_functor{}, subcol, buf, out_cols 
            // PERFORMANCE TEST 4
            /*, valid_count*/);
      }     
      // DEBUG --------------------------
         cudaDeviceSynchronize();
         copy_time.end();
         total_copy_time += copy_time.total_time_ms();
         total_copies += split_table.size();
      // DEBUG --------------------------
   
      result.push_back(contiguous_split_result{cudf::table_view{out_cols}, std::move(device_buf)});
   }
 
   // DEBUG --------------------------
      printf("  alloc time : %.2f (%d allocs)\n", total_alloc_time, total_allocs);
      printf("  copy time : %.2f (%d copies)\n", total_copy_time, total_copies);
   // DEBUG --------------------------

   // DEBUG --------------------------
         cudaDeviceSynchronize();
   // DEBUG --------------------------
   
   return result;
}


void verify_split_results( cudf::experimental::table const& src_table, 
                           std::vector<contiguous_split_result> const &dst_tables,
                           std::vector<size_type> const& splits,
                           int verbosity = 0)
{     
   table_view src_v(src_table.view());

   // printf("Verification : \n");
   // printf("%d, %d\n", src_v.num_columns(), (int)splits.size());   

   int col_count = 0;
   for(size_t c_idx = 0; c_idx<(size_t)src_v.num_columns(); c_idx++){
      for(size_t s_idx=0; s_idx<splits.size(); s_idx+=2){         
         // grab the subpiece of the src table
         auto src_subcol = cudf::experimental::slice(src_v.column(c_idx), std::vector<size_type>{splits[s_idx], splits[s_idx+1]});         
         
         // make sure it's the same as the output subtable's equivalent column
         size_t subtable_index = s_idx/2;         
         cudf::test::expect_columns_equal(src_subcol[0], dst_tables[subtable_index].table.column(c_idx), true);
         if(verbosity > 0 && (col_count % verbosity == 0)){
            printf("----------------------------\n");            
            print_column(src_subcol[0], false, 20);
            print_column(dst_tables[subtable_index].table.column(c_idx), false, 20);
            printf("----------------------------\n");
         }
         col_count++;        
      }
   }   
}

float frand()
{
   return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 65535.0f;
}

void single_split_test( int64_t total_desired_bytes, 
                        int64_t num_cols,                     
                        int64_t num_rows,
                        int64_t num_splits,
                        bool include_validity)
{
   printf("total data size : %.2f GB\n", (float)total_desired_bytes / (float)(1024 * 1024 * 1024));
   
   srand(31337);

   // generate input columns and table   
   std::vector<std::unique_ptr<column>> columns(num_cols);
   scope_timer_manual src_table_gen("src table gen");
   src_table_gen.start();
   std::vector<bool> all_valid(num_rows);
   if(include_validity){
      for(int idx=0; idx<num_rows; idx++){
         all_valid[idx] = true;
      }
   }   
   std::vector<int> icol(num_rows);
   std::vector<float> fcol(num_rows);
   for(int idx=0; idx<num_cols; idx++){
      if(idx % 2 == 0){                  
         for(int e_idx=0; e_idx<num_rows; e_idx++){
            icol[e_idx] = rand();
         }
         if(include_validity){            
            wrapper<int> cw(icol.begin(), icol.end(), all_valid.begin());            
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         } else {
            wrapper<int> cw(icol.begin(), icol.end());
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         }         
      } else {         
         for(int e_idx=0; e_idx<num_rows; e_idx++){
            fcol[e_idx] = frand();
         }
         if(include_validity){
            wrapper<float> cw(fcol.begin(), fcol.end(), all_valid.begin());
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         } else {
            wrapper<float> cw(fcol.begin(), fcol.end());
            columns[idx] = cw.release();
            columns[idx]->set_null_count(0);
         }         
      }
   }
   cudf::experimental::table src_table(std::move(columns));
   src_table_gen.end();
   printf("# columns : %d\n", (int)num_cols);

   // generate splits   
   int split_stride = num_rows / num_splits;
   std::vector<size_type> splits;  
   scope_timer_manual split_gen("split gen");
   split_gen.start();
   for(int idx=0; idx<num_rows; idx+=split_stride){
      splits.push_back(idx);
      splits.push_back(min(idx + split_stride, (int)num_rows));
   }
   split_gen.end();

   printf("# splits : %d\n", (int)splits.size() / 2);
   /*
   printf("splits : ");
   for(size_t idx=0; idx<splits.size(); idx+=2){
      printf("(%d, %d) ", splits[idx], splits[idx+1]);
   }
   */

   // do the split
   scope_timer_manual split_time("contiguous_split total");
   split_time.start();
   auto dst_tables = contiguous_split(src_table.view(), splits);
   cudaDeviceSynchronize();
   split_time.end();

   scope_timer_manual verify_time("verify_split_results");
   verify_time.start();
   verify_split_results(src_table, dst_tables, splits);
   verify_time.end();

   scope_timer_manual free_time("free buffers");
   free_time.start();   
   for(size_t idx=0; idx<dst_tables.size(); idx++){
      rmm::device_buffer *buf = dst_tables[idx].all_data.release();
      delete buf;   
   }
   cudaDeviceSynchronize();
   free_time.end();   
}

void large_split_tests()
{      
   // single_split_test does ints and floats only
   int el_size = 4;
      
   /*
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 2.00 GB
      // src table gen : 8442.80 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 77.27 (256 allocs)     <------
      //    copy time : 436.92 (131072 copies)  <------
      // contiguous_split total : 524.31 ms     <------
      // verify_split_results : 6763.76 ms
      // free buffers : 0.18 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)2 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   }
   */  
   
   /*
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 2.00 GB
      // src table gen : 9383.00 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 43.93 (256 allocs)     <------
      //    copy time : 413.77 (131072 copies)  <------
      // contiguous_split total : 469.21 ms     <------
      // verify_split_results : 11387.72 ms
      // free buffers : 0.20 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)2 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }
   */  

   /*      
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 4.00 GB
      // src table gen : 16917.02 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 79.27 (256 allocs)     <------
      //    copy time : 454.59 (131072 copies)  <------
      // contiguous_split total : 541.54 ms     <------
      // verify_split_results : 6777.47 ms
      // free buffers : 0.18 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)4 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   } 
   */        

   /* 
   {
      // Tesla T4, 16 GB (all times in milliseconds)
      // total data size : 4.00 GB
      // src table gen : 18649.68 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 47.73 (256 allocs)     <------
      //    copy time : 446.58 (131072 copies)  <------
      // contiguous_split total : 503.26 ms     <------
      // verify_split_results : 11802.98 ms
      // free buffers : 0.26 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)4 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }   
   */
   
   /*
   {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 25230.81 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 48.01 (256 allocs)     <------
      //    copy time : 416.30 (131072 copies)  <------
      // contiguous_split total : 471.48 ms     <------
      // verify_split_results : 53921.47 ms
      // free buffers : 0.20 ms                 <------

      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   } 
   */
   
   /*
   {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 27897.44 ms
      // # columns : 512
      // split gen : 0.00 ms
      // # splits : 256
      //    alloc time : 61.25 (256 allocs)     <------
      //    copy time : 447.05 (131072 copies)  <------
      // contiguous_split total : 517.05 ms     <------
      // verify_split_results : 13794.44 ms
      // free buffers : 0.20 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 512;      
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = num_cols / 2;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }*/ 
   
   /*
   {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 28402.29 ms
      // # columns : 10
      // split gen : 0.01 ms
      // # splits : 257
      //    alloc time : 45.74 (257 allocs)     <------
      //    copy time : 70.60 (2570 copies)     <------
      // contiguous_split total : 116.76 ms     <------
      // verify_split_results : 1962.77 ms
      // free buffers : 0.24 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 10;
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = 256;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, false);
   }
   */

    {
      // Tesla T4, 16 GB
      // total data size : 6.00 GB
      // src table gen : 30930.70 ms
      // # columns : 10
      // split gen : 0.00 ms
      // # splits : 257
      //    alloc time : 42.77 (257 allocs)     <------
      //    copy time : 72.51 (2570 copies)     <------
      // contiguous_split total : 115.61 ms     <------
      // verify_split_results : 2088.58 ms
      // free buffers : 0.25 ms                 <------
 
      // pick some numbers
      int64_t total_desired_bytes = (int64_t)6 * (1024 * 1024 * 1024);
      int64_t num_cols = 10;
      int64_t num_rows = total_desired_bytes / (num_cols * el_size);
      int64_t num_splits = 256;
      single_split_test(total_desired_bytes, num_cols, num_rows, num_splits, true);
   }
}