


/*
 ============================================================================
 Name        : testing-pointer-functions.cu
 Author      : felipe
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */
#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "helper/utils.cuh"

template <typename T> class ReciprocalFunctor {
	public:
	__host__ __device__ T operator()(const T &x) {
		return reciprocal(x);
	}
};

struct shift_left: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	shift_left(gdf_valid_type num_bits): num_bits(num_bits){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
    return x << num_bits;
  }
};

struct shift_right: public thrust::unary_function<gdf_valid_type,gdf_valid_type>
{

	gdf_valid_type num_bits;
	shift_right(gdf_valid_type num_bits): num_bits(num_bits){

	}

  __host__ __device__
  gdf_valid_type operator()(gdf_valid_type x) const
  {
	    //if you want to force the shift to be fill bits with 0 you need to use an unsigned type

	  return *((unsigned char *) &x) >> num_bits;

  }
};



struct bit_or: public thrust::unary_function<thrust::tuple<gdf_valid_type,gdf_valid_type>,gdf_valid_type>
{

  __host__ __device__
  gdf_valid_type operator()(thrust::tuple<gdf_valid_type,gdf_valid_type> x) const
  {
    return thrust::get<0>(x) | thrust::get<1>(x);
  }
};


 

size_t count_valid_bytes(size_t column_size) {
    return sizeof(gdf_valid_type) * (column_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
}

void printValid(thrust::device_vector<gdf_valid_type> output_device_bits, size_t size){
	char * host_valid_out = new char[5];



//			thrust::copy(output_device_bits.begin(),output_device_bits.end(),thrust::detail::make_normal_iterator(thrust::raw_pointer_cast(host_valid_out)));
		size_t n_bytes = count_valid_bytes(size);

        cudaMemcpy(host_valid_out,output_device_bits.data().get(), n_bytes,cudaMemcpyDeviceToHost);

        /*int prevColPosition = 0;
        for(int i = 0; i < size; i++){
            int col_position = i / 8;
            if(prevColPosition != col_position){
                std::cout<<std::endl;
                prevColPosition = col_position;
            }
            int bit_offset = 7 - (i % 8);
            std::cout<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1);

        }*/
        for (int i = 0; i < n_bytes; i++) {
            int length = n_bytes != i+1 ? GDF_VALID_BITSIZE : size - GDF_VALID_BITSIZE * (n_bytes - 1);
            print_binary(host_valid_out[i], length);
        }
        std::cout<<std::endl;
        delete[] host_valid_out;
}

/*using val_tuple = thrust::tuple<float, int >;
struct smaller_tuple {
    val_tuple operator () (val_tuple a, val_tuple b) {
        if ( thrust::get<0>(a) < thrust::get<0>(b) )
            return a;
        else
            return b;
    }
};

int min_index (thrust::device_vector<float> & values) {
    thrust::counting_iterator<int> begin_counting (0);
    thrust::counting_iterator<int> end_counting (values.size());

    val_tuple init (values[0], 0);

    val_tuple smallest = thrust::reduce (
        thrust::make_zip_iterator(thrust::make_tuple(values.begin(), begin_counting)),
        thrust::make_zip_iterator(thrust::make_tuple(values.end(), end_counting)),
        init,
        smaller_tuple() 
    );
    return thrust::get<1>(smallest);
}
*/

size_t  get_last_left_byte_length(size_t column_size) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (n_bytes == 1 ) {
        length = column_size;
    }
    return  length;
}

size_t  get_right_byte_length(size_t column_size, size_t iter, size_t left_length) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (iter == n_bytes - 1) { // the last one
        if (left_length + length > GDF_VALID_BITSIZE) {
            length = GDF_VALID_BITSIZE - left_length;
        }
    }
    else {
        length = GDF_VALID_BITSIZE - left_length;
    }
    return length;
}

 bool last_with_too_many_bits(size_t column_size, size_t iter, size_t left_length) {
    size_t n_bytes = get_number_of_bytes_for_valid(column_size);
    size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
    if (iter == n_bytes) { // the last one
        // the last one has to many bits
        if (left_length + length > GDF_VALID_BITSIZE) {
            return true;
        }
    }
    return false;
}

 gdf_valid_type concat_bins (gdf_valid_type A, gdf_valid_type B, int len_a, int len_b, bool has_next, size_t right_length){
    std::cout << "A | B\n";
    print_binary(A, len_a);
    print_binary(B, len_b);
    std::cout << has_next << "\t" << right_length <<  "\n";

    A = A << len_b;
    if (!has_next) {
        B = B << len_a;
        B = B >> len_a;
    } else {
        B = B >> right_length - len_b;
    }
    std::cout << "A | B\n";
    print_binary(A, len_a);
    print_binary(B, len_b);
    std::cout << "\n";
    return  (A | B);
}

TEST(TestPointerFunctions, CaseMain)
{

	//int GDF_VALID_BITSIZE = 8;

	int left_bits;
	gdf_valid_type * left_bits_char = (gdf_valid_type *) &left_bits;
	left_bits_char[0] = 0b100;
	left_bits_char[1] = 0b00001101; 


	int right_bits;
	gdf_valid_type * right_bits_char = (gdf_valid_type *) &right_bits;
	right_bits_char[0] = 0b1000000;
	right_bits_char[1] = 0b11111111;
	right_bits_char[2] = 0b1101010; 

	//this does not work because right shift fills with 1's to force it to fill with 0's you have to make it unsigned
	print_binary(right_bits_char[0]);
    char test = right_bits_char[0] >> 7; //gives you -1 e.g. 11111111
    print_binary(test);

	test = *((unsigned char *) &right_bits_char[0]) >> 7; //gives you 1 e.g. 00000001
    print_binary(test);

	//used for setting output to null, yo udon't have to do this i just use this for testing purposes
	long long out_bits = 0b0000000000000000000000000000000000000000000000000000000000000000;

	int length_left_bits = 3; //e.g. tienes 15 elementos en la lista aca
	int length_right_bits = 7;

	

	gdf_valid_type shift_bits = (GDF_VALID_BITSIZE - (length_left_bits % GDF_VALID_BITSIZE));
	if(shift_bits == 8){
		shift_bits = 0;
	}
    
	std::cout<<"shift_bits is "<< (int)shift_bits << std::endl;
    
    //number of  bytes
    // int left_num_chars = (length_left_bits + (GDF_VALID_BITSIZE - 1) )/ GDF_VALID_BITSIZE; 
	// int right_num_chars = ((length_right_bits - shift_bits) + (GDF_VALID_BITSIZE - 1) )/ GDF_VALID_BITSIZE;
    
    int left_num_chars = count_valid_bytes(length_left_bits);
    int right_num_chars = count_valid_bytes(length_right_bits);
    

    std::cout << "left num chars : " << left_num_chars << std::endl;
    std::cout << "right num chars : " << right_num_chars << std::endl;

    thrust::device_vector<gdf_valid_type> left_device_bits(left_num_chars); //4 bytes = 32 bits
	thrust::device_vector<gdf_valid_type> right_device_bits(right_num_chars);
	thrust::device_vector<gdf_valid_type> output_device_bits(left_num_chars + right_num_chars);




	thrust::copy((char *) &left_bits, ((char * ) &left_bits) + left_num_chars, left_device_bits.begin());

	//cudaMemcpy(left_device_bits.data().get(),&left_bits,4,cudaMemcpyHostToDevice);

	thrust::copy((char *) &right_bits, ((char * ) &right_bits) + right_num_chars, right_device_bits.begin());
	thrust::copy((char *) &out_bits, ((char * ) &out_bits) + left_num_chars + right_num_chars, output_device_bits.begin());


	std::cout<<"left is "<<std::endl;
    printValid(left_device_bits,length_left_bits);
	std::cout<<"***********************"<<std::endl;

    std::cout<<"right is "<<std::endl;
    printValid(right_device_bits,length_right_bits);
	std::cout<<"***********************"<<std::endl;

	/*//testing trasnform iterator this is junk code (es basura esto)
	std::cout<<"before left copy but with transform test left shift"<<std::endl;
	thrust::transform(right_device_bits.begin(),right_device_bits.begin() + right_num_chars,output_device_bits.begin(),shift_left(shift_bits));
	printValid(output_device_bits,64);
	thrust::copy((char *) &out_bits, ((char * ) &out_bits) + 5, output_device_bits.begin());
	std::cout<<"before left copy but with transform test right shift "<<std::endl;
	thrust::transform(right_device_bits.begin(),right_device_bits.begin() + right_num_chars,output_device_bits.begin(),shift_right(8 - shift_bits));
		printValid(output_device_bits,64);
    */
	
    thrust::copy((char *) &out_bits, ((char * ) &out_bits) + 5, output_device_bits.begin()); //reset it

	//ahora se acaba la basura

	//what we would actually do
	thrust::copy(left_device_bits.begin(),left_device_bits.begin() + left_num_chars, output_device_bits.begin());

	std::cout<<"after left copy is "<<std::endl;
	printValid(output_device_bits, length_left_bits + length_right_bits);
	//now output is 0b1111000011001100************************ where * means we dont know what it has since we havent written it yet
	//and the last 0 is really not a valid thing to look at becuase our length in bits is 15 and this is the 16th bit

	//so we want to take the right bits and shift them to the left, bringing over the bit from the next char
	//to do this we will use a special iterator that shifts


    auto valid_left_length = [](size_t column_size, size_t n_bytes) {
         size_t length = column_size - GDF_VALID_BITSIZE * (n_bytes - 1);
        if (n_bytes == 1 ) {
            length = column_size;
        }
        return  length;
    };
	//now we make a zipped iterator that allows us to access both values we need for this process
    if (right_num_chars == 0) {
        std::cout<<"End is"<<std::endl;
	    printValid(output_device_bits, length_left_bits + length_right_bits);
        return;
    }
	if(shift_bits == 0){
		//we are in luck and its easy we just copy
		//esto debe ser facil solo tienes que copiar
        std::cout << "shift_bits = 0 : " << (int)shift_bits << std::endl;

        //cudaMemcpyAsync(output->valid, rhs->valid, sizeof(gdf_valid_type) * rnbytes, cudaMemcpyDeviceToDevice, stream);
        thrust::copy(right_device_bits.begin(),right_device_bits.end(), output_device_bits.begin() + left_num_chars);

	}else{
        thrust::host_vector<gdf_valid_type> last_byte (2);
        thrust::copy (left_device_bits.end() - 1, left_device_bits.end(), last_byte.begin());
        thrust::copy (right_device_bits.begin(), right_device_bits.begin() + 1, last_byte.begin() + 1);
        size_t last_right_byte_length = length_right_bits - GDF_VALID_BITSIZE * (right_num_chars - 1);
        
        size_t prev_len = get_last_left_byte_length(length_left_bits);
        size_t curr_len = get_right_byte_length(length_right_bits, 0, prev_len);
        auto flag = last_with_too_many_bits(length_right_bits, 0 + 1, prev_len);

        size_t length = length_right_bits - GDF_VALID_BITSIZE * (right_num_chars - 1);
        last_byte[0] = concat_bins(last_byte[0], last_byte[1], prev_len, curr_len, flag, length);

        /*if(right_num_chars > 1)  {
            if (shift_bits + last_right_byte_length > GDF_VALID_BITSIZE) {
                std::cout << "case 1: sum last_bytes: " << (int)last_right_byte_length << std::endl;
                print_binary(last_byte[0] << last_right_byte_length, GDF_VALID_BITSIZE);
                print_binary(last_byte[1], GDF_VALID_BITSIZE);
                last_byte[0] = last_byte[0] << last_right_byte_length | last_byte[1]; 
                std::cout << "equals: " << std::endl;
                print_binary(last_byte[0], GDF_VALID_BITSIZE);

            }
        }
         else {
            std::cout << "case 2: sum last_bytes: " << (int)shift_bits << std::endl;
            print_binary(last_byte[0] << shift_bits, GDF_VALID_BITSIZE);
            print_binary(last_byte[1] >> GDF_VALID_BITSIZE - shift_bits, GDF_VALID_BITSIZE);
            
            last_byte[0] = last_byte[0] << shift_bits | last_byte[1] >> GDF_VALID_BITSIZE - shift_bits; 
            
            std::cout << "equals: " << std::endl;
            print_binary(last_byte[0], GDF_VALID_BITSIZE);
        }*/
        thrust::copy( last_byte.begin(), last_byte.begin() + 1, output_device_bits.begin() + left_num_chars - 1);
        //thrust::copy( last_byte.begin(), last_byte.end(), output_device_bits.begin() + left_num_chars - 1);
        
        if(right_num_chars > 1)  {
            using first_iterator_type = thrust::transform_iterator<shift_left,thrust::device_vector<gdf_valid_type>::iterator>;
            using second_iterator_type = thrust::transform_iterator<shift_right,thrust::device_vector<gdf_valid_type>::iterator>;
            
            using offset_tuple = thrust::tuple<first_iterator_type, second_iterator_type>;
            using zipped_offset = thrust::zip_iterator<offset_tuple>;

            zipped_offset  zipped_offset_iter(
                    thrust::make_tuple(
                            thrust::make_transform_iterator<shift_left, thrust::device_vector<gdf_valid_type>::iterator >(
                                    right_device_bits.begin(),
                                    shift_left(shift_bits)),
                            
                            thrust::make_transform_iterator<shift_right, thrust::device_vector<gdf_valid_type>::iterator >(
                                    right_device_bits.begin() + 1,
                                    shift_right(GDF_VALID_BITSIZE - shift_bits))
                    )
            );

            //so what this does is give you an iterator which gives you a tuple where you have your char, and the char after you, so you can get the last bits!

            using transformed_or = thrust::transform_iterator<bit_or, zipped_offset>;
            //now we want to make a transform iterator that ands these values together
            transformed_or ored_offset_iter =
                    thrust::make_transform_iterator<bit_or,zipped_offset> (
                            zipped_offset_iter,
                            bit_or()
                    );
            //because one of the iterators is + 1 we dont want to read the last char here since it could be past the end of our allocation
            thrust::copy( ored_offset_iter,ored_offset_iter + right_num_chars - 1, output_device_bits.begin() + left_num_chars);

        }
          
	}


	//nowe we need to deal with the ver last char athat was output by the left hand side
	//to do this we will copy the last value from left, the first value from right, shift and process on cpu
	//then copy this one value back to the gpu
	//eastiest way is to take the last char from the left side

	if(right_num_chars > 1 && shift_bits != 0){
        std::cout << "shift_bits: " << (int)shift_bits << std::endl;
        
        thrust::host_vector<gdf_valid_type> last_byte (right_device_bits.end() - 1, right_device_bits.end());
        
        print_binary(last_byte[0], GDF_VALID_BITSIZE - shift_bits);
        print_binary(last_byte[0], GDF_VALID_BITSIZE - shift_bits - shift_bits);

        thrust::copy( right_device_bits.end() - 1,right_device_bits.end(), output_device_bits.begin() + left_num_chars + right_num_chars - 1);

		//handle the edges
		//get first character and last character for right
		//get last character of left

		//do the right shift on the first character of right side, or it to the last character of left
		//copy it back to the gpu int he position it was

		//do the left shift on the last character of the right side
		//copy it back to the gpu in the last character that we need to set e.g. (output_device_bits.dta().get() + left_num_chars + right_num_chars - 1)
	}

	//TODO:

	std::cout<<"End is"<<std::endl;
	printValid(output_device_bits, length_left_bits + length_right_bits);






}