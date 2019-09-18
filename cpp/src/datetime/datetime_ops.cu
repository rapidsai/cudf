/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 William Scott Malpica <william@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <rmm/thrust_rmm_allocator.h>

/*  Portions of the code below is borrowed from a paper by Howard Hinnant dated 2013-09-07  http://howardhinnant.github.io/date_algorithms.html  as seen on July 2nd, 2018
 The piece of code borrowed and modified is:

 **************************************************************************************
// Returns year/month/day triple in civil calendar
// Preconditions:  z is number of days since 1970-01-01 and is in the range:
//                   [numeric_limits<Int>::min(), numeric_limits<Int>::max()-719468].
template <class Int>
constexpr
std::tuple<Int, unsigned, unsigned>
civil_from_days(Int z) noexcept
{
    static_assert(std::numeric_limits<unsigned>::digits >= 18,
             "This algorithm has not been ported to a 16 bit unsigned integer");
    static_assert(std::numeric_limits<Int>::digits >= 20,
             "This algorithm has not been ported to a 16 bit signed integer");
    z += 719468;
    const Int era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);          // [0, 146096]
    const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;  // [0, 399]
    const Int y = static_cast<Int>(yoe) + era * 400;
    const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);                // [0, 365]
    const unsigned mp = (5*doy + 2)/153;                                   // [0, 11]
    const unsigned d = doy - (153*mp+2)/5 + 1;                             // [1, 31]
    const unsigned m = mp + (mp < 10 ? 3 : -9);                            // [1, 12]
    return std::tuple<Int, unsigned, unsigned>(y + (m <= 2), m, d);
}
******************************************************************************************
 */


struct gdf_extract_year_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{

	int64_t units_per_day;
	__host__ __device__
	gdf_extract_year_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s)   // second
			units_per_day = 86400;
		else if (unit == TIME_UNIT_ms)   // millisecond
			units_per_day = 86400000;
		else if (unit == TIME_UNIT_us)   // microsecond
			units_per_day = 86400000000;
		else if (unit == TIME_UNIT_ns)   // nanosecond
			units_per_day = 86400000000000;
		else
			units_per_day = 86400000;   // default to millisecond
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - (units_per_day - 1)) / units_per_day) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const int y = static_cast<int>(yoe) + era * 400;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		const unsigned m = mp + (mp < 10 ? 3 : -9);
		if (m <= 2)
			return y + 1;
		else
			return y;
	}
};

struct gdf_extract_month_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{
	int64_t units_per_day;
	__host__ __device__
	gdf_extract_month_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s)   // second
			units_per_day = 86400;
		else if (unit == TIME_UNIT_ms)   // millisecond
			units_per_day = 86400000;
		else if (unit == TIME_UNIT_us)   // microsecond
			units_per_day = 86400000000;
		else if (unit == TIME_UNIT_ns)   // nanosecond
			units_per_day = 86400000000000;
		else
			units_per_day = 86400000;   // default to millisecond
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - (units_per_day - 1)) / units_per_day) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		return mp + (mp < 10 ? 3 : -9);
	}
};

struct gdf_extract_day_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{

	int64_t units_per_day;
	__host__ __device__
	gdf_extract_day_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s)   // second
			units_per_day = 86400;
		else if (unit == TIME_UNIT_ms)   // millisecond
			units_per_day = 86400000;
		else if (unit == TIME_UNIT_us)   // microsecond
			units_per_day = 86400000000;
		else if (unit == TIME_UNIT_ns)   // nanosecond
			units_per_day = 86400000000000;
		else
			units_per_day = 86400000;   // default to millisecond
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - (units_per_day - 1)) / units_per_day) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		return doy - (153*mp+2)/5 + 1;
	}
};

struct gdf_extract_weekday_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{

	int64_t units_per_day;
	__host__ __device__
	gdf_extract_weekday_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s)   // second
			units_per_day = 86400;
		else if (unit == TIME_UNIT_ms)   // millisecond
			units_per_day = 86400000;
		else if (unit == TIME_UNIT_us)   // microsecond
			units_per_day = 86400000000;
		else if (unit == TIME_UNIT_ns)   // nanosecond
			units_per_day = 86400000000000;
		else
			units_per_day = 86400000;   // default to millisecond
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - (units_per_day - 1)) / units_per_day) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		const unsigned d =  doy - (153*mp+2)/5 + 1;
		int m = mp + (mp < 10 ? 3 : -9);
		int y = static_cast<int>(yoe) + era * 400;

		// apply Zeller's algorithm		
		if (m <= 2) {
			y += 1;
		}

		if (m == 1) {
			m = 13;
			y -= 1;
		}

		if (m == 2) {
			m = 14;
			y-= 1;
		}

		const unsigned k = y % 100;
		const unsigned j = y / 100;
		const unsigned h = (d + 13*(m+1)/5 + k + k/4 + j/4 + 5*j) % 7; 

		return (h - 2 + 7) % 7; // pandas convention Monday = 0

	}
};

struct gdf_extract_hour_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{
	int64_t units_per_day;
	int64_t units_per_hour;
	__host__ __device__
	gdf_extract_hour_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s) {   // second
			units_per_day = 86400;
			units_per_hour = 3600;
		} else if (unit == TIME_UNIT_ms) {   // millisecond
			units_per_day = 86400000;
			units_per_hour = 3600000;
		} else if (unit == TIME_UNIT_us) {  // microsecond
			units_per_day = 86400000000;
			units_per_hour = 3600000000;
		} else if (unit == TIME_UNIT_ns) {  // nanosecond
			units_per_day = 86400000000000;
			units_per_hour = 3600000000000;
		} 	else {
			units_per_day = 86400000;   // default to millisecond
			units_per_hour = 3600000;
		}
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return unixTime >= 0 ? ((unixTime % units_per_day)/units_per_hour) : ((units_per_day + (unixTime % units_per_day))/units_per_hour);
	}
};

struct gdf_extract_minute_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{
	int64_t units_per_hour;
	int64_t units_per_minute;
	__host__ __device__
	gdf_extract_minute_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s) {   // second
			units_per_hour = 3600;
			units_per_minute = 60;
		} else if (unit == TIME_UNIT_ms) {   // millisecond
			units_per_hour = 3600000;
			units_per_minute = 60000;
		} else if (unit == TIME_UNIT_us) {  // microsecond
			units_per_hour = 3600000000;
			units_per_minute = 60000000;
		} else if (unit == TIME_UNIT_ns) {  // nanosecond
			units_per_hour = 3600000000000;
			units_per_minute = 60000000000;
		} 	else {  // default to millisecond
			units_per_hour = 3600000;
			units_per_minute = 60000;
		}
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return unixTime >= 0 ? ((unixTime % units_per_hour)/units_per_minute) :  ((units_per_hour + (unixTime % units_per_hour))/units_per_minute);
	}
};

struct gdf_extract_second_from_unixtime_op : public thrust::unary_function<int64_t, int16_t>
{
	int64_t units_per_minute;
	int64_t units_per_second;
	__host__ __device__
	gdf_extract_second_from_unixtime_op(gdf_time_unit unit){
		if (unit == TIME_UNIT_s) {   // second
			units_per_minute = 60;
			units_per_second = 1;
		} else if (unit == TIME_UNIT_ms) {   // millisecond
			units_per_minute = 60000;
			units_per_second = 1000;
		} else if (unit == TIME_UNIT_us) {  // microsecond
			units_per_minute = 60000000;
			units_per_second = 1000000;
		} else if (unit == TIME_UNIT_ns) {  // nanosecond
			units_per_minute = 60000000000;
			units_per_second = 1000000000;
		} 	else {  // default to millisecond
			units_per_minute = 60000;
			units_per_second = 1000;
		}
	}

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return unixTime >= 0 ? ((unixTime % units_per_minute)/units_per_second) : ((units_per_minute + (unixTime % units_per_minute))/units_per_second);
	}
};


struct gdf_extract_year_from_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		const int z = unixDate + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const int y = static_cast<int>(yoe) + era * 400;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		const unsigned m = mp + (mp < 10 ? 3 : -9);
		if (m <= 2)
			return y + 1;
		else
			return y;
	}
};

struct gdf_extract_month_from_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		const int z = unixDate + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;

		return mp + (mp < 10 ? 3 : -9);
	}
};

struct gdf_extract_day_from_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		const int z = unixDate + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		return doy - (153*mp+2)/5 + 1;
	}
};


struct gdf_extract_weekday_from_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		const int z = unixDate + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);
		const unsigned mp = (5*doy + 2)/153;
		const unsigned d = doy - (153*mp+2)/5 + 1;
		int m = mp + (mp < 10 ? 3 : -9);
		int y = static_cast<int>(yoe) + era * 400;

		// apply Zeller's algorithm
		if (m <= 2) {
			y += 1;
		}

		if (m == 1) {
			m = 13;
			y -= 1;
		}

		if (m == 2) {
			m = 14;
			y-= 1;
		}

		const unsigned k = y % 100;
		const unsigned j = y / 100;
		const unsigned h = (d + 13*(m+1)/5 + k + k/4 + j/4 + 5*j) % 7; 

		return (h - 2 + 7) % 7; // pandas convention Monday = 0

	}
};


gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(input->size);
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_year_from_unixtime_op op(TIME_UNIT_ms);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_year_from_date32_op op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_year_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}

gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_month_from_unixtime_op op(TIME_UNIT_ms);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_month_from_date32_op op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_month_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}

gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }
	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_day_from_unixtime_op op(TIME_UNIT_ms);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_day_from_date32_op op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_day_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}

gdf_error gdf_extract_datetime_weekday(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = gdf_num_bitmask_elements(input->size);
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }
	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_weekday_from_unixtime_op op(TIME_UNIT_ms);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_weekday_from_date32_op op;
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_weekday_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}

gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);
	GDF_REQUIRE(input->dtype != GDF_DATE32, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_hour_from_unixtime_op op(TIME_UNIT_ms);;
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_hour_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}

gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);
	GDF_REQUIRE(input->dtype != GDF_DATE32, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }
	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_minute_from_unixtime_op op(TIME_UNIT_ms);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_minute_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}

gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);
	GDF_REQUIRE(input->dtype != GDF_DATE32, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);


    if (input->valid){
      gdf_size_type num_bitmask_elements = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
      thrust::copy(rmm::exec_policy(stream)->on(stream), input->valid, input->valid + num_bitmask_elements, output->valid); // copy over valid bitmask
    }
	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_second_from_unixtime_op op(TIME_UNIT_ms);;
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_TIMESTAMP) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_second_from_unixtime_op op(input->dtype_info.time_unit);
		thrust::transform(rmm::exec_policy(stream)->on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}
