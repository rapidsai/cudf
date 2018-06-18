
#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>


#include <cuda_runtime.h>
#include <vector>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/device_vector.h>




/*
 * Brainstorming
 *
 * functions needed:
 * extract_datetime_year
 * extract_datetime_month
 * extract_datetime_day
 * extract_datetime_hour
 * extract_datetime_minute
 * extract_datetime_second
 *
 * input formats:
 * date64
 * date32
 * timestamp with second, milisecond, microsecond, nanosecond
 *
 * these functions depend on starting dateformat may need to just implement them all?
 *
 *
 */


/* original unittime to date functions
 *
 * int unixDate = unixTime/86400000;
		int totalDays = 719469 + unixDate;

		int year = 400*totalDays/146097;
		totalDays -= (365*year + year/4 - year/100 + year/400);
		int month = (totalDays*5 + 457)/153;

		int day = totalDays - ((153*month - 457)/5);

		if (day == 0){
			month--;
			day = totalDays - ((153*month - 457)/5);
		}

		if(month > 12){
			month -= 12;
			year++;
		}

		another one more complex
 *
 * from http://howardhinnant.github.io/date_algorithms.html
 * int z = unixTime/86400000;
 * z += 719468;
    const Int era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);          // [0, 146096]
    const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;  // [0, 399]
    const Int y = static_cast<Int>(yoe) + era * 400;
    const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);                // [0, 365]
    const unsigned mp = (5*doy + 2)/153;                                   // [0, 11]
    const unsigned d = doy - (153*mp+2)/5 + 1;                             // [1, 31]
    const unsigned m = mp + (mp < 10 ? 3 : -9);                            // [1, 12]
    return std::tuple<Int, unsigned, unsigned>(y + (m <= 2), m, d);
 */






struct gdf_extract_datetime_year_date64_op : public thrust::unary_function<int64_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - 86399999) / 86400000) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);          // [0, 146096]
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;  // [0, 399]
		const int y = static_cast<int>(yoe) + era * 400;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);                // [0, 365]
		const unsigned mp = (5*doy + 2)/153;                                   // [0, 11]
		const unsigned d = doy - (153*mp+2)/5 + 1;                             // [1, 31]
		const unsigned m = mp + (mp < 10 ? 3 : -9);                            // [1, 12]
		if (m <= 2)
			return y + 1;
		else
			return y;
	}
};

struct gdf_extract_datetime_month_date64_op : public thrust::unary_function<int64_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - 86399999) / 86400000) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);          // [0, 146096]
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;  // [0, 399]
		const int y = static_cast<int>(yoe) + era * 400;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);                // [0, 365]
		const unsigned mp = (5*doy + 2)/153;                                   // [0, 11]
		const unsigned d = doy - (153*mp+2)/5 + 1;                             // [1, 31]
		return mp + (mp < 10 ? 3 : -9);                            // [1, 12]

	}
};

struct gdf_extract_datetime_day_date64_op : public thrust::unary_function<int64_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		const int z = ((unixTime >= 0 ? unixTime : unixTime - 86399999) / 86400000) + 719468;
		const int era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = static_cast<unsigned>(z - era * 146097);          // [0, 146096]
		const unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;  // [0, 399]
		const int y = static_cast<int>(yoe) + era * 400;
		const unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);                // [0, 365]
		const unsigned mp = (5*doy + 2)/153;                                   // [0, 11]
		return doy - (153*mp+2)/5 + 1;                             // [1, 31]
	}
};

/*

1528996790 unix time
5:19:50 pm UTC | Thursday, June 14, 2018

1528935590000
'2018-06-14 00:19:50'
 */

struct gdf_extract_datetime_hour_date64_op : public thrust::unary_function<int64_t, int16_t>
{
	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return (unixTime % 86400000)/3600000;
	}
};

struct gdf_extract_datetime_minute_date64_op : public thrust::unary_function<int64_t, int16_t>
{
	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return (unixTime % 3600000)/60000 ;
	}
};

struct gdf_extract_datetime_second_date64_op : public thrust::unary_function<int64_t, int16_t>
{
	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return (unixTime % 60000)/1000;
	}
};

struct gdf_extract_datetime_millisecond_date64_op : public thrust::unary_function<int64_t, int16_t>
{
	__host__ __device__
	int16_t operator()(int64_t unixTime) // unixTime is milliseconds since the UNIX epoch
	{
		return unixTime % 1000;
	}
};

struct gdf_extract_datetime_year_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		int totalDays = 719469 + unixDate;
		int year = 400*totalDays/146097;
		totalDays -= (365*year + year/4 - year/100 + year/400);
		int month = (totalDays*5 + 457)/153;
		if ((totalDays - ((153*month - 457)/5)) == 0){
			month--;
		}
		if(month > 12){
			year++;
		}
		return year;
	}
};

struct gdf_extract_datetime_month_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		int totalDays = 719469 + unixDate;
		int year = 400*totalDays/146097;
		totalDays -= (365*year + year/4 - year/100 + year/400);
		int month = (totalDays*5 + 457)/153;
		if ((totalDays - ((153*month - 457)/5)) == 0){
			month--;
		}
		if(month > 12){
			month -= 12;
		}
		return month;
	}
};

struct gdf_extract_datetime_day_date32_op : public thrust::unary_function<int32_t, int16_t>
{

	__host__ __device__
	int16_t operator()(int32_t unixDate) // unixDate is days since the UNIX epoch
	{
		int totalDays = 719469 + unixDate;

		int year = 400*totalDays/146097;
		totalDays -= (365*year + year/4 - year/100 + year/400);
		int month = (totalDays*5 + 457)/153;

		int day = totalDays - ((153*month - 457)/5);

		if (day == 0){
			month--;
			day = totalDays - ((153*month - 457)/5);
		}
		return day;
	}
};





gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output) {

	GDF_REQUIRE(input->size == output->size, GDF_COLUMN_SIZE_MISMATCH);
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);  // WSM do we want extracted values to be other than 16-bit?

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// WSM DO WE NEED TO DO THIS? DO WE WANT TO DO IT THIS WAY?
	gdf_size_type num_chars_bitmask = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::copy(thrust::cuda::par.on(stream), input->valid, input->valid + num_chars_bitmask, output->valid); // copy over valid bitmask

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_year_date64_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_year_date32_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
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
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);  // WSM do we want extracted values to be other than 16-bit?

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// WSM DO WE NEED TO DO THIS? DO WE WANT TO DO IT THIS WAY?
	gdf_size_type num_chars_bitmask = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::copy(thrust::cuda::par.on(stream), input->valid, input->valid + num_chars_bitmask, output->valid); // copy over valid bitmask

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_month_date64_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_month_date32_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
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
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);  // WSM do we want extracted values to be other than 16-bit?

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// WSM DO WE NEED TO DO THIS? DO WE WANT TO DO IT THIS WAY?
	gdf_size_type num_chars_bitmask = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::copy(thrust::cuda::par.on(stream), input->valid, input->valid + num_chars_bitmask, output->valid); // copy over valid bitmask

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_day_date64_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	}else if (input->dtype == GDF_DATE32) {
		thrust::device_ptr<int32_t> input_ptr((int32_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_day_date32_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
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
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);  // WSM do we want extracted values to be other than 16-bit?
	GDF_REQUIRE(input->dtype != GDF_DATE32, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// WSM DO WE NEED TO DO THIS? DO WE WANT TO DO IT THIS WAY?
	gdf_size_type num_chars_bitmask = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::copy(thrust::cuda::par.on(stream), input->valid, input->valid + num_chars_bitmask, output->valid); // copy over valid bitmask

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_hour_date64_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
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
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);  // WSM do we want extracted values to be other than 16-bit?
	GDF_REQUIRE(input->dtype != GDF_DATE32, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// WSM DO WE NEED TO DO THIS? DO WE WANT TO DO IT THIS WAY?
	gdf_size_type num_chars_bitmask = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::copy(thrust::cuda::par.on(stream), input->valid, input->valid + num_chars_bitmask, output->valid); // copy over valid bitmask

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_minute_date64_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
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
	GDF_REQUIRE(output->dtype == GDF_INT16, GDF_UNSUPPORTED_DTYPE);  // WSM do we want extracted values to be other than 16-bit?
	GDF_REQUIRE(input->dtype != GDF_DATE32, GDF_UNSUPPORTED_DTYPE);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// WSM DO WE NEED TO DO THIS? DO WE WANT TO DO IT THIS WAY?
	gdf_size_type num_chars_bitmask = ( ( input->size +( GDF_VALID_BITSIZE - 1)) / GDF_VALID_BITSIZE );
	thrust::copy(thrust::cuda::par.on(stream), input->valid, input->valid + num_chars_bitmask, output->valid); // copy over valid bitmask

	if ( input->dtype == GDF_DATE64 ) {
		thrust::device_ptr<int64_t> input_ptr((int64_t *) input->data);
		thrust::device_ptr<int16_t> output_ptr((int16_t *) output->data);
		gdf_extract_datetime_second_date64_op op;
		thrust::transform(thrust::cuda::par.on(stream), thrust::detail::make_normal_iterator(input_ptr),
				thrust::detail::make_normal_iterator(input_ptr) + input->size, thrust::detail::make_normal_iterator(output_ptr), op);

	} else {
		return GDF_UNSUPPORTED_DTYPE;
	}

	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	return GDF_SUCCESS;
}



