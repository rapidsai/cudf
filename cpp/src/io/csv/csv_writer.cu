/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/**
 * @file csv_writer.cu  code to create csv file
 *
 * CSV Writer
 */

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvstrings/NVStrings.h>
#include "cudf.h"
#include "utilities/error_utils.hpp"

// called by write_csv method
NVStrings* column_to_strings_csv( gdf_column* col, const char* delimiter, const char* strue, const char* sfalse )
{
    NVStrings* rtn = 0;
    unsigned int rows = (unsigned int)col->size;
    unsigned char* valid = (unsigned char*)col->valid;
    switch( col->dtype )
    {
        case GDF_STRING:
            rtn = ((NVStrings*)col->data)->copy();
            break;
        case GDF_INT8:
            rtn = NVStrings::create_from_bools((const bool*)col->data,rows,strue,sfalse,valid);
            break;
        case GDF_INT32:
            rtn = NVStrings::itos((const int*)col->data,rows,valid);
            break;
        case GDF_INT64:
            rtn = NVStrings::ltos((const long*)col->data,rows,valid);
            break;
        case GDF_FLOAT32:
            rtn = NVStrings::ftos((const float*)col->data,rows,valid);
            break;
        case GDF_FLOAT64:
            rtn = NVStrings::dtos((const double*)col->data,rows,valid);
            break;
        case GDF_DATE64:
            rtn = NVStrings::long2timestamp((const unsigned long*)col->data,rows,NVStrings::seconds,valid);
            break;
        default:
            break; // should not happen
    }
    if( !rtn )
        return 0;

    // probably could collapse this more
    bool bquoted = (col->dtype==GDF_STRING);
    // check for delimeters and quotes
    bool* bmatches = 0;
    cudaMalloc(&bmatches,rows*sizeof(bool));
    if( rtn->contains("\"",bmatches) > 0 )
    {
        NVStrings* esc = rtn->replace("\"","\"\"");
        NVStrings::destroy(rtn);
        rtn = esc;
    }
    else if( rtn->contains(",",bmatches) > 0 )
        bquoted = true;
    cudaFree(bmatches);
    if( bquoted )
    {
        // prepend and append quotes
        NVStrings* pre = rtn->slice_replace("\"",0,0);
        NVStrings::destroy(rtn);
        rtn = pre->slice_replace("\"",-1,-1);
        NVStrings::destroy(pre);
    }
    if( delimiter && *delimiter )
    {
        NVStrings* dstr = rtn->slice_replace(delimiter,-1,-1);
        NVStrings::destroy(rtn);
        rtn = dstr;
    }
    return rtn;
}

//
// The args structure is interpretted as documented in io_types.h
// This will create a CSV format by allocating host memory for the
// entire output and determine pointers for each row/column entry.
// Each column is converted to an NVStrings instance and then
// copied into their position in the output memory. This way,
// one column is processed at a time minimizing device memory usage.
//
gdf_error write_csv(csv_write_arg* args)
{
    gdf_error rc = gdf_error::GDF_SUCCESS;

    gdf_column** columns = args->data;
    unsigned int count = (unsigned int)args->num_cols;
    unsigned int rows = (unsigned int)columns[0]->size;
    const char* filepath = args->filepath;
    char delimiter[2] = {',','\0'};
    if( args->delimiter )
        delimiter[0] = args->delimiter;
    char terminator[3] = {'\n','\0','\0'};
    if( args->windows_line )
    {
        terminator[0] = '\r';
        terminator[1] = '\n';
    }
    else if( args->line_terminator )
        terminator[0] = args->line_terminator;
    const char* true_value = (args->true_value ? args->true_value : "true");
    const char* false_value = (args->false_value ? args->false_value : "false");

    // check for issues here
    if( filepath==0 )
    {
        std::cerr << "write_csv: filepath not specified\n";
        return GDF_INVALID_API_CALL;
    }
    if( count==0 )
    {
        std::cerr << "write_csv: num_cols is required\n";
        return GDF_INVALID_API_CALL;
    }
    if( columns==0 )
    {
        std::cerr << "write_csv: invalid data values\n";
        return GDF_INVALID_API_CALL;
    }

    // check all columns are the same size
    for( int idx=0; idx < args->num_cols; ++idx )
    {
        gdf_column* col = args->data[idx];
        NVStrings* strs = 0;
        //printf("%d: %p %s %d %d\n", idx, col, col->col_name, (int)col->dtype, (int)col->size);
        switch( col->dtype )
        {
            case GDF_INT8:
            case GDF_INT32:
            case GDF_INT64:
            case GDF_FLOAT32:
            case GDF_FLOAT64:
            case GDF_DATE64:
                if( (unsigned int)col->size != rows )
                {
                    std::cerr << "write_csv: columns sizes do not match (" << (int)col->size << "!=" << rows << ")\n";
                    return GDF_COLUMN_SIZE_MISMATCH;
                }
                break;
            case GDF_STRING:
                strs = (NVStrings*)col->data;
                if( strs->size() != rows )
                {
                    std::cerr << "write_csv: columns sizes do not match (" << strs->size() << "!=" << rows << ")\n";
                    return GDF_COLUMN_SIZE_MISMATCH;
                }
                break;
            default:
                std::cerr << "write_csv: unknown column type: " << (int)col->dtype << "\n";
                return GDF_UNSUPPORTED_DTYPE;
        }
    }

    // check the file can be written
    FILE* fh = fopen(filepath,"wb");
    if( !fh )
    {
        std::cerr << "write_csv: file [" << filepath << "] could not be opened\n";
        return GDF_FILE_ERROR;
    }

    //
    // It would be good if we could chunk this.
    // Use the rows*count calculation and a memory threshold to
    // output a subset of rows at a time instead of the whole thing at once.
    // The entire CSV must fit in CPU memory before writing it out.
    //
    // compute sizes we need
    // build a matrix of pointers
    int* datalens = new int[rows*count]; // matrix
    size_t memsize = 0;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        gdf_column* col = columns[idx];
        const char* delim = ((idx+1)<count ? delimiter : terminator);
        NVStrings* sdata = column_to_strings_csv(col,delim,true_value,false_value);
        if( sdata )
            memsize += sdata->byte_count(datalens + (idx*rows),false);
        if( sdata )
            NVStrings::destroy(sdata);
    }

    // cols/rows are transposed in this diagram
    // datalens
    //    1,  1,  2, 11, 12,  7,  7 =  41
    //    1,  1,  2,  2,  3,  7,  6 =  22
    //   20, 20, 20, 20, 20, 20, 20 = 140
    //    5,  6,  4,  6,  4,  4,  5 =  34
    //   --------------------------------
    //   27, 28, 28, 39, 39, 38, 38 = 237
    //
    //
    // this is a vertical-scan plus carry:
    // dataptrs
    //     0,  27,  55,  83, 122, 161, 199
    //     1,  28,  57,  94, 134, 168, 206
    //     2,  29,  59,  96, 137, 175, 213
    //    22,  49,  79, 116, 157, 195, 233
    //
    // looks like if we transposed, this would be exclusive-scan but then we have to untranspose;
    // so it may be better to fixup in place
    //
    //printf("memsize=%ld\n",memsize);
    char* buffer = new char[memsize+1];
    memset(buffer,',',memsize); // fill with commas
    size_t* dataptrs = new size_t[rows*count]; // this will hold all the memory pointers for each column
    dataptrs[0] = 0; // first one is always 0
    // compute offsets into dataptrs
    // need figure out a good way to parallelize this
    // this is just math and could be done on the GPU
    size_t offset = 0;
    for( unsigned int jdx=0; jdx < rows; ++jdx )
    {
        // add up column values for each row
        // this is essentially an exclusive-scan
        dataptrs[jdx] = (size_t)(buffer + offset); // initialize first item
        for( unsigned int idx=0; idx < count; ++idx )
        {
            int* in = datalens + (idx*rows);
            int len = in[jdx];
            offset += (len>0 ? len:0);
            if( (idx+1) < count )
            {
                size_t* out = dataptrs + ((idx+1)*rows);
                out[jdx] = (size_t)(buffer + offset);
            }
        }
    }

    // now fill in the memory one column at a time
    for( unsigned int idx=0; idx < count; ++idx )
    {
        gdf_column* col = columns[idx];
        const char* delim = ((idx+1)<count ? delimiter : terminator);
        NVStrings* sdata = column_to_strings_csv(col,delim,true_value,false_value);
        if( sdata )
        {
            size_t* colptrs = dataptrs + (idx*rows);
            // to_host places all the strings into their correct positions in host memory
            sdata->to_host((char**)colptrs,0,rows);
            NVStrings::destroy(sdata);
        }
    }
    buffer[memsize] = 0; // just so we can printf
    //printf("\n%s\n",buffer);
    delete datalens;
    delete dataptrs;
    // write buffer to file
    // first write the header
    for( unsigned int idx=0; idx < count; ++idx )
    {
        gdf_column* col = columns[idx];
        const char* delim = ((idx+1)<count ? delimiter : terminator);
        if( col->col_name )
            fprintf(fh,"\"%s\"",col->col_name);
        fprintf(fh,"%s",delim);
    }
    // now write the data
    fwrite(buffer,memsize,1,fh);
    fclose(fh);
    delete buffer;
    return rc;
}
