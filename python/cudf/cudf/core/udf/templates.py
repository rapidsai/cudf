
unmasked_input_initializer_template = """\
        d_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], True)
"""

masked_input_initializer_template = """\
        d_{idx}, m_{idx} = input_col_{idx}
        masked_{idx} = Masked(d_{idx}[i], mask_get(m_{idx}, i + offset_{idx}))
"""

row_initializer_template = """\
        row["{name}"] = masked_{idx}
"""

row_kernel_template = """\
def _kernel(retval, size, {input_columns}, {input_offsets}, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    if i < size:
        # Create a structured array with the desired fields
        rows = cuda.local.array(1, dtype=row_type)

        # one element of that array
        row = rows[0]

{masked_input_initializers}
{row_initializers}

        # pass the assembled row into the udf
        ret = f_(row, {extra_args})

        # pack up the return values and set them
        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
"""

lambda_kernel_template = """
def _kernel(retval, size, input_col, input_offset, {extra_args}):
    i = cuda.grid(1)
    ret_data_arr, ret_mask_arr = retval
    input_data_arr, input_mask_arr = input_col
    
    if i < size:
        data = input_data_arr[i]
        mask = mask_get(input_mask_arr, i + input_offset)
        
        masked = Masked(data, mask)

        ret = f_(masked, {extra_args})

        ret_masked = pack_return(ret)
        ret_data_arr[i] = ret_masked.value
        ret_mask_arr[i] = ret_masked.valid
"""
