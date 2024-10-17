# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# This function is from the rapidsai/clx repo at below link
# https://github.com/rapidsai/clx/blob/267c6d30805c9dcbf80840f222bf31c5c4b7068a/python/clx/analytics/_perfect_hash.py
import numpy as np

PRIME = np.uint64(281474976710677)

# Coefficients ranges for inner hash - This are important to set to be
# large so that we have randomness in the bottom bits when modding
A_SECOND_LEVEL_POW = np.uint64(48)
B_SECOND_LEVEL_POW = np.uint64(7)

A_LBOUND_SECOND_LEVEL_HASH = 2**16
A_HBOUND_SECOND_LEVEL_HASH = 2**A_SECOND_LEVEL_POW

B_LBOUND_SECOND_LEVEL_HASH = 0
B_HBOUND_SECOND_LEVEL_HASH = 2**B_SECOND_LEVEL_POW

# Extremely generous and should not ever happen. This limit is imposed
# To ensure we can bit pack all the information needed for the bin hash
# functions - a, b and table size
MAX_SIZE_FOR_INITIAL_BIN = 2**8 - 1


# Shifts for bit packing
A_SECOND_LEVEL_SHIFT_AMT = np.uint64(64 - A_SECOND_LEVEL_POW)
B_SECOND_LEVEL_SHIFT_AMT = np.uint64(
    64 - A_SECOND_LEVEL_POW - B_SECOND_LEVEL_POW
)
BITS_FOR_INNER_TABLE_SIZE = np.uint64(8)

NOT_FOUND = -1


def _sdbm_hash(string):
    hv = 0
    mask = (1 << 48) - 1
    for c in string:
        hv = ord(c) + (hv << 6) + (hv << 16) - hv
        hv &= mask
    return hv


def _hash_func(k, a, b, size):
    k = np.uint64(k)
    a = np.uint64(a)
    b = np.uint64(b)
    size = np.uint64(size)
    return ((a * k + b) % PRIME) % size


def _longest_bin_length(bins):
    return len(max(bins, key=len))


def _make_bins(data, num_bins, a, b):
    bins = [[] for i in range(num_bins)]

    for item in data:
        bins[_hash_func(item, a, b, num_bins)].append(item)
    return bins


def _new_bin_length(orig_length):
    return int(orig_length)


def _get_space_util(bins, init_bins):
    return sum(_new_bin_length(len(b)) for b in bins) + 2 * init_bins


def _pick_initial_a_b(data, max_constant, init_bins, rng):
    while True:
        a = rng.integers(2**12, 2**15)
        b = rng.integers(2**12, 2**15)
        bins = _make_bins(data, init_bins, a, b)
        score = _get_space_util(bins, init_bins) / len(data)

        longest = _new_bin_length(_longest_bin_length(bins))

        if score <= max_constant and longest <= MAX_SIZE_FOR_INITIAL_BIN:
            print(f"Attempting to build table using {score:.6f}n space")
            print(f"Longest bin was {longest}")
            break

    return bins, a, b


def _find_hash_for_internal(hash_bin, rng):
    if not hash_bin:
        return [[], 0, 0]

    new_length = _new_bin_length(len(hash_bin))

    while True:
        a = rng.integers(
            A_LBOUND_SECOND_LEVEL_HASH,
            A_HBOUND_SECOND_LEVEL_HASH,
        )
        b = rng.integers(
            B_LBOUND_SECOND_LEVEL_HASH, B_HBOUND_SECOND_LEVEL_HASH
        )
        bins = _make_bins(hash_bin, new_length, a, b)

        max_length = len(max(bins, key=len))
        if max_length == 1:
            bins = [b[0] if b else 0 for b in bins]
            return bins, a, b


def _perfect_hash(integers, max_constant, rng):
    num_top_level_bins = len(integers) // 4

    init_bins, init_a, init_b = _pick_initial_a_b(
        integers, max_constant, num_top_level_bins, rng
    )
    flattened_bins = []

    internal_table_coeffs = np.zeros(
        shape=[num_top_level_bins], dtype=np.uint64
    )
    offset_into_flattened_table = np.zeros(
        shape=[num_top_level_bins + 1], dtype=np.uint64
    )

    max_bin_length = 0
    for i, b in enumerate(init_bins):
        if i % 500 == 0:
            print(f"Processing bin {i} / {len(init_bins)} of size = {len(b)}")
        internal_table, coeff_a, coeff_b = _find_hash_for_internal(b, rng)
        bin_length = len(internal_table)
        max_bin_length = max(bin_length, max_bin_length)
        internal_table_coeffs[i] = (
            np.uint64(coeff_a) << A_SECOND_LEVEL_SHIFT_AMT
            | np.uint64(coeff_b) << B_SECOND_LEVEL_SHIFT_AMT
            | np.uint64(bin_length)
        )
        offset_into_flattened_table[i + 1] = offset_into_flattened_table[
            i
        ] + np.uint64(bin_length)
        flattened_bins.extend(internal_table)

    print(
        "Final table size {} elements compared to {} for original".format(
            len(flattened_bins), len(integers)
        )
    )

    print("Max bin length was", max_bin_length)

    return (
        init_a,
        init_b,
        num_top_level_bins,
        flattened_bins,
        internal_table_coeffs,
        offset_into_flattened_table,
    )


def _pack_keys_and_values(flattened_hash_table, original_dict):
    for i in range(len(flattened_hash_table)):
        if flattened_hash_table[i] in original_dict:
            value = original_dict[flattened_hash_table[i]]
            flattened_hash_table[i] <<= 16
            flattened_hash_table[i] |= value


def _load_vocab_dict(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        counter = 0
        for line in f:
            vocab[line.strip()] = counter
            counter += 1

    return vocab


def _store_func(
    out_name,
    outer_a,
    outer_b,
    num_outer_bins,
    hash_table,
    inner_table_coeffs,
    offsets_into_ht,
    unk_tok_id,
    first_token_id,
    sep_token_id,
):
    with open(out_name, mode="w+") as f:
        f.write(f"{outer_a}\n")
        f.write(f"{outer_b}\n")
        f.write(f"{num_outer_bins}\n")
        f.writelines(
            f"{coeff} {offset}\n"
            for coeff, offset in zip(inner_table_coeffs, offsets_into_ht)
        )
        f.write(f"{len(hash_table)}\n")
        f.writelines(f"{kv}\n" for kv in hash_table)
        f.writelines(
            f"{tok_id}\n"
            for tok_id in [unk_tok_id, first_token_id, sep_token_id]
        )


def _retrieve(
    k,
    outer_a,
    outer_b,
    num_outer_bins,
    hash_table,
    inner_table_coeffs,
    offsets_into_ht,
):
    bin_hash = _hash_func(k, outer_a, outer_b, num_outer_bins)
    start_offset_in_ht = offsets_into_ht[bin_hash]
    inner_table_values = inner_table_coeffs[bin_hash]

    one = np.uint64(1)

    inner_a = inner_table_values >> A_SECOND_LEVEL_SHIFT_AMT
    inner_b = (inner_table_values >> B_SECOND_LEVEL_SHIFT_AMT) & (
        (one << B_SECOND_LEVEL_POW) - one
    )
    size = inner_table_values & ((one << BITS_FOR_INNER_TABLE_SIZE) - one)

    inner_offset = _hash_func(k, inner_a, inner_b, size)
    kv = hash_table[start_offset_in_ht + inner_offset]

    key, value = kv >> 16, kv & ((1 << 16) - 1)
    indicator = key == k

    return indicator * value + (not indicator) * NOT_FOUND


def hash_vocab(
    vocab_path,
    output_path,
    unk_tok="[UNK]",
    first_token="[CLS]",
    sep_token="[SEP]",
):
    """
    Write the vocab vocabulary hashtable to the output_path
    """
    rng = np.random.default_rng(seed=1243342)
    vocab = _load_vocab_dict(vocab_path)
    keys = list(map(_sdbm_hash, vocab.keys()))

    hashed_vocab = {_sdbm_hash(key): value for key, value in vocab.items()}

    error_message = (
        "A collision occurred and only sdbm token hash is currently "
        "supported. This can be extended to use random hashes if needed."
    )
    assert len(hashed_vocab) == len(vocab), error_message

    (
        outer_a,
        outer_b,
        num_outer_bins,
        hash_table,
        inner_table_coeffs,
        offsets_into_ht,
    ) = _perfect_hash(keys, 10, rng)

    _pack_keys_and_values(hash_table, hashed_vocab)
    _store_func(
        output_path,
        outer_a,
        outer_b,
        num_outer_bins,
        hash_table,
        inner_table_coeffs,
        offsets_into_ht,
        vocab[unk_tok],
        vocab[first_token],
        vocab[sep_token],
    )

    for key, value in hashed_vocab.items():
        val = _retrieve(
            key,
            outer_a,
            outer_b,
            num_outer_bins,
            hash_table,
            inner_table_coeffs,
            offsets_into_ht,
        )
        assert (
            val == value
        ), f"Incorrect value found. Got {val} expected {value}"

    print("All present tokens return correct value.")
