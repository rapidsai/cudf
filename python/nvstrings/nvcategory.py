import nvstrings as nvs
import pyniNVCategory


def to_device(strs):
    """
    Create a nvcategory object from a list of Python strings.

    Parameters
    ----------
    strs : list
        List of Python strings.

    Examples
    --------
    >>> import nvcategory
    >>> c = nvcategory.to_device(['apple','pear','banana','orange','pear'])
    >>> print(c.keys(),c.values())
    ['apple', 'banana', 'orange', 'pear'] [0, 3, 1, 2, 3]

    """
    rtn = pyniNVCategory.n_createCategoryFromHostStrings(strs)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_offsets(sbuf, obuf, scount, nbuf=None, ncount=0, bdevmem=False):
    """
    Create nvcategory object from byte-array of characters encoded in UTF-8.

    Parameters
    ----------
    sbuf : CPU memory address or buffer
        Strings characters encoded as UTF-8.
    obuf : CPU memory address or buffer
        Array of int32 byte offsets to beginning of each string in sbuf.
        There should be scount+1 values where the last value is the
        number of bytes in sbuf.
    scount : int
        Number of strings.
    nbuf : CPU memory address or buffer, optional
        Optional null bitmask in arrow format.
        Strings with no lengths are empty strings unless specified as
        null by this bitmask.
    ncount : int, optional
        Optional number of null strings (the default is 0).
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    Examples
    --------
    >>> import numpy as np
    >>> import nvcategory

    Create numpy array for 'a','p','p','l','e' with the utf8 int8 values
    of 97,112,112,108,101

    >>> values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
    >>> print("values",values.tobytes())
    values b'apple'
    >>> offsets = np.array([0,1,2,3,4,5], dtype=np.int32)
    >>> print("offsets",offsets)
    offsets [0 1 2 3 4 5]
    >>> c = nvcategory.from_offsets(values,offsets,5)
    >>> print(c.keys(),c.values())
    ['a', 'e', 'l', 'p'] [0, 3, 3, 2, 1]

    """
    rtn = pyniNVCategory.n_createFromOffsets(
        sbuf, obuf, scount, nbuf, ncount, bdevmem
    )
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_strings(*args):
    """
    Create a nvcategory object from a nvstrings object.

    Parameters
    ----------
    args : variadic
        1 or more nvstrings objects

    Examples
    --------
    >>> import nvcategory, nvstrings
    >>> s1 = nvstrings.to_device(['apple','pear','banana'])
    >>> s2 = nvstrings.to_device(['orange','pear'])
    >>> c = nvcategory.from_strings(s1,s2)
    >>> print(c.keys(),c.values())
    ['apple', 'banana', 'orange', 'pear'] [0, 3, 1, 2, 3]

    """
    strs = []
    for arg in args:
        strs.append(arg)
    rtn = pyniNVCategory.n_createCategoryFromNVStrings(strs)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_strings_list(list):
    """
    Create a nvcategory object from a list of nvstrings.

    Parameters
    ----------
    list : list
        1 or more nvstrings objects

    Examples
    --------
    >>> import nvcategory, nvstrings
    >>> s1 = nvstrings.to_device(['apple','pear','banana'])
    >>> s2 = nvstrings.to_device(['orange','pear'])
    >>> c = nvcategory.from_strings_list([s1,s2])
    >>> print(c.keys(),c.values())
    ['apple', 'banana', 'orange', 'pear'] [0, 3, 1, 2, 3]

    """
    rtn = pyniNVCategory.n_createCategoryFromNVStrings(list)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def from_numbers(narr, nulls=None):
    """
    Create a nvcategory object from an array of numbers.

    Parameters
    ----------
    narr : ndarray
        Array of numbers in host or device memory
    nulls: ndarray
        Array of type int8 indicating which indexed values are null.

    Examples
    --------
    >>> import nvcategory
    >>> import numpy as np
    >>> nc = nvcategory.from_numbers(np.array([4, 1, 2, 3, 2, 1, 4, 1, 1]))
    >>> print(nc.keys(),nc.values())
    [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3, 0, 0]

    """
    rtn = pyniNVCategory.n_createCategoryFromNumbers(narr, nulls)
    if rtn is not None:
        rtn = nvcategory(rtn)
    return rtn


def bind_cpointer(cptr, own=True):
    """Bind an NVCategory C-pointer to a new instance."""
    rtn = None
    if cptr != 0:
        rtn = nvcategory(cptr)
        rtn._own = own
    return rtn


class nvcategory:
    """
    Instance manages a dictionary of strings (keys) in device memory
    and a mapping of indexes (values).

    """

    #
    m_cptr = 0

    def __init__(self, cptr):
        """For internal use only."""
        self.m_cptr = cptr
        self._own = True

    def __del__(self):
        if self._own:
            pyniNVCategory.n_destroyCategory(self.m_cptr)
        self.m_cptr = 0

    def __str__(self):
        return str(self.keys())

    def __repr__(self):
        return "<nvcategory[{}] keys={},values={}>".format(
            self.keys_type(), self.keys_size(), self.size()
        )

    def get_cpointer(self):
        """
        Returns memory pointer to underlying C++ class instance.
        """
        return self.m_cptr

    def size(self):
        """
        The number of values.

        Returns
        -------
        int
            Number of values

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.values())
        [2, 0, 2, 1]
        >>> print(c.size())
        4

        """
        return pyniNVCategory.n_size(self.m_cptr)

    def keys_size(self):
        """
        The number of keys.

        Returns
        -------
        int
            Number of keys

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.keys())
        ['aaa','dddd','eee']
        >>> print(c.keys_size())
        3

        """
        return pyniNVCategory.n_keys_size(self.m_cptr)

    def keys(self, narr=None):
        """
        Return the unique keys for this category.
        String keys are returned as nvstrings instance.
        Numeric keys require a buffer to fill.
        Buffer must be able to hold at least keys_size() elements
        of type keys_type().

        Returns
        -------
        nvstrings or None

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.keys())
        ['aaa','dddd','eee']
        >>> import numpy as np
        >>> narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
        >>> nc = nvcategory.from_numbers(narr)
        >>> keys = np.empty([cat.keys_size()], dtype=narr.dtype)
        >>> nc.keys(keys)
        >>> keys.tolist()
        [1.0, 1.25, 1.5, 2.0]

        """
        rtn = pyniNVCategory.n_get_keys(self.m_cptr, narr)
        if rtn is None:
            return rtn
        if isinstance(rtn, list):
            return rtn
        return nvs.nvstrings(rtn)

    def keys_type(self):
        """
        Return string with name of the keys type.

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
        >>> nc = nvcategory.from_numbers(narr)
        >>> nc.keys_type()
        'float64'

        """
        return pyniNVCategory.n_keys_type(self.m_cptr)

    def indexes_for_key(self, key, devptr=0):
        """
        Return all index values for given key.

        Parameters
        ----------
        key : str or number
            key whose values should be returned
        devptr : GPU memory pointer or ndarray
            Where index values will be written.
            Must be able to hold int32 values for this key.

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.indexes_for_key('aaa'))
        [1]
        >>> print(c.indexes_for_key('eee'))
        [0, 2]
        >>> import numpy as np
        >>> narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
        >>> nc = nvcategory.from_numbers(narr)
        >>> count = nc.indexes_for_key(1)
        >>> idxs = np.empty([count], dtype=np.int32)
        >>> count = nc.indexes_for_key(1, idxs)
        >>> idxs.tolist()
        [1, 4, 6, 7]

        """
        return pyniNVCategory.n_get_indexes_for_key(self.m_cptr, key, devptr)

    def value_for_index(self, idx):
        """
        Return the category value for the given index.

        Parameters
        ----------
        idx : int
            index value to retrieve

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.value_for_index(3))
        1

        """
        return pyniNVCategory.n_get_value_for_index(self.m_cptr, idx)

    def value(self, str):
        """
        Return the category value for the given string.

        Parameters
        ----------
        str : str
            key to retrieve

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.value('aaa'))
        0
        >>> print(c.value('eee'))
        2

        """
        return pyniNVCategory.n_get_value_for_string(self.m_cptr, str)

    def values(self, devptr=0):
        """
        Return all values for this instance.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where index values will be written.
            Must be able to hold size() of int32 values.

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.values())
        [2, 0, 2, 1]
        >>> import numpy as np
        >>> narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
        >>> nc = nvcategory.from_numbers(narr)
        >>> values = np.empty([cat.size()], dtype=np.int32)
        >>> nc.values(values)
        >>> values.tolist()
        [3, 0, 1, 2, 0, 1, 0, 0, 3]

        """
        return pyniNVCategory.n_get_values(self.m_cptr, devptr)

    def values_cpointer(self):
        """
        Returns memory pointer to underlying device memory array
        of int32 values for this instance.
        """
        return pyniNVCategory.n_get_values_cpointer(self.m_cptr)

    def add_strings(self, nvs):
        """
        Create new category incorporating specified strings.
        This will return a new nvcategory with new key values.
        The index values will appear as if appended.

        Parameters
        ----------
        nvs : nvstrings
            New strings to be added.

        Examples
        --------
        >>> import nvcategory, nvstrings
        >>> s1 = nvstrings.to_device(["eee","aaa","eee","dddd"])
        >>> s2 = nvstrings.to_device(["ggg","eee","aaa"])
        >>> c1 = nvcategory.from_strings(s1)
        >>> c2 = c1.add_strings(s2)
        >>> print(c1.keys())
        ['aaa','dddd','eee']
        >>> print(c1.values())
        [2, 0, 2, 1]
        >>> print(c2.keys())
        ['aaa','dddd','eee','ggg']
        >>> print(c2.values())
        [2, 0, 2, 1, 3, 2, 0]

        """
        rtn = pyniNVCategory.n_add_strings(self.m_cptr, nvs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def remove_strings(self, nvs):
        """
        Create new category without the specified strings.
        The returned category will have new set of key values and indexes.

        Parameters
        ----------
        nvs : nvstrings
            strings to be removed.

        Examples
        --------
        >>> import nvcategory, nvstrings
        >>> s1 = nvstrings.to_device(["eee","aaa","eee","dddd"])
        >>> s2 = nvstrings.to_device(["aaa"])
        >>> c1 = nvcategory.from_strings(s1)
        >>> c2 = c1.remove_strings(s2)
        >>> print(c1.keys())
        ['aaa','dddd','eee']
        >>> print(c1.values())
        [2, 0, 2, 1]
        >>> print(c2.keys())
        ['dddd', 'eee']
        >>> print(c2.values())
        [1, 1, 0]

        """
        rtn = pyniNVCategory.n_remove_strings(self.m_cptr, nvs)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def to_strings(self):
        """
        Return nvstrings instance represented by the values in this instance.

        Returns
        -------
        nvstrings
            Full strings list based on values indexes

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.keys())
        ['aaa','dddd','eee']
        >>> print(c.values())
        [2, 0, 2, 1]
        >>> print(c.to_strings())
        ['eee','aaa','eee','dddd']

        """
        rtn = pyniNVCategory.n_to_strings(self.m_cptr)
        if rtn is not None:
            rtn = nvs.nvstrings(rtn)
        return rtn

    def to_numbers(self, narr, nulls=None):
        """
        Fill array with key numbers as represented by the values
        in this instance.

        Parameters
        ----------
        narr : ndarray
            Array to fill with numbers. Must be of the same type
            as the keys for this instance and must be able to hold
            size() values.
        nulls : ndarray
            Array to fill with bits identifying null and non-null
            entries.

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
        >>> nc = nvcategory.from_numbers(narr)
        >>> nbrs = np.empty([cat.size()], dtype=narr.dtype)
        >>> nc.to_numbers(nbrs)
        >>> nbrs.tolist()
        [2.0, 1.0, 1.25, 1.5, 1.0, 1.25, 1.0, 1.0, 2.0]

        """
        return pyniNVCategory.n_to_numbers(self.m_cptr, narr, nulls)

    def gather_strings(self, indexes, count=0):
        """
        Return nvstrings instance represented using the specified indexes.

        Parameters
        ----------
        indexes : List of ints or GPU memory pointer
            0-based indexes of keys to return as an nvstrings object
        count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Returns
        -------
        nvstrings
            strings list based on indexes

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["eee","aaa","eee","dddd"])
        >>> print(c.keys())
        ['aaa','dddd','eee']
        >>> print(c.values())
        [2, 0, 2, 1]
        >>> print(c.gather_strings([0,2,0]))
        ['aaa','eee','aaa']

        """
        rtn = pyniNVCategory.n_gather_strings(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvs.nvstrings(rtn)
        return rtn

    def gather_numbers(self, indexes, narr, nulls=None):
        """
        Fill buffer with keys values specified by the given indexes.

        Parameters
        ----------
        indexes : ndarray of type int32
            List of integers identifying keys to copy to the output.
        narr : ndarray
            Type must match the keys for this instance and hold
            the number of values indicated by the indexes parameter.
        nulls : ndarray
            Bits are set to indicate null and non-null entries in
            the narr output array.

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> narr = np.array([2, 1, 1.25, 1.5, 1, 1.25, 1, 1, 2])
        >>> nc = nvcategory.from_numbers(narr)
        >>> idxs = np.array([0, 2, 0], dtype=np.int32)
        >>> nbrs = np.empty([idxs.size], dtype=narr.dtype)
        >>> nc.gather_numbers(idxs, nbrs)
        >>> nbrs.tolist()
        [1.0, 1.5, 1.0]

        """
        return pyniNVCategory.n_gather_numbers(
            self.m_cptr, indexes, narr, nulls
        )

    def gather_and_remap(self, indexes, count=0):
        """
        Return nvcategory instance using the specified indexes
        to gather strings from this instance.
        Index values will be remapped if any keys are not
        represented.
        This is equivalent to calling nvcategory.from_strings()
        using the nvstrings object returned from a call to
        gather_strings().

        Parameters
        ----------
        indexes : list or GPU memory pointer
            List of ints or GPU memory pointer to array of int32 values.
        count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Returns
        -------
        nvcategory
            keys and values based on indexes provided

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["aa","bb","bb","ff","cc","ff"])
        >>> print(c.keys(),c.values())
        ['aa', 'bb', 'cc', 'ff'] [0, 1, 1, 3, 2, 3]
        >>> c = c.gather([1,3,2,3,1,2])
        >>> print(c.keys(),c.values())
        ['bb', 'cc', 'ff'] [0, 2, 1, 2, 0, 1]
        >>> import numpy as np
        >>> nc = nvcategory.from_numbers(np.array([2, 1, 5, 4, 1, 5, 1, 1]))
        >>> indexes = np.array([1, 3, 2, 3, 1, 2], dtype=np.int32)
        >>> nc1 = nc.gather_and_remap(indexes)
        >>> print(nc1.keys(),nc1.values())
        [2, 4, 5] [0, 2, 1, 2, 0, 1]

        """
        rtn = pyniNVCategory.n_gather_and_remap(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def gather(self, indexes, count=0):
        """
        Return nvcategory instance using the keys for this option
        and copying the values from the indexes argument.

        Parameters
        ----------
        indexes : list or GPU memory pointer
            List of ints or GPU memory pointer to array of int32 values.
        count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Returns
        -------
        nvcategory
            keys and values based on indexes provided

        Examples
        --------
        >>> import nvcategory
        >>> c = nvcategory.to_device(["aa","bb","bb","ff","cc","ff"])
        >>> print(c.keys(),c.values())
        ['aa', 'bb', 'cc', 'ff'] [0, 1, 1, 3, 2, 3]
        >>> c = c.gather([1,3,2,3,1,2])
        >>> print(c.keys(),c.values())
        ['aa', 'bb', 'cc', 'ff'] [1, 3, 2, 3, 1, 2]
        >>> import numpy as np
        >>> nc = nvcategory.from_numbers(np.array([2, 1, 5, 4, 1, 5, 1, 1]))
        >>> indexes = np.array([1, 3, 2, 3, 1, 2], dtype=np.int32)
        >>> nc1 = nc.gather(indexes)
        >>> print(nc1.keys(),nc1.values())
        [1, 2, 4, 5] [1, 3, 2, 3, 1, 2]

        """
        rtn = pyniNVCategory.n_gather(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def merge_category(self, nvcat):
        """
        Create new category incorporating the specified category keys
        and values. This will return a new nvcategory with new key values.
        The index values will appear as if appended. Any matching keys
        will preserve their values and any new keys will get new values.

        Parameters
        ----------
        nvcat : nvcategory
            New category to be merged.

        """
        rtn = pyniNVCategory.n_merge_category(self.m_cptr, nvcat)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def merge_and_remap(self, nvcat):
        """
        Create new category incorporating the specified category keys
        and values. This will return a new nvcategory with new key values.
        The index values will appear as if appended.
        Values are appended and will be remapped to the new keys.

        Parameters
        ----------
        nvcat : nvcategory
            New category to be merged.

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> cat1 = nvcategory.from_numbers(np.array([4, 1, 2, 3, 2, 1, 4]))
        >>> print(cat1.keys(),cat1.values())
        [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3]
        >>> cat2 = nvcategory.from_numbers(np.array([2, 4, 3, 0]))
        >>> print(cat2.keys(),cat2.values())
        [0, 2, 3, 4] [1, 3, 2, 0]
        >>> nc = cat1.merge_and_remap(cat2)
        >>> print(nc.keys(),nc.values())
        [0, 1, 2, 3, 4] [4, 1, 2, 3, 2, 1, 4, 2, 4, 3, 0]

        """
        rtn = pyniNVCategory.n_merge_and_remap(self.m_cptr, nvcat)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def add_keys(self, keys, nulls=None):
        """
        Create new category adding the specified keys and remapping
        values to the new key indexes.

        Parameters
        ----------
        keys : nvstrings or ndarray
            keys to be added to existing keys
        nulls: bitmask
            ndarray of type uint8

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> c = nvcategory.from_numbers(np.array([4, 1, 2, 3, 2, 1, 4]))
        >>> print(c.keys(),c.values())
        [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3]
        >>> nc = c.add_keys(np.array([2,1,0,3]))
        >>> print(nc.keys(),nc.values())
        [0, 1, 2, 3, 4] [4, 1, 2, 3, 2, 1, 4]

        """
        rtn = pyniNVCategory.n_add_keys(self.m_cptr, keys, nulls)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def remove_keys(self, keys, nulls=None):
        """
        Create new category removing the specified keys and remapping
        values to the new key indexes. Values with removed keys are
        mapped to -1.

        Parameters
        ----------
        keys : nvstrings or ndarray
            keys to be removed from existing keys
        nulls: bitmask
            ndarray of type uint8

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> c = nvcategory.from_numbers(np.array([4, 1, 2, 3, 2, 1, 4]))
        >>> print(c.keys(),c.values())
        [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3]
        >>> nc = c.remove_keys(np.array([4,0]))
        >>> print(nc.keys(),nc.values())
        [1, 2, 3] [-1, 0, 1, 2, 1, 0, -1]

        """
        rtn = pyniNVCategory.n_remove_keys(self.m_cptr, keys, nulls)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def remove_unused_keys(self):
        """
        Create new category removing any keys that have no corresponding
        values. Values are remapped to match the new keyset.

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> c = nvcategory.from_numbers(np.array([4, 1, 2, 3, 2, 1, 4]))
        >>> print(c.keys(),c.values())
        [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3]
        >>> nc = c.add_keys(np.array([4,0]))
        >>> print(nc.keys(),nc.values())
        [1, 2, 3] [-1, 0, 1, 2, 1, 0, -1]
        >>> nc1 = nc.remove_unused_keys()
        >>> print(nc1.keys(),nc1.values())
        [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3]

        """
        rtn = pyniNVCategory.n_remove_unused_keys(self.m_cptr)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn

    def set_keys(self, keys, nulls=None):
        """
        Create new category using the specified keys and remapping
        values to the new key indexes. Matching names will have
        remapped values. Values with removed keys are mapped to -1.

        Parameters
        ----------
        strs : nvstrings or ndarray
            keys to be used for new category
        nulls: bitmask
            ndarray of type uint8

        Examples
        --------
        >>> import nvcategory
        >>> import numpy as np
        >>> c = nvcategory.from_numbers(np.array([4, 1, 2, 3, 2, 1, 4]))
        >>> print(c.keys(),c.values())
        [1, 2, 3, 4] [3, 0, 1, 2, 1, 0, 3]
        >>> nc = c.set_keys(np.array([2, 4, 3, 0]))
        >>> print(nc.keys(),nc.values())
        [0, 2, 3, 4] [3, -1, 1, 2, 1, -1, 3]

        """
        rtn = pyniNVCategory.n_set_keys(self.m_cptr, keys, nulls)
        if rtn is not None:
            rtn = nvcategory(rtn)
        return rtn
