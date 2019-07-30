import pyniNVStrings


def to_device(strs):
    """
    Create nvstrings instance from list of Python strings.

    Parameters
    ----------
    strs : list
        List of Python strings.

    Examples
    --------
    >>> import nvstrings
    >>> s = nvstrings.to_device(['apple','pear','banana','orange'])
    >>> print(s)
    ['apple', 'pear', 'banana', 'orange']

    """
    rtn = pyniNVStrings.n_createFromHostStrings(strs)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def from_strings(*args):
    """
    Create a nvstrings object from other nvstrings objects.

    Parameters
    ----------
    args : variadic
        1 or more nvstrings objects

    Examples
    --------
    >>> import nvstrings
    >>> s1 = nvstrings.to_device(['apple','pear','banana'])
    >>> s2 = nvstrings.to_device(['orange','pear'])
    >>> s3 = nvstrings.from_strings(s1,s2)
    >>> print(s3)
    ['apple', 'pear', banana', 'orange', 'pear']

    """
    strs = []
    for arg in args:
        if isinstance(arg, list):
            for s in arg:
                strs.append(s)
        else:
            strs.append(arg)
    rtn = pyniNVStrings.n_createFromNVStrings(strs)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def from_csv(csv, column, lines=0, flags=0):
    """
    Reads a column of values from a CSV file into a new nvstrings instance.
    The CSV file must be formatted as UTF-8.

    Parameters
    ----------
    csv : str
        Path to the csv file from which to load data
    column : int
        0-based index of the column to read into an nvstrings object
    lines : int
        maximum number of lines to read from the file
    flags : int
        values may be combined
        1 - sort by length
        2 - sort by name
        8 - nulls are empty strings

    Returns
    -------
    nvstrings
        A new nvstrings instance pointing to strings loaded onto the GPU

    Examples
    --------
    >>> import nvstrings

    For CSV file (file.csv) containing 2 rows and 3 columns:
    header1,header2,header3
    r1c1,r1c2,r1c3
    r2c1,r2c2,r2c3

    >>> s = nvstrings.from_csv("file.csv",2)
    >>> print(s)
    ['r1c3','r2c3']

    """
    rtn = pyniNVStrings.n_createFromCSV(csv, column, lines, flags)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def from_offsets(sbuf, obuf, scount, nbuf=None, ncount=0, bdevmem=False):
    """
    Create nvstrings object from byte-array of characters encoded in UTF-8.

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
    >>> import nvstrings
    >>> import numpy as np

    Create numpy array for 'a','p','p','l','e' with the utf8 int8 values
    of 97,112,112,108,101

    >>> values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
    >>> print("values",values.tobytes())
    values b'apple'
    >>> offsets = np.array([0,1,2,3,4,5], dtype=np.int32)
    >>> print("offsets",offsets)
    offsets [0 1 2 3 4 5]
    >>> s = nvstrings.from_offsets(values,offsets,5)
    >>> print(s)
    ['a', 'p', 'p', 'l', 'e']

    """
    rtn = pyniNVStrings.n_createFromOffsets(sbuf, obuf, scount, nbuf, ncount,
                                            bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def itos(values, count=0, nulls=None, bdevmem=False):
    """
    Create strings from an array of int32 values.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of int32 values to convert to strings.
    count : int
        Number of integers in values.
        This is only required if values is a memptr.
    nulls : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromInt32s(values, count, nulls, bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def ltos(values, count=0, nulls=None, bdevmem=False):
    """
    Create strings from an array of int64 values.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of int64 values to convert to strings.
    count : int
        Number of integers in values.
        This is only required if values is a memptr.
    nulls : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromInt64s(values, count, nulls, bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def ftos(values, count=0, nulls=None, bdevmem=False):
    """
    Create strings from an array of float32 values.
    Scientific notation may be used to show up to 10 significant digits.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of float32 values to convert to strings.
    count : int
        Number of floats in values.
        This is only required if values is a memptr.
    nulls : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromFloat32s(values, count, nulls, bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def dtos(values, count=0, nulls=None, bdevmem=False):
    """
    Create strings from an array of float64 values.
    Scientific notation may be used to show up to 10 significant digits.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of float64 values to convert to strings.
    count : int
        Number of floats in values.
        This is only required if values is a memptr.
    nulls : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromFloat64s(values, count, nulls, bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def int2ip(values, count=0, nulls=None, bdevmem=False):
    """
    Create ip address strings from an array of uint32 values.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of uint32 values (IPv4) to convert to strings.
    countc : int
        Number of integers in values.
        This is only required if values is a memory pointer.
    nullsc : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    bdevmemc : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromIPv4Integers(values, count, nulls, bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def int2timestamp(values, count=0, nulls=None,
                  format=None, units='s', bdevmem=False):
    """
    Create date/time strings from an array of int64 values.
    The values must be in units as specified by the units parameter.
    The values is expected to be from epoch time and in UTC.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of int64 values to convert to date-time strings.
    count : int
        Number of integers in values.
        This is only required if values is a memory pointer.
    nulls : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    format : str
        May include the following strftime specifiers only:
        %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z
        Default format is "%Y-%m-%dT%H:%M:%SZ"
    units : str
        The units of the values and must be one of the following:
        Y,M,D,h,m,s,ms,us,ns
        Default is 's' for seconds
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromTimestamp(values, count, nulls,
                                              format, units,
                                              bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def from_booleans(values, count=0, nulls=None,
                  true='True', false='False', bdevmem=False):
    """
    Create strings from an array of bool values.
    Each string will be created using the true and false strings provided.

    Parameters
    ----------
    values : list, memory address or buffer
        Array of boolean values to convert to strings.
        Memory pointers should be at least size() of int8 and
        should have values of 1 or 0 only.
    count : int
        Number of integers in values.
        This is only required if values is a memory pointer.
    nulls : list, memory address or buffer
        Bit array indicating which values should be considered null.
        Uses the arrow format for valid bitmask.
    true : str
        This string will be used to represent values that are 1 or True.
    false : str
        This string will be used to represent values that are 0 or False.
    bdevmem : boolean
        Default (False) interprets memory pointers as CPU memory.

    """
    rtn = pyniNVStrings.n_createFromBools(values, count, nulls,
                                          true, false, bdevmem)
    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def create_from_ipc(ipc_data):
    """
    Returns a valid NVStrings object from IPC data.

    Parameters
    ----------
    ipc_data : list, IPC handlers data
    """
    rtn = pyniNVStrings.n_createFromIPC(ipc_data)

    if rtn is not None:
        rtn = nvstrings(rtn)
    return rtn


def free(dstrs):
    """Force free resources for the specified instance."""
    if dstrs is not None:
        pyniNVStrings.n_destroyStrings(dstrs.m_cptr)
        dstrs.m_cptr = 0


def bind_cpointer(cptr, own=True):
    """Bind an NVStrings C-pointer to a new instance."""
    rtn = None
    if cptr != 0:
        rtn = nvstrings(cptr)
        rtn._own = own
    return rtn


# this will be documented with all the public methods
class nvstrings:
    """
    Instance manages a list of strings in device memory.

    Operations are across all of the strings and their results reside in
    device memory. Strings in the list are immutable.
    Methods that modify any string will create a new nvstrings instance.
    """
    #
    m_cptr = 0

    def __init__(self, cptr):
        """
        Use to_device() to create new instance from Python array of strings.
        """
        self.m_cptr = cptr
        self._own = True

    def __del__(self):
        if self._own:
            pyniNVStrings.n_destroyStrings(self.m_cptr)
        self.m_cptr = 0

    def __str__(self):
        return str(pyniNVStrings.n_createHostStrings(self.m_cptr))

    def __repr__(self):
        return "<nvstrings count={}>".format(self.size())

    def __getitem__(self, key):
        """
        Implemented for [] operator on nvstrings.
        Parameter must be integer, slice, or list of integers.
        """
        if key is None:
            raise KeyError("key must not be None")
        if isinstance(key, list):
            return self.gather(key)
        if isinstance(key, int):
            return self.gather([key])
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            end = self.size() if key.stop is None else key.stop
            step = 1 if key.step is None or key.step == 0 else key.step
            # negative slicing check
            end = self.size()+end if end < 0 else end
            start = self.size()+start if start < 0 else start
            rtn = pyniNVStrings.n_sublist(self.m_cptr, start, end, step)
            if rtn is not None:
                rtn = nvstrings(rtn)
            return rtn
        # gather can handle almost anything now
        return self.gather(key)

    def __iter__(self):
        raise TypeError("iterable not supported by nvstrings")

    def __len__(self):
        return self.size()

    def get_cpointer(self):
        """
        Returns memory pointer to underlying C++ class instance.
        """
        return self.m_cptr

    def get_ipc_data(self):
        """
        Returns IPC data handler to underlying C++ object from NVStrings.

        Returns
        -------
        list
            A list containing the IPC handler data from NVstrings.
        Examples
        --------
        >>> import nvstrings
        >>> ipc_data = nvstrings.get_ipc_data()
        >>> print(ipc_data)
        """
        ipc_data = pyniNVStrings.n_getIPCData(self.m_cptr)
        return ipc_data

    def to_host(self):
        """
        Copies strings back to CPU memory into a Python array.

        Returns
        -------
        list
            A list of strings

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","world"])
        >>> h = s.upper().to_host()
        >>> print(h)
        ["HELLO","WORLD"]

        """
        return pyniNVStrings.n_createHostStrings(self.m_cptr)

    def to_offsets(self, sbuf, obuf, nbuf=0, bdevmem=False):
        """
        Store byte-array of characters encoded in UTF-8 and offsets
        and optional null-bitmask into provided memory.

        Parameters
        ----------
        sbuf : memory address or buffer
            Strings characters are stored contiguously encoded as UTF-8.
        obuf : memory address or buffer
            Stores array of int32 byte offsets to beginning of each
            string in sbuf. This should be able to hold size()+1 values.
        nbuf : memory address or buffer
            Optional: stores null bitmask in arrow format.
        bdevmem : boolean
            Default (False) interprets memory pointers as CPU memory.

        Examples
        --------
        >>> import nvstrings
        >>> import numpy as np
        >>> s = nvstrings.to_device(['a','p','p','l','e'])
        >>> values = np.empty(s.size(), dtype=np.int8)
        >>> offsets = np.empty(s.size()+1, dtype=np.int32)
        >>> s.to_offsets(values,offsets)
        >>> print("values",values.tobytes())
        values b'apple'
        >>> print("offsets",offsets)
        offsets [0 1 2 3 4 5]

        """
        return pyniNVStrings.n_create_offsets(self.m_cptr, sbuf, obuf, nbuf,
                                              bdevmem)

    def size(self):
        """
        The number of strings managed by this instance.

        Returns
        -------
        int
            number of strings

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","world"])
        >>> print(s.size())
        2

        """
        return pyniNVStrings.n_size(self.m_cptr)

    def len(self, devptr=0):
        """
        Returns the number of characters of each string.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where string length values will be written.
            Must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> from librmm_cffi import librmm
        >>> import numpy as np

        Example passing device memory pointer

        >>> s = nvstrings.to_device(["abc","d","ef"])
        >>> arr = np.arange(s.size(),dtype=np.int32)
        >>> d_arr = librmm.to_device(arr)
        >>> s.len(d_arr.device_ctypes_pointer.value)
        >>> print(d_arr.copy_to_host())
        [3,1,2]

        """
        rtn = pyniNVStrings.n_len(self.m_cptr, devptr)
        return rtn

    def byte_count(self, vals=0, bdevmem=False):
        """
        Fills the argument with the number of bytes of each string.
        Returns the total number of bytes.

        Parameters
        ----------
        vals : memory pointer
            Where byte length values will be written.
            Must be able to hold at least size() of int32 values.
            None can be specified if only the total count is required.

        Examples
        --------
        >>> import nvstrings
        >>> import numpy as np
        >>> from librmm_cffi import librmm

        Example passing device memory pointer

        >>> s = nvstrings.to_device(["abc","d","ef"])
        >>> arr = np.arange(s.size(),dtype=np.int32)
        >>> d_arr = librmm.to_device(arr)
        >>> s.byte_count(d_arr.device_ctypes_pointer.value,True)
        >>> print(d_arr.copy_to_host())
        [3,1,2]

        """
        rtn = pyniNVStrings.n_byte_count(self.m_cptr, vals, bdevmem)
        return rtn

    def set_null_bitmask(self, nbuf, bdevmem=False):
        """
        Store null-bitmask into provided memory.

        Parameters
        ----------
        nbuf : memory address or buffer
            Stores null bitmask in arrow format.
        bdevmem : boolean
            Default (False) interprets nbuf as CPU memory.

        Examples
        --------
        >>> import nvstrings
        >>> import numpy as np
        >>> s = nvstrings.to_device(['a',None,'p','l','e'])
        >>> nulls = np.empty(int(s.size()/8)+1, dtype=np.int8)
        >>> s.set_null_bitmask(nulls)
        >>> print("nulls",nulls.tobytes())
        nulls b\'\\x1d\'

        """
        return pyniNVStrings.n_set_null_bitmask(self.m_cptr, nbuf, bdevmem)

    def null_count(self, emptyisnull=False):
        """
        Returns the number of null strings in this instance.

        Parameters
        ----------
        emptyisnull : boolean
            If True, empty strings are counted as null.
            Default is False.

        Examples
        --------
        >>> import nvstrings

        Example passing device memory pointer

        >>> s = nvstrings.to_device(["abc","",None])
        >>> print("nulls",s.null_count())
        nulls 1
        >>> print("nulls+empty", s.null_count(True))
        nulls+empty 2

        """
        return pyniNVStrings.n_null_count(self.m_cptr, emptyisnull)

    def compare(self, str, devptr=0):
        """
        Compare each string to the supplied string.
        Returns value of 0 for strings that match str.
        Returns < 0 when first different character is lower
        than argument string or argument string is shorter.
        Returns > 0 when first different character is greater
        than the argument string or the argument string is longer.

        Parameters
        ----------
        str : str
            String to compare all strings in this instance.
        devptr : GPU memory pointer
            Where string result values will be written.
            Must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","world"])
        >>> print(s.compare('hello'))
        [0,15]

        """
        rtn = pyniNVStrings.n_compare(self.m_cptr, str, devptr)
        return rtn

    def hash(self, devptr=0):
        """
        Returns hash values represented by each string.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where string hash values will be written.
            Must be able to hold at least size() of uint32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","world"])
        >>> s.hash()
        [99162322, 113318802]

        """
        rtn = pyniNVStrings.n_hash(self.m_cptr, devptr)
        return rtn

    def stoi(self, devptr=0):
        """
        Returns integer values represented by each string.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where resulting integer values will be written.
            Memory must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["1234","-876","543.2","-0.12",".55""])
        >>> print(s.stoi())
        [1234, -876, 543, 0, 0]

        """
        rtn = pyniNVStrings.n_stoi(self.m_cptr, devptr)
        return rtn

    def stol(self, devptr=0):
        """
        Returns int64 values represented by each string.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where resulting integer values will be written.
            Memory must be able to hold at least size() of int64 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["1234","-876","543.2","-0.12",".55""])
        >>> print(s.stol())
        [1234, -876, 543, 0, 0]

        """
        rtn = pyniNVStrings.n_stol(self.m_cptr, devptr)
        return rtn

    def stof(self, devptr=0):
        """
        Returns float values represented by each string.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where resulting float values will be written.
            Memory must be able to hold at least size() of float32 values

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["1234","-876","543.2","-0.12",".55"])
        >>> print(s.stof())
        [1234.0, -876.0, 543.2000122070312,
         -0.11999999731779099, 0.550000011920929]

        """
        rtn = pyniNVStrings.n_stof(self.m_cptr, devptr)
        return rtn

    def stod(self, devptr=0):
        """
        Returns float64 values represented by each string.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where resulting float values will be written.
            Memory must be able to hold at least size() of float64 values

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["1234","-876","543.2","-0.12",".55"])
        >>> print(s.stod())
        [1234.0, -876.0, 543.2000122070312,
         -0.11999999731779099, 0.550000011920929]

        """
        rtn = pyniNVStrings.n_stod(self.m_cptr, devptr)
        return rtn

    def htoi(self, devptr=0):
        """
        Returns integer value represented by each string.
        String is interpretted to have hex (base-16) characters.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where resulting integer values will be written.
            Memory must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["1234","ABCDEF","1A2","cafe"])
        >>> print(s.htoi())
        [4660, 11259375, 418, 51966]

        """
        rtn = pyniNVStrings.n_htoi(self.m_cptr, devptr)
        return rtn

    def to_booleans(self, true="True", devptr=0):
        """
        Returns boolean value represented by each string.

        Parameters
        ----------
        true : str
            String to use for True values. All others are set to False.
        devptr : GPU memory pointer
            Where resulting integer values will be written.
            Memory must be able to hold at least size() of uint32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["True","False","",None])
        >>> print(s.to_booleans())
        [True, False, False, None]

        """
        rtn = pyniNVStrings.n_to_bools(self.m_cptr, true, devptr)
        return rtn

    def ip2int(self, devptr=0):
        """
        Returns integer value represented by each string.
        String is interpretted to be IPv4 format.

        Parameters
        ----------
        devptr : GPU memory pointer
            Where resulting integer values will be written.
            Memory must be able to hold at least size() of uint32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["192.168.0.1","10.0.0.1"])
        >>> print(s.ip2int())
        [3232235521, 167772161]

        """
        rtn = pyniNVStrings.n_ip2int(self.m_cptr, devptr)
        return rtn

    def timestamp2int(self, format=None, units='s', devptr=0):
        """
        Returns integer value represented by each string.
        String is interpretted using the format provided.
        The values are returned in the units specified based
        on epoch time and in UTC.

        Parameters
        ----------
        format : str
            May include the following strptime specifiers only:
            %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z
            Default format is "%Y-%m-%dT%H:%M:%SZ"
        units : str
            The units of the values and must be one of the following:
            Y,M,D,h,m,s,ms,us,ns
            Default is 's' for seconds
        devptr : GPU memory pointer
            Where resulting integer values will be written.
            Memory must be able to hold at least size() of int64 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["2019-03-20T12:34:56Z"])
        >>> print(s.timestamp2int())
        [1553085296]

        """
        rtn = pyniNVStrings.n_timestamp2int(self.m_cptr,
                                            format, units,
                                            devptr)
        return rtn

    def cat(self, others=None, sep=None, na_rep=None):
        """
        Appends the given strings to this list of strings and
        returns as new nvstrings.

        Parameters
        ----------
        others : nvstrings or list of nvstrings
            Strings to be appended.
            The number of strings in the arg(s) must match size() of
            this instance.
        sep : str
            If specified, this separator will be appended to each string
            before appending the others.
        na_rep : char
            This character will take the place of any null strings
            (not empty strings) in either list.

        Examples
        --------
        >>> import nvstrings
        >>> s1 = nvstrings.to_device(['hello', None,'goodbye'])
        >>> s2 = nvstrings.to_device(['world','globe', None])
        >>> print(s1.cat(s2,sep=':', na_rep='_'))
        ["hello:world","_:globe","goodbye:_"]

        """
        rtn = pyniNVStrings.n_cat(self.m_cptr, others, sep, na_rep)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def join(self, sep=''):
        """
        Concatentate this list of strings into a single string.

        Parameters
        ----------
        sep : str
            This separator will be appended to each string before
            appending the next.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye"])
        >>> print(s.join(sep=':'))
        ['hello:goodbye']

        """
        rtn = pyniNVStrings.n_join(self.m_cptr, sep)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def split_record(self, delimiter=None, n=-1):
        """
        Returns an array of nvstrings each representing the split
        of each individual string.

        Parameters
        ----------
        delimiter : str
            The character used to locate the split points of
            each string. Default is space.
        n : int
            Maximum number of strings to return for each split.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello world","goodbye","well said"])
        >>> for result in s.split_record(' '):
        ...     print(result)
        ["hello","world"]
        ["goodbye"]
        ["well","said"]

        """
        strs = pyniNVStrings.n_split_record(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rsplit_record(self, delimiter=None, n=-1):
        """
        Returns an array of nvstrings each representing the split of each
        individual string. The delimiter is searched for from the end of
        each string.

        Parameters
        ----------
        delimiter : str
            The character used to locate the split points of each
            string. Default is space.
        n : int
            Maximum number of strings to return for each split.

        Examples
        --------
        >>> import nvstrings
        >>> strs = nvstrings.to_device(["hello world","goodbye","up in arms"])
        >>> for s in strs.rsplit_record(' ',2):
        ...     print(s)
        ['hello', 'world']
        ['goodbye']
        ['up in', 'arms']

        """
        strs = pyniNVStrings.n_rsplit_record(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def partition(self, delimiter=' '):
        """
        Each string is split into two strings on the first delimiter found.

        Three strings are returned for each string:
        beginning, delimiter, end.

        Parameters
        ----------
        delimiter : str
            The character used to locate the split points of each
            string. Default is space.

        Examples
        --------
        >>> import nvstrings
        >>> strs = nvstrings.to_device(["hello world","goodbye","up in arms"])
        >>> for s in strs.partition(' '):
        ...     print(s)
        ['hello', ' ', 'world']
        ['goodbye', '', '']
        ['up', ' ', 'in arms']

        """
        strs = pyniNVStrings.n_partition(self.m_cptr, delimiter)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rpartition(self, delimiter=' '):
        """
        Each string is split into two strings on the first delimiter found.
        Delimiter is searched for from the end.

        Three strings are returned for each string: beginning, delimiter, end.

        Parameters
        ----------
        delimiter : str
            The character used to locate the split points of each string.
            Default is space.

        Examples
        --------
        >>> import nvstrings
        >>> strs = nvstrings.to_device(["hello world","goodbye","up in arms"])
        >>> for s in strs.rpartition(' '):
        ...     print(s)
        ['hello', ' ', 'world']
        ['', '', 'goodbye']
        ['up in', ' ', 'arms']

        """
        strs = pyniNVStrings.n_rpartition(self.m_cptr, delimiter)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def split(self, delimiter=None, n=-1):
        """
        A new set of columns (nvstrings) is created by splitting
        the strings vertically.

        Parameters
        ----------
        delimiter : str
            The character used to locate the split points of each string.
            Default is space.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello world","goodbye","well said"])
        >>> for result in s.split(' '):
        ...     print(result)
        ["hello","goodbye","well"]
        ["world",None,"said"]

        """
        strs = pyniNVStrings.n_split(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def rsplit(self, delimiter=None, n=-1):
        """
        A new set of columns (nvstrings) is created by splitting
        the strings vertically. Delimiter is searched from the end.

        Parameters
        ----------
        delimiter : str
            The character used to locate the split points of each string.
            Default is space.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello world","goodbye","well said"])
        >>> for result in s.rsplit(' '):
        ...     print(result)
        ["hello","goodbye","well"]
        ["world",None,"said"]

        """
        strs = pyniNVStrings.n_rsplit(self.m_cptr, delimiter, n)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def get(self, i):
        """
        Returns the character specified in each string as a new string.
        The nvstrings returned contains a list of single character strings.

        Parameters
        ----------
        i : int
            The character position identifying the character
            in each string to return.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello world","goodbye","well said"])
        >>> print(s.get(0))
        ['h', 'g', 'w']

        """
        rtn = pyniNVStrings.n_get(self.m_cptr, i)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def repeat(self, repeats):
        """
        Appends each string with itself the specified number of times.
        This returns a nvstrings instance with the new strings.

        Parameters
        ----------
        repeats : int
           The number of times each string should be repeated.
           Repeat count of 0 or 1 will just return copy of each string.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye","well"])
        >>> print(s.repeat(2))
        ['hellohello', 'goodbyegoodbye', 'wellwell']

        """
        rtn = pyniNVStrings.n_repeat(self.m_cptr, repeats)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def pad(self, width, side='left', fillchar=' '):
        """
        Add specified padding to each string.
        Side:{'left','right','both'}, default is 'left'.

        Parameters
        ----------
        fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        side : str
            Either one of "left", "right", "both". The default is "left"
            "left" performs a padding on the left – same as rjust()
            "right" performs a padding on the right – same as ljust()
            "both" performs equal padding on left and right – same as center()

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye","well"])
        >>> print(s.pad(5))
        ['  hello', 'goodbye', '   well']

        """
        rtn = pyniNVStrings.n_pad(self.m_cptr, width, side, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def ljust(self, width, fillchar=' '):
        """
        Pad the end of each string to the minimum width.

        Parameters
        ----------
        width : int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

        fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye","well"])
        >>> print(s.ljust(width=6))
        ['hello ', 'goodbye', 'well  ']

        """
        rtn = pyniNVStrings.n_ljust(self.m_cptr, width, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def center(self, width, fillchar=' '):
        """
        Pad the beginning and end of each string to the minimum width.

        Parameters
        ----------
        width : int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

        fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye","well"])
        >>> print(s.center(7))
        [' hello ', 'goodbye', ' well  ']

        """
        rtn = pyniNVStrings.n_center(self.m_cptr, width, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def rjust(self, width, fillchar=' '):
        """
        Pad the beginning of each string to the minimum width.

        Parameters
        ----------
        width : int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

        fillchar : char
            The character used to do the padding.
            Default is space character. Only the first character is used.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye","well"])
        >>> print(s.ljust(width=6))
        [' hello', 'goodbye', '  well']

        """
        rtn = pyniNVStrings.n_rjust(self.m_cptr, width, fillchar)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def zfill(self, width):
        """
        Pads the strings with leading zeros.
        It will handle prefix sign characters correctly for strings
        containing leading number characters.

        Parameters
        ----------
        width : int
            The minimum width of characters of the new string.
            If the width is smaller than the existing string,
            no padding is performed.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","1234","-9876","+5.34"])
        >>> print(s.zfill(width=6))
        ['0hello', '001234', '-09876', '+05.34']

        """
        rtn = pyniNVStrings.n_zfill(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def wrap(self, width):
        """
        This will place new-line characters in whitespace so each line
        is no more than width characters. Lines will not be truncated.

        Parameters
        ----------
        width : int
            The maximum width of characters per newline in the new string.
            If the width is smaller than the existing string, no newlines
            will be inserted.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello there","goodbye all","well ok"])
        >>> print(s.wrap(3))
        ['hello\\nthere', 'goodbye\\nall', 'well\\nok']

        """
        rtn = pyniNVStrings.n_wrap(self.m_cptr, width)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice(self, start, stop=None, step=None):
        """
        Returns a substring of each string.

        Parameters
        ----------
        start : int
            Beginning position of the string to extract.
            Default is beginning of the each string.
        stop : int
            Ending position of the string to extract.
            Default is end of each string.
        step : str
            Characters that are to be captured within the specified section.
            Default is every character.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye"])
        >>> print(s.slice(2,5))
        ['llo', 'odb']

        """
        rtn = pyniNVStrings.n_slice(self.m_cptr, start, stop, step)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice_from(self, starts=0, stops=0):
        """
        Return substring of each string using positions for each string.

        The starts and stops parameters are device memory pointers.
        If specified, each must contain size() of int32 values.

        Parameters
        ----------
        starts : GPU memory pointer
            Beginning position of each the string to extract.
            Default is beginning of the each string.
        stops : GPU memory pointer
            Ending position of the each string to extract.
            Default is end of each string.
            Use -1 to specify to the end of that string.


        Examples
        --------
        >>> import nvstrings
        >>> import numpy as np
        >>> from numba import cuda
        >>> s = nvstrings.to_device(["hello","there"])
        >>> darr = cuda.to_device(np.asarray([2,3],dtype=np.int32))
        >>> print(s.slice_from(starts=darr.device_ctypes_pointer.value))
        ['llo','re']

        """
        rtn = pyniNVStrings.n_slice_from(self.m_cptr, starts, stops)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace the specified section of each string with a new string.

        Parameters
        ----------
        start : int
            Beginning position of the string to replace.
            Default is beginning of the each string.
        stop : int
            Ending position of the string to replace.
            Default is end of each string.
        repl : str
            String to insert into the specified position values.

        Examples
        --------
        >>> import nvstrings
        >>> strs = nvstrings.to_device(["abcdefghij","0123456789"])
        >>> print(strs.slice_replace(2,5,'z'))
        ['abzfghij', '01z56789']

        """
        rtn = pyniNVStrings.n_slice_replace(self.m_cptr, start, stop, repl)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def insert(self, start=0, repl=None):
        """
        Insert the specified string into each string in the specified
        position.

        Parameters
        ----------
        start : int
            Beginning position of the string to replace.
            Default is beginning of the each string.
            Specify -1 to insert at the end of each string.
        repl : str
            String to insert into the specified position valus.

        Examples
        --------
        >>> import nvstrings
        >>> strs = nvstrings.to_device(["abcdefghij","0123456789"])
        >>> print(strs.insert(2,'_'))
        ['ab_cdefghij', '01_23456789']

        """
        rtn = pyniNVStrings.n_insert(self.m_cptr, start, repl)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def replace(self, pat, repl, n=-1, regex=True):
        """
        Replace a string (pat) in each string with another string (repl).

        Parameters
        ----------
        pat : str
            String to be replaced.
            This can also be a regex expression -- not a compiled regex.
        repl : str
            String to replace found string in the output instance.
        regex : boolean
            Set to True if pat is a regular expression string.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye"])
        >>> print(s.replace('e', ''))
        ['hllo', 'goodby']

        """
        rtn = pyniNVStrings.n_replace(self.m_cptr, pat, repl, n, regex)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def replace_multi(self, pats, repls, regex=True):
        """
        Replace multiple strings (pats) in each string with corresponding
        strings (repls).

        Parameters
        ----------
        pats : list or nvstrings
            Strings to be replaced.
            These can also be a regex expressions patterns.
            If so, this must not be an nvstrings instance.
        repls : list or nvstrings
            Strings to replace found pattern/string.
            Must be the same number of strings as pats.
            Alternately, this can be a single str instance
            and would be used as replacement for each string found.
        regex : boolean
            Set to True if pats are regular expression strings.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","goodbye"])
        >>> print(s.replace_multi(['e', 'o'],['E','O']))
        ['hEllO', 'gOOdbyE']

        """
        if regex is True:
            if isinstance(pats, list) is False:
                raise ValueError("pats must be list of str")
        else:
            if isinstance(pats, list):
                pats = to_device(pats)
        if isinstance(repls, str):
            repls = to_device([repls])
        if isinstance(repls, list):
            repls = to_device(repls)

        rtn = pyniNVStrings.n_replace_multi(self.m_cptr, pats, repls.m_cptr,
                                            regex)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def replace_with_backrefs(self, pat, repl):
        """
        Use the repl back-ref template to create a new string with
        the extracted elements found using the pat expression.

        Parameters
        ----------
        pat : str
            Regex with groupings to identify extract sections.
            This should not be a compiled regex.
        repl : str
            String template containing back-reference indicators.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["A543","Z756"])
        >>> print(s.replace_with_backrefs('(\\d)(\\d)', 'V\\2\\1'))
        ['V45', 'V57']

        """
        rtn = pyniNVStrings.n_replace_with_backrefs(self.m_cptr, pat, repl)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def fillna(self, repl):
        """
        Create new instance, replacing all nulls with the given string(s).

        Parameters
        ----------
        repl : str or nvstrings
            String to be used in place of nulls.
            This may be an empty string but may not be None.
            This may also be another nvstrings instance with the size.
            Corresponding strings are replaced only if null.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello", None, "goodbye"])
        >>> print(s.fillna(''))
        ['hello', '', 'goodbye']

        """
        rtn = pyniNVStrings.n_fillna(self.m_cptr, repl)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def lstrip(self, to_strip=None):
        """
        Strip leading characters from each string.

        Parameters
        ----------
        to_strip : str
            Characters to be removed from leading edge of each string

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["oh","hello","goodbye"])
        >>> print(s.lstrip('o'))
        ['h', 'hello', 'goodbye']

        """
        rtn = pyniNVStrings.n_lstrip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def strip(self, to_strip=None):
        """
        Strip leading and trailing characters from each string.

        Parameters
        ----------
        to_strip : str
            Characters to be removed from both ends of each string

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["oh, hello","goodbye"])
        >>> print(s.strip('o'))
        ['h, hell', 'goodbye']

        """
        rtn = pyniNVStrings.n_strip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def rstrip(self, to_strip=None):
        """
        Strip trailing characters from each string.

        Parameters
        ----------
        to_strip : str
            Characters to be removed from trailing edge of each string

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["oh","hello","goodbye"])
        >>> print(s.rstrip('o'))
        ['oh', 'hell', 'goodbye']

        """
        rtn = pyniNVStrings.n_rstrip(self.m_cptr, to_strip)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def lower(self):
        """
        Convert each string to lowercase.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["Hello, Friend","Goodbye, Friend"])
        >>> print(s.lower())
        ['hello, friend', 'goodbye, friend']

        """
        rtn = pyniNVStrings.n_lower(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def upper(self):
        """
        Convert each string to uppercase.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["Hello, friend","Goodbye, friend"])
        >>> print(s.lower())
        ['HELLO, FRIEND', 'GOODBYE, FRIEND']

        """
        rtn = pyniNVStrings.n_upper(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def capitalize(self):
        """
        Capitalize first character of each string.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello, friend","goodbye, friend"])
        >>> print(s.lower())
        ['Hello, friend", "Goodbye, friend"]

        """
        rtn = pyniNVStrings.n_capitalize(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def swapcase(self):
        """
        Change each lowercase character to uppercase and vice versa.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["Hello, Friend","Goodbye, Friend"])
        >>> print(s.lower())
        ['hELLO, fRIEND', 'gOODBYE, fRIEND']

        """
        rtn = pyniNVStrings.n_swapcase(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def title(self):
        """
        Uppercase the first letter of each letter after a space
        and lowercase the rest.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["Hello friend","goodnight moon"])
        >>> print(s.title())
        ['Hello Friend', 'Goodnight Moon']

        """
        rtn = pyniNVStrings.n_title(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def index(self, sub, start=0, end=None, devptr=0):
        """
        Same as find but throws an error if arg is not found in all strings.

        Parameters
        ----------
        sub : str
            String to find
        start : int
            Beginning of section to search from.
            Default is 0 (beginning of each string).
        end : int
            End of section to search. Default is end of each string.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","world"])
        >>> print(s.index('l'))
        [2,3]

        """
        rtn = pyniNVStrings.n_index(self.m_cptr, sub, start, end, devptr)
        return rtn

    def rindex(self, sub, start=0, end=None, devptr=0):
        """
        Same as rfind but throws an error if arg is not found in all strings.

        Parameters
        ----------
        sub : str
            String to find
        start : int
            Beginning of section to search from.
            Default is 0 (beginning of each string).
        end : int
            End of section to search. Default is end of each string.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.

        Examples
        --------
        >>>import nvstrings
        >>> s = nvstrings.to_device(["hello","world"])
        >>> print(s.rindex('l'))
        [3,3]

        """
        rtn = pyniNVStrings.n_rindex(self.m_cptr, sub, start, end, devptr)
        return rtn

    def find(self, sub, start=0, end=None, devptr=0):
        """
        Find the specified string sub within each string.
        Return -1 for those strings where sub is not found.

        Parameters
        ----------
        sub : str
            String to find
        start : int
            Beginning of section to search from.
            Default is 0 (beginning of each string).
        end : int
            End of section to find. Default is end of each string.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.find('o'))
        [4,-1,1]

        """
        rtn = pyniNVStrings.n_find(self.m_cptr, sub, start, end, devptr)
        return rtn

    def find_from(self, sub, starts=0, ends=0, devptr=0):
        """
        Find the specified string within each string starting at the
        specified character positions.
        The starts and ends parameters are device memory pointers.
        If specified, each must contain size() of int32 values.
        Returns -1 for those strings where sub is not found.

        Parameters
        ----------
        sub : str
            String to find
        starts : GPU memory pointer
            Pointer to GPU array of int32 values of beginning of sections to
            search, one per string.
        ends : GPU memory pointer
            Pointer to GPU array of int32 values of end of sections to search.
            Use -1 to specify to the end of that string.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> import numpy as np
        >>> from numba import cuda
        >>> s = nvstrings.to_device(["hello","there"])
        >>> darr = cuda.to_device(np.asarray([2,3],dtype=np.int32))
        >>> print(s.find_from('e',starts=darr.device_ctypes_pointer.value))
        [-1,4]

        """
        rtn = pyniNVStrings.n_find_from(self.m_cptr, sub, starts, ends, devptr)
        return rtn

    def rfind(self, sub, start=0, end=None, devptr=0):
        """
        Find the specified string within each string.
        Search from the end of the string.

        Return -1 for those strings where sub is not found.

        Parameters
        ----------
        sub : str
            String to find
        start : int
            Beginning of section to find.
            Default is 0(beginning of each string).
        end : int
            End of section to find. Default is end of each string.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.rfind('o'))
        [4, -1, 1]

        """
        rtn = pyniNVStrings.n_rfind(self.m_cptr, sub, start, end, devptr)
        return rtn

    def findall_record(self, pat):
        """
        Find all occurrences of regular expression pattern in each string.
        A new array of nvstrings is created for each string in this instance.

        Parameters
        ----------
        pat : str
            The regex pattern used to search for substrings

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hare","bunny","rabbit"])
        >>> for result in s.findall_record('[ab]'):
        ...     print(result)
        ["a"]
        ["b"]
        ["a","b","b"]

        """
        strs = pyniNVStrings.n_findall_record(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def findall(self, pat):
        """
        A new set of nvstrings is created by organizing substring
        results vertically.

        Parameters
        ----------
        pat : str
            The regex pattern to search for substrings

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hare","bunny","rabbit"])
        >>> for result in s.findall('[ab]'):
        ...     print(result)
        ["a","b","a"]
        [None,None,"b"]
        [None,None,"b"]

        """
        strs = pyniNVStrings.n_findall(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def contains(self, pat, regex=True, devptr=0):
        """
        Find the specified string within each string.
        Default expects regex pattern.
        Returns an array of boolean values where
        True if `pat` is found, False if not.

        Parameters
        ----------
        pat : str
            Pattern or string to search for in each string of this instance.
        regex : bool
            If `True`, pat is interpreted as a regex string.
            If `False`, pat is a string to be searched for in each instance.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Must be able to hold at least size() of np.byte values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.contains('o'))
        [True, False, True]

        """
        rtn = pyniNVStrings.n_contains(self.m_cptr, pat, regex, devptr)
        return rtn

    def match(self, pat, devptr=0):
        """
        Return array of boolean values where True is set if the specified
        pattern matches the beginning of the corresponding string.

        Parameters
        ----------
        pat : str
            Pattern to find
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of
            np.byte values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.match('h'))
        [True, False, False]

        """
        rtn = pyniNVStrings.n_match(self.m_cptr, pat, devptr)
        return rtn

    def match_strings(self, strs, devptr=0):
        """
        Return array of boolean values where True is set for those
        strings in strs that match exactly to the corresponding
        strings in this instance.

        Parameters
        ----------
        strs : nvstrings
            Strings to match against.
            Number of strings must match those in this instance.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size() of
            np.byte or np.int8 values.

        Examples
        --------
        >>> import nvstrings
        >>> s1 = nvstrings.to_device(["hello","there","world"])
        >>> s2 = nvstrings.to_device(["hello","here","world"])
        >>> print(s1.match_strings(s2))
        [True, False, True]

        """
        rtn = pyniNVStrings.n_match_strings(self.m_cptr, strs, devptr)
        return rtn

    def count(self, pat, devptr=0):
        """
        Count occurrences of pattern in each string.

        Parameters
        ----------
        pat : str
            Pattern to find
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory must be able to hold at least size() of int32 values.

        """
        rtn = pyniNVStrings.n_count(self.m_cptr, pat, devptr)
        return rtn

    def startswith(self, pat, devptr=0):
        """
        Return array of boolean values with True for the strings where the
        specified string is at the beginning.

        Parameters
        ----------
        pat : str
            Pattern to find. Regular expressions are not accepted.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory must be able to hold at least size() of np.byte values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.startswith('h'))
        [True, False, False]

        """
        rtn = pyniNVStrings.n_startswith(self.m_cptr, pat, devptr)
        return rtn

    def endswith(self, pat, devptr=0):
        """
        Return array of boolean values with True for the strings
        where the specified string is at the end.

        Parameters
        ----------
        pat : str
            Pattern to find. Regular expressions are not accepted.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory must be able to hold at least size() of np.byte values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.endswith('d'))
        [False, False, True]

        """
        rtn = pyniNVStrings.n_endswith(self.m_cptr, pat, devptr)
        return rtn

    def extract_record(self, pat):
        """
        Extract string from the first match of regular expression pat.
        A new array of nvstrings is created for each string in this instance.

        Parameters
        ----------
        pat : str
            The regex pattern with group capture syntax

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["a1","b2","c3"])
        >>> for result in s.extract_record('([ab])(\\d)'):
        ...     print(result)
        ["a","1"]
        ["b","2"]
        [None,None]

        """
        strs = pyniNVStrings.n_extract_record(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def extract(self, pat):
        """
        Extract string from the first match of regular expression pat.
        A new array of nvstrings is created by organizing group results
        vertically.

        Parameters
        ----------
        pat : str
            The regex pattern with group capture syntax

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["a1","b2","c3"])
        >>> for result in s.extract('([ab])(\\d)'):
        ...     print(result)
        ["a","b"]
        ["1","2"]
        [None,None]

        """
        strs = pyniNVStrings.n_extract(self.m_cptr, pat)
        rtn = []
        for cptr in strs:
            if cptr != 0:
                rtn.append(nvstrings(cptr))
            else:
                rtn.append(None)
        return rtn

    def isalnum(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only alpha-numeric characters.
        Equivalent to: isalpha() or isdigit() or isnumeric() or isdecimal()

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
        >>> print(s.isalnum())
        [True, True, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isalnum(self.m_cptr, devptr)
        return rtn

    def isalpha(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only alphabetic characters.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
        >>> print(s.isalpha())
        [False, True, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isalpha(self.m_cptr, devptr)
        return rtn

    def isdigit(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only decimal and digit characters.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
        >>> print(s.isdigit())
        [True, False, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isdigit(self.m_cptr, devptr)
        return rtn

    def isspace(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only whitespace characters.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
        >>> print(s.isspace())
        [False, False, False, False, False, True]

        """
        rtn = pyniNVStrings.n_isspace(self.m_cptr, devptr)
        return rtn

    def isdecimal(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain only
        decimal characters -- those that can be used to extract base10 numbers.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
        >>> print(s.isdecimal())
        [True, False, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isdecimal(self.m_cptr, devptr)
        return rtn

    def isnumeric(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only numeric characters. These include digit and numeric characters.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['1234', 'de', '1.75', '-34', '+9.8', ' '])
        >>> print(s.isnumeric())
        [True, False, False, False, False, False]

        """
        rtn = pyniNVStrings.n_isnumeric(self.m_cptr, devptr)
        return rtn

    def islower(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only lowercase characters.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['hello', 'Goodbye'])
        >>> print(s.islower())
        [True, False]

        """
        rtn = pyniNVStrings.n_islower(self.m_cptr, devptr)
        return rtn

    def isupper(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        only uppercase characters.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['hello', 'Goodbye'])
        >>> print(s.isupper())
        [False, True]

        """
        rtn = pyniNVStrings.n_isupper(self.m_cptr, devptr)
        return rtn

    def is_empty(self, devptr=0):
        """
        Return array of boolean values with True for strings that contain
        at least one character.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['hello', 'goodbye', '', None])
        >>> print(s.isempty())
        [True, True,False,False]

        """
        rtn = pyniNVStrings.n_is_empty(self.m_cptr, devptr)
        return rtn

    def translate(self, table):
        """
        Translate individual characters to new characters using
        the provided table.

        Parameters
        ----------
        pat : dict
            Use str.maketrans() to build the mapping table.
            Unspecified characters are unchanged.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.translate(str.maketrans('elh','ELH')))
        ['HELLo', 'tHErE', 'worLd]
        """
        rtn = pyniNVStrings.n_translate(self.m_cptr, table)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def sort(self, stype=2, asc=True, nullfirst=True):
        """
        Sort this list by name (2) or length (1) or both (3).
        Sorting can help improve performance for other operations.

        Parameters
        ----------
        stype : int
            Type of sort to use.
            If stype is 1, strings will be sorted by length
            If stype is 2, strings will be sorted alphabetically by name
            If stype is 3, strings will be sorted by length and then
            alphabetically
        asc : bool
            Whether to sort ascending (True) or descending (False)
        nullfirst : bool
            Null strings are sorted to the beginning by default

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["aaa", "bb", "aaaabb"])
        >>> print(s.sort(3))
        ['bb', 'aaa', 'aaaabb']

        """
        rtn = pyniNVStrings.n_sort(self.m_cptr, stype, asc, nullfirst)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def order(self, stype=2, asc=True, nullfirst=True, devptr=0):
        """
        Sort this list by name (2) or length (1) or both (3).
        This sort only provides the new indexes and does not reorder the
        managed strings.

        Parameters
        ----------
        stype : int
            Type of sort to use.
            If stype is 1, strings will be sorted by length
            If stype is 2, strings will be sorted alphabetically by name
            If stype is 3, strings will be sorted by length and then
            alphabetically
        asc : bool
            Whether to sort ascending (True) or descending (False)
        nullfirst : bool
            Null strings are sorted to the beginning by default
        devptr : GPU memory pointer
            Where index values will be written.
            Must be able to hold at least size() of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["aaa", "bb", "aaaabb"])
        >>> print(s.order(2))
        [1, 0, 2]

        """
        rtn = pyniNVStrings.n_order(self.m_cptr, stype, asc, nullfirst, devptr)
        return rtn

    def sublist(self, indexes, count=0):
        """ Calls gather() """
        return self.gather(indexes, count)

    def gather(self, indexes, count=0):
        """
        Return a new list of strings from this instance.

        Parameters
        ----------
        indexes : List of ints or GPU memory pointer
            0-based indexes of strings to return from an nvstrings object.
            Values must be of type int32.
        count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.gather([0, 2]))
        ['hello', 'world']

        """
        rtn = pyniNVStrings.n_gather(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def scatter(self, strs, indexes):
        """
        Return a new list of strings combining this instance
        with the provided strings using the specified indexes.

        Parameters
        ----------
        strs : nvstrings
            Strings to be combined with this instance.
        indexes : List of ints or GPU memory pointer
            0-based indexes of strings indicating which strings
            should be replaced by the corresponding element in strs.
            Values must be of type int32. The number of values
            should be the same as strs.size(). Repeated indexes
            will cause undefined results.

        Examples
        --------
        >>> import nvstrings
        >>> s1 = nvstrings.to_device(["a","b","c","d"])
        >>> s2 = nvstrings.to_device(["e","f"])
        >>> print(s1.scatter(s2, [1, 3]))
        ['a', 'e', 'c', 'f']

        """
        rtn = pyniNVStrings.n_scatter(self.m_cptr, strs, indexes)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def scalar_scatter(self, str, indexes, count):
        """
        Return a new list of strings placing the specified
        string at the provided indexes.

        Parameters
        ----------
        str : str
            String to be placed
        indexes : List of ints or GPU memory pointer
            0-based indexes of strings indicating the positions
            where the str parameter should be placed.
            Existing strings at those positions will be replaced.
            Values must be of type int32.
        count : int
            Number of elements in the indexes parameter.

        Examples
        --------
        >>> import nvstrings
        >>> s1 = nvstrings.to_device(["a","b","c","d"])
        >>> print(s1.scalar_scatter("_", [1, 3]))
        ['a', '_', 'c', '_']

        """
        rtn = pyniNVStrings.n_scalar_scatter(self.m_cptr, str, indexes, count)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def remove_strings(self, indexes, count=0):
        """
        Remove the specified strings and return a new instance.

        Parameters
        ----------
        indexes : List of ints
            0-based indexes of strings to remove from an nvstrings object
            If this parameter is pointer to device memory, count parm is
            required.
        count : int
            Number of ints if indexes parm is a device pointer.
            Otherwise it is ignored.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hello","there","world"])
        >>> print(s.remove_strings([0, 2]))
        ['there']

        """
        rtn = pyniNVStrings.n_remove_strings(self.m_cptr, indexes, count)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def add_strings(self, strs):
        """
        Add the specified strings to the end of these strings
        and return a new instance.

        Parameters
        ----------
        strs : nvstrings or list
            1 or more nvstrings objects

        Examples
        --------
        >>> import nvstrings
        >>> s1 = nvstrings.to_device(['apple','pear','banana'])
        >>> s2 = nvstrings.to_device(['orange','pear'])
        >>> s3 = s1.add_strings(s2)
        >>> print(s3)
        ['apple', 'pear', banana', 'orange', 'pear']

        """
        rtn = pyniNVStrings.n_add_strings(self.m_cptr, strs)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def copy(self):
        """
        Return a new instance as a copy of this instance.

        Examples
        --------
        >>> import nvstrings
        >>> s1 = nvstrings.to_device(['apple','pear','banana'])
        >>> s2 = s1.copy()
        >>> print(s2)
        ['apple', 'pear', banana']

        """
        rtn = pyniNVStrings.n_copy(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def find_multiple(self, strs, devptr=0):
        """
        Return a 'matrix' of find results for each of the string in the
        strs parameter.

        Each row is an array of integers identifying the first location
        of the corresponding provided string.

        Parameters
        ----------
        strs : nvstrings
            Strings to find in each of the strings in this instance.
        devptr : GPU memory pointer
            Optional device memory pointer to hold the results.
            Memory size must be able to hold at least size()*strs.size()
            of int32 values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["hare","bunny","rabbit"])
        >>> t = nvstrings.to_device(["a","e","i","o","u"])
        >>> print(s.find_multiple(t))
        [[1, 3, -1, -1, -1], [-1, -1, -1, -1, 1], [1, -1, 4, -1, -1]]

        """
        rtn = pyniNVStrings.n_find_multiple(self.m_cptr, strs, devptr)
        return rtn

    def get_info(self):
        """
        Return a dictionary of information about the strings
        in this instance. This could be helpful in understanding
        the makeup of the data and how the operations may perform.

        """
        rtn = pyniNVStrings.n_get_info(self.m_cptr)
        return rtn

    def url_encode(self):
        """
        URL-encode each string and return as a new instance.
        No format checking is performed. All characters are encoded except
        for ASCII letters, digits, and these characters: '.','_','-','~'.
        Encoding converts to hex using UTF-8 encoded bytes.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(["a/b-c/d","E F.G","1-2,3"])
        >>> print(s.url_encode())
        ['a%2Fb-c%2Fd', 'E%20F.G', '1-2%2C3']

        """
        rtn = pyniNVStrings.n_url_encode(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn

    def url_decode(self):
        """
        URL-decode each string and return as a new instance.
        No format checking is performed. All characters are
        expected to be encoded as UTF-8 hex values.

        Examples
        --------
        >>> import nvstrings
        >>> s = nvstrings.to_device(['A%2FB-C%2FD', 'e%20f.g', '4-5%2C6'])
        >>> print(s.url_decode())
        ['A/B-C/D', 'e f.g', '4-5,6']

        """
        rtn = pyniNVStrings.n_url_decode(self.m_cptr)
        if rtn is not None:
            rtn = nvstrings(rtn)
        return rtn
