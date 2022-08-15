import gdb
import sys

if sys.version_info[0] > 2:
    Iterator = object
else:
    # "Polyfill" for Python2 Iterator interface
    class Iterator:
        def next(self):
            return self.__next__()


class HostIterator(Iterator):
    """Iterates over arrays in host memory"""

    def __init__(self, start, size):
        self.item = start
        self.size = size
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        elt = self.item.dereference()
        count = self.count
        self.item = self.item + 1
        self.count = self.count + 1
        return ('[%d]' % count, elt)


class DeviceIterator(Iterator):
    """Iterates over arrays in device memory by copying chunks into host memory"""

    def __init__(self, start, size):
        self.exec = exec
        self.item = start
        self.size = size
        self.count = 0
        self.buffer = None
        self.sizeof = self.item.dereference().type.sizeof
        self.buffer_start = 0
        # At most 1 MB or size, at least 1
        self.buffer_size = min(size, max(1, 2 ** 20 // self.sizeof))
        self.buffer = gdb.parse_and_eval(
            '(void*)malloc(%s)' % (self.buffer_size * self.sizeof))
        self.buffer.fetch_lazy()
        self.buffer_count = self.buffer_size
        self.update_buffer()

    def update_buffer(self):
        if self.buffer_count >= self.buffer_size:
            self.buffer_item = gdb.parse_and_eval(
                hex(self.buffer)).cast(self.item.type)
            self.buffer_count = 0
            self.buffer_start = self.count
            device_addr = hex(self.item.dereference().address)
            buffer_addr = hex(self.buffer)
            size = min(self.buffer_size, self.size -
                        self.buffer_start) * self.sizeof
            status = gdb.parse_and_eval(
                '(cudaError)cudaMemcpy(%s, %s, %d, cudaMemcpyDeviceToHost)' % (buffer_addr, device_addr, size))
            if status != 0:
                raise gdb.MemoryError(
                    'memcpy from device failed: %s' % status)

    def __del__(self):
        gdb.parse_and_eval('(void)free(%s)' %
                            hex(self.buffer)).fetch_lazy()

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.size:
            raise StopIteration
        self.update_buffer()
        elt = self.buffer_item.dereference()
        self.buffer_item = self.buffer_item + 1
        self.buffer_count = self.buffer_count + 1
        count = self.count
        self.item = self.item + 1
        self.count = self.count + 1
        return ('[%d]' % count, elt)


class CudfHostSpanPrinter(gdb.printing.PrettyPrinter):
    """Print a cudf::host_span"""

    def __init__(self, val):
        self.val = val
        self.pointer = val['_data']
        self.size = int(val['_size'])

    def children(self):
        return HostIterator(self.pointer, self.size)

    def to_string(self):
        typename = str(self.val.type)
        return ('%s of length %d at %s' % (typename, self.size, hex(self.pointer)))

    def display_hint(self):
        return 'array'


class CudfDeviceSpanPrinter(gdb.printing.PrettyPrinter):
    """Print a cudf::device_span"""

    def __init__(self, val):
        self.val = val
        self.pointer = val['_data']
        self.size = int(val['_size'])

    def children(self):
        return DeviceIterator(self.pointer, self.size)

    def to_string(self):
        typename = str(self.val.type)
        return ('%s of length %d at %s' % (typename, self.size, hex(self.pointer)))

    def display_hint(self):
        return 'array'


class ThrustVectorPrinter(gdb.printing.PrettyPrinter):
    """Print a thrust::*_vector"""

    def __init__(self, val):
        self.val = val
        self.pointer = val['m_storage']['m_begin']['m_iterator']
        self.size = int(val['m_size'])
        self.capacity = int(val['m_storage']['m_size'])
        self.is_device = False
        if str(self.pointer.type).startswith("thrust::device_ptr"):
            self.pointer = self.pointer['m_iterator']
            self.is_device = True

    def children(self):
        if self.is_device:
            return DeviceIterator(self.pointer, self.size)
        else:
            return HostIterator(self.pointer, self.size)

    def to_string(self):
        typename = str(self.val.type)
        return ('%s of length %d, capacity %d' % (typename, self.size, self.capacity))

    def display_hint(self):
        return 'array'


class ThrustReferencePrinter(gdb.printing.PrettyPrinter):
    """Print a thrust::device_reference"""

    def __init__(self, val):
        self.val = val
        self.pointer = val['ptr']['m_iterator']
        self.type = self.pointer.dereference().type
        sizeof = self.type.sizeof
        self.buffer = gdb.parse_and_eval('(void*)malloc(%s)' % sizeof)
        device_addr = hex(self.pointer)
        buffer_addr = hex(self.buffer)
        status = gdb.parse_and_eval('(cudaError)cudaMemcpy(%s, %s, %d, cudaMemcpyDeviceToHost)' % (
            buffer_addr, device_addr, sizeof))
        if status != 0:
            raise gdb.MemoryError('memcpy from device failed: %s' % status)
        self.buffer_val = gdb.parse_and_eval(
            hex(self.buffer)).cast(self.pointer.type).dereference()

    def __del__(self):
        gdb.parse_and_eval('(void)free(%s)' % hex(self.buffer)).fetch_lazy()

    def children(self):
        return []

    def to_string(self):
        typename = str(self.val.type)
        return ('(%s) @%s: %s' % (typename, self.pointer, self.buffer_val))

    def display_hint(self):
        return None


class RmmDeviceUVectorPrinter(gdb.printing.PrettyPrinter):
    """Print a rmm::device_uvector"""

    def __init__(self, val):
        self.val = val
        el_type = val.type.template_argument(0)
        self.pointer = val['_storage']['_data'].cast(el_type.pointer())
        self.size = int(val['_storage']['_size']) // el_type.sizeof
        self.capacity = int(val['_storage']['_capacity']) // el_type.sizeof

    def children(self):
        return DeviceIterator(self.pointer, self.size)        

    def to_string(self):
        typename = str(self.val.type)
        return ('%s of length %d, capacity %d' % (typename, self.size, self.capacity))

    def display_hint(self):
        return 'array'


# Workaround to avoid using the pretty printer on things like std::vector<int>::iterator
def is_template_type_not_alias(typename):
    loc = typename.find('<')
    if loc is None:
        return False
    depth = 0
    for char in typename[typename.find('<'):-1]:
        if char == '<':
            depth += 1
        if char == '>':
            depth -= 1
        if depth == 0:
            return False
    return True


def template_match(typename, template_name):
    return typename.startswith(template_name + "<") and typename.endswith(">")


def lookup_cudf_type(val):
    if not str(val.type.unqualified()).startswith('cudf::'):
        return None
    suffix = str(val.type.unqualified())[6:]
    if not is_template_type_not_alias(suffix):
        return None
    if template_match(suffix, 'host_span'):
        return CudfHostSpanPrinter(val)
    if template_match(suffix, 'device_span'):
        return CudfDeviceSpanPrinter(val)
    return None


def lookup_rmm_type(val):
    if not str(val.type.unqualified()).startswith('rmm::'):
        return None
    suffix = str(val.type.unqualified())[5:]
    if not is_template_type_not_alias(suffix):
        return None
    if template_match(suffix, 'device_uvector'):
        return RmmDeviceUVectorPrinter(val)
    return None


def lookup_thrust_type(val):
    if not str(val.type.unqualified()).startswith('thrust::'):
        return None
    suffix = str(val.type.unqualified())[8:]
    if not is_template_type_not_alias(suffix):
        return None
    if template_match(suffix, 'host_vector') or template_match(suffix, 'device_vector'):
        return ThrustVectorPrinter(val)
    elif int(gdb.VERSION.split(".")[0]) >= 10 and template_match(suffix, 'device_reference'):
        return ThrustReferencePrinter(val)
    return None


gdb.pretty_printers.append(lookup_cudf_type)
gdb.pretty_printers.append(lookup_rmm_type)
gdb.pretty_printers.append(lookup_thrust_type)
