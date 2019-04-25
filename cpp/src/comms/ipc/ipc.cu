#include <cudf.h>

#pragma diag_suppress set_but_not_used
#include <cudf/ipc_generated/Schema_generated.h>
#pragma diag_default set_but_not_used
#include <cudf/ipc_generated/Message_generated.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>

using namespace org::apache::arrow;

namespace {

using namespace arrow;

#if ARROW_VERSION < 800
static std::string GetBufferTypeName(BufferType type) {
  switch (type) {
    case BufferType::DATA:
      return "DATA";
    case BufferType::OFFSET:
      return "OFFSET";
    case BufferType::TYPE:
      return "TYPE";
    case BufferType::VALIDITY:
      return "VALIDITY";
    default:
      break;
  }
  return "UNKNOWN";
}
#endif

static std::string GetTypeName(Type::type id) {
    switch (id) {
    #define SHOW_TYPE_NAME(K) case Type::K: return #K;
    SHOW_TYPE_NAME(NA)
    SHOW_TYPE_NAME(BOOL)
    SHOW_TYPE_NAME(UINT8)
    SHOW_TYPE_NAME(INT8)
    SHOW_TYPE_NAME(UINT16)
    SHOW_TYPE_NAME(INT16)
    SHOW_TYPE_NAME(UINT32)
    SHOW_TYPE_NAME(INT32)
    SHOW_TYPE_NAME(UINT64)
    SHOW_TYPE_NAME(INT64)
    SHOW_TYPE_NAME(HALF_FLOAT)
    SHOW_TYPE_NAME(FLOAT)
    SHOW_TYPE_NAME(DOUBLE)
    SHOW_TYPE_NAME(STRING)
    SHOW_TYPE_NAME(BINARY)
    SHOW_TYPE_NAME(FIXED_SIZE_BINARY)
    SHOW_TYPE_NAME(DATE32)
    SHOW_TYPE_NAME(DATE64)
    SHOW_TYPE_NAME(TIMESTAMP)
    SHOW_TYPE_NAME(TIME32)
    SHOW_TYPE_NAME(TIME64)
    SHOW_TYPE_NAME(INTERVAL)
    SHOW_TYPE_NAME(DECIMAL)
    SHOW_TYPE_NAME(LIST)
    SHOW_TYPE_NAME(STRUCT)
    SHOW_TYPE_NAME(UNION)
    SHOW_TYPE_NAME(DICTIONARY)
    SHOW_TYPE_NAME(MAP)
    #undef SHOW_TYPE_NAME
  }
  return "UNKNOWN";
}
}

class IpcParser {
public:

    typedef std::unique_ptr<const char []> unique_bytes_type;

    class ParseError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    struct MessageInfo {
        const void *header;
        int64_t body_length;
        flatbuf::MessageHeader type;
        flatbuf::MetadataVersion version;
    };

    struct LayoutDesc {
        int bitwidth;
        std::string vectortype;
    };

    struct FieldDesc {
        std::string name;
        std::string type;
        std::vector<LayoutDesc> layouts;
    };

    struct BufferDesc {
        int64_t offset, length;
    };

    struct DTypeDesc {
        std::string name;
        int bitwidth;
    };

    struct NodeDesc {
        std::string name;
        int64_t length;
        int64_t null_count;
        BufferDesc null_buffer, data_buffer;
        DTypeDesc dtype;
    };

    IpcParser()
    :_d_buffer(nullptr), _d_curptr(nullptr), _d_data_body(nullptr), _failed(false)
    { /* empty */ }

    void open(const uint8_t *schema, size_t length) {
        try {
            read_schema(schema, length);
        } catch ( ParseError const &e ) {
            std::ostringstream oss;
            oss << "ParseError: " << e.what();
            _error_message = oss.str();
            _failed = true;
        }
    }

    void open_recordbatches(const uint8_t *recordbatches, size_t length) {
        try {
            read_record_batch(recordbatches, length);
        } catch ( ParseError const &e ) {
            std::ostringstream oss;
            oss << "ParseError: " << e.what();
            _error_message = oss.str();
            _failed = true;
        }
    }

    bool is_failed() const {
        return _failed;
    }

    const std::string& get_error() const {
        return _error_message;
    }

    /*
     * Returns the GPU pointer to the start of the data region.
     */
    const void* get_data() const {
        return static_cast<const void*>(_d_data_body);
    }

    int64_t get_data_offset() const {
        return _d_data_body - _d_buffer;
    }

    /*
     * Returns the layout information in json.
     * The json contains a list metadata for each column.
     */
    const std::string& get_layout_json() {
        if ( _json_output.size() == 0 ) {
            std::ostringstream oss;
            oss << "[";
            int ct = 0;
            for (auto i=_nodes.begin(); i!=_nodes.end(); ++i, ++ct) {
                if ( ct > 0 ) {
                    oss << ", ";
                }
                jsonify_node(oss, *i);
            }
            oss << "]";
            _json_output = oss.str();
        }
        return _json_output;
    }

    const std::string& get_schema_json() {
        if ( _json_schema_output.size() == 0 ) {
            // To JSON
#if ARROW_VERSION < 800
	    std::unique_ptr<arrow::ipc::JsonWriter> json_writer;	    
            arrow::ipc::JsonWriter::Open(_schema, &json_writer);
            json_writer->Finish(&_json_schema_output);
#else
	    std::unique_ptr<arrow::ipc::internal::json::JsonWriter> json_writer;
            arrow::ipc::internal::json::JsonWriter::Open(_schema, &json_writer);
            json_writer->Finish(&_json_schema_output);
#endif
        }
        return _json_schema_output;
    }

protected:

    void jsonify_node(std::ostream &os, const NodeDesc &node) {
        os << "{";

        os << "\"name\": " << '"' << node.name << '"';
        os << ", ";

        os << "\"length\": " << node.length;
        os << ", ";

        os << "\"null_count\": " << node.null_count;
        os << ", ";

        os << "\"dtype\": ";
        jsonify_dtype(os, node.dtype);
        os << ", ";

        os << "\"data_buffer\": ";
        jsonify_buffer(os, node.data_buffer);
        os << ", ";

        os << "\"null_buffer\": ";
        jsonify_buffer(os, node.null_buffer);

        os << "}";
    }

    void jsonify_dtype(std::ostream &os, const DTypeDesc &dtype) {
        os << "{";

        os << "\"name\": " << '"' << dtype.name << '"';
        os << ", ";

        os << "\"bitwidth\": " << dtype.bitwidth;

        os << "}";
    }

    void jsonify_buffer(std::ostream &os, const BufferDesc &buffer) {
        os << "{";

        os << "\"length\": " << buffer.length;
        os << ", ";

        os << "\"offset\": " << buffer.offset;

        os << "}";
    }

    void read_schema(const uint8_t *schema_buf, size_t length) {
        if (_fields.size() || _nodes.size()) {
            throw ParseError("cannot open more than once");
        }
        // Use Arrow to load the schema
        const auto payload = std::make_shared<arrow::Buffer>(schema_buf, length);
        auto buffer = std::make_shared<io::BufferReader>(payload);
#if ARROW_VERSION < 800
	std::shared_ptr<ipc::RecordBatchStreamReader> reader;
#else
        std::shared_ptr<ipc::RecordBatchReader> reader;
#endif
        auto status = ipc::RecordBatchStreamReader::Open(buffer, &reader);
        if ( !status.ok() ) {
	  throw ParseError(status.message());
	}
        _schema = reader->schema();
        if (!_schema) throw ParseError("failed to parse schema");
        // Parse the schema
        parse_schema(_schema);
    }

    void read_record_batch(const uint8_t *recordbatches, size_t length) {
        _d_curptr = _d_buffer = recordbatches;

        int size = read_msg_size();
        auto header_buf = read_bytes(size);
        auto header = parse_msg_header(header_buf);

#if ARROW_VERSION < 800
	if ( header.version != flatbuf::MetadataVersion_V3 )
	  throw ParseError("unsupported metadata version, expected V3 got "\
			   + std::string(flatbuf::EnumNameMetadataVersion(header.version)));
#else
	if ( header.version != flatbuf::MetadataVersion_V4 )
	  throw ParseError("unsupported metadata version, expected V4 got "\
			   + std::string(flatbuf::EnumNameMetadataVersion(header.version)));
#endif
        if ( header.body_length <= 0) {
            throw ParseError("recordbatch should have a body");
        }
        // store the current ptr as the data ptr
        _d_data_body = _d_curptr;

        parse_record_batch(header);
    }

    MessageInfo parse_msg_header(const unique_bytes_type & header_buf) {
        auto msg = flatbuf::GetMessage(header_buf.get());
        MessageInfo mi;
        mi.header = msg->header();
        mi.body_length = msg->bodyLength();
        mi.type = msg->header_type();
        mi.version = msg->version();
        return mi;
    }

    void parse_schema(std::shared_ptr<arrow::Schema> schema) {
        auto fields = schema->fields();

        _fields.reserve(fields.size());
        
        for (auto field : fields) {
            _fields.push_back(FieldDesc());
            auto & out_field = _fields.back();

            out_field.name = field->name();
            out_field.type = GetTypeName(field->type()->id());
#if ARROW_VERSION < 800
            auto layouts = field->type()->GetBufferLayout();
            for ( int j=0; j < layouts.size(); ++j ) {
                auto layout = layouts[j];
                LayoutDesc layout_desc;
                layout_desc.bitwidth = layout.bit_width();
                layout_desc.vectortype = GetBufferTypeName(layout.type());
                out_field.layouts.push_back(layout_desc);
            }
#endif
        }
    }

    void parse_record_batch(MessageInfo msg) {
        if ( msg.type != flatbuf::MessageHeader_RecordBatch ) {
            throw ParseError("expecting recordbatch type");
        }
        auto rb = static_cast<const flatbuf::RecordBatch*>(msg.header);
        int node_ct = rb->nodes()->Length();
        int buffer_ct = rb->buffers()->Length();

        int buffer_per_node = 2;
        if ( node_ct * buffer_per_node != buffer_ct ) {
            throw ParseError("unexpected: more than 2 buffers per node!?");
        }

        _nodes.reserve(node_ct);
        for ( int i=0; i < node_ct; ++i ) {
            const auto &fd = _fields[i];
            auto node = rb->nodes()->Get(i);

            _nodes.push_back(NodeDesc());
            auto &out_node = _nodes.back();
	    
            for ( int j=0; j < buffer_per_node; ++j ) {
                auto buf = rb->buffers()->Get(i * buffer_per_node + j);
#if ARROW_VERSION < 800
                if ( buf->page() != -1 ) {
                    std::cerr << "buf.Page() != -1; metadata format changed!\n";
                }
#endif
                BufferDesc bufdesc;
                bufdesc.offset = buf->offset();
                bufdesc.length = buf->length();

#if ARROW_VERSION < 800
                const auto &layout = fd.layouts[j];
                if ( layout.vectortype == "DATA" ) {
                    out_node.data_buffer = bufdesc;
                    out_node.dtype.name = fd.type;
                    out_node.dtype.bitwidth = layout.bitwidth;
                } else if ( layout.vectortype == "VALIDITY" ) {
                    out_node.null_buffer = bufdesc;
                } else {
                    throw ParseError("unsupported vector type");
                }
#else
		if (j==0) // assuming first buffer is null bitmap
                    out_node.null_buffer = bufdesc;
                else {
                    out_node.data_buffer = bufdesc;
                    out_node.dtype.name = fd.type;
                    out_node.dtype.bitwidth = (bufdesc.length / node->length()) * 8;
		}
#endif
            }

	    assert(out_node.null_buffer.length <= out_node.data_buffer.length); // check the null bitmap assumption

            out_node.name = fd.name;
            out_node.length = node->length();
            out_node.null_count = node->null_count();

        }
    }

    unique_bytes_type read_bytes(size_t size) {
        if (size <= 0) {
            throw ParseError("attempt to read zero or negative bytes");
        }
        char *buf = new char[size];
        if (cudaSuccess != cudaMemcpy(buf, _d_curptr,  size,
                                      cudaMemcpyDeviceToHost) )
            throw ParseError("cannot read value");
        _d_curptr += size;
        return unique_bytes_type(buf);
    }

    template<typename T>
    void read_value(T &val) {
        if (cudaSuccess != cudaMemcpy(&val, _d_curptr,  sizeof(T),
                                      cudaMemcpyDeviceToHost) )
            throw ParseError("cannot read value");
        _d_curptr += sizeof(T);
    }

    int read_msg_size() {
        int size;
        read_value(size);
        if (size <= 0) {
            throw ParseError("non-positive message size");
        }
        return size;
    }

private:
    const uint8_t *_d_buffer;
    const uint8_t *_d_curptr;
    const uint8_t *_d_data_body;

    std::shared_ptr<arrow::Schema> _schema;

    std::vector<FieldDesc> _fields;
    std::vector<NodeDesc> _nodes;
    bool _failed;
    std::string _error_message;
    // cache
    std::string _json_output;
    std::string _json_schema_output;
};

gdf_ipc_parser_type* cffi_wrap(IpcParser* obj){
    return reinterpret_cast<gdf_ipc_parser_type*>(obj);
}

IpcParser* cffi_unwrap(gdf_ipc_parser_type* hdl){
    return reinterpret_cast<IpcParser*>(hdl);
}

gdf_ipc_parser_type* gdf_ipc_parser_open(const uint8_t *schema, size_t length) {
    IpcParser *parser = new IpcParser;
    

    parser->open(schema, length);

    return cffi_wrap(parser);
}

void gdf_ipc_parser_close(gdf_ipc_parser_type *handle) {
    delete cffi_unwrap(handle);
}

int gdf_ipc_parser_failed(gdf_ipc_parser_type *handle) {
    return cffi_unwrap(handle)->is_failed();
}


const char *gdf_ipc_parser_get_schema_json(gdf_ipc_parser_type *handle) {
    return cffi_unwrap(handle)->get_schema_json().c_str();
}


const char* gdf_ipc_parser_get_layout_json(gdf_ipc_parser_type *handle) {
    return cffi_unwrap(handle)->get_layout_json().c_str();
}

const char* gdf_ipc_parser_get_error(gdf_ipc_parser_type *handle) {
    return cffi_unwrap(handle)->get_error().c_str();
}

const void* gdf_ipc_parser_get_data(gdf_ipc_parser_type *handle) {
    return cffi_unwrap(handle)->get_data();
}

int64_t gdf_ipc_parser_get_data_offset(gdf_ipc_parser_type *handle) {
    return cffi_unwrap(handle)->get_data_offset();
}

void gdf_ipc_parser_open_recordbatches(gdf_ipc_parser_type *handle,
                                       const uint8_t *recordbatches,
                                       size_t length)
{
    return cffi_unwrap(handle)->open_recordbatches(recordbatches, length);
}
