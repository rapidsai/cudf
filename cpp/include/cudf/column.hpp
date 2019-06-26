#include "bitmask.hpp"
#include "types.hpp"

#include <rmm/device_buffer.hpp>

namespace rmm {
// forward decl
class device_buffer;
}  // namespace rmm

namespace cudf {

// forward decl
class column_view;

class column {
  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a size, type, and option to
   * allocate bitmask.
   *
   * Both the data and bitmask are unintialized.
   *
   * @param[in] type The element type
   * @param[in] size The number of elements in the column
   * @param[in] allocate_bitmask Optionally allocate an appropriate sized
   * bitmask
   *---------------------------------------------------------------------------**/
  column(data_type type, int size, bool allocate_bitmask = false);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and a device_buffer for
   * data that will be *deep* copied.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data device_buffer whose data will be *deep* copied
   *---------------------------------------------------------------------------**/
  column(data_type dtype, int size, rmm::device_buffer data);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and a device_buffer for
   * data that will be shallow copied.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements in the column
   * @param[in] data device_buffer whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, int size, rmm::device_buffer&& data);

  /**---------------------------------------------------------------------------*
   * @brief Column constructor that deep copies a `device_buffer` and `bitmask`.
   *
   * @param[in] dtype The element type
   * @param[in] size The number of elements
   * @param[in] data The device buffer to copy
   * @param[in] mask The bitmask to copy
   *---------------------------------------------------------------------------**/
  column(data_type dtype, int size, rmm::device_buffer data, bitmask mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, and device_buffers for data and
   * bitmask that will be *shallow* copied.
   *
   * This constructor uses move semantics to take ownership of the
   *device_buffer's device memory. The `device_buffer` passed into this
   *constructor will not longer be valid to use. Furthermore, it will result in
   *undefined behavior if the device_buffer`s associated memory is modified or
   *freed after invoking this constructor.
   *
   * @param dtype The element type
   * @param[in] size The number of elements in the column
   * @param data device_buffer whose data will be moved from into this column
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, int size, rmm::device_buffer&& data, bitmask&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and deep copied device
   * buffer for data, and moved bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be *deep* copied
   * @param mask bitmask whose data will be moved into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, int size, rmm::device_buffer data, bitmask&& mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column from a type, size, and moved device
   * buffer for data, and deep copied bitmask.
   *
   * @param dtype The element type
   * @param size The number of elements
   * @param data device_buffer whose data will be moved into this column
   * @param mask bitmask whose data will be deep copied into this column
   *---------------------------------------------------------------------------**/
  column(data_type dtype, int size, rmm::device_buffer&& data, bitmask mask);

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column by deep copying the device memory of another
   * column.
   *
   * @param other The other column to copy
   *---------------------------------------------------------------------------**/
  // This won't work because of the unique_ptr member
  // column(column const& other) = default;

  /**---------------------------------------------------------------------------*
   * @brief Construct a new column object by moving the device memory from
   *another column.
   *
   * @param other The other column whose device memory will be moved to the new
   * column
   *---------------------------------------------------------------------------**/
  column(column&& other) = default;

  ~column() = default;
  column& operator=(column const& other) = delete;
  column& operator=(column&& other) = delete;

  column_view view() noexcept {}

  column_view const view() const noexcept {}

 private:
  rmm::device_buffer _data{};  ///< Dense, contiguous, type erased device memory
                               ///< buffer containing the column elements
  bitmask _mask{};             ///< Validity bitmask for columne elements
  data_type _type{INVALID};    ///< Logical type of elements in the column
  std::unique_ptr<column> _other{
      nullptr};  ///< Depending on column's type, may point to
                 ///< another column, e.g., a Dicitionary
};
}  // namespace cudf