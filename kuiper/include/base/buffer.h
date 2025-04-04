#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;
  void* ptr_ = nullptr;
  bool use_external_ = false;  //是否拥有这块数据的所有权, 在~Buffer()中来定义是否需要释放这块数据
                                // use_external_=false 说明该buffer拥有该内存,这块资源需要buffer来管理
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  void copy_from(const Buffer& buffer) const;

  void copy_from(const Buffer* buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  std::shared_ptr<Buffer> get_shared_from_this();

  bool is_external() const;
};
}  // namespace base

#endif