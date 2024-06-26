/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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
*******************************************************************************/

#include "xpu/sycl/usm_memory_storage.hpp"

#include "common/memory.hpp"
#include "common/memory_map_manager.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

namespace {
template <::sycl::access_mode mode>
memory_arg_t<mode> get_memory_arg(const usm_memory_storage_t *storage,
        stream_t *stream, ::sycl::handler &cgh) {
    auto *sycl_stream_impl
            = utils::downcast<xpu::sycl::stream_impl_t *>(stream->impl());
    return {storage->usm_ptr(),
            sycl_stream_impl->get_dummy_accessor<mode>(cgh)};
}

} // namespace

struct map_usm_tag;

status_t usm_memory_storage_t::map_data(
        void **mapped_ptr, stream_t *stream, size_t size) const {
    void *usm_ptr = this->usm_ptr(); // shadowing is bad

    if (is_host_accessible()) {
        *mapped_ptr = usm_ptr;
        return status::success;
    }

    if (!usm_ptr || size == 0) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    if (!stream) CHECK(engine()->get_service_stream(stream));

    ::sycl::queue sycl_queue
            = *utils::downcast<xpu::sycl::stream_impl_t *>(stream->impl())
                       ->queue();

    void *host_ptr = ::sycl::malloc_host(size, sycl_queue.get_context());
    if (!host_ptr) return status::out_of_memory;

    sycl_queue.wait_and_throw();
    sycl_queue.memcpy(host_ptr, usm_ptr, size).wait();

    *mapped_ptr = host_ptr;
    auto unmap_callback = [usm_ptr, size](stream_t *stream, void *mapped_ptr) {
        ::sycl::queue sycl_queue
                = *utils::downcast<xpu::sycl::stream_impl_t *>(stream->impl())
                           ->queue();
        sycl_queue.wait_and_throw();
        sycl_queue.memcpy(usm_ptr, mapped_ptr, size).wait();
        ::sycl::free(mapped_ptr, sycl_queue.get_context());
        return status::success;
    };

    auto &map_manager = memory_map_manager_t<map_usm_tag>::instance();
    return map_manager.map(this, stream, *mapped_ptr, unmap_callback);
}

status_t usm_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr || is_host_accessible()) return status::success;

    if (!stream) CHECK(engine()->get_service_stream(stream));
    auto &map_manager = memory_map_manager_t<map_usm_tag>::instance();
    return map_manager.unmap(this, stream, mapped_ptr);
}

in_memory_arg_t usm_memory_storage_t::get_in_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::read>(this, stream, cgh);
}

out_memory_arg_t usm_memory_storage_t::get_out_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::write>(this, stream, cgh);
}

inout_memory_arg_t usm_memory_storage_t::get_inout_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) const {
    return get_memory_arg<::sycl::access::mode::read_write>(this, stream, cgh);
}
} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
