/*******************************************************************************
* Copyright 2016-2025 Intel Corporation
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

#ifndef COMMON_SOFTMAX_PD_HPP
#define COMMON_SOFTMAX_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

#define VDISPATCH_SOFTMAX(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, softmax, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_SOFTMAX_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, softmax, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

namespace dnnl {
namespace impl {

struct softmax_fwd_pd_t;

struct softmax_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::softmax;

    const softmax_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::primitive_kind:
                *(primitive_kind_t *)result = desc()->primitive_kind;
                break;
            case query::alg_kind:
                *(alg_kind_t *)result = desc()->alg_kind;
                break;
            case query::axis_s32: *(int *)result = desc()->softmax_axis; break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common softmax aux functions */

    dim_t MB() const { return dst_desc().dims[0]; }
    dim_t C() const { return dst_desc().dims[1]; }
    dim_t D() const { return ndims() >= 5 ? dst_desc().dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? dst_desc().dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? dst_desc().dims[ndims() - 1] : 1; }

    dim_t outer_size() const {
        return utils::array_product(dst_desc().dims, axis());
    }
    dim_t axis_size(bool padded = false) const {
        return padded ? dst_desc().padded_dims[axis()]
                      : dst_desc().dims[axis()];
    }
    dim_t inner_size() const {
        return utils::array_product(
                dst_desc().dims + axis() + 1, ndims() - 1 - axis());
    }

    dim_t axis_stride() const {
        const memory_desc_wrapper dst_d(dst_desc());
        return dst_d.blocking_desc().strides[axis()];
    }

    int axis() const { return desc_.softmax_axis; }
    int ndims() const { return dst_desc().ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(dst_desc()).has_zero_dim();
    }

    alg_kind_t alg_kind() const { return desc()->alg_kind; }
    bool is_softmax() const {
        return utils::one_of(alg_kind(), alg_kind::softmax_accurate,
                alg_kind::softmax_accurate_inf_as_zero);
    }
    bool is_logsoftmax() const { return alg_kind() == alg_kind::softmax_log; }

protected:
    softmax_desc_t desc_;
    const softmax_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t dst_md_;

    softmax_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<softmax_desc_t>(adesc))
        , hint_fwd_pd_(hint_fwd_pd)
        , dst_md_(desc_.dst_desc) {}

private:
    const memory_desc_t &dst_desc() const { return dst_md_; }
};

// NOLINTBEGIN(google-default-arguments)
struct softmax_fwd_pd_t : public softmax_pd_t {
    using base_class = softmax_fwd_pd_t;
    using hint_class = softmax_fwd_pd_t;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE)
            return !types::is_zero_md(workspace_md()) ? arg_usage_t::output
                                                      : arg_usage_t::unused;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return softmax_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->src_desc : &src_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override { return 1 + n_binary_po_inputs(); }
    int n_outputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }

protected:
    memory_desc_t src_md_;

    softmax_fwd_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(adesc, attr, hint_fwd_pd), src_md_(desc_.src_desc) {}

    status_t set_default_formats() {
        if (dst_md()->format_kind != format_kind::any) return status::success;

        if (src_md()->format_kind != format_kind::blocked)
            return status::unimplemented;

        return memory_desc_init_by_blocking_desc(
                dst_md_, src_md_.format_desc.blocking);
    }

    bool attr_scales_ok(const std::vector<int> &supported_args
            = {DNNL_ARG_SRC, DNNL_ARG_DST}) const {
        const auto &scales = attr()->scales_;
        bool ok = scales.has_default_values(supported_args);

        for (const auto &arg : supported_args) {
            if (scales.has_default_values(arg)) continue;

            // TODO: disallow non-int8 scales?
            // const data_type_t dt = arg_md(arg)->data_type;
            // ok = ok && utils::one_of(dt, s8, u8);
            ok = ok && scales.get_mask(arg) == 0;
        }
        return ok;
    }
};
// NOLINTEND(google-default-arguments)

// NOLINTBEGIN(google-default-arguments)
struct softmax_bwd_pd_t : public softmax_pd_t {
    using base_class = softmax_bwd_pd_t;
    using hint_class = softmax_fwd_pd_t;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_DST, DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE)
            return !types::is_zero_md(workspace_md()) ? arg_usage_t::input
                                                      : arg_usage_t::unused;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_DST: return dst_md(0, user_input);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0, user_input);
            default: return softmax_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0) return user_input ? &desc()->dst_desc : &dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_dst_desc : &diff_dst_md_;
        return &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(
            int index = 0, bool user_input = false) const override {
        if (index == 0)
            return user_input ? &desc()->diff_src_desc : &diff_src_md_;
        return &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + (!types::is_zero_md(workspace_md()));
    }
    int n_outputs() const override { return 1; }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;

    softmax_bwd_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}

    status_t set_default_formats() {
        status_t st = status::invalid_arguments;
        if (diff_dst_md_.format_kind == format_kind::any) {
            st = memory_desc_init_by_md_and_dt(
                    diff_dst_md_, dst_md_, diff_dst_md_.data_type);
            if (st != status::success) return st;
        }
        if (diff_src_md_.format_kind == format_kind::any) {
            st = memory_desc_init_by_md_and_dt(
                    diff_src_md_, diff_dst_md_, diff_src_md_.data_type);
            if (st != status::success) return st;
        }
        return status::success;
    }
};
// NOLINTEND(google-default-arguments)

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
