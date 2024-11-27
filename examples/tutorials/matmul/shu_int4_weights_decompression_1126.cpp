/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

namespace {

void init_vector(std::vector<float> &v) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(0, 1);
    for (auto &e : v)
        e = u(gen);
}

void init_vector(std::vector<int32_t> &v) {
    std::mt19937 gen;
    std::uniform_int_distribution<int32_t> u(0, 511);
    for (auto &e : v)
        e = u(gen);
}

// Transpose the INT4 data for the src vector which packs 8 INT4 values as INT32,
// for example, the data babababa is transposed to abababab.
void transpose_s4(const std::vector<int32_t> &src, std::vector<int32_t> &dst, int32_t K, int32_t N) {
    // Ensure dst is the correct size
    dst.resize(src.size());

    // Iterate over the src vector to transpose the INT4 data
    for (int32_t k = 0; k < K; ++k) {
        for (int32_t n = 0; n < N; ++n) {
            // Extract INT4 values from src
            int32_t src_byte = src[k * N + n];
            for (int32_t i = 0; i < 8; ++i) {
                int8_t src_int4 = (src_byte >> (i * 4)) & 0x0F;

                // Calculate destination indices
                int32_t dst_index = (n * K + k) / 8;
                int32_t dst_offset = (n * K + k) % 8;

                // Pack INT4 values into dst
                dst[dst_index] &= ~(0x0F << (dst_offset * 4)); // Clear the destination INT4
                dst[dst_index] |= (src_int4 << (dst_offset * 4)); // Set the destination INT4
            }
        }
    }
}

} // namespace

int32_t number_of_runs = 1;

// Create a MatMul primitive descriptor for the following op:
// C_f16 = A_f16 * (B_s4 - zp_B) * sc_B[:]
//
// Here:
// - Matrices A and C are of f16 data type.
// - The B matrix is stored as s4 with format tag ab, its zero point is zp_B, 
//   and all its dimensions are known. This matrix can be a matrix of compressed
//   weights in an MLP topology.
// - The weights scaling and zero point values are not known at the primitive creation time.
matmul::primitive_desc matmul_pd_create(
        int64_t M, int64_t N, int64_t K, int64_t G_SC, int64_t G_ZP, const engine &eng) {

    memory::desc a_md({M, K}, memory::data_type::f16, memory::format_tag::ab); // M x K layout
    // oneDNN doesn't have a notion of format for zero-points and it's always considered as tag::ab
    // In this example, we align the weights format to match the format tag::ab of the zero-points
    memory::desc b_s4_md({K, N}, memory::data_type::s4, memory::format_tag::ab); // N x K layout
    memory::desc c_md({M, N}, memory::data_type::f16, memory::format_tag::ab); // M x N layout

    // Create attributes and indicate that the alpha and zero points are
    // runtime parameters
    primitive_attr attr;
    // Set scales with multiple scales along K and N dimensions and with groups along K.
    attr.set_scales(DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1), {G_SC, 1}, memory::data_type::f16);

    // Set zero points with s4 data type both along K and N dimensions
    attr.set_zero_points(
            DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {G_ZP, 1}, memory::data_type::s4);

    // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
    // integral primitives (in this example, matmul).
    attr.set_fpmath_mode(fpmath_mode::f16, true);

    // Create a MatMul primitive descriptor
    return matmul::primitive_desc(eng, a_md, b_s4_md, c_md, attr);
}

void prepare_input(memory &A_f16_mem, memory &sc_B_mem, memory &zp_B_mem) {
    int64_t M = A_f16_mem.get_desc().get_dims()[0];
    int64_t N = sc_B_mem.get_desc().get_dims()[0];
    int64_t K = A_f16_mem.get_desc().get_dims()[1];
    int64_t NUM_G_SC = sc_B_mem.get_desc().get_dims()[1];
    int64_t NUM_G_ZP_s4 = zp_B_mem.get_desc().get_dims()[1];

    std::vector<float> A_f32(M * K);
    init_vector(A_f32);
    // Fill A_f16_mem f16 data from A_f32 data filled as f32
    write_to_dnnl_memory(A_f32.data(), A_f16_mem);

    std::vector<float> sc_B(NUM_G_SC * N);
    init_vector(sc_B);
    // Fill sc_B_mem f16 data from sc_B data filled as f32
    write_to_dnnl_memory(sc_B.data(), sc_B_mem);

    // 8 INT4 values are packed as INT32
    int64_t NUM_G_ZP_s32 = NUM_G_ZP_s4 / 8 + 1;
    std::vector<int32_t> zp_transpose_B(NUM_G_ZP_s32 * 1);
    init_vector(zp_transpose_B);
    // Transpose the s4 data to match the format tag::ab
    std::vector<int32_t> zp_B(NUM_G_ZP_s32 * 1);
    transpose_s4(zp_transpose_B, zp_B, NUM_G_ZP_s32 , 1);
    // Fill zp_B_mem s4 data from zp_B data filled as s32
    write_to_dnnl_memory(zp_B.data(), zp_B_mem);
}

void infer(const matmul &matmul_p, int64_t M, int64_t N, int64_t K, int64_t G_SC,
        int64_t G_ZP, const memory &B_s4_mem, const engine &eng) {
    // input of the current layer / operation
    memory A_f16_mem({{M, K}, memory::data_type::f16, {K, 1}}, eng);
    // De-quantization parameters (eg. Scale and Shift)
    const int64_t n_sc_groups = K / G_SC;
    memory sc_B_mem({{N, n_sc_groups}, memory::data_type::f16, {1, N}}, eng);
    // number of groups for zero points 
    const int64_t n_zp_groups = K / G_ZP;
    memory zp_B_mem({{1, n_zp_groups}, memory::data_type::s4, {1, 1}}, eng);

    // the function below fills dnnl::memory with some values
    // these memories, typically, come from the previous layers / operations
    // with meaningful data inside
    prepare_input(A_f16_mem, sc_B_mem, zp_B_mem);

    // output - no initialization required
    memory C_f16_mem({{M, N}, memory::data_type::f16, {N, 1}}, eng);

    stream s(eng);
    for (int32_t run = 0; run < number_of_runs; ++run)
        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_f16_mem}, {DNNL_ARG_WEIGHTS, B_s4_mem},
                        {DNNL_ARG_DST, C_f16_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_B_mem},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                                zp_B_mem}});
    s.wait();
}

void int4_weights_decompression_matmul(engine::kind engine_kind) {
    engine eng(engine_kind, 0);

    const int64_t K = 96;
    const int64_t N = 1000;
    const int64_t M = 100;
    // Quantization Group size for scales
    const int64_t G_SC = K / 2;
    // Quantization Group size for zero points
    const int64_t G_ZP = K / 4;

    auto matmul_pd = matmul_pd_create(M, N, K, G_SC, G_ZP, eng);

    // Original weights stored by packing 8 INT4 values as INT32 in a format tag::ba.
    // oneDNN doesn't have a notion of format for zero-points and it's always considered as tag::ab.
    // The example of memory::desc for transposed weights with format tag::ba is below.
    // memory::desc B_s32_trans_md({K / 8, N}, memory::data_type::s32, memory::format_tag::ba);
    // The example of memory::desc for weights with format tag::ab is below.
    // memory::desc B_s32_md({K / 8, N}, memory::data_type::s32, memory::format_tag::ab);
    // In this example, we transpose the weights data to match the format tag::ab of the zero-points.
    std::vector<int32_t> B_s32_trans_data(N * K / 8);
    init_vector(B_s32_trans_data);

    // Transpose the s4 data to match the format tag::ab
    std::vector<int32_t> B_s32_data(K / 8 * N);
    transpose_s4(B_s32_trans_data, B_s32_data, K / 8, N);

    // This way of constrcuting memory causes segfault on GPU:
    // Fill B_s4_mem data using handle from B_s32 data filled as INT32
    // memory B_s4_mem(matmul_pd.weights_desc(), eng, B_s32_data.data());

    // Fill B_s4_mem data using the write_to_dnnl_memory function from B_s32 data filled as INT32
    memory B_s4_mem(matmul_pd.weights_desc(), eng);
    write_to_dnnl_memory(B_s32_data.data(), B_s4_mem);

    matmul matmul_p(matmul_pd);

    infer(matmul_p, M, N, K, G_SC, G_ZP, B_s4_mem, eng);
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return handle_example_errors(int4_weights_decompression_matmul, engine_kind);
}
