/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
 */

#include <cub/cub.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<class TFrom>
__device__ __forceinline__ float cvt_to_float(TFrom v)
{
    static_assert(std::is_same_v<TFrom, __half> || std::is_same_v<TFrom, __nv_bfloat16>,
                  "cvt_to_float expects a 16-bit type");
    if constexpr (std::is_same_v<TFrom, __half>) {
        return __half2float(v);
    } else {
        return __bfloat162float(v);
    }
}

template<class TTo>
__device__ __forceinline__ TTo cvt_from_float(float v)
{
    static_assert(std::is_same_v<TTo, __half> || std::is_same_v<TTo, __nv_bfloat16>,
                  "cvt_from_float expects a 16-bit type");
    if constexpr (std::is_same_v<TTo, __half>) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

template<class TOut, class TTable, int vec_size>
__global__ void
embeddingLookupKernel(TOut* dst, int dst_stride, const TTable* src, int src_stride, const int* ids, int dim)
{
    const int ti = blockIdx.x;

    const int64_t idx = ids[ti];

    src += idx * src_stride;
    dst += ti * dst_stride;

    for (int di = threadIdx.x * vec_size; di < dim; di += blockDim.x * vec_size) {
        Array<TTable, vec_size> in;
        Ldg(in, &src[di]);
        Array<TOut, vec_size> out;
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            if constexpr (std::is_same_v<TOut, TTable>) {
                out[i] = in[i];
            } else {
                out[i] = cvt_from_float<TOut>(cvt_to_float(in[i]));
            }
        }
        Store(&dst[di], out);
    }
}

void invokeEmbeddingLookup(Ref<Tensor>         out_,
                           const Buffer_<int>& token_ids,
                           const Tensor&       embedding_table,
                           cudaStream_t        st)
{
    auto& out = out_.get();

    TM_CHECK_EQ(out.shape(0), token_ids.size());
    TM_CHECK_EQ(out.shape(1), embedding_table.shape(1));

    int num, dim;
    std::tie(num, dim) = out.shapes(0, 1);

    const bool out_half    = out.dtype() == kHalf;
    const bool out_bf16    = out.dtype() == kBfloat16;
    const bool table_half  = embedding_table.dtype() == kHalf;
    const bool table_bf16  = embedding_table.dtype() == kBfloat16;

    TM_CHECK(out_half || out_bf16)
        << "embedding lookup supports fp16/bf16 out; got " << out.dtype();
    TM_CHECK(table_half || table_bf16)
        << "embedding lookup supports fp16/bf16 tables; got " << embedding_table.dtype();

    auto invoke = [&](auto t_out, auto t_table) {
        using TOut             = decltype(t_out);
        using TTable           = decltype(t_table);
        constexpr int vec_size = sizeof(uint4) / sizeof(TOut);
        TM_CHECK(dim % vec_size == 0) << dim << " " << vec_size;
        const int threads = std::min(dim / vec_size, 1024);
        const int blocks  = num;
        TM_CHECK(out_.get());
        TM_CHECK(token_ids);
        TM_CHECK(embedding_table);
        embeddingLookupKernel<TOut, TTable, vec_size><<<blocks, threads, 0, st>>>(
            (TOut*)out.raw_data(),
            out.stride(0),
            (const TTable*)embedding_table.raw_data(),
            embedding_table.stride(0),
            token_ids.data(),
            dim);
    };

    if (out_half) {
        table_half ? invoke(__half{}, __half{}) : invoke(__half{}, __nv_bfloat16{});
    } else {
        table_half ? invoke(__nv_bfloat16{}, __half{}) : invoke(__nv_bfloat16{}, __nv_bfloat16{});
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T>
__global__ void embeddingLookupInt8Kernel(T*            dst,
                                          int           dst_stride,
                                          const int8_t* src,
                                          int           src_stride,
                                          const T*      scales,
                                          int           scales_stride,
                                          const int*    ids,
                                          int           dim,
                                          int           group)
{
    const int     ti   = blockIdx.x;
    const int64_t idx  = ids[ti];
    const int8_t* row  = src + idx * src_stride;
    const T*      srow = scales + idx * scales_stride;
    T*            out  = dst + ti * dst_stride;

    for (int di = threadIdx.x; di < dim; di += blockDim.x) {
        out[di] = (T)((float)row[di] * (float)srow[di / group]);
    }
}

void invokeEmbeddingLookupInt8(Ref<Tensor>         out_,
                               const Buffer_<int>& token_ids,
                               const Tensor&       table,
                               const Tensor&       scales,
                               int                 group,
                               cudaStream_t        st)
{
    auto& out = out_.get();
    TM_CHECK_EQ(out.shape(0), token_ids.size());
    TM_CHECK_EQ(out.shape(1), table.shape(1));
    TM_CHECK_EQ(table.dtype(), kInt8);
    TM_CHECK_EQ(scales.dtype(), out.dtype());
    TM_CHECK_EQ((int)table.shape(1) % group, 0);
    TM_CHECK_EQ((int)scales.shape(1), (int)table.shape(1) / group);

    const int num = (int)out.shape(0);
    const int dim = (int)out.shape(1);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        const int threads = std::min(dim, 1024);
        embeddingLookupInt8Kernel<T><<<num, threads, 0, st>>>((T*)out.raw_data(),
                                                              (int)out.stride(0),
                                                              (const int8_t*)table.raw_data(),
                                                              (int)table.stride(0),
                                                              (const T*)scales.raw_data(),
                                                              (int)scales.stride(0),
                                                              token_ids.data(),
                                                              dim,
                                                              group);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    if (out.dtype() == kHalf) {
        invoke(half_t{});
    }
    else if (out.dtype() == kBfloat16) {
        invoke(bfloat16_t{});
    }
    else {
        TM_LOG_FATAL("invokeEmbeddingLookupInt8: unsupported out dtype");
    }
}

template<class T>
__global__ void embeddingLookupInt4Kernel(T*             dst,
                                          int            dst_stride,
                                          const uint8_t* src,
                                          int            src_stride,
                                          const T*       scales,
                                          const uint8_t* zeros,
                                          int            scales_stride,
                                          const int*     ids,
                                          int            dim,
                                          int            group)
{
    const int      ti   = blockIdx.x;
    const int64_t  idx  = ids[ti];
    const uint8_t* row  = src + idx * src_stride;
    const T*       srow = scales + idx * scales_stride;
    const uint8_t* zrow = zeros + idx * scales_stride;
    T*             out  = dst + ti * dst_stride;

    for (int di = threadIdx.x; di < dim; di += blockDim.x) {
        const uint8_t byte = row[di >> 1];
        const int     nib  = (di & 1) ? (byte >> 4) : (byte & 0xF);
        const int     gi   = di / group;
        out[di] = (T)(((float)nib - (float)zrow[gi]) * (float)srow[gi]);
    }
}

void invokeEmbeddingLookupInt4(Ref<Tensor>         out_,
                               const Buffer_<int>& token_ids,
                               const Tensor&       table,
                               const Tensor&       scales,
                               const Tensor&       zeros,
                               int                 group,
                               cudaStream_t        st)
{
    auto& out = out_.get();
    TM_CHECK_EQ(out.shape(0), token_ids.size());
    TM_CHECK_EQ((int)table.shape(1) * 2, (int)out.shape(1));
    TM_CHECK_EQ(table.dtype(), kUint8);
    TM_CHECK_EQ(zeros.dtype(), kUint8);
    TM_CHECK_EQ(scales.dtype(), out.dtype());
    TM_CHECK_EQ((int)scales.shape(1), (int)out.shape(1) / group);
    TM_CHECK_EQ((int)zeros.shape(0), (int)table.shape(0));
    TM_CHECK_EQ((int)zeros.shape(1), (int)scales.shape(1));
    TM_CHECK_EQ((int)zeros.stride(0), (int)scales.stride(0));

    const int num = (int)out.shape(0);
    const int dim = (int)out.shape(1);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        const int threads = std::min(dim, 1024);
        embeddingLookupInt4Kernel<T><<<num, threads, 0, st>>>((T*)out.raw_data(),
                                                              (int)out.stride(0),
                                                              (const uint8_t*)table.raw_data(),
                                                              (int)table.stride(0),
                                                              (const T*)scales.raw_data(),
                                                              (const uint8_t*)zeros.raw_data(),
                                                              (int)scales.stride(0),
                                                              token_ids.data(),
                                                              dim,
                                                              group);
        TM_CUDA_CHECK(cudaGetLastError());
    };

    if (out.dtype() == kHalf) {
        invoke(half_t{});
    }
    else if (out.dtype() == kBfloat16) {
        invoke(bfloat16_t{});
    }
    else {
        TM_LOG_FATAL("invokeEmbeddingLookupInt4: unsupported out dtype");
    }
}

// TODO Add half2 implementation
template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1 * dim2) {
        const int input_dim2_index = index % dim2;
        index                      = (index - input_dim2_index) / dim2;
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;

        out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 + input_dim2_index] =
            in[input_dim0_index * dim1 * dim2 + input_dim1_index * dim2 + input_dim2_index];
    }
}

template<typename T>
void invokeTransposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 * dim2 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
    TM_CUDA_CHECK(cudaGetLastError());
}

template void
invokeTransposeAxis01(float* out, float* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(half* out, half* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(int* out, int* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(uint16_t* out, uint16_t* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

template void
invokeTransposeAxis01(uint8_t* out, uint8_t* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeTransposeAxis01(
    __nv_bfloat16* out, __nv_bfloat16* in, const int dim0, const int dim1, const int dim2, cudaStream_t stream);
#endif

template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1)
{
    // out: [dim1, dim0]
    // in: [dim0, dim1]
    // in_skipping_dim1: [dim1]

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1) {
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;
        const int in_offset        = in_skipping_dim1 == nullptr ? 0 : in_skipping_dim1[input_dim1_index] * dim1;

        out[input_dim1_index * dim0 + input_dim0_index] = in[in_offset + input_dim0_index * dim1 + input_dim1_index];
    }
}

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, in_skipping_dim1, dim0, dim1);
    TM_CUDA_CHECK(cudaGetLastError());
}

template void invokeTransposeAxis01(
    int* out, int* in, const int* in_skipping_dim1, const int dim0, const int dim1, cudaStream_t stream);

template<int TILE_DIM, int BLOCK_ROWS, class T>
__global__ void transpose_2d_kernel(T* __restrict__ dst, const T* __restrict__ src, int rows, int cols, bool swap_xy)
{
    __shared__ T smem[TILE_DIM][TILE_DIM + 1];

    const int block_idx_x = swap_xy ? blockIdx.y : blockIdx.x;
    const int block_idx_y = swap_xy ? blockIdx.x : blockIdx.y;

    {
        const int j = block_idx_x * TILE_DIM + threadIdx.x;
        const int i = block_idx_y * TILE_DIM + threadIdx.y;

#pragma unroll
        for (int y = 0; y < TILE_DIM; y += BLOCK_ROWS) {
            if (i + y < rows && j < cols) {
                smem[threadIdx.y + y][threadIdx.x] = src[(i + y) * cols + j];
            }
        }
    }

    __syncthreads();

    {
        const int j = block_idx_y * TILE_DIM + threadIdx.x;
        const int i = block_idx_x * TILE_DIM + threadIdx.y;

#pragma unroll
        for (int y = 0; y < TILE_DIM; y += BLOCK_ROWS) {
            if (i + y < cols && j < rows) {
                dst[(i + y) * rows + j] = smem[threadIdx.x][threadIdx.y + y];
            }
        }
    }
}

template<class T>
void invokeTranspose2D_(T* dst, const T* src, int rows, int cols, cudaStream_t st)
{
    constexpr int TILE_DIM   = 32;  // warp size
    constexpr int BLOCK_ROWS = 8;

    const dim3 block(TILE_DIM, BLOCK_ROWS);

    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,  //
              (rows + TILE_DIM - 1) / TILE_DIM);
    bool swap_xy = false;

    if (grid.y > 65535) {  // max dim for grid.y
        std::swap(grid.x, grid.y);
        swap_xy = true;
    }

    transpose_2d_kernel<TILE_DIM, BLOCK_ROWS><<<grid, block, 0, st>>>(dst, src, rows, cols, swap_xy);
    TM_CUDA_CHECK(cudaGetLastError());
}

template void invokeTranspose2D_(uint32_t*, const uint32_t*, int, int, cudaStream_t);

}  // namespace turbomind
