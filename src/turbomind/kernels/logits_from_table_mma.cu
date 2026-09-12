#include "src/turbomind/kernels/gpt_kernels.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace turbomind {

// Fused logits-from-table GEMM for the shared embedding head (SM80+ only).
// A is the vocabulary-major table (row-major over vocab) and B is the
// activation x; the output is [tokens, vocab].
//
//   FORMAT 0: fp16/bf16 x and table, whose dtypes may differ (converted at
//             fragment-load time); table rows staged to smem as-is.
//   FORMAT 1: int8 table; the raw bytes are staged to smem and dequantized in
//             registers with a per-128-group scale at fragment-load time.
//   FORMAT 2: int4 table; the packed nibbles are staged as raw bytes and
//             dequantized in registers with a per-128-group scale and zero.
//
// The epilogue scatters the accumulator fragments through a smem tile, then
// writes rows of 128 consecutive vocab entries with consecutive threads.
// Decode (tokens <= 16) uses the N_TILE=16 instance (K_STAGE=32 for 16-bit
// tables, K_STAGE=64 for int8/int4 so each raw row copy is a full 64 B);
// prefill uses N_TILE=64 with K_STAGE=32 for every format.

namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Fragment helpers are parameterized on the 16-bit operand type T (bf16 or
// fp16). Both types occupy one register per k-pair, so the mma register
// contract ({a0..a3}, {b0,b1}) and every layout/stride below are unchanged.
template<class T>
__device__ __forceinline__ uint32_t load_f2(const T* p)
{
    return *reinterpret_cast<const uint32_t*>(p);
}

template<class T>
__device__ __forceinline__ T cvt_f(float v)
{
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "cvt_f expects a 16-bit mma operand type");
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(v);
    } else {
        return __float2bfloat16(v);
    }
}

// Pack two 16-bit operands, even k in the low half, matching load_f2's
// little-endian pair order.
template<class T>
__device__ __forceinline__ uint32_t pack_f2(T lo, T hi)
{
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "pack_f2 expects a 16-bit mma operand type");
    if constexpr (std::is_same_v<T, __half>) {
        return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
    } else {
        return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
    }
}

// Convert a packed pair of 16-bit table values to the mma operand type.
template<class TFrom, class TTo>
__device__ __forceinline__ uint32_t cvt_pack(uint32_t v)
{
    if constexpr (std::is_same_v<TFrom, TTo>) {
        return v;
    } else {
        const unsigned short lo = (unsigned short)(v & 0xFFFFu);
        const unsigned short hi = (unsigned short)(v >> 16);
        float f_lo, f_hi;
        if constexpr (std::is_same_v<TFrom, __half>) {
            f_lo = __half2float(__ushort_as_half(lo));
            f_hi = __half2float(__ushort_as_half(hi));
        } else {
            f_lo = __bfloat162float(__ushort_as_bfloat16(lo));
            f_hi = __bfloat162float(__ushort_as_bfloat16(hi));
        }
        return pack_f2<TTo>(cvt_f<TTo>(f_lo), cvt_f<TTo>(f_hi));
    }
}

template<class T>
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1)
{
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "mma_m16n8k16 expects a 16-bit mma operand type");
    if constexpr (std::is_same_v<T, __half>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src)
{
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
}

__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Raw quantized A tile row stride, bytes: the smallest multiple of 16 that
// holds the K_STAGE row footprint and makes the fragment row rotation
// conflict-free (stride/4 * r mod 32 injective over the 8 group rows
// r = 0..7). 32-byte rows -> 48 (12r mod 32); 64-byte rows -> 80 (20r mod 32).
template<int k_stage>
constexpr int raw_stride_of()
{
    static_assert(k_stage <= 64, "raw staging supports K_STAGE up to 64 (80 B row stride)");
    return k_stage <= 32 ? 48 : 80;
}

// Shared A storage: FORMAT 0 keeps the 16-bit tile (K_STRIDE = K_STAGE + 8
// spreads the banks); FORMAT 1/2 stage the raw quantized tile and dequantize
// in registers at fragment-load time. The format-bool specialization makes
// every instantiation allocate only its own variant.
template<int FORMAT, int M_TILE_, int K_STAGE_, int STAGES_, class T, bool QUANT = (FORMAT != 0)>
struct AStorage;

template<int FORMAT, int M_TILE_, int K_STAGE_, int STAGES_, class T>
struct AStorage<FORMAT, M_TILE_, K_STAGE_, STAGES_, T, false> {
    T tile[STAGES_][M_TILE_ * (K_STAGE_ + 8)];
};

template<int FORMAT, int M_TILE_, int K_STAGE_, int STAGES_, class T>
struct AStorage<FORMAT, M_TILE_, K_STAGE_, STAGES_, T, true> {
    unsigned char tile[STAGES_][M_TILE_ * raw_stride_of<K_STAGE_>()];
};

// Dequantize a k-pair (even k in the low byte/nibble) from a raw smem row to
// the operand type; the packed layout matches what load_f2 reads for the
// 16-bit table path.
template<class T>
__device__ __forceinline__ uint32_t dequant_int8x2(const unsigned char* p, float s)
{
    const uint16_t v = *reinterpret_cast<const uint16_t*>(p);
    return pack_f2<T>(cvt_f<T>((float)(int8_t)(v & 0xFF) * s), cvt_f<T>((float)(int8_t)(v >> 8) * s));
}

template<class T>
__device__ __forceinline__ uint32_t dequant_int4x2(const unsigned char* p, float s, float z)
{
    const uint8_t v = *p;
    return pack_f2<T>(cvt_f<T>(((float)(v & 0xF) - z) * s), cvt_f<T>(((float)(v >> 4) - z) * s));
}

#endif  // __CUDA_ARCH__ >= 800

// FORMAT: 0 = 16-bit float table, 1 = int8, 2 = int4. T is the operand type
// (__nv_bfloat16 or __half) of x/scale/logits; TA is the FORMAT 0 table
// operand type (__nv_bfloat16 or __half), which may differ from T and is
// converted at fragment-load time. FORMAT 1/2 ignore TA.
template<int N_TILE, int K_STAGE, int FORMAT, class T, class TA>
__global__ void logitsFromTableMmaKernel(void* __restrict__ logits_,
                                         int                  vocab,
                                         const void* __restrict__ x_,
                                         int                  tokens,
                                         const void* __restrict__ table_,
                                         const void* __restrict__ scale_,
                                         const void* __restrict__ zero_,
                                         int                  dim)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int M_TILE   = 128;
    constexpr int K_STRIDE = K_STAGE + 8;  // 16-bit row stride (pad to spread smem banks)

    // Pipeline depth. FORMAT 0 uses the 16-bit double buffer. The raw
    // quantized decode path stages a narrower footprint per row than 16-bit
    // (production instance K_STAGE=64 for that reason), so it needs a third
    // buffer to keep two copies in flight over the per-stage latency;
    // quantized prefill is already bandwidth-bound and stays double buffered
    // (a third buffer would exceed the 48 KB static smem limit).
    constexpr int STAGES = (FORMAT != 0 && N_TILE <= 16) ? 3 : 2;

    auto logits = (T*)logits_;
    auto x      = (const T*)x_;

    // Static smem budget (K_STRIDE = K_STAGE + 8, kRawStride = raw_stride_of()):
    //   FORMAT 0 decode   (N_TILE=16, K_STAGE=32):  2*128*40*2 + 2*16*40*2 + 16*136*2 = 27392 B
    //   FORMAT 0 prefill  (N_TILE=64, K_STAGE=32):  2*128*40*2 + 2*64*40*2 + 64*136*2 = 48128 B
    //   FORMAT 1/2 decode (N_TILE=16, K_STAGE=64):  3*128*80   + 3*16*72*2 + 16*136*2 = 41984 B
    //   FORMAT 1/2 prefill(N_TILE=64, K_STAGE=32):  2*128*48   + 2*64*40*2 + 64*136*2 = 39936 B
    // All < 48 KB static. Every cp.async destination is 16 B aligned: the
    // 16-bit row stride is K_STRIDE * 2, the raw row stride is a multiple of
    // 16, and every chunk offset is a multiple of 16 B.
    __shared__ __align__(16) AStorage<FORMAT, M_TILE, K_STAGE, STAGES, TA> A_s;
    __shared__ __align__(16) T Bs[STAGES][N_TILE * K_STRIDE];

    auto& As = A_s.tile;  // TA array (FORMAT 0) or raw bytes (FORMAT 1/2)

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;   // groupID
    const int tg   = lane & 3;    // thread in group

    const int m0 = blockIdx.x * M_TILE;
    const int t0 = blockIdx.y * N_TILE;

    float acc[N_TILE / 8][4];
#pragma unroll
    for (int n = 0; n < N_TILE / 8; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[n][i] = 0.f;
        }
    }

    static_assert(K_STAGE % 8 == 0, "16 B cp.async chunks need K_STAGE % 8 == 0");
    static_assert(128 % K_STAGE == 0, "a K_STAGE-wide tile must lie in one 128-element dequant group");

    // Stage one K_STAGE-wide tile into As[buf] / Bs[buf]. All formats use
    // cp.async: FORMAT 0 copies 16 B (8 operands) chunks of the table; FORMAT
    // 1/2 copy the raw quantized tile (int8: K_STAGE bytes; int4: K_STAGE/2
    // bytes) in 16 B chunks and dequantize in registers later. The host
    // guarantees `dim % 128 == 0` for FORMAT 1/2, so K is always in bounds
    // there and only the row predicate remains.
    constexpr int CHUNK_COLS = K_STAGE / 8;  // 16 B (8-operand) chunks per row

    auto stage = [&](int k0, int buf) {
        if constexpr (FORMAT == 0) {
            const auto* b16 = (const TA*)table_;
            for (int i = tid; i < M_TILE * CHUNK_COLS; i += blockDim.x) {
                const int r = i / CHUNK_COLS;
                const int c = i % CHUNK_COLS;
                if (m0 + r < vocab && k0 + c * 8 < dim) {
                    cp_async16(&As[buf][r * K_STRIDE + c * 8], b16 + (size_t)(m0 + r) * dim + k0 + c * 8);
                } else {
                    *reinterpret_cast<uint4*>(&As[buf][r * K_STRIDE + c * 8]) = make_uint4(0, 0, 0, 0);
                }
            }
        } else {
            constexpr int CHUNKS_PER_ROW = FORMAT == 1 ? K_STAGE / 16 : K_STAGE / 32;
            constexpr int kRawStride     = raw_stride_of<K_STAGE>();
            const auto*   q              = (const unsigned char*)table_;
            for (int i = tid; i < M_TILE * CHUNKS_PER_ROW; i += blockDim.x) {
                const int   r   = i / CHUNKS_PER_ROW;
                const int   c   = i % CHUNKS_PER_ROW;
                void*       dst = &As[buf][r * kRawStride + c * 16];
                if (m0 + r < vocab) {
                    const size_t src = FORMAT == 1 ? (size_t)(m0 + r) * dim + k0 + c * 16
                                                   : (size_t)(m0 + r) * (dim / 2) + k0 / 2 + c * 16;
                    cp_async16(dst, q + src);
                } else {
                    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        // B (16-bit x) is staged with cp.async for every format.
        for (int i = tid; i < N_TILE * CHUNK_COLS; i += blockDim.x) {
            const int n = i / CHUNK_COLS;
            const int c = i % CHUNK_COLS;
            if (t0 + n < tokens && k0 + c * 8 < dim) {
                cp_async16(&Bs[buf][n * K_STRIDE + c * 8], x + (size_t)(t0 + n) * dim + k0 + c * 8);
            } else {
                *reinterpret_cast<uint4*>(&Bs[buf][n * K_STRIDE + c * 8]) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_commit();
    };

    // Multi-stage pipeline: STAGES-1 copies stay in flight so the copy for
    // tile s+STAGES-1 overlaps the mma of tile s. The FORMAT 0 instantiation
    // is STAGES == 2, the plain double buffer.
    const int nstages = (dim + K_STAGE - 1) / K_STAGE;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nstages) {
            stage(s * K_STAGE, s % STAGES);
        }
    }
    for (int s = 0; s < nstages; ++s) {
        const int k0   = s * K_STAGE;
        const int buf  = s % STAGES;
        const int next = s + STAGES - 1;
        if (next < nstages) {
            stage(next * K_STAGE, next % STAGES);
            cp_wait<STAGES - 1>();  // retire the oldest in-flight tile (s)
        } else {
            cp_wait<0>();  // no successor left: drain
        }
        __syncthreads();

        // FORMAT 1/2: a K_STAGE-wide tile lies inside one 128-element dequant
        // group, so each fragment row's scale (and int4 zero) is loaded once
        // per stage and reused from registers across kk.
        [[maybe_unused]] const auto* srow = (const T*)scale_;
        [[maybe_unused]] const auto* zrow = (const uint8_t*)zero_;
        [[maybe_unused]] const int   rs   = dim / 128;
        [[maybe_unused]] const int   gs   = k0 / 128;
        [[maybe_unused]] const int   r0   = warp * 16 + g;
        [[maybe_unused]] const int   r1   = r0 + 8;
        [[maybe_unused]] float       sf0  = 0.f;
        [[maybe_unused]] float       sf1  = 0.f;
        [[maybe_unused]] float       zf0  = 0.f;
        [[maybe_unused]] float       zf1  = 0.f;
        if constexpr (FORMAT != 0) {
            if (m0 + r0 < vocab) {
                sf0 = (float)srow[(size_t)(m0 + r0) * rs + gs];
                if constexpr (FORMAT == 2) {
                    zf0 = (float)zrow[(size_t)(m0 + r0) * rs + gs];
                }
            }
            if (m0 + r1 < vocab) {
                sf1 = (float)srow[(size_t)(m0 + r1) * rs + gs];
                if constexpr (FORMAT == 2) {
                    zf1 = (float)zrow[(size_t)(m0 + r1) * rs + gs];
                }
            }
        }

#pragma unroll
        for (int kk = 0; kk < K_STAGE; kk += 16) {
            uint32_t a0, a1, a2, a3;
            if constexpr (FORMAT == 0) {
                const auto* arow = &As[buf][(warp * 16 + g) * K_STRIDE + kk + tg * 2];
                a0 = cvt_pack<TA, T>(load_f2<TA>(arow));
                a1 = cvt_pack<TA, T>(load_f2<TA>(arow + 8 * K_STRIDE));
                a2 = cvt_pack<TA, T>(load_f2<TA>(arow + 8));
                a3 = cvt_pack<TA, T>(load_f2<TA>(arow + 8 * K_STRIDE + 8));
            } else if constexpr (FORMAT == 1) {
                constexpr int kRawStride = raw_stride_of<K_STAGE>();
                const int     k          = kk + tg * 2;  // even, so the byte pair is aligned
                const auto*   row0       = &As[buf][r0 * kRawStride + k];
                const auto*   row1       = &As[buf][r1 * kRawStride + k];
                a0 = dequant_int8x2<T>(row0, sf0);
                a1 = dequant_int8x2<T>(row1, sf1);
                a2 = dequant_int8x2<T>(row0 + 8, sf0);
                a3 = dequant_int8x2<T>(row1 + 8, sf1);
            } else {
                constexpr int kRawStride = raw_stride_of<K_STAGE>();
                const int     k          = kk + tg * 2;
                const auto*   row0       = &As[buf][r0 * kRawStride + k / 2];
                const auto*   row1       = &As[buf][r1 * kRawStride + k / 2];
                a0 = dequant_int4x2<T>(row0, sf0, zf0);
                a1 = dequant_int4x2<T>(row1, sf1, zf1);
                a2 = dequant_int4x2<T>(row0 + 4, sf0, zf0);
                a3 = dequant_int4x2<T>(row1 + 4, sf1, zf1);
            }
#pragma unroll
            for (int n = 0; n < N_TILE / 8; ++n) {
                const auto* brow = &Bs[buf][(n * 8 + g) * K_STRIDE + kk + tg * 2];
                const uint32_t b0 = load_f2<T>(brow);
                const uint32_t b1 = load_f2<T>(brow + 8);
                mma_m16n8k16<T>(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();  // all warps done reading before the buffer is refilled
    }

    // Transposed epilogue: scatter the fragments into a smem tile, then write
    // rows of 128 consecutive vocab entries with consecutive threads.
    constexpr int E_STRIDE = M_TILE + 8;
    __shared__ T Es[N_TILE * E_STRIDE];

#pragma unroll
    for (int n = 0; n < N_TILE / 8; ++n) {
        const int t_lo = n * 8 + tg * 2;
        const int m    = warp * 16 + g;
        Es[t_lo * E_STRIDE + m]           = cvt_f<T>(acc[n][0]);
        Es[(t_lo + 1) * E_STRIDE + m]     = cvt_f<T>(acc[n][1]);
        Es[t_lo * E_STRIDE + m + 8]       = cvt_f<T>(acc[n][2]);
        Es[(t_lo + 1) * E_STRIDE + m + 8] = cvt_f<T>(acc[n][3]);
    }
    __syncthreads();
    for (int i = tid; i < N_TILE * M_TILE; i += blockDim.x) {
        const int n = i / M_TILE;
        const int m = i % M_TILE;
        if (t0 + n < tokens && m0 + m < vocab) {
            logits[(size_t)(t0 + n) * vocab + m0 + m] = Es[n * E_STRIDE + m];
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

}  // namespace

void invokeLogitsFromTable(Ref<Tensor>   logits,
                           const Tensor& x,
                           const Tensor& table,
                           const Tensor& scale,
                           const Tensor& zero,
                           int           group,
                           cudaStream_t  st)
{
    TM_CHECK(getSMVersion() >= 80)
        << "shared-table head requires SM80 or newer; use LMDEPLOY_DISABLE_EMBED_QUANT=1 or regenerate the "
           "sidecar without sharing";

    const bool x_is_half      = x.dtype() == kHalf;
    const bool x_is_bf16      = x.dtype() == kBfloat16;
    const bool table_is_int8  = table.dtype() == kInt8;
    const bool table_is_int4  = table.dtype() == kUint8;
    const bool table_is_half  = table.dtype() == kHalf;
    const bool table_is_bf16  = table.dtype() == kBfloat16;
    const bool table_is_16bit = table_is_half || table_is_bf16;

    TM_CHECK((x_is_half || x_is_bf16) && (table_is_16bit || table_is_int8 || table_is_int4))
        << "shared-table head supports fp16/bf16 x with an fp16/bf16 or int8/int4 table; got "
           "x.dtype="
        << x.dtype() << ", table.dtype=" << table.dtype()
        << "; use LMDEPLOY_DISABLE_EMBED_QUANT=1 or regenerate the sidecar via scripts/quantize_embedding.py";

    const int tokens = (int)x.shape(0);
    const int dim    = (int)x.shape(1);
    const int vocab  = (int)table.shape(0);

    TM_CHECK(dim % 16 == 0)
        << "shared-table head requires dim % 16 == 0 to cover the mma K tiles; got dim=" << dim
        << "; use LMDEPLOY_DISABLE_EMBED_QUANT=1 or regenerate the sidecar via scripts/quantize_embedding.py";
    TM_CHECK((int)x.stride(0) == dim)
        << "shared-table head requires contiguous x rows (x.stride(0) == dim); got x.stride(0)=" << (int)x.stride(0)
        << ", dim=" << dim
        << "; use LMDEPLOY_DISABLE_EMBED_QUANT=1 or regenerate the sidecar via scripts/quantize_embedding.py";

    const int table_row_stride = table_is_int4 ? dim / 2 : dim;  // int4 packs two hidden elements per byte
    TM_CHECK((int)table.shape(1) == table_row_stride && (int)table.stride(0) == table_row_stride)
        << "shared-table head requires an unrepacked, contiguous table whose rows hold " << table_row_stride
        << " elements (table.shape(1) and table.stride(0)); got table.shape(1)=" << table.shape(1)
        << ", table.stride(0)=" << table.stride(0)
        << "; use LMDEPLOY_DISABLE_EMBED_QUANT=1 or regenerate the sidecar via scripts/quantize_embedding.py";

    TM_CHECK(logits.get().dtype() == x.dtype() && (int)logits.get().shape(0) == tokens
              && (int)logits.get().shape(1) == vocab && (int)logits.get().stride(0) == vocab
              && (int)logits.get().stride(1) == 1)
        << "shared-table head requires contiguous logits of shape [" << tokens << ", " << vocab
        << "] with x's dtype; got logits.dtype=" << logits.get().dtype() << ", logits.shape=("
        << logits.get().shape(0) << ", " << logits.get().shape(1) << "), logits.stride=("
        << logits.get().stride(0) << ", " << logits.get().stride(1) << ")";

    if (table_is_int8 || table_is_int4) {
        TM_CHECK(group == 128)
            << "shared-table head requires group == 128 for int8/int4 tables; got group=" << group
            << "; regenerate the sidecar via scripts/quantize_embedding.py";
        TM_CHECK(dim % 128 == 0)
            << "shared-table head requires dim % 128 == 0 for int8/int4 tables; got dim=" << dim
            << "; regenerate the sidecar via scripts/quantize_embedding.py";
        const int ngroups = dim / 128;
        TM_CHECK(scale.dtype() == x.dtype())
            << "shared-table head requires the int8/int4 scale sidecar dtype to match x; got scale.dtype="
            << scale.dtype() << ", x.dtype=" << x.dtype()
            << "; regenerate the sidecar via scripts/quantize_embedding.py";
        TM_CHECK(scale.ndim() == 2)
            << "shared-table head requires a 2-D int8/int4 scale sidecar [vocab, dim / 128]; got scale.ndim="
            << scale.ndim() << "; regenerate the sidecar via scripts/quantize_embedding.py";
        TM_CHECK((int)scale.shape(0) == vocab && (int)scale.shape(1) == ngroups && (int)scale.stride(0) == ngroups)
            << "shared-table head requires a contiguous scale sidecar of shape [" << vocab << ", " << ngroups
            << "]; got scale.shape=(" << scale.shape(0) << ", " << scale.shape(1)
            << "), scale.stride(0)=" << scale.stride(0)
            << "; regenerate the sidecar via scripts/quantize_embedding.py";
        if (table_is_int4) {
            TM_CHECK(zero.dtype() == kUint8)
                << "shared-table head requires the int4 zero sidecar to be uint8; got zero.dtype=" << zero.dtype()
                << "; regenerate the sidecar via scripts/quantize_embedding.py";
            TM_CHECK(zero.ndim() == 2)
                << "shared-table head requires a 2-D int4 zero sidecar [vocab, dim / 128]; got zero.ndim="
                << zero.ndim() << "; regenerate the sidecar via scripts/quantize_embedding.py";
            TM_CHECK(zero.shape() == scale.shape() && (int)zero.shape(0) != 0 && (int)zero.stride(0) == ngroups)
                << "shared-table head requires a non-empty, contiguous int4 zero sidecar matching the scale sidecar "
                   "shape; got zero.shape=("
                << zero.shape(0) << ", " << zero.shape(1) << "), zero.stride(0)=" << zero.stride(0)
                << "; regenerate the sidecar via scripts/quantize_embedding.py";
        }
    }

    // Decode (tokens <= 16) keeps the N_TILE=16 instance, but quantized decode
    // uses K_STAGE=64 so each raw row copy is a full 64 B (better DRAM sector
    // efficiency and half the pipeline stages); 16-bit tables keep K_STAGE=32.
    // Prefill takes the N_TILE=64 instance (FORMAT 0: 48128 B static smem;
    // FORMAT 1/2: 39936 B; see the kernel comment). A runtime N or a ternary
    // over the distinct template instantiations would not compile.
#define LAUNCH_LOGITS_MMA(TT, TA, NT, KS, FMT)                                                      \
    logitsFromTableMmaKernel<NT, KS, FMT, TT, TA>                                                   \
        <<<dim3((vocab + 127) / 128, (tokens + NT - 1) / NT), 256, 0, st>>>(logits.get().raw_data(), \
                                                                            vocab,                  \
                                                                            x.raw_data(),           \
                                                                            tokens,                 \
                                                                            table.raw_data(),       \
                                                                            scale.data_or<void>((void*)nullptr), \
                                                                            zero.data_or<void>((void*)nullptr),  \
                                                                            dim)

#define LAUNCH_FORMAT_BY_TOKENS(TT, TA)                        \
    if (tokens <= 16) {                                        \
        if (format == 0) {                                     \
            LAUNCH_LOGITS_MMA(TT, TA, 16, 32, 0);              \
        } else if (format == 1) {                              \
            LAUNCH_LOGITS_MMA(TT, TA, 16, 64, 1);              \
        } else {                                               \
            LAUNCH_LOGITS_MMA(TT, TA, 16, 64, 2);              \
        }                                                      \
    } else {                                                   \
        if (format == 0) {                                     \
            LAUNCH_LOGITS_MMA(TT, TA, 64, 32, 0);              \
        } else if (format == 1) {                              \
            LAUNCH_LOGITS_MMA(TT, TA, 64, 32, 1);              \
        } else {                                               \
            LAUNCH_LOGITS_MMA(TT, TA, 64, 32, 2);              \
        }                                                      \
    }

    const int format = table_is_int4 ? 2 : table_is_int8 ? 1 : 0;
    if (format == 0) {
        if (x_is_half) {
            if (table_is_half) {
                LAUNCH_FORMAT_BY_TOKENS(__half, __half);
            } else {
                LAUNCH_FORMAT_BY_TOKENS(__half, __nv_bfloat16);
            }
        } else {
            if (table_is_half) {
                LAUNCH_FORMAT_BY_TOKENS(__nv_bfloat16, __half);
            } else {
                LAUNCH_FORMAT_BY_TOKENS(__nv_bfloat16, __nv_bfloat16);
            }
        }
    } else {
        if (x_is_half) {
            LAUNCH_FORMAT_BY_TOKENS(__half, __half);
        } else {
            LAUNCH_FORMAT_BY_TOKENS(__nv_bfloat16, __nv_bfloat16);
        }
    }
#undef LAUNCH_FORMAT_BY_TOKENS
#undef LAUNCH_LOGITS_MMA
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
