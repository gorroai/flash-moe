// nax_gemm.metal — NAX tensor matmul for M5+ (Metal 4.0)
// Compile: xcrun -sdk macosx metal -std=metal4.0 nax_gemm.metal -o nax_gemm.air
// This file is compiled separately from shaders.metal (which uses Metal 3.1)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ============================================================================
// Dequantize 4-bit packed weights to half buffer
// W[N, K/8] uint32 → W_out[N, K] half
// ============================================================================

kernel void nax_dequant_4bit(
    const device uint32_t* W       [[buffer(0)]],
    const device uint16_t* scales  [[buffer(1)]],  // bf16 stored as uint16
    const device uint16_t* biases  [[buffer(2)]],  // bf16 stored as uint16
    device half* W_out             [[buffer(3)]],
    constant uint& N_dim           [[buffer(4)]],
    constant uint& K_dim           [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]],
    uint tsize                     [[threads_per_grid]])
{
    const uint GROUP_SIZE = 64;
    uint packed_cols = K_dim / 8;
    uint num_groups = K_dim / GROUP_SIZE;
    uint total_groups = N_dim * num_groups;

    for (uint idx = tid; idx < total_groups; idx += tsize) {
        uint row = idx / num_groups;
        uint g = idx % num_groups;
        // bf16 → float: shift left 16 bits
        float s = as_type<float>(uint(scales[row * num_groups + g]) << 16);
        float b = as_type<float>(uint(biases[row * num_groups + g]) << 16);
        uint base = g * (GROUP_SIZE / 8);
        for (uint p = 0; p < GROUP_SIZE / 8; p++) {
            uint32_t packed = W[row * packed_cols + base + p];
            uint out_base = row * K_dim + g * GROUP_SIZE + p * 8;
            for (uint n = 0; n < 8; n++) {
                float nibble = (float)((packed >> (n * 4)) & 0xF);
                W_out[out_base + n] = (half)fma(nibble, s, b);
            }
        }
    }
}

// ============================================================================
// Also dequant input float32 → half (for NAX which needs half inputs)
// ============================================================================

kernel void nax_f32_to_half(
    const device float* src        [[buffer(0)]],
    device half* dst               [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid                       [[thread_position_in_grid]],
    uint tsize                     [[threads_per_grid]])
{
    for (uint i = tid; i < count; i += tsize) {
        dst[i] = (half)src[i];
    }
}

// ============================================================================
// NAX GEMM: C[M,N] = A[M,K] @ B[N,K]^T
// Uses tensor_inline with cooperative_tensor accumulation
// Tile: BM=64, BN=32, BK=32, 4 simdgroups (128 threads)
// ============================================================================

kernel void nax_gemm_half(
    const device half* A_ptr       [[buffer(0)]],   // [M, K]
    const device half* B_ptr       [[buffer(1)]],   // [N, K] (transposed)
    device float* C_ptr            [[buffer(2)]],   // [M, N]
    constant uint& M_dim           [[buffer(3)]],
    constant uint& N_dim           [[buffer(4)]],
    constant uint& K_dim           [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    constexpr int BM = 32, BN = 32, BK = 32;  // 32×32 output tile, 4 simdgroups

    int bm = (int)tgid.y * BM;
    int bn = (int)tgid.x * BN;

    threadgroup half sa[BM * BK];
    threadgroup half sb[BN * BK];

    // We want C[m,n] = sum_k A[m,k] * B[n,k]   (B is transposed)
    // In column-major tensor land (inner dim first):
    //   A stored row-major [M,K] → tensor(K, M)
    //   B stored row-major [N,K] → tensor(K, N)
    //   C stored row-major [M,N] → tensor(N, M)
    // matmul: C(N,M) = B(K,N)^T @ A(K,M) → left=B transposed, right=A not transposed
    // descriptor(m=N_tile, n=M_tile, k=K_tile, transpose_left=true, transpose_right=false)
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK,
        true, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;

    // Dummy tensors for type deduction
    auto tB_ref = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(BK, BN));  // B tile: (K, N)
    auto tA_ref = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(BK, BM));  // A tile: (K, M)
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB_ref), decltype(tA_ref), float>();

    for (int k = 0; k < (int)K_dim; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile: A[bm:bm+BM, k:k+BK] → sa[BK, BM] (column-major: K inner)
        for (int i = (int)lid; i < BM * BK; i += 128) {
            int m_local = i / BK;  // row in A = M dimension
            int k_local = i % BK;  // col in A = K dimension
            int gm = bm + m_local, gk = k + k_local;
            // Store as (k, m) in threadgroup for column-major tensor
            sa[k_local * BM + m_local] = (gm < (int)M_dim && gk < (int)K_dim) ? A_ptr[gm * K_dim + gk] : (half)0;
        }
        // Load B tile: B[bn:bn+BN, k:k+BK] → sb[BK, BN] (column-major: K inner)
        for (int i = (int)lid; i < BN * BK; i += 128) {
            int n_local = i / BK;
            int k_local = i % BK;
            int gn = bn + n_local, gk = k + k_local;
            sb[k_local * BN + n_local] = (gn < (int)N_dim && gk < (int)K_dim) ? B_ptr[gn * K_dim + gk] : (half)0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sb, dextents<int32_t, 2>(BK, BN));
        auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sa, dextents<int32_t, 2>(BK, BM));
        mm.run(tB, tA, cT);
    }

    // Store: cT writes in the matmul2d's natural layout
    // With descriptor(BM, BN, BK, transpose_left=true, transpose_right=false):
    //   output is (M_tile, N_tile) where M is the "m" dimension
    // We store to C_ptr as [N, M] so that C_ptr[n*M + m] = result[m, n]
    // The caller reads with stride M to extract row m=0
    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_ptr, dextents<int32_t, 2>(M_dim, N_dim));
    auto tC_slice = tC.slice((int)(tgid.y * BM), (int)(tgid.x * BN));
    cT.store(tC_slice);
}

// ============================================================================
// Extract row 0 from column-major output: dst[n] = src[n * stride]
// ============================================================================

kernel void nax_extract_row0(
    const device float* src        [[buffer(0)]],   // [M, N] column-major
    device float* dst              [[buffer(1)]],   // [N] row-major output
    constant uint& N_dim           [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],   // M_padded
    uint tid [[thread_position_in_grid]])
{
    if (tid < N_dim) {
        dst[tid] = src[tid * stride];
    }
}

// ============================================================================
// NAX GEMM with float32 input: C[M,N] = A_f32[M,K] @ B_half[N,K]^T
// Converts A from float32→half on the fly in threadgroup memory
// Eliminates the separate f32→half conversion pass
// Output is column-major: C[m,n] at offset m + n*M_dim
// ============================================================================

kernel void nax_gemm_f32_input(
    const device float* A_ptr      [[buffer(0)]],   // [M, K] float32
    const device half* B_ptr       [[buffer(1)]],   // [N, K] half (pre-dequantized)
    device float* C_ptr            [[buffer(2)]],   // [M, N] float32 (column-major)
    constant uint& M_dim           [[buffer(3)]],
    constant uint& N_dim           [[buffer(4)]],
    constant uint& K_dim           [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    constexpr int BM = 32, BN = 32, BK = 32;

    int bm = (int)tgid.y * BM;
    int bn = (int)tgid.x * BN;

    threadgroup half sa[BM * BK];
    threadgroup half sb[BN * BK];

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK,
        true, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;

    auto tB_ref = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(BK, BN));
    auto tA_ref = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(BK, BM));
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB_ref), decltype(tA_ref), float>();

    for (int k = 0; k < (int)K_dim; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A (float32) → threadgroup half, column-major (K, M)
        for (int i = (int)lid; i < BM * BK; i += 128) {
            int m_local = i / BK;
            int k_local = i % BK;
            int gm = bm + m_local, gk = k + k_local;
            float val = (gm < (int)M_dim && gk < (int)K_dim) ? A_ptr[gm * K_dim + gk] : 0.0f;
            sa[k_local * BM + m_local] = (half)val;  // f32→half inline
        }
        // Load B (half) → threadgroup, column-major (K, N)
        for (int i = (int)lid; i < BN * BK; i += 128) {
            int n_local = i / BK;
            int k_local = i % BK;
            int gn = bn + n_local, gk = k + k_local;
            sb[k_local * BN + n_local] = (gn < (int)N_dim && gk < (int)K_dim) ? B_ptr[gn * K_dim + gk] : (half)0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sb, dextents<int32_t, 2>(BK, BN));
        auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sa, dextents<int32_t, 2>(BK, BM));
        mm.run(tB, tA, cT);
    }

    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_ptr, dextents<int32_t, 2>(M_dim, N_dim));
    auto tC_slice = tC.slice((int)(tgid.y * BM), (int)(tgid.x * BN));
    cT.store(tC_slice);
}
