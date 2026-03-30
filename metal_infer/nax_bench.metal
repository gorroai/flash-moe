#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ============================================================================
// Baseline: FMA dequant matvec (current production kernel, simplified)
// ============================================================================

kernel void dequant_matvec_fma(
    const device uint32_t* W       [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device half* biases      [[buffer(2)]],
    const device float* x          [[buffer(3)]],
    device float* y                [[buffer(4)]],
    constant int& out_dim          [[buffer(5)]],
    constant int& in_dim           [[buffer(6)]],
    uint tid                       [[thread_position_in_grid]],
    uint tsize                     [[threads_per_grid]])
{
    const int GROUP_SIZE = 64;
    int packed_cols = in_dim / 8;
    int num_groups = in_dim / GROUP_SIZE;

    for (int row = (int)tid; row < out_dim; row += (int)tsize) {
        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            float s = (float)scales[row * num_groups + g];
            float b = (float)biases[row * num_groups + g];
            int base = g * (GROUP_SIZE / 8);
            for (int p = 0; p < GROUP_SIZE / 8; p++) {
                uint32_t packed = W[row * packed_cols + base + p];
                int xi = g * GROUP_SIZE + p * 8;
                for (int n = 0; n < 8; n++) {
                    float nibble = (float)((packed >> (n * 4)) & 0xF);
                    float sx = s * x[xi + n];
                    float bx = b * x[xi + n];
                    acc = fma(nibble, sx, bx + acc);
                }
            }
        }
        y[row] = acc;
    }
}

// ============================================================================
// Dequantize 4-bit weights to half buffer (separate pass)
// ============================================================================

kernel void dequant_4bit_to_half(
    const device uint32_t* W       [[buffer(0)]],
    const device half* scales      [[buffer(1)]],
    const device half* biases      [[buffer(2)]],
    device half* W_out             [[buffer(3)]],
    constant int& N_dim            [[buffer(4)]],
    constant int& K_dim            [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]],
    uint tsize                     [[threads_per_grid]])
{
    const int GROUP_SIZE = 64;
    int packed_cols = K_dim / 8;
    int num_groups = K_dim / GROUP_SIZE;
    int total_groups = N_dim * num_groups;

    for (int idx = (int)tid; idx < total_groups; idx += (int)tsize) {
        int row = idx / num_groups;
        int g = idx % num_groups;
        half s = scales[row * num_groups + g];
        half b = biases[row * num_groups + g];
        int base = g * (GROUP_SIZE / 8);
        for (int p = 0; p < GROUP_SIZE / 8; p++) {
            uint32_t packed = W[row * packed_cols + base + p];
            int out_base = row * K_dim + g * GROUP_SIZE + p * 8;
            for (int n = 0; n < 8; n++) {
                float nibble = (float)((packed >> (n * 4)) & 0xF);
                W_out[out_base + n] = (half)fma(nibble, (float)s, (float)b);
            }
        }
    }
}

// ============================================================================
// NAX GEMM: y[M,N] = x[M,K] @ W[N,K]^T
// Tensor args auto-constructed by Metal from bound buffers
// ============================================================================

// NAX GEMM following llama.cpp pattern: cooperative_tensor + store
// y[M,N] = x[M,K] @ W[N,K]^T
// Uses threadgroup memory for tiling, tensor_inline for I/O
kernel void gemm_nax(
    const device half* A_ptr       [[buffer(0)]],   // [M, K] row-major
    const device half* B_ptr       [[buffer(1)]],   // [N, K] row-major (transposed)
    device float* C_ptr            [[buffer(2)]],   // [M, N] row-major
    constant int& M_dim            [[buffer(3)]],
    constant int& N_dim            [[buffer(4)]],
    constant int& K_dim            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    constexpr int BM = 64, BN = 32, BK = 32;

    // Tile offsets
    int bm = (int)tgid.y * BM;
    int bn = (int)tgid.x * BN;

    // Load tiles into threadgroup memory
    threadgroup half sa[BM * BK];  // A tile
    threadgroup half sb[BN * BK];  // B tile (transposed)

    // NAX matmul descriptor: BM×BN with BK inner
    constexpr auto desc = matmul2d_descriptor(
        BN, BM, BK,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;

    // Get cooperative tensor for accumulation
    auto tA_dummy = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sa, dextents<int32_t, 2>(BK, BM));
    auto tB_dummy = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        sb, dextents<int32_t, 2>(BN, BK));
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB_dummy), decltype(tA_dummy), float>();

    // Loop over K
    for (int k = 0; k < K_dim; k += BK) {
        // Cooperative load A[bm:bm+BM, k:k+BK] into sa
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int)lid; i < BM * BK; i += 128) {
            int r = i / BK, c = i % BK;
            int gr = bm + r, gc = k + c;
            sa[r * BK + c] = (gr < M_dim && gc < K_dim) ? A_ptr[gr * K_dim + gc] : (half)0;
        }
        // Cooperative load B[bn:bn+BN, k:k+BK] into sb
        for (int i = (int)lid; i < BN * BK; i += 128) {
            int r = i / BK, c = i % BK;
            int gr = bn + r, gc = k + c;
            sb[r * BK + c] = (gr < N_dim && gc < K_dim) ? B_ptr[gr * K_dim + gc] : (half)0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tensor views over threadgroup tiles
        auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sa, dextents<int32_t, 2>(BK, BM));
        auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            sb, dextents<int32_t, 2>(BN, BK));

        mm.run(tB, tA, cT);
    }

    // Store results
    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_ptr, dextents<int32_t, 2>(N_dim, M_dim));
    auto tC_slice = tC.slice((int)(tgid.x * BN), (int)(tgid.y * BM));
    cT.store(tC_slice);
}

// ============================================================================
// Half GEMM baseline (simdgroup_matrix, no NAX) for comparison
// y[M,N] = x[M,K] @ W[N,K]^T
// ============================================================================

kernel void gemm_half_baseline(
    const device half* A           [[buffer(0)]],   // [M, K]
    const device half* B           [[buffer(1)]],   // [N, K]
    device float* C                [[buffer(2)]],   // [M, N]
    constant int& M_dim            [[buffer(3)]],
    constant int& N_dim            [[buffer(4)]],
    constant int& K_dim            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]])
{
    // Simple tiled GEMM using simdgroup_matrix (Metal 3 baseline)
    const int BM = 64, BN = 32, BK = 32;
    const int BK_PAD = BK + 8;

    int bm = (int)tgid.y * BM;
    int bn = (int)tgid.x * BN;
    int sm = ((int)sgid / 2) * 32;
    int sn = ((int)sgid % 2) * 16;

    threadgroup half As[BM * BK_PAD];
    threadgroup half Bs[BN * BK_PAD];

    simdgroup_matrix<float, 8, 8> acc[4][2];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

    for (int k = 0; k < K_dim; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A[BM, BK] and B[BN, BK] (B is transposed: [N,K])
        for (int i = (int)lid; i < BM * BK; i += 128) {
            int r = i / BK, c = i % BK;
            int gr = bm + r, gc = k + c;
            As[r * BK_PAD + c] = (gr < M_dim && gc < K_dim) ? A[gr * K_dim + gc] : (half)0;
        }
        for (int i = (int)lid; i < BN * BK; i += 128) {
            int r = i / BK, c = i % BK;
            int gr = bn + r, gc = k + c;
            Bs[r * BK_PAD + c] = (gr < N_dim && gc < K_dim) ? B[gr * K_dim + gc] : (half)0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Matmul with simdgroup_matrix
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a_frag[4], b_frag[2];
            for (int i = 0; i < 4; i++)
                simdgroup_load(a_frag[i], As + (sm + i*8) * BK_PAD + kk, BK_PAD);
            for (int j = 0; j < 2; j++)
                simdgroup_load(b_frag[j], Bs + (sn + j*8) * BK_PAD + kk, BK_PAD);

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    }

    // Store
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 2; j++) {
            int r = bm + sm + i*8, c = bn + sn + j*8;
            if (r + 7 < M_dim && c + 7 < N_dim) {
                // Need to store via threadgroup for float output
                threadgroup float tmp[8*8];
                simdgroup_store(acc[i][j], tmp, 8);
                for (int ii = (int)slid; ii < 64; ii += 32) {
                    int rr = ii / 8, cc = ii % 8;
                    C[(r + rr) * N_dim + (c + cc)] = tmp[ii];
                }
            }
        }
}
