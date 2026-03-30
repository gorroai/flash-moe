/*
 * nax_bench.m — Benchmark NAX tensor matmul vs simdgroup baseline
 * Tests attention projection (4096×4096) and LM head (248320×4096)
 * Build: clang -O2 -framework Foundation -framework Metal nax_bench.m -o nax_bench
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("Device: %s\n", [[dev name] UTF8String]);

        BOOL hasNAX = NO;
        if (@available(macOS 26.2, *)) {
            if ([dev supportsFamily:(MTLGPUFamily)5002]) {
                printf("Metal 4 (NAX): YES\n");
                hasNAX = YES;
            }
        }

        // Compile shaders
        NSError *err;
        NSString *src = [NSString stringWithContentsOfFile:@"nax_bench.metal"
                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "Cannot read nax_bench.metal\n"); return 1; }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 26.2, *)) {
            opts.languageVersion = (MTLLanguageVersion)0x40000;  // Metal 4.0
        }
        printf("Compiling Metal 4.0 shaders...\n");
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { printf("Compile error: %s\n", [[err description] UTF8String]); return 1; }

        // PSOs
        id<MTLComputePipelineState> pso_fma = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"dequant_matvec_fma"] error:&err];
        id<MTLComputePipelineState> pso_deq = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"dequant_4bit_to_half"] error:&err];
        id<MTLComputePipelineState> pso_baseline = [dev newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"gemm_half_baseline"] error:&err];

        id<MTLComputePipelineState> pso_nax = nil;
        if (hasNAX) {
            double t0 = now_ms();
            pso_nax = [dev newComputePipelineStateWithFunction:
                [lib newFunctionWithName:@"gemm_nax"] error:&err];
            printf("NAX PSO: %.0f ms (threadExecWidth=%lu)\n",
                   now_ms() - t0, (unsigned long)[pso_nax threadExecutionWidth]);
            if (!pso_nax) printf("NAX PSO failed: %s\n", [[err description] UTF8String]);
        }
        if (!pso_baseline) printf("Baseline PSO failed: %s\n", [[err description] UTF8String]);

        id<MTLCommandQueue> queue = [dev newCommandQueue];

        // Test cases: [M, N, K]
        typedef struct { const char *name; int M, N, K; } Case;
        Case cases[] = {
            {"Q_proj",   1,   4096,  4096},
            {"Q_proj",   4,   4096,  4096},
            {"Q_proj",  16,   4096,  4096},
            {"Q_proj",  64,   4096,  4096},
            {"Q_proj", 128,   4096,  4096},
            {"LM_head",  1, 248320, 4096},
            {"LM_head",  4, 248320, 4096},
            {"LM_head", 16, 248320, 4096},
        };
        int n_cases = sizeof(cases)/sizeof(cases[0]);
        int warmup = 3, iters = 10;

        printf("\n%-12s %5s | %9s %9s %9s | %s  %s\n",
               "Test", "M", "FMA(ms)", "SIMD(ms)", "NAX(ms)", "NAX/FMA", "NAX/SIMD");
        printf("%-12s-%5s-+-%9s-%9s-%9s-+-%s--%s\n",
               "------------", "-----", "---------", "---------", "---------", "-------", "--------");

        for (int c = 0; c < n_cases; c++) {
            Case tc = cases[c];
            int M = tc.M, N = tc.N, K = tc.K;
            int packed_cols = K / 8;
            int num_groups = K / 64;

            // Buffers
            id<MTLBuffer> bW = [dev newBufferWithLength:N*packed_cols*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bS = [dev newBufferWithLength:N*num_groups*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB = [dev newBufferWithLength:N*num_groups*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bX_f32 = [dev newBufferWithLength:M*K*4 options:MTLResourceStorageModeShared];
            // Dequantized weight + half input + outputs
            id<MTLBuffer> bW_half = [dev newBufferWithLength:(size_t)N*K*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bX_half = [dev newBufferWithLength:M*K*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bY_f32 = [dev newBufferWithLength:(size_t)M*N*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bY_nax = [dev newBufferWithLength:(size_t)M*N*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bY_base = [dev newBufferWithLength:(size_t)M*N*4 options:MTLResourceStorageModeShared];

            // Random fill
            arc4random_buf([bW contents], N*packed_cols*4);
            arc4random_buf([bX_f32 contents], M*K*4);
            // Scale/bias: small values
            uint16_t *sp = [bS contents], *bp = [bB contents];
            for (int i = 0; i < N*num_groups; i++) {
                __fp16 sv = (__fp16)(0.01f * (arc4random_uniform(100)+1));
                __fp16 bv = (__fp16)(0.001f * (int)(arc4random_uniform(200)-100));
                sp[i] = *(uint16_t*)&sv;
                bp[i] = *(uint16_t*)&bv;
            }
            // Half input
            float *xf = [bX_f32 contents]; uint16_t *xh = [bX_half contents];
            for (int i = 0; i < M*K; i++) { __fp16 h = (__fp16)(xf[i]*0.001f); xh[i] = *(uint16_t*)&h; }

            // Pre-dequantize weights for GEMM kernels
            {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:pso_deq];
                [enc setBuffer:bW offset:0 atIndex:0];
                [enc setBuffer:bS offset:0 atIndex:1];
                [enc setBuffer:bB offset:0 atIndex:2];
                [enc setBuffer:bW_half offset:0 atIndex:3];
                [enc setBytes:&N length:4 atIndex:4];
                [enc setBytes:&K length:4 atIndex:5];
                int t = MIN(N*num_groups, 65536);
                [enc dispatchThreads:MTLSizeMake(t,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(t,1024),1,1)];
                [enc endEncoding];
                [cmd commit]; [cmd waitUntilCompleted];
            }

            // === FMA baseline (M=1 only) ===
            double fma_ms = -1;
            if (M == 1 && pso_fma) {
                for (int iter = 0; iter < warmup+iters; iter++) {
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pso_fma];
                    [enc setBuffer:bW offset:0 atIndex:0];
                    [enc setBuffer:bS offset:0 atIndex:1];
                    [enc setBuffer:bB offset:0 atIndex:2];
                    [enc setBuffer:bX_f32 offset:0 atIndex:3];
                    [enc setBuffer:bY_f32 offset:0 atIndex:4];
                    [enc setBytes:&N length:4 atIndex:5];
                    [enc setBytes:&K length:4 atIndex:6];
                    [enc dispatchThreads:MTLSizeMake(1024,1,1) threadsPerThreadgroup:MTLSizeMake(1024,1,1)];
                    [enc endEncoding];
                    [cmd commit]; [cmd waitUntilCompleted];
                    if (iter == warmup) fma_ms = 0;
                    if (iter >= warmup) fma_ms += ([cmd GPUEndTime]-[cmd GPUStartTime])*1000.0;
                }
                fma_ms /= iters;
            }

            // === Simdgroup baseline GEMM ===
            double simd_ms = -1;
            if (pso_baseline) {
                for (int iter = 0; iter < warmup+iters; iter++) {
                    memset([bY_base contents], 0, (size_t)M*N*4);
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pso_baseline];
                    [enc setBuffer:bX_half offset:0 atIndex:0];
                    [enc setBuffer:bW_half offset:0 atIndex:1];
                    [enc setBuffer:bY_base offset:0 atIndex:2];
                    [enc setBytes:&M length:4 atIndex:3];
                    [enc setBytes:&N length:4 atIndex:4];
                    [enc setBytes:&K length:4 atIndex:5];
                    int gx = (N+31)/32, gy = (M+63)/64;
                    [enc dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    [enc endEncoding];
                    [cmd commit]; [cmd waitUntilCompleted];
                    if (iter == warmup) simd_ms = 0;
                    if (iter >= warmup) simd_ms += ([cmd GPUEndTime]-[cmd GPUStartTime])*1000.0;
                }
                simd_ms /= iters;
            }

            // === NAX GEMM ===
            double nax_ms = -1;
            if (pso_nax) {
                for (int iter = 0; iter < warmup+iters; iter++) {
                    memset([bY_nax contents], 0, (size_t)M*N*4);
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:pso_nax];
                    [enc setBuffer:bX_half offset:0 atIndex:0];  // A[M,K]
                    [enc setBuffer:bW_half offset:0 atIndex:1];  // B[N,K]
                    [enc setBuffer:bY_nax offset:0 atIndex:2];   // C[M,N]
                    [enc setBytes:&M length:4 atIndex:3];
                    [enc setBytes:&N length:4 atIndex:4];
                    [enc setBytes:&K length:4 atIndex:5];
                    int gx = (N+31)/32, gy = (M+63)/64;
                    [enc dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    [enc endEncoding];
                    [cmd commit]; [cmd waitUntilCompleted];
                    if (iter == warmup) nax_ms = 0;
                    if (iter >= warmup) nax_ms += ([cmd GPUEndTime]-[cmd GPUStartTime])*1000.0;
                }
                nax_ms /= iters;
            }

            // Verify NAX vs baseline (spot check)
            if (nax_ms >= 0 && simd_ms >= 0 && M*N <= 4096*4096) {
                float *y_nax = [bY_nax contents], *y_base = [bY_base contents];
                float max_err = 0;
                for (int i = 0; i < M*N; i++) {
                    float diff = fabsf(y_nax[i] - y_base[i]);
                    float ref = fabsf(y_base[i]);
                    if (ref > 0.001f && diff/ref > max_err) max_err = diff/ref;
                }
                if (max_err > 0.1f) printf("  WARNING: max relative error = %.4f\n", max_err);
            }

            // Print
            char label[32]; snprintf(label, sizeof(label), "%s", tc.name);
            printf("%-12s %5d |", label, M);
            if (fma_ms >= 0) printf(" %7.3f ms", fma_ms); else printf(" %9s", "-");
            if (simd_ms >= 0) printf(" %7.3f ms", simd_ms); else printf(" %9s", "-");
            if (nax_ms >= 0) printf(" %7.3f ms", nax_ms); else printf(" %9s", "-");
            printf(" |");
            if (fma_ms > 0 && nax_ms > 0) printf(" %5.1fx", fma_ms/nax_ms); else printf(" %7s", "-");
            if (simd_ms > 0 && nax_ms > 0) printf("  %5.1fx", simd_ms/nax_ms); else printf("  %7s", "-");
            printf("\n");
        }

        printf("\nNotes:\n");
        printf("  FMA: Current 4-bit dequant matvec (M=1 only)\n");
        printf("  SIMD: simdgroup_matrix GEMM on pre-dequantized half weights\n");
        printf("  NAX: Metal 4 tensor matmul2d on pre-dequantized half weights\n");
        printf("  Both SIMD and NAX include dequant cost in pre-processing (not timed)\n");
    }
    return 0;
}
