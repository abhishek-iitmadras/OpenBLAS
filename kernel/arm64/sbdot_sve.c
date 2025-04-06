// #ifdef USE_SVE

// #define HAVE_SBDOT_ACCL_KERNEL 1
// #include "common.h"
// #include <arm_sve.h>

// /* 
//  * Manual conversion from bfloat16 to float32 using SVE
//  * This function extracts the bfloat16 values and converts them to float32
//  */
// static inline svfloat32_t bf16_to_f32_z(svbool_t pg, svbfloat16_t bf16_vec) {
//     // Convert bfloat16 to uint16_t bits
//     svuint16_t bf16_bits = svreinterpret_u16_bf16(bf16_vec);
    
//     // Zero-extend to 32 bits - using alternative approach without svextw_u32_z
//     // First, interleave with zeros
//     svuint32_t tmp = svreinterpret_u32_u16(bf16_bits);
    
//     // Shift left by 16 bits to properly position the exponent and mantissa
//     svuint32_t f32_bits = svlsl_n_u32_z(pg, tmp, 16);
    
//     // Reinterpret as float32
//     return svreinterpret_f32_u32(f32_bits);
// }

// static float sbdot_accl_kernel(BLASLONG n, bfloat16 *x, bfloat16 *y)
// {
//     // Get the current vector length in bytes and elements
//     const int sve_vl_bytes = svcntb();
//     const int sve_vl_bf16 = sve_vl_bytes / sizeof(bfloat16);
    
//     // Initialize accumulators
//     svfloat32_t accum0 = svdup_f32(0.0f);
//     svfloat32_t accum1 = svdup_f32(0.0f);
//     float final_result = 0.0f;
    
//     // Full predicate for float32 operations
//     svbool_t pred_f32 = svptrue_b32();
    
//     if (n >= sve_vl_bf16) {
//         // Process vectors in pairs when possible
//         BLASLONG i = 0;
//         BLASLONG tail_index_2x = n & ~(2 * sve_vl_bf16 - 1);
        
//         // Create a predicate for a full bf16 vector
//         svbool_t pred_full = svptrue_b16();
        
//         for (; i < tail_index_2x; i += 2 * sve_vl_bf16) {
//             // Load bf16 vectors - cast to __bf16* to match SVE function signature
//             svbfloat16_t x_vec0 = svld1_bf16(pred_full, (__bf16*)&x[i]);
//             svbfloat16_t y_vec0 = svld1_bf16(pred_full, (__bf16*)&y[i]);
            
//             svbfloat16_t x_vec1 = svld1_bf16(pred_full, (__bf16*)&x[i + sve_vl_bf16]);
//             svbfloat16_t y_vec1 = svld1_bf16(pred_full, (__bf16*)&y[i + sve_vl_bf16]);
            
//             // Convert bf16 to f32 and multiply-accumulate
//             svfloat32_t x_f32_0 = bf16_to_f32_z(pred_f32, x_vec0);
//             svfloat32_t y_f32_0 = bf16_to_f32_z(pred_f32, y_vec0);
//             accum0 = svmla_f32_z(pred_f32, accum0, x_f32_0, y_f32_0);
            
//             svfloat32_t x_f32_1 = bf16_to_f32_z(pred_f32, x_vec1);
//             svfloat32_t y_f32_1 = bf16_to_f32_z(pred_f32, y_vec1);
//             accum1 = svmla_f32_z(pred_f32, accum1, x_f32_1, y_f32_1);
//         }
        
//         // Process remaining full vector
//         BLASLONG tail_index_1x = n & ~(sve_vl_bf16 - 1);
        
//         for (; i < tail_index_1x; i += sve_vl_bf16) {
//             svbfloat16_t x_vec = svld1_bf16(pred_full, (__bf16*)&x[i]);
//             svbfloat16_t y_vec = svld1_bf16(pred_full, (__bf16*)&y[i]);
            
//             svfloat32_t x_f32 = bf16_to_f32_z(pred_f32, x_vec);
//             svfloat32_t y_f32 = bf16_to_f32_z(pred_f32, y_vec);
//             accum0 = svmla_f32_z(pred_f32, accum0, x_f32, y_f32);
//         }
        
//         // Process remaining elements (less than one vector)
//         if (i < n) {
//             // Create predicate for remaining elements, using consistent types (uint64_t)
//             uint64_t remaining = n - i;
//             svbool_t pred_tail = svwhilelt_b16((uint64_t)0, remaining);
            
//             svbfloat16_t x_vec = svld1_bf16(pred_tail, (__bf16*)&x[i]);
//             svbfloat16_t y_vec = svld1_bf16(pred_tail, (__bf16*)&y[i]);
            
//             // Create a corresponding predicate for float32 operations
//             svbool_t pred_f32_tail = svpnext_b32(svpfalse_b(), pred_tail);
            
//             svfloat32_t x_f32 = bf16_to_f32_z(pred_f32_tail, x_vec);
//             svfloat32_t y_f32 = bf16_to_f32_z(pred_f32_tail, y_vec);
//             accum1 = svmla_f32_z(pred_f32_tail, accum1, x_f32, y_f32);
//         }
//     } else if (n > 0) {
//         // Small arrays (less than one vector)
//         uint64_t remaining = n;
//         svbool_t pred_tail = svwhilelt_b16((uint64_t)0, remaining);
        
//         svbfloat16_t x_vec = svld1_bf16(pred_tail, (__bf16*)&x[0]);
//         svbfloat16_t y_vec = svld1_bf16(pred_tail, (__bf16*)&y[0]);
        
//         // Create a corresponding predicate for float32 operations
//         svbool_t pred_f32_tail = svpnext_b32(svpfalse_b(), pred_tail);
        
//         svfloat32_t x_f32 = bf16_to_f32_z(pred_f32_tail, x_vec);
//         svfloat32_t y_f32 = bf16_to_f32_z(pred_f32_tail, y_vec);
//         accum0 = svmla_f32_z(pred_f32_tail, accum0, x_f32, y_f32);
//     }
    
//     // Combine the two accumulators
//     accum0 = svadd_f32_z(pred_f32, accum0, accum1);
    
//     // Horizontal sum
//     final_result = svaddv_f32(pred_f32, accum0);
    
//     return final_result;
// }

// #endif // USE_SVE

// #ifdef USE_SVE

// #define HAVE_SBDOT_ACCL_KERNEL 1
// #include "common.h"
// #include <arm_sve.h>

// /* 
//  * Manual conversion from bfloat16 to float32 using SVE
//  * Only used when hardware BF16 operations aren't available
//  */
// static inline svfloat32_t bf16_to_f32_z(svbool_t pg, svbfloat16_t bf16_vec) {
//     // Convert bfloat16 to uint16_t bits
//     svuint16_t bf16_bits = svreinterpret_u16_bf16(bf16_vec);
    
//     // Zero-extend to 32 bits
//     svuint32_t tmp = svreinterpret_u32_u16(bf16_bits);
    
//     // Shift left by 16 bits to properly position the exponent and mantissa
//     svuint32_t f32_bits = svlsl_n_u32_z(pg, tmp, 16);
    
//     // Reinterpret as float32
//     return svreinterpret_f32_u32(f32_bits);
// }

// static float sbdot_accl_kernel(BLASLONG n, bfloat16 *x, bfloat16 *y)
// {
//     // Early return for empty inputs
//     if (n <= 0) return 0.0f;
    
//     // Get the current vector length
//     const uint64_t sve_vl = svcntb() / sizeof(bfloat16);
    
//     // Initialize accumulators
//     svfloat32_t accum0 = svdup_f32(0.0f);
//     svfloat32_t accum1 = svdup_f32(0.0f);
    
//     // Full predicates
//     const svbool_t all_true_bf16 = svptrue_b16();
//     const svbool_t all_true_f32 = svptrue_b32();
    
//     // Main vector processing with 2x vector unrolling
//     BLASLONG i = 0;
//     for (; i <= n - 2*sve_vl; i += 2*sve_vl) {
//         // Load two vectors at a time
//         svbfloat16_t x0 = svld1_bf16(all_true_bf16, (__bf16*)&x[i]);
//         svbfloat16_t y0 = svld1_bf16(all_true_bf16, (__bf16*)&y[i]);
//         svbfloat16_t x1 = svld1_bf16(all_true_bf16, (__bf16*)&x[i+sve_vl]);
//         svbfloat16_t y1 = svld1_bf16(all_true_bf16, (__bf16*)&y[i+sve_vl]);
        
// #if defined(__ARM_FEATURE_SVE_BF16) || defined(__ARM_FEATURE_SVE2_BF16)
//         // Native BF16 conversions if available
//         accum0 = svbfdot_f32(accum0, x0, y0);
//         accum1 = svbfdot_f32(accum1, x1, y1);
// #else
//         // Manual conversion and multiply-accumulate
//         svfloat32_t x0_f32 = bf16_to_f32_z(all_true_f32, x0);
//         svfloat32_t y0_f32 = bf16_to_f32_z(all_true_f32, y0);
//         accum0 = svmla_f32_z(all_true_f32, accum0, x0_f32, y0_f32);
        
//         svfloat32_t x1_f32 = bf16_to_f32_z(all_true_f32, x1);
//         svfloat32_t y1_f32 = bf16_to_f32_z(all_true_f32, y1);
//         accum1 = svmla_f32_z(all_true_f32, accum1, x1_f32, y1_f32);
// #endif
//     }
    
//     // Process one more full vector if possible
//     if (i <= n - sve_vl) {
//         svbfloat16_t x_vec = svld1_bf16(all_true_bf16, (__bf16*)&x[i]);
//         svbfloat16_t y_vec = svld1_bf16(all_true_bf16, (__bf16*)&y[i]);
        
// #if defined(__ARM_FEATURE_SVE_BF16) || defined(__ARM_FEATURE_SVE2_BF16)
//         accum0 = svbfdot_f32(accum0, x_vec, y_vec);
// #else
//         svfloat32_t x_f32 = bf16_to_f32_z(all_true_f32, x_vec);
//         svfloat32_t y_f32 = bf16_to_f32_z(all_true_f32, y_vec);
//         accum0 = svmla_f32_z(all_true_f32, accum0, x_f32, y_f32);
// #endif
//         i += sve_vl;
//     }
    
//     // Process remaining elements (tail)
//     if (i < n) {
//         // Create predicate for the tail
//         svbool_t tail_pred = svwhilelt_b16((uint64_t)0, (uint64_t)(n - i));
        
//         // Load remainder with predication
//         svbfloat16_t x_vec = svld1_bf16(tail_pred, (__bf16*)&x[i]);
//         svbfloat16_t y_vec = svld1_bf16(tail_pred, (__bf16*)&y[i]);
        
// #if defined(__ARM_FEATURE_SVE_BF16) || defined(__ARM_FEATURE_SVE2_BF16)
//         accum1 = svbfdot_f32(accum1, x_vec, y_vec);
// #else
//         // Create appropriate predicate for float32 operations
//         svbool_t f32_tail_pred = svwhilelt_b32((uint64_t)0, (uint64_t)(n - i));
        
//         svfloat32_t x_f32 = bf16_to_f32_z(f32_tail_pred, x_vec);
//         svfloat32_t y_f32 = bf16_to_f32_z(f32_tail_pred, y_vec);
//         accum1 = svmla_f32_z(f32_tail_pred, accum1, x_f32, y_f32);
// #endif
//     }
    
//     // Combine accumulators and compute horizontal sum
//     svfloat32_t combined = svadd_f32_z(all_true_f32, accum0, accum1);
//     float result = svaddv_f32(all_true_f32, combined);
    
//     return result;
// }

// #endif // USE_SVE


#ifdef USE_SVE

#define HAVE_SBDOT_ACCL_KERNEL 1
#include "common.h"
#include <arm_sve.h>

static float sbdot_accl_kernel(BLASLONG n, bfloat16 *x, bfloat16 *y) {
    if (n <= 0) return 0.0f;

    const uint64_t vl = svcntb() / sizeof(bfloat16);
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    const svbool_t all_bf16 = svptrue_b16();
    const svbool_t all_f32 = svptrue_b32();

    // Process 2 vectors per iteration
    BLASLONG i = 0;
    for (; i <= n - 2*vl; i += 2*vl) {
        svbfloat16_t x0 = svld1_bf16(all_bf16, (__bf16*)&x[i]);
        svbfloat16_t y0 = svld1_bf16(all_bf16, (__bf16*)&y[i]);
        svbfloat16_t x1 = svld1_bf16(all_bf16, (__bf16*)&x[i+vl]);
        svbfloat16_t y1 = svld1_bf16(all_bf16, (__bf16*)&y[i+vl]);

#if defined(__ARM_FEATURE_SVE_BF16)
        acc0 = svbfdot_f32(acc0, x0, y0);
        acc1 = svbfdot_f32(acc1, x1, y1);
#else
        svfloat32_t x0_f32 = svcvt_f32_bf16_z(all_f32, x0);
        svfloat32_t y0_f32 = svcvt_f32_bf16_z(all_f32, y0);
        acc0 = svmla_f32(acc0, x0_f32, y0_f32);
        
        svfloat32_t x1_f32 = svcvt_f32_bf16_z(all_f32, x1);
        svfloat32_t y1_f32 = svcvt_f32_bf16_z(all_f32, y1);
        acc1 = svmla_f32(acc1, x1_f32, y1_f32);
#endif
    }

    // Process remaining full vector
    if (i <= n - vl) {
        svbfloat16_t xv = svld1_bf16(all_bf16, (__bf16*)&x[i]);
        svbfloat16_t yv = svld1_bf16(all_bf16, (__bf16*)&y[i]);
        
#if defined(__ARM_FEATURE_SVE_BF16)
        acc0 = svbfdot_f32(acc0, xv, yv);
#else
        acc0 = svmla_f32(acc0, 
                        svcvt_f32_bf16_z(all_f32, xv),
                        svcvt_f32_bf16_z(all_f32, yv));
#endif
        i += vl;
    }

    // Process tail elements (1-31 remaining)
    if (i < n) {
        const uint64_t rem = n - i;
        const svbool_t pg_bf16 = svwhilelt_b16(i, n);
        
        svbfloat16_t xv = svld1_bf16(pg_bf16, (__bf16*)&x[i]);
        svbfloat16_t yv = svld1_bf16(pg_bf16, (__bf16*)&y[i]);

#if defined(__ARM_FEATURE_SVE_BF16)
        const uint64_t pairs = (rem + 1) >> 1;
        const svbool_t pg_f32 = svwhilelt_b32(0, pairs);
        acc1 = svbfdot_f32_m(pg_f32, acc1, xv, yv);
#else
        const svbool_t pg_f32 = svwhilelt_b32(0, rem);
        acc1 = svmla_f32_z(pg_f32, acc1,
                          svcvt_f32_bf16_z(pg_f32, xv),
                          svcvt_f32_bf16_z(pg_f32, yv));
#endif
    }

    // Combine and reduce
    return svaddv_f32(all_f32, svadd_f32_z(all_f32, acc0, acc1));
}

#endif // USE_SVE