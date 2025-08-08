/***************************************************************************
Copyright (c) 2024, The OpenBLAS Project
All rights reserved.
*****************************************************************************/

#include <arm_neon.h>

// ASIMD/NEON optimized axpby kernel
// Computes: y = alpha * x + beta * y
static int axpby_kernel_asimd(BLASLONG n, FLOAT alpha, FLOAT * restrict x, BLASLONG inc_x,
                              FLOAT beta, FLOAT * restrict y, BLASLONG inc_y)
{
    BLASLONG i = 0;
    BLASLONG ix = 0, iy = 0;
    
#if !defined(DOUBLE)
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vbeta = vdupq_n_f32(beta);
    float32x4_t vx0, vx1, vy0, vy1;
#else
    float64x2_t valpha = vdupq_n_f64(alpha);
    float64x2_t vbeta = vdupq_n_f64(beta);
    float64x2_t vx0, vx1, vy0, vy1;
#endif
    
    // Optimized path for unit stride
    if (inc_x == 1 && inc_y == 1) {
        BLASLONG n_simd;
        
#if !defined(DOUBLE)
        n_simd = n & ~7;  // Process 8 elements at a time
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0 - use memset for better performance
                memset(y, 0, n * sizeof(FLOAT));
                return 0;
            } else {
                // y = alpha * x
                for (i = 0; i < n_simd; i += 8) {
                    vx0 = vld1q_f32(&x[i]);
                    vx1 = vld1q_f32(&x[i + 4]);
                    vy0 = vmulq_f32(vx0, valpha);
                    vy1 = vmulq_f32(vx1, valpha);
                    vst1q_f32(&y[i], vy0);
                    vst1q_f32(&y[i + 4], vy1);
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y
            for (i = 0; i < n_simd; i += 8) {
                vy0 = vld1q_f32(&y[i]);
                vy1 = vld1q_f32(&y[i + 4]);
                vy0 = vmulq_f32(vy0, vbeta);
                vy1 = vmulq_f32(vy1, vbeta);
                vst1q_f32(&y[i], vy0);
                vst1q_f32(&y[i + 4], vy1);
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n_simd; i += 8) {
                vx0 = vld1q_f32(&x[i]);
                vx1 = vld1q_f32(&x[i + 4]);
                vy0 = vld1q_f32(&y[i]);
                vy1 = vld1q_f32(&y[i + 4]);
                vy0 = vmlaq_f32(vmulq_f32(vy0, vbeta), vx0, valpha);
                vy1 = vmlaq_f32(vmulq_f32(vy1, vbeta), vx1, valpha);
                vst1q_f32(&y[i], vy0);
                vst1q_f32(&y[i + 4], vy1);
            }
        }
        
        // Handle remaining elements
        if (beta == 0.0) {
            if (alpha == 0.0) {
                for (; i < n; i++) {
                    y[i] = 0.0;
                }
            } else {
                for (; i < n; i++) {
                    y[i] = alpha * x[i];
                }
            }
        } else if (alpha == 0.0) {
            for (; i < n; i++) {
                y[i] = beta * y[i];
            }
        } else {
            for (; i < n; i++) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }
#else
        n_simd = n & ~3;  // Process 4 elements at a time for double
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0 - use memset for better performance
                memset(y, 0, n * sizeof(FLOAT));
                return 0;
            } else {
                // y = alpha * x
                for (i = 0; i < n_simd; i += 4) {
                    vx0 = vld1q_f64(&x[i]);
                    vx1 = vld1q_f64(&x[i + 2]);
                    vy0 = vmulq_f64(vx0, valpha);
                    vy1 = vmulq_f64(vx1, valpha);
                    vst1q_f64(&y[i], vy0);
                    vst1q_f64(&y[i + 2], vy1);
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y
            for (i = 0; i < n_simd; i += 4) {
                vy0 = vld1q_f64(&y[i]);
                vy1 = vld1q_f64(&y[i + 2]);
                vy0 = vmulq_f64(vy0, vbeta);
                vy1 = vmulq_f64(vy1, vbeta);
                vst1q_f64(&y[i], vy0);
                vst1q_f64(&y[i + 2], vy1);
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n_simd; i += 4) {
                vx0 = vld1q_f64(&x[i]);
                vx1 = vld1q_f64(&x[i + 2]);
                vy0 = vld1q_f64(&y[i]);
                vy1 = vld1q_f64(&y[i + 2]);
                vy0 = vfmaq_f64(vmulq_f64(vy0, vbeta), vx0, valpha);
                vy1 = vfmaq_f64(vmulq_f64(vy1, vbeta), vx1, valpha);
                vst1q_f64(&y[i], vy0);
                vst1q_f64(&y[i + 2], vy1);
            }
        }
        
        // Handle remaining elements
        if (beta == 0.0) {
            if (alpha == 0.0) {
                for (; i < n; i++) {
                    y[i] = 0.0;
                }
            } else {
                for (; i < n; i++) {
                    y[i] = alpha * x[i];
                }
            }
        } else if (alpha == 0.0) {
            for (; i < n; i++) {
                y[i] = beta * y[i];
            }
        } else {
            for (; i < n; i++) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }
#endif
        
    } else if (inc_y == 1) {
        // x has non-unit stride, y has unit stride
        BLASLONG n_simd;
        
#if !defined(DOUBLE)
        n_simd = n & ~3;
        ix = 0;
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0
                memset(y, 0, n * sizeof(FLOAT));
                return 0;
            } else {
                // y = alpha * x (gather x)
                for (i = 0; i < n_simd; i += 4) {
                    float32x4_t vx_gathered = (float32x4_t){x[ix], x[ix + inc_x], 
                                               x[ix + 2*inc_x], x[ix + 3*inc_x]};
                    vy0 = vmulq_f32(vx_gathered, valpha);
                    vst1q_f32(&y[i], vy0);
                    ix += 4 * inc_x;
                }
                // Handle remaining
                for (; i < n; i++) {
                    y[i] = alpha * x[ix];
                    ix += inc_x;
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y
            for (i = 0; i < n_simd; i += 4) {
                vy0 = vld1q_f32(&y[i]);
                vy0 = vmulq_f32(vy0, vbeta);
                vst1q_f32(&y[i], vy0);
            }
            for (; i < n; i++) {
                y[i] = beta * y[i];
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n_simd; i += 4) {
                float32x4_t vx_gathered = (float32x4_t){x[ix], x[ix + inc_x], 
                                           x[ix + 2*inc_x], x[ix + 3*inc_x]};
                vy0 = vld1q_f32(&y[i]);
                vy0 = vmlaq_f32(vmulq_f32(vy0, vbeta), vx_gathered, valpha);
                vst1q_f32(&y[i], vy0);
                ix += 4 * inc_x;
            }
            // Handle remaining
            for (; i < n; i++) {
                y[i] = alpha * x[ix] + beta * y[i];
                ix += inc_x;
            }
        }
#else
        n_simd = n & ~1;
        ix = 0;
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0
                memset(y, 0, n * sizeof(FLOAT));
                return 0;
            } else {
                // y = alpha * x (gather x)
                for (i = 0; i < n_simd; i += 2) {
                    float64x2_t vx_gathered = (float64x2_t){x[ix], x[ix + inc_x]};
                    vy0 = vmulq_f64(vx_gathered, valpha);
                    vst1q_f64(&y[i], vy0);
                    ix += 2 * inc_x;
                }
                // Handle remaining
                if (i < n) {
                    y[i] = alpha * x[ix];
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y
            for (i = 0; i < n_simd; i += 2) {
                vy0 = vld1q_f64(&y[i]);
                vy0 = vmulq_f64(vy0, vbeta);
                vst1q_f64(&y[i], vy0);
            }
            if (i < n) {
                y[i] = beta * y[i];
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n_simd; i += 2) {
                float64x2_t vx_gathered = (float64x2_t){x[ix], x[ix + inc_x]};
                vy0 = vld1q_f64(&y[i]);
                vy0 = vfmaq_f64(vmulq_f64(vy0, vbeta), vx_gathered, valpha);
                vst1q_f64(&y[i], vy0);
                ix += 2 * inc_x;
            }
            // Handle remaining element
            if (i < n) {
                y[i] = alpha * x[ix] + beta * y[i];
            }
        }
#endif
        
    } else if (inc_x == 1) {
        // x has unit stride, y has non-unit stride
        BLASLONG n_simd;
        
#if !defined(DOUBLE)
        n_simd = n & ~3;
        iy = 0;
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0 (scatter)
                for (i = 0; i < n; i++) {
                    y[iy] = 0.0;
                    iy += inc_y;
                }
            } else {
                // y = alpha * x (scatter)
                for (i = 0; i < n_simd; i += 4) {
                    vx0 = vld1q_f32(&x[i]);
                    vy0 = vmulq_f32(vx0, valpha);
                    y[iy] = vgetq_lane_f32(vy0, 0);
                    y[iy + inc_y] = vgetq_lane_f32(vy0, 1);
                    y[iy + 2*inc_y] = vgetq_lane_f32(vy0, 2);
                    y[iy + 3*inc_y] = vgetq_lane_f32(vy0, 3);
                    iy += 4 * inc_y;
                }
                for (; i < n; i++) {
                    y[iy] = alpha * x[i];
                    iy += inc_y;
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y (gather-scatter)
            for (i = 0; i < n; i++) {
                y[iy] = beta * y[iy];
                iy += inc_y;
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n_simd; i += 4) {
                vx0 = vld1q_f32(&x[i]);
                vx0 = vmulq_f32(vx0, valpha);
                y[iy] = vgetq_lane_f32(vx0, 0) + beta * y[iy];
                y[iy + inc_y] = vgetq_lane_f32(vx0, 1) + beta * y[iy + inc_y];
                y[iy + 2*inc_y] = vgetq_lane_f32(vx0, 2) + beta * y[iy + 2*inc_y];
                y[iy + 3*inc_y] = vgetq_lane_f32(vx0, 3) + beta * y[iy + 3*inc_y];
                iy += 4 * inc_y;
            }
            for (; i < n; i++) {
                y[iy] = alpha * x[i] + beta * y[iy];
                iy += inc_y;
            }
        }
#else
        n_simd = n & ~1;
        iy = 0;
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0 (scatter)
                for (i = 0; i < n; i++) {
                    y[iy] = 0.0;
                    iy += inc_y;
                }
            } else {
                // y = alpha * x (scatter)
                for (i = 0; i < n_simd; i += 2) {
                    vx0 = vld1q_f64(&x[i]);
                    vy0 = vmulq_f64(vx0, valpha);
                    y[iy] = vgetq_lane_f64(vy0, 0);
                    y[iy + inc_y] = vgetq_lane_f64(vy0, 1);
                    iy += 2 * inc_y;
                }
                if (i < n) {
                    y[iy] = alpha * x[i];
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y (gather-scatter)
            for (i = 0; i < n; i++) {
                y[iy] = beta * y[iy];
                iy += inc_y;
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n_simd; i += 2) {
                vx0 = vld1q_f64(&x[i]);
                vx0 = vmulq_f64(vx0, valpha);
                y[iy] = vgetq_lane_f64(vx0, 0) + beta * y[iy];
                y[iy + inc_y] = vgetq_lane_f64(vx0, 1) + beta * y[iy + inc_y];
                iy += 2 * inc_y;
            }
            // Handle remaining element
            if (i < n) {
                y[iy] = alpha * x[i] + beta * y[iy];
            }
        }
#endif
        
    } else {
        // Both have non-unit stride - scalar fallback
        ix = 0;
        iy = 0;
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                // y = 0
                for (i = 0; i < n; i++) {
                    y[iy] = 0.0;
                    iy += inc_y;
                }
            } else {
                // y = alpha * x
                for (i = 0; i < n; i++) {
                    y[iy] = alpha * x[ix];
                    ix += inc_x;
                    iy += inc_y;
                }
            }
        } else if (alpha == 0.0) {
            // y = beta * y
            for (i = 0; i < n; i++) {
                y[iy] = beta * y[iy];
                iy += inc_y;
            }
        } else {
            // y = alpha * x + beta * y
            for (i = 0; i < n; i++) {
                y[iy] = alpha * x[ix] + beta * y[iy];
                ix += inc_x;
                iy += inc_y;
            }
        }
    }
    
    return 0;
}