/***************************************************************************
Copyright (c) 2024, The OpenBLAS Project
All rights reserved.
*****************************************************************************/

#include <arm_sve.h>

// SVE optimized axpby kernel for unit stride
// Computes: y = alpha * x + beta * y
static int axpby_kernel_sve(BLASLONG n, FLOAT alpha, FLOAT * restrict x, 
                            FLOAT beta, FLOAT * restrict y)
{
    BLASLONG i = 0;
    
#if !defined(DOUBLE)
    // Single precision path
    svfloat32_t valpha = svdup_f32(alpha);
    svfloat32_t vbeta = svdup_f32(beta);
    svfloat32_t vx, vy;
    svbool_t pg;
    size_t step = svcntw();  // Number of 32-bit elements
    
    if (beta == 0.0f) {
        if (alpha == 0.0f) {
            // y = 0 - use memset for unit stride
            memset(y, 0, n * sizeof(FLOAT));
        } else {
            // y = alpha * x
            for (i = 0; i < n; i += step) {
                pg = svwhilelt_b32(i, n);
                vx = svld1_f32(pg, &x[i]);
                vy = svmul_f32_z(pg, vx, valpha);
                svst1_f32(pg, &y[i], vy);
            }
        }
    } else if (alpha == 0.0f) {
        // y = beta * y
        for (i = 0; i < n; i += step) {
            pg = svwhilelt_b32(i, n);
            vy = svld1_f32(pg, &y[i]);
            vy = svmul_f32_z(pg, vy, vbeta);
            svst1_f32(pg, &y[i], vy);
        }
    } else {
        // y = alpha * x + beta * y
        // Use FMA for better performance
        for (i = 0; i < n; i += step) {
            pg = svwhilelt_b32(i, n);
            vx = svld1_f32(pg, &x[i]);
            vy = svld1_f32(pg, &y[i]);
            vy = svmul_f32_z(pg, vy, vbeta);        // β*y
            vy = svmla_f32_m(pg, vy, vx, valpha);   // + α*x
            svst1_f32(pg, &y[i], vy);
        }
    }
#else
    // Double precision path
    svfloat64_t valpha = svdup_f64(alpha);
    svfloat64_t vbeta = svdup_f64(beta);
    svfloat64_t vx, vy;
    svbool_t pg;
    size_t step = svcntd();  // Number of 64-bit elements
    
    if (beta == 0.0) {
        if (alpha == 0.0) {
            // y = 0 - use memset for unit stride
            memset(y, 0, n * sizeof(FLOAT));
        } else {
            // y = alpha * x
            for (i = 0; i < n; i += step) {
                pg = svwhilelt_b64(i, n);
                vx = svld1_f64(pg, &x[i]);
                vy = svmul_f64_z(pg, vx, valpha);
                svst1_f64(pg, &y[i], vy);
            }
        }
    } else if (alpha == 0.0) {
        // y = beta * y
        for (i = 0; i < n; i += step) {
            pg = svwhilelt_b64(i, n);
            vy = svld1_f64(pg, &y[i]);
            vy = svmul_f64_z(pg, vy, vbeta);
            svst1_f64(pg, &y[i], vy);
        }
    } else {
        // y = alpha * x + beta * y
        // Use FMA for better performance
        for (i = 0; i < n; i += step) {
            pg = svwhilelt_b64(i, n);
            vx = svld1_f64(pg, &x[i]);
            vy = svld1_f64(pg, &y[i]);
            vy = svmul_f64_z(pg, vy, vbeta);        // β*y
            vy = svmla_f64_m(pg, vy, vx, valpha);   // + α*x
            svst1_f64(pg, &y[i], vy);
        }
    }
#endif
    
    return 0;
}
