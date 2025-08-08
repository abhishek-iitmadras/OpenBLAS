/***************************************************************************
Copyright (c) 2024, The OpenBLAS Project
All rights reserved.
*****************************************************************************/

#include "common.h"

// Check for SVE support
#ifdef HAVE_SVE
#if defined __has_include 
#if __has_include(<arm_sve.h>) && __ARM_FEATURE_SVE
#define USE_SVE
#endif 
#endif
#endif

#ifdef USE_SVE
#ifdef AXPBY_KERNEL_SVE
#include AXPBY_KERNEL_SVE
#else
#include "axpby_kernel_sve.c"
#endif
#endif

#include "axpby_kernel_asimd.c"

#if defined(SMP)
extern int blas_level1_thread(int mode, BLASLONG m, BLASLONG n,
	BLASLONG k, void *alpha, void *a, BLASLONG lda, void *b, BLASLONG ldb,
	void *c, BLASLONG ldc, int (*function)(), int nthreads);

#ifdef DYNAMIC_ARCH
extern char* gotoblas_corename(void);
#endif

// Global variables for thread communication
static FLOAT global_alpha;
static FLOAT global_beta;

// Optimal thread count determination
static inline int get_axpby_optimal_nthreads(BLASLONG n) {
    int ncpu = num_cpu_avail(1);
    
    // Tuned thresholds for ARM64 architectures
    if (n <= 50000L)
        return 1;
    else if (n <= 200000L)
        return MIN(ncpu, 2);
    else if (n <= 500000L)
        return MIN(ncpu, 4);
    else if (n <= 1000000L)
        return MIN(ncpu, 8);
    else
        return ncpu;
}
#endif

// Core compute function
static int axpby_compute(BLASLONG n, FLOAT alpha, FLOAT *x, BLASLONG inc_x, 
                         FLOAT beta, FLOAT *y, BLASLONG inc_y)
{
    if (n <= 0) return 0;
    
    // Handle special case where either increment is zero
    if (inc_x == 0 || inc_y == 0) {
        BLASLONG i = 0;
        BLASLONG ix = 0, iy = 0;
        
        if (beta == 0.0) {
            if (alpha == 0.0) {
                while (i < n) {
                    y[iy] = 0.0;
                    iy += inc_y;
                    i++;
                }
            } else {
                while (i < n) {
                    y[iy] = alpha * x[ix];
                    ix += inc_x;
                    iy += inc_y;
                    i++;
                }
            }
        } else {
            if (alpha == 0.0) {
                while (i < n) {
                    y[iy] = beta * y[iy];
                    iy += inc_y;
                    i++;
                }
            } else {
                while (i < n) {
                    y[iy] = alpha * x[ix] + beta * y[iy];
                    ix += inc_x;
                    iy += inc_y;
                    i++;
                }
            }
        }
        return 0;
    }
    
#ifdef USE_SVE
    // Use SVE kernel for unit stride cases when available
    if (inc_x == 1 && inc_y == 1) {
        return axpby_kernel_sve(n, alpha, x, beta, y);
    }
#endif
    
    // Fall back to ASIMD kernel
    return axpby_kernel_asimd(n, alpha, x, inc_x, beta, y, inc_y);
}

#if defined(SMP)
// Thread function for parallel execution
// Note: OpenBLAS threading passes parameters in a specific way
static int axpby_thread_function(BLASLONG n, BLASLONG dummy0,
    BLASLONG dummy1, FLOAT dummy2, FLOAT *x, BLASLONG inc_x, FLOAT *y,
    BLASLONG inc_y, FLOAT *dummy3, BLASLONG dummy4)
{
    // Use global variables for alpha and beta
    axpby_compute(n, global_alpha, x, inc_x, global_beta, y, inc_y);
    return 0;
}
#endif

// Main AXPBY function
int CNAME(BLASLONG n, FLOAT alpha, FLOAT *x, BLASLONG inc_x, 
          FLOAT beta, FLOAT *y, BLASLONG inc_y)
{
#if defined(SMP)
    int nthreads;
    int mode;
#endif
    
    if (n <= 0) return 0;
    
    // Handle negative increments
    if (inc_x < 0) {
        x = x - (n - 1) * inc_x;
        inc_x = -inc_x;
    }
    if (inc_y < 0) {
        y = y - (n - 1) * inc_y;
        inc_y = -inc_y;
    }
    
#if defined(SMP)
    // Determine optimal thread count
    if (inc_x == 0 || inc_y == 0) {
        nthreads = 1;
    } else {
        nthreads = get_axpby_optimal_nthreads(n);
    }
    
    if (nthreads == 1) {
        return axpby_compute(n, alpha, x, inc_x, beta, y, inc_y);
    } else {
        // Setup for parallel execution
#if !defined(DOUBLE)
        mode = BLAS_SINGLE | BLAS_REAL;
#else
        mode = BLAS_DOUBLE | BLAS_REAL;
#endif
        
        // Store alpha and beta in global variables for thread access
        global_alpha = alpha;
        global_beta = beta;
        
        // Use OpenBLAS threading infrastructure
        blas_level1_thread(mode, n, 0, 0, (void *)&alpha, x, inc_x, y, inc_y,
                          (void *)&beta, 0, (void *)axpby_thread_function, nthreads);
    }
#else
    return axpby_compute(n, alpha, x, inc_x, beta, y, inc_y);
#endif
    
    return 0;
}