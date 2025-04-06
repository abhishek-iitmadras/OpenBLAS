#include "common.h"

// Some compilers will report feature support for SVE without the appropriate
// header available
#ifdef HAVE_SVE
#if defined __has_include
#if __has_include(<arm_sve.h>) && __ARM_FEATURE_SVE
#define USE_SVE
#endif
#endif
#endif

#ifdef USE_SVE
#include "sbdot_sve.c"
#endif

static float sbdot_compute(BLASLONG n, bfloat16 *x, BLASLONG inc_x,
                           bfloat16 *y, BLASLONG inc_y)
{
    float d = 0.0f;

#ifdef HAVE_SBDOT_ACCL_KERNEL
    // If increments are both 1, attempt to use the SVE-accelerated kernel
    if ((inc_x == 1) && (inc_y == 1)) {
        return sbdot_accl_kernel(n, x, y);
    }
#endif

    // Fallback path: convert BF16 → FP32, then use a standard float dot
    float *x_fp32 = (float *)malloc(sizeof(float) * n);
    float *y_fp32 = (float *)malloc(sizeof(float) * n);

    if (x_fp32 == NULL || y_fp32 == NULL) {
        // Fallback if allocation fails; not optimal but avoids crash
        if (x_fp32) free(x_fp32);
        if (y_fp32) free(y_fp32);
        return 0.0f;
    }

    // Convert BF16 → FP32
    SBF16TOS_K(n, x, inc_x, x_fp32, 1);
    SBF16TOS_K(n, y, inc_y, y_fp32, 1);

    // Standard float dot
    d = SDOTU_K(n, x_fp32, 1, y_fp32, 1);

    free(x_fp32);
    free(y_fp32);

    return d;
}

#if defined(SMP)
static int sbdot_thread_func(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, bfloat16 dummy2,
                             bfloat16 *x, BLASLONG inc_x, bfloat16 *y, BLASLONG inc_y,
                             float *result, BLASLONG dummy3)
{
    *(float *)result = sbdot_compute(n, x, inc_x, y, inc_y);
    return 0;
}

extern int blas_level1_thread_with_return_value(int mode, BLASLONG m, BLASLONG n, BLASLONG k,
                                                void *alpha, void *a, BLASLONG lda,
                                                void *b, BLASLONG ldb, void *c, BLASLONG ldc,
                                                int (*function)(), int nthreads);
#endif

float CNAME(BLASLONG n, bfloat16 *x, BLASLONG inc_x,
            bfloat16 *y, BLASLONG inc_y)
{
    float dot_result = 0.0f;

    if (n <= 0) return 0.0f;

#if defined(SMP)
    int nthreads;
    int thread_thres = 40960;
    bfloat16 dummy_alpha;

    if (inc_x == 0 || inc_y == 0 || n <= thread_thres)
        nthreads = 1;
    else
        nthreads = num_cpu_avail(1);

    int best_threads = (int)((float)n / (float)thread_thres + 0.5f);
    if (best_threads < nthreads) {
        nthreads = best_threads;
    }

    if (nthreads <= 1) {
        // Single-threaded
        dot_result = sbdot_compute(n, x, inc_x, y, inc_y);
    } else {
        // Multi-threaded path
        char thread_result[MAX_CPU_NUMBER * sizeof(double) * 2];
        int mode = BLAS_BFLOAT16 | BLAS_REAL;

        blas_level1_thread_with_return_value(mode, n, 0, 0, &dummy_alpha,
                                             x, inc_x,
                                             y, inc_y,
                                             thread_result, 0,
                                             (void *)sbdot_thread_func, nthreads);

        float *ptr = (float *)thread_result;
        for (int i = 0; i < nthreads; i++) {
            dot_result += (*ptr);
            ptr = (float *)(((char *)ptr) + sizeof(double) * 2);
        }
    }
#else
    // Single-threaded if SMP is not defined
    dot_result = sbdot_compute(n, x, inc_x, y, inc_y);
#endif

    return dot_result;
}
