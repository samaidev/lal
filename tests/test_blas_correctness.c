/* test_blas_correctness.c — verify OpenBLAS cblas_sgemv matches reference matmul
 *
 * GPT-2 Conv1D: W is [in_dim, out_dim] row-major, y = W^T @ x + b
 * cblas_sgemv(RowMajor, Trans, M=in_dim, N=out_dim, 1.0, W, lda=out_dim, x, 1, 0.0, y, 1)
 *
 * Compile: gcc -O2 -DUSE_OPENBLAS -o test_blas test_blas.c -lopenblas -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

int main(void) {
    int in_dim = 768, out_dim = 2304;  /* c_attn size */
    float *W = malloc(in_dim * out_dim * sizeof(float));
    float *x = malloc(in_dim * sizeof(float));
    float *b = malloc(out_dim * sizeof(float));
    float *y_ref = malloc(out_dim * sizeof(float));
    float *y_blas = malloc(out_dim * sizeof(float));

    srand(42);
    for (int i = 0; i < in_dim * out_dim; i++) W[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    for (int i = 0; i < in_dim; i++) x[i] = (rand()/(float)RAND_MAX - 0.5f) * 2;
    for (int j = 0; j < out_dim; j++) b[j] = (rand()/(float)RAND_MAX - 0.5f) * 0.5f;

    /* Reference: y[j] = sum_i W[i][j] * x[i] + b[j]
     * W[i][j] = W[i*out_dim + j] (row-major) */
    for (int j = 0; j < out_dim; j++) {
        float s = b[j];
        for (int i = 0; i < in_dim; i++) s += W[i * out_dim + j] * x[i];
        y_ref[j] = s;
    }

    /* BLAS: y = W^T @ x + b
     * W is [in_dim, out_dim] row-major. W^T is [out_dim, in_dim].
     * cblas_sgemv(RowMajor, Trans, M, N, alpha, A, lda, x, incx, beta, y, incy)
     *   op(A) has shape [N, M] when trans, A is [M, N] with lda >= N (row-major)
     *   result y has len N
     * So: M=in_dim, N=out_dim, A=W[in_dim][out_dim], lda=out_dim
     *   op(A) = A^T = [out_dim, in_dim]
     *   y = A^T @ x = [out_dim, in_dim] @ [in_dim] = [out_dim] ✓ */
    cblas_sgemv(CblasRowMajor, CblasTrans, in_dim, out_dim,
                1.0f, W, out_dim, x, 1, 0.0f, y_blas, 1);
    cblas_saxpy(out_dim, 1.0f, b, 1, y_blas, 1);

    /* Compare */
    float max_diff = 0, mean_diff = 0;
    for (int j = 0; j < out_dim; j++) {
        float d = fabsf(y_ref[j] - y_blas[j]);
        if (d > max_diff) max_diff = d;
        mean_diff += d;
    }
    mean_diff /= out_dim;
    printf("Reference vs BLAS:\n");
    printf("  max diff = %.6e\n", max_diff);
    printf("  mean diff = %.6e\n", mean_diff);
    printf("  y_ref[0..3] = %.4f %.4f %.4f %.4f\n", y_ref[0], y_ref[1], y_ref[2], y_ref[3]);
    printf("  y_blas[0..3] = %.4f %.4f %.4f %.4f\n", y_blas[0], y_blas[1], y_blas[2], y_blas[3]);

    if (max_diff < 1e-3f) {
        printf("PASS: BLAS matches reference\n");
        return 0;
    } else {
        printf("FAIL: BLAS mismatch!\n");
        return 1;
    }
}
