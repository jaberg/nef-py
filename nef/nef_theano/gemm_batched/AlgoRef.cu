
// X[bb] is M by K rowmajor contiguous
// Y[bb] is K by N rowmajor contiguous
// The strategy is to compute a whole product by iterating over dM x dN
// blocks, loading entire row blocks and column blocks into shared memory
__global__ void refk(const int B, const int M, const int N, const int K,
        float *alpha, int sa0,
        float *X, int sx0, int sx1, int sx2,
        float *Y, int sy0, int sy1, int sy2,
        float *beta, int sb0,
        float *Z, int sz0, int sz1, int sz2)
{
    const int bb = blockIdx.x;
    const int dN = blockDim.x;
    const int dM = blockDim.y;
    const int nn_rel = threadIdx.x;
    const int mm_rel = threadIdx.y;

    X += bb * sx0;
    Y += bb * sy0;
    Z += bb * sz0;
    float alpha_bb = alpha[bb * sa0];
    float beta_bb = beta[bb * sb0];

    for (int mm = mm_rel; mm < M; mm += dM)
    {
        for (int nn = nn_rel; nn < N; nn += dN)
        {
            float ksum = 0.0;
            for (int kk = 0; kk < K; ++kk)
            {
                float Xmk = X[mm * sx1 + kk * sx2];
                float Ykn = Y[kk * sy1 + nn * sy2];
                ksum += Xmk * Ykn;
            }
            float tmp = Z[mm * sz1 + nn * sz2];
            Z[mm * sz1 + nn * sz2] = beta_bb * tmp + alpha_bb * ksum;
        }
    }
}
