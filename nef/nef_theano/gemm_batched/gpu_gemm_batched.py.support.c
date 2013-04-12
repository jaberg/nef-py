#define MAX_N_THREADS 512
#define MAX_SHARED_FLOATS 3800

__global__ void %(name)s_asdf(const int K, const int B,
        float *alpha, int sa0,
        float *X, int sx0, int sx1, int sx2,
        float *Y, int sy0, int sy1, int sy2,
        float *beta, int sb0,
        float *Z, int sz0, int sz1, int sz2)
{
    extern __shared__ float buf[];
    const int bb8 = blockIdx.x;
    const int N = blockDim.x;
    const int M = blockDim.y;
    const int nn = threadIdx.x;
    const int mm = threadIdx.y;
    const int bbrel = threadIdx.z;
    const int nthreads = M * N * 8;
    const int bb = bb8 * 8 + bbrel;
    if (bb >= B) return;
    float ksum = 0.0;
    X += bb * sx0;
    Y += bb * sy0;

    float * Xbuf = buf; //+ (M* K + N * K) * bbrel;
    float * Ybuf = buf + M * K * 8;

    //-- copy X and Y into shared memory
    for (int ii = nn + mm * N + bbrel * M * N; ii < M * K; ii += M * N * 8) Xbuf[ii] = X[ii];
    for (int ii = nn + mm * N + bbrel * M * N; ii < N * K; ii += M * N * 8) Ybuf[ii] = Y[ii];
    __syncthreads();
 
    Xbuf += bbrel * M * K + nn;
    Ybuf += bbrel * N * K + mm * K;
    for (int kk = 0; kk < K; ++kk)
    {
        ksum += Ybuf[kk * N] * Xbuf[kk];
    }
    Z += bb * sz0;
    alpha += bb * sa0;
    beta += bb * sb0;
    float tmp = Z[mm * sz1 + nn * sz2];
    Z[mm * sz1 + nn * sz2] = beta[0] * tmp + alpha[0] * ksum;
}

__global__ void %(name)s_MNlt512_Xrowmajcontig_Yrowmajcontig_Ksmall(const int K,
        float *alpha, int sa0,
        float *X, int sx0, int sx1, int sx2,
        float *Y, int sy0, int sy1, int sy2,
        float *beta, int sb0,
        float *Z, int sz0, int sz1, int sz2)
{
    extern __shared__ float buf[];
    const int bb = blockIdx.x;
    const int N = blockDim.x;
    const int M = blockDim.y;
    const int nn = threadIdx.x;
    const int mm = threadIdx.y;
    const int nthreads = M * N;
    float ksum = 0.0;
    X += bb * sx0;
    Y += bb * sy0;
    Z += bb * sz0;
    alpha += bb * sa0;
    beta += bb * sb0;

    float * Xbuf = buf;
    float * Ybuf = buf + M * K;

    //-- copy X and Y into shared memory
    for (int ii = nn + mm * N; ii < M * K; ii += M * N) Xbuf[ii] = X[ii];
    for (int ii = nn + mm * N; ii < N * K; ii += M * N) Ybuf[ii] = Y[ii];
    __syncthreads();
 
    Xbuf += nn;
    Ybuf += mm * K;
    for (int kk = 0; kk < K; ++kk)
    {
        ksum += Ybuf[kk * N] * Xbuf[kk];
    }
    float tmp = Z[mm * sz1 + nn * sz2];
    Z[mm * sz1 + nn * sz2] = beta[0] * tmp + alpha[0] * ksum;
}


// X[bb] is M by K rowmajor contiguous
// Y[bb] is K by N rowmajor contiguous
// The strategy is to compute a whole product by iterating over dM x dN
// blocks, loading entire row blocks and column blocks into shared memory
__global__ void %(name)s_full_row_col(const int K,
        const int M, const int N,
        float *alpha, int sa0,
        float *X, int sx0, int sx1, int sx2,
        float *Y, int sy0, int sy1, int sy2,
        float *beta, int sb0,
        float *Z, int sz0, int sz1, int sz2)
{
    extern __shared__ float buf[];
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

    float * Xbuf = buf;
    float * Ybuf = buf + dM * K;

    for (int mm = mm_rel; mm < M; mm += dM)
    {
        // copy dM full rows of X into shared memory
        for (int kk = nn_rel; kk < K; kk += dN)
        {
            Xbuf[mm_rel * K + kk] = X[mm * sx1 + kk * sx2];
        }
        for (int nn = nn_rel; nn < N; nn += dN)
        {
            // copy dN full columns of Y into shared memory
            for (int kk = mm_rel; kk < K; kk += dM)
            {
                Ybuf[kk * dN + nn_rel] = Y[kk * sx1 + nn * sx2];
            }

            __syncthreads();
            float ksum = 0.0;
            for (int kk = 0; kk < K; ++kk)
            {
                ksum += Xbuf[kk * dN + nn_rel] * Ybuf[mm_rel * K + kk];
            }
            float tmp = Z[mm * sz1 + nn * sz2];
            Z[mm * sz1 + nn * sz2] = beta_bb * tmp + alpha_bb * ksum;
            __syncthreads();
        }
    }
}

// X[bb] is M by K rowmajor contiguous
// Y[bb] is K by 1 contiguous
// The strategy is to compute a whole product by iterating over dM x dN
// blocks, loading entire row blocks and column blocks into shared memory
__global__ void %(name)s_N1(const int K,
        const int M,
        float *alpha, int sa0,
        float *X, int sx0, int sx1, int sx2,
        float *Y, int sy0, int sy1, int sy2,
        float *beta, int sb0,
        float *Z, int sz0, int sz1, int sz2)
{
    extern __shared__ float Ybuf[];
    const int bb = blockIdx.x;

    X += bb * sx0;
    Y += bb * sy0;
    Z += bb * sz0;
    float alpha_bb = alpha[bb * sa0];
    float beta_bb = beta[bb * sb0];

    for (int kk = threadIdx.x; kk < K; kk += blockDim.x)
    {
        Ybuf[kk] = Y[kk * sy1];
    }
    __syncthreads();

    for (int mm = threadIdx.x; mm < M; mm += blockDim.x)
    {
        float ksum = 0.0;
        for (int kk = 0; kk < K; ++kk)
        {
            ksum += X[mm * sx1 + kk * sx2] * Ybuf[kk];
        }
        float tmp = Z[mm * sz1];
        Z[mm * sz1] = beta_bb * tmp + alpha_bb * ksum;
        }
    }
