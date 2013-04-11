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
