{
  int elemsize = 4;
  float fbeta;
  double dbeta;
  int B = 0;
  int M = 0;
  int N = 0;
  int K = 0;

  if ((%(alpha)s)->nd != 1)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(alpha) != 1");
      %(fail)s;
    }
  if ((%(X)s)->nd != 3)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(X) != 2");
      %(fail)s;
    }
  if ((%(Y)s)->nd != 3)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(Y) != 3");
      %(fail)s;
    }
  if ((%(beta)s)->nd != 1)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(beta) != 0");
      %(fail)s;
    }
  if ((%(Z)s)->nd != 3)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(Z) != 3");
      %(fail)s;
    }


  B = CudaNdarray_HOST_DIMS(%(Z)s)[0];
  M = CudaNdarray_HOST_DIMS(%(Z)s)[1];
  N = CudaNdarray_HOST_DIMS(%(Z)s)[2];

  // alpha dim check
  if (CudaNdarray_HOST_DIMS(%(alpha)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: alpha.shape[0] != Z.shape[0]");
      %(fail)s;
    }

  // X dim check
  if (CudaNdarray_HOST_DIMS(%(X)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: X.shape[0] != Z.shape[0]");
      %(fail)s;
    }
  if (CudaNdarray_HOST_DIMS(%(X)s)[1] != M)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: X.shape[1] != Z.shape[1]");
      %(fail)s;
    }
  K = CudaNdarray_HOST_DIMS(%(X)s)[2];

  // Y dim check
  if (CudaNdarray_HOST_DIMS(%(Y)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: Y.shape[0] != Z.shape[0]");
      %(fail)s;
    }
  if (CudaNdarray_HOST_DIMS(%(Y)s)[1] != K)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: Y.shape[1] != X.shape[2]");
      %(fail)s;
    }
  if (CudaNdarray_HOST_DIMS(%(Y)s)[2] != N)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: Y.shape[2] != Z.shape[2]");
      %(fail)s;
    }

  // beta dim check
  if (CudaNdarray_HOST_DIMS(%(beta)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: beta.shape[0] != Z.shape[0]");
      %(fail)s;
    }


  // set up destination zz
  if (%(destructive)s)
    {
      // just set up zz as alias of Z
      if (%(zz)s != %(Z)s)
        {
          if (%(zz)s)
            {
              Py_DECREF(%(zz)s);
            }
          %(zz)s = %(Z)s;
          Py_INCREF(%(zz)s);
        }
    }
  else
    {
      if ((NULL == %(zz)s)
          || (CudaNdarray_HOST_DIMS(%(zz)s)[0] != CudaNdarray_HOST_DIMS(%(Z)s)[0])
          || (CudaNdarray_HOST_DIMS(%(zz)s)[1] != CudaNdarray_HOST_DIMS(%(Z)s)[1])
          || (CudaNdarray_HOST_DIMS(%(zz)s)[2] != CudaNdarray_HOST_DIMS(%(Z)s)[2]))
        {
          // zz is unusable
          //fprintf(stderr, "allocate new output %%i %%i %%i\n",
          //      CudaNdarray_HOST_DIMS(%(Z)s)[0],
          //      CudaNdarray_HOST_DIMS(%(Z)s)[1],
          //      CudaNdarray_HOST_DIMS(%(Z)s)[2]
          //        );
          if (%(zz)s) Py_XDECREF(%(zz)s);
          %(zz)s = (CudaNdarray *) CudaNdarray_Copy(%(Z)s);
          if(!%(zz)s)
            {
              //PyErr_SetString(PyExc_MemoryError,
              //                "failed to alloc GpuGemmBatched output");
              %(fail)s;
            }
        }
      else
        {
          // zz is usable, but has wrong values
          if (CudaNdarray_CopyFromCudaNdarray(%(zz)s, %(Z)s))
            {
              PyErr_SetString(PyExc_MemoryError,
                              "failed to copy in GpuGemmBatched");
              %(fail)s;
            }
        }
    }

  {
    if (0)
    {
    }
    else if ((N * M * 8 < MAX_N_THREADS) && (B > 16 * 8) && (8 * (N * K + M * K) < MAX_SHARED_FLOATS))
    {
        dim3 n_threads(N, M, 8);
        //printf("N threads (8-way) %%i %%i %%i\n", M, N, K);
        int n_shared = sizeof(float) * (M * K + N * K) * 8;
        %(name)s_asdf<<<B/8, n_threads, n_shared>>>(
                K, B,
                CudaNdarray_DEV_DATA(%(alpha)s),
                CudaNdarray_HOST_STRIDES(%(alpha)s)[0],
                CudaNdarray_DEV_DATA(%(X)s),
                CudaNdarray_HOST_STRIDES(%(X)s)[0],
                CudaNdarray_HOST_STRIDES(%(X)s)[1],
                CudaNdarray_HOST_STRIDES(%(X)s)[2],
                CudaNdarray_DEV_DATA(%(Y)s),
                CudaNdarray_HOST_STRIDES(%(Y)s)[0],
                CudaNdarray_HOST_STRIDES(%(Y)s)[1],
                CudaNdarray_HOST_STRIDES(%(Y)s)[2],
                CudaNdarray_DEV_DATA(%(beta)s),
                CudaNdarray_HOST_STRIDES(%(beta)s)[0],
                CudaNdarray_DEV_DATA(%(Z)s),
                CudaNdarray_HOST_STRIDES(%(Z)s)[0],
                CudaNdarray_HOST_STRIDES(%(Z)s)[1],
                CudaNdarray_HOST_STRIDES(%(Z)s)[2]
                );

        CNDA_THREAD_SYNC;

        cudaError_t cudaStat;    
        if (cudaSuccess != (cudaStat=cudaGetLastError()))
        {
            fprintf(stderr, "Calling %(name)s_general with %%i %%i %%i %%i n_shared=%%i\n",
                    B, N, M, K, n_shared);
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s",
                    cudaGetErrorString(cudaStat) );
            %(fail)s;
        }

    }
    else if ((N * M < MAX_N_THREADS) && (N * K + M * K < MAX_SHARED_FLOATS))
    {
        dim3 n_threads(N, M);
        //printf("N threads %%i %%i %%i\n", M, N, K);
        int n_shared = sizeof(float) * (M * K + N * K);
        %(name)s_MNlt512_Xrowmajcontig_Yrowmajcontig_Ksmall<<<B, n_threads, n_shared>>>(
                K,
                CudaNdarray_DEV_DATA(%(alpha)s),
                CudaNdarray_HOST_STRIDES(%(alpha)s)[0],
                CudaNdarray_DEV_DATA(%(X)s),
                CudaNdarray_HOST_STRIDES(%(X)s)[0],
                CudaNdarray_HOST_STRIDES(%(X)s)[1],
                CudaNdarray_HOST_STRIDES(%(X)s)[2],
                CudaNdarray_DEV_DATA(%(Y)s),
                CudaNdarray_HOST_STRIDES(%(Y)s)[0],
                CudaNdarray_HOST_STRIDES(%(Y)s)[1],
                CudaNdarray_HOST_STRIDES(%(Y)s)[2],
                CudaNdarray_DEV_DATA(%(beta)s),
                CudaNdarray_HOST_STRIDES(%(beta)s)[0],
                CudaNdarray_DEV_DATA(%(Z)s),
                CudaNdarray_HOST_STRIDES(%(Z)s)[0],
                CudaNdarray_HOST_STRIDES(%(Z)s)[1],
                CudaNdarray_HOST_STRIDES(%(Z)s)[2]
                );

        CNDA_THREAD_SYNC;

        cudaError_t cudaStat;    
        if (cudaSuccess != (cudaStat=cudaGetLastError()))
        {
            fprintf(stderr, "Calling %(name)s_general with %%i %%i %%i %%i n_shared=%%i\n",
                    B, N, M, K, n_shared);
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s",
                    cudaGetErrorString(cudaStat) );
            %(fail)s;
        }

    }
    else
    {
        cublasOperation_t cT = CUBLAS_OP_T;
        cublasOperation_t cN = CUBLAS_OP_N;
        int unit = 0;

        const int * S = CudaNdarray_HOST_STRIDES(%(X)s);
        const int * D = CudaNdarray_HOST_DIMS(%(X)s);
        unit |= ((S[2] == 1) || (D[2] == 1) ? 0x0 :
                (S[1] == 1) || (D[1] == 1) ?  0x1 : 0x2) << 8;
        int Xlda1 = (D[1] > 1) ? S[1] : (D[2] + 1);
        int Xlda2 = (D[2] > 1) ? S[2] : (D[1] + 1);

        S = CudaNdarray_HOST_STRIDES(%(Y)s);
        D = CudaNdarray_HOST_DIMS(%(Y)s);
        unit |= ((S[2] == 1) || (D[2] == 1)  ? 0x0 :
                (S[1] == 1) || (D[1] == 1) ?  0x1 : 0x2) << 4;
        int Ylda1 = (D[1] > 1) ? S[1] : (D[2] + 1);
        int Ylda2 = (D[2] > 1) ? S[2] : (D[1] + 1);

        S = CudaNdarray_HOST_STRIDES(%(zz)s);
        D = CudaNdarray_HOST_DIMS(%(zz)s);
        unit |= ((S[2] == 1) || (D[2] == 1) ? 0x0 :
                (S[1] == 1) || (D[1] == 1) ?  0x1 : 0x2) << 0;
        int Zlda1 = (D[1] > 1) ? S[1] : (D[2] + 1);
        int Zlda2 = (D[2] > 1) ? S[2] : (D[1] + 1);

        float ** ptrs = (float **) malloc(B * 3 * sizeof(float*));
        for (int bb = 0; bb < B ; ++ bb)
        {
            ptrs[ 0 * B + bb ] = CudaNdarray_DEV_DATA(%(X)s) + bb * CudaNdarray_HOST_STRIDES(%(X)s)[0];
            ptrs[ 1 * B + bb ] = CudaNdarray_DEV_DATA(%(Y)s) + bb * CudaNdarray_HOST_STRIDES(%(Y)s)[0];
            ptrs[ 2 * B + bb ] = CudaNdarray_DEV_DATA(%(zz)s) + bb * CudaNdarray_HOST_STRIDES(%(zz)s)[0];
        }
        float ** dptrs;
        if (cudaSuccess != cudaMalloc(&dptrs, B * 3 * sizeof(float*)))
        {
            fprintf(stderr, "malloc fail");
            assert (0);
        }
        if (cudaSuccess != cudaMemcpy(dptrs, ptrs, B * 3 * sizeof(float*), cudaMemcpyHostToDevice))
        {
            fprintf(stderr, "copy fail");
            assert (0);
        }

        float alpha1 = 1.0;
        float beta1 = 1.0;
        float * alpha = &alpha1;
        float * beta = &beta1;
        const float ** x = (const float **)(dptrs + 0 * B);
        const float ** y = (const float **)(dptrs + 1 * B);
        float ** z = dptrs + 2 * B;
        cudaError_t cudaStat;    
        cublasStatus_t stat;
        cublasHandle_t handle;
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            PyErr_SetString(PyExc_AssertionError, "CUBLAS init failed");
            %(fail)s;
        }

        //fprintf(stderr, "Calling sgemm %%x %%i %%i %%i %%i\n", unit, B, N, M, K);
        switch(unit)
        {
            case 0x000: stat = cublasSgemmBatched(handle, cN, cN, N, M, K, alpha, y, Ylda1, x, Xlda1, beta, z, Zlda1, B); break;
            case 0x100: stat = cublasSgemmBatched(handle, cN, cT, N, M, K, alpha, y, Ylda1, x, Xlda2, beta, z, Zlda1, B); break;
            case 0x010: stat = cublasSgemmBatched(handle, cT, cN, N, M, K, alpha, y, Ylda2, x, Xlda1, beta, z, Zlda1, B); break;
            case 0x110: stat = cublasSgemmBatched(handle, cT, cT, N, M, K, alpha, y, Ylda2, x, Xlda2, beta, z, Zlda1, B); break;
            case 0x001: stat = cublasSgemmBatched(handle, cT, cT, M, N, K, alpha, x, Xlda1, y, Ylda1, beta, z, Zlda2, B); break;
            case 0x101: stat = cublasSgemmBatched(handle, cN, cT, M, N, K, alpha, x, Xlda2, y, Ylda1, beta, z, Zlda2, B); break;
            case 0x011: stat = cublasSgemmBatched(handle, cT, cN, M, N, K, alpha, x, Xlda1, y, Ylda2, beta, z, Zlda2, B); break;
            case 0x111: stat = cublasSgemmBatched(handle, cN, cN, M, N, K, alpha, x, Xlda2, y, Ylda2, beta, z, Zlda2, B); break;
            default: PyErr_SetString(PyExc_AssertionError, "some matrix has no unit stride");
                cublasDestroy(handle);
                cudaFree(dptrs);
                free(ptrs);
                     %(fail)s;
        };
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            cudaGetLastError(); // clear the error flag
            PyErr_Format(PyExc_RuntimeError, "failed GpuGemmBatched call (%%i)",
                    (int) stat);
            cublasDestroy(handle);
            cudaFree(dptrs);
            free(ptrs);
            %(fail)s;
        }
        else
        {
            assert (cudaSuccess == cudaGetLastError());
        }
        cudaFree(dptrs);
        free(ptrs);

        stat = cublasDestroy(handle);
        assert (stat == CUBLAS_STATUS_SUCCESS);
        //fprintf(stderr, "    %%s \n", cudaGetErrorString(bar));
        assert (cudaSuccess == cudaGetLastError());
      }
    }

}
