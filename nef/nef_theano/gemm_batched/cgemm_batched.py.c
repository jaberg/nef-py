{
  int elemsize ;
  float fbeta;
  double dbeta;
  int B = 0;
  int M = 0;
  int N = 0;
  int K = 0;

  if (PyArray_NDIM(%(alpha)s) != 1)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(alpha) != 1");
      %(fail)s;
    }
  if (PyArray_NDIM(%(X)s) != 3)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(X) != 2");
      %(fail)s;
    }
  if (PyArray_NDIM(%(Y)s) != 3)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(Y) != 3");
      %(fail)s;
    }
  if (PyArray_NDIM(%(beta)s) != 1)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(beta) != 0");
      %(fail)s;
    }
  if (PyArray_NDIM(%(Z)s) != 3)
    {
      PyErr_SetString(PyExc_NotImplementedError, "GemmBatched: rank(Z) != 3");
      %(fail)s;
    }


  // check for consistent dtype
  if (PyArray_DESCR(%(X)s)->type_num != PyArray_DESCR(%(Y)s)->type_num)
    {
      PyErr_SetString(PyExc_TypeError, "GemmBatched: X vs. Y");
      %(fail)s;
    }
  if (PyArray_DESCR(%(X)s)->type_num != PyArray_DESCR(%(Z)s)->type_num)
    {
      PyErr_SetString(PyExc_TypeError, "GemmBatched: X vs. Z");
      %(fail)s;
    }
  if (PyArray_DESCR(%(X)s)->type_num != PyArray_DESCR(%(alpha)s)->type_num)
    {
      PyErr_SetString(PyExc_TypeError, "GemmBatched: X vs. alpha");
      %(fail)s;
    }
  if (PyArray_DESCR(%(X)s)->type_num != PyArray_DESCR(%(beta)s)->type_num)
    {
      PyErr_SetString(PyExc_TypeError, "GemmBatched: X vs. beta");
      %(fail)s;
    }

  B = PyArray_DIMS(%(Z)s)[0];
  M = PyArray_DIMS(%(Z)s)[1];
  N = PyArray_DIMS(%(Z)s)[2];

  // alpha dim check
  if (PyArray_DIMS(%(alpha)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: alpha.shape[0] != Z.shape[0]");
      %(fail)s;
    }

  // X dim check
  if (PyArray_DIMS(%(X)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: X.shape[0] != Z.shape[0]");
      %(fail)s;
    }
  if (PyArray_DIMS(%(X)s)[1] != M)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: X.shape[1] != Z.shape[1]");
      %(fail)s;
    }
  K = PyArray_DIMS(%(X)s)[2];

  // Y dim check
  if (PyArray_DIMS(%(Y)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: Y.shape[0] != Z.shape[0]");
      %(fail)s;
    }
  if (PyArray_DIMS(%(Y)s)[1] != K)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: Y.shape[1] != X.shape[2]");
      %(fail)s;
    }
  if (PyArray_DIMS(%(Y)s)[2] != N)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: Y.shape[2] != Z.shape[2]");
      %(fail)s;
    }

  // beta dim check
  if (PyArray_DIMS(%(beta)s)[0] != B)
    {
      PyErr_SetString(PyExc_ValueError,
                      "Shape mismatch: beta.shape[0] != Z.shape[0]");
      %(fail)s;
    }

  // set elemsize
  if (PyArray_DESCR(%(Z)s)->type_num == NPY_DOUBLE)
  {
    elemsize = 8;
  }
  else if (PyArray_DESCR(%(Z)s)->type_num == NPY_FLOAT)
  {
    elemsize = 4;
  }
  else
  {
    PyErr_SetString(PyExc_NotImplementedError, "complex GemmBatched");
    %(fail)s;
  }

  if (!(%(X)s = %(name)s_as_positive_strided12(%(X)s, elemsize))) %(fail)s;
  if (!(%(Y)s = %(name)s_as_positive_strided12(%(Y)s, elemsize))) %(fail)s;
  if (!(%(Z)s = %(name)s_as_positive_strided12(%(Z)s, elemsize))) %(fail)s;

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
          || (PyArray_DIMS(%(zz)s)[0] != PyArray_DIMS(%(Z)s)[0])
          || (PyArray_DIMS(%(zz)s)[1] != PyArray_DIMS(%(Z)s)[1])
          || (PyArray_DIMS(%(zz)s)[2] != PyArray_DIMS(%(Z)s)[2]))
        {
          // zz is unusable
          if (%(zz)s) Py_XDECREF(%(zz)s);
          %(zz)s = (PyArrayObject *) PyArray_Copy(%(Z)s);
          if(!%(zz)s)
            {
              PyErr_SetString(PyExc_MemoryError,
                              "failed to alloc GemmBatched output");
              %(fail)s;
            }
        }
      else
        {
          // zz is usable, but has wrong values
          if (PyArray_CopyInto(%(zz)s, %(Z)s))
            {
              PyErr_SetString(PyExc_MemoryError,
                              "failed to copy in GemmBatched");
              %(fail)s;
            }
        }
    }

  {
    char cT = 'T';
    char cN = 'N';
    int unit = 0;
    unit |= encode_strides_for_gemm(%(X)s, elemsize, 1, 2) << 8;
    unit |= encode_strides_for_gemm(%(Y)s, elemsize, 1, 2) << 4;
    unit |= encode_strides_for_gemm(%(zz)s, elemsize, 1, 2) << 0;

    int Xlda0, Xlda1;
    int Ylda0, Ylda1;
    int Zlda0, Zlda1;
    extract_lda(%(X)s, elemsize, 1, 2, &Xlda0, &Xlda1);
    extract_lda(%(Y)s, elemsize, 1, 2, &Ylda0, &Ylda1);
    extract_lda(%(zz)s, elemsize, 1, 2, &Zlda0, &Zlda1);

    switch (PyArray_DESCR(%(zz)s)->type_num)
    {
    case NPY_FLOAT:
        {
          if ( M * N * K < 256 ) // TODO: autotune this
            {
              int sa0 = PyArray_STRIDES(%(alpha)s)[0] / elemsize;
              int sx0 = PyArray_STRIDES(%(X)s)[0] / elemsize;
              int sx1 = PyArray_STRIDES(%(X)s)[1] / elemsize;
              int sx2 = PyArray_STRIDES(%(X)s)[2] / elemsize;
              int sy0 = PyArray_STRIDES(%(Y)s)[0] / elemsize;
              int sy1 = PyArray_STRIDES(%(Y)s)[1] / elemsize;
              int sy2 = PyArray_STRIDES(%(Y)s)[2] / elemsize;
              int sb0 = PyArray_STRIDES(%(beta)s)[0] / elemsize;
              int sz0 = PyArray_STRIDES(%(zz)s)[0] / elemsize;
              int sz1 = PyArray_STRIDES(%(zz)s)[1] / elemsize;
              int sz2 = PyArray_STRIDES(%(zz)s)[2] / elemsize;
              const float * __restrict__ alpha_ = (float*)(PyArray_DATA(%(alpha)s));
              const float * __restrict__ X_ = (float*)(PyArray_DATA(%(X)s));
              const float * __restrict__ Y_ = (float*)(PyArray_DATA(%(Y)s));
              const float * __restrict__ beta_ = (float*)(PyArray_DATA(%(beta)s));
              float * __restrict__ Z_ = (float*)(PyArray_DATA(%(zz)s));
#pragma omp parallel for schedule(static, 16)
              for (int bb = 0; bb < B; ++bb)
                {
                  for (int mm = 0; mm < M; ++mm)
                    {
                      if (K > 1)
                        {
                          for (int nn = 0; nn < N; ++nn)
                            {
                              float ksum = 0.0;
                              for (int kk = 0; kk < K; ++kk)
                                {
                                  float xi = X_[sx0 * bb + sx1 * mm + sx2 * kk];
                                  float yi = Y_[sy0 * bb + sy1 * kk + sy2 * nn];
                                  ksum += xi * yi;
                                }
                              Z_[sz0 * bb + sz1 * mm + sz2 * nn] =
                                Z_[sz0 * bb + sz1 * mm + sz2 * nn] * beta_[sb0 * bb]
                                + ksum * alpha_[sa0 * bb];
                            }
                        }
                      else
                        {
                          for (int nn = 0; nn < N; ++nn)
                            {
                              float ksum =
                                  X_[sx0 * bb + sx1 * mm] *
                                  Y_[sy0 * bb + sy2 * nn];
                              Z_[sz0 * bb + sz1 * mm + sz2 * nn] =
                                Z_[sz0 * bb + sz1 * mm + sz2 * nn] * beta_[sb0 * bb] 
                                + ksum * alpha_[sa0 * bb];
                            }
                        }
                    }
                }
            }
          else
            {
              int failbit = 0;
#pragma omp parallel for schedule(static, 16)
              for (int bb = 0; bb < B; ++bb)
                {

                  float* alpha = (float*)((char*)PyArray_DATA(%(alpha)s) + PyArray_STRIDES(%(alpha)s)[0] * bb);
                  float* beta = (float*)((char*)PyArray_DATA(%(beta)s) + PyArray_STRIDES(%(beta)s)[0] * bb);
                  float* x = (float*)((char*)PyArray_DATA(%(X)s) + PyArray_STRIDES(%(X)s)[0] * bb);
                  float* y = (float*)((char*)PyArray_DATA(%(Y)s) + PyArray_STRIDES(%(Y)s)[0] * bb);
                  float* z = (float*)((char*)PyArray_DATA(%(zz)s) + PyArray_STRIDES(%(zz)s)[0] * bb);
                  //fprintf(stderr, "Calling sgemm %%i %%i %%i %%i\\n", unit, N, M, Nx1);
                  switch(unit)
                  {
                    case 0x000: sgemm_(&cN, &cN, &N, &M, &K, alpha, y, &Ylda0, x, &Xlda0, beta, z, &Zlda0); break;
                    case 0x100: sgemm_(&cN, &cT, &N, &M, &K, alpha, y, &Ylda0, x, &Xlda1, beta, z, &Zlda0); break;
                    case 0x010: sgemm_(&cT, &cN, &N, &M, &K, alpha, y, &Ylda1, x, &Xlda0, beta, z, &Zlda0); break;
                    case 0x110: sgemm_(&cT, &cT, &N, &M, &K, alpha, y, &Ylda1, x, &Xlda1, beta, z, &Zlda0); break;
                    case 0x001: sgemm_(&cT, &cT, &M, &N, &K, alpha, x, &Xlda0, y, &Ylda0, beta, z, &Zlda1); break;
                    case 0x101: sgemm_(&cN, &cT, &M, &N, &K, alpha, x, &Xlda1, y, &Ylda0, beta, z, &Zlda1); break;
                    case 0x011: sgemm_(&cT, &cN, &M, &N, &K, alpha, x, &Xlda0, y, &Ylda1, beta, z, &Zlda1); break;
                    case 0x111: sgemm_(&cN, &cN, &M, &N, &K, alpha, x, &Xlda1, y, &Ylda1, beta, z, &Zlda1); break;
                    default: PyErr_SetString(PyExc_AssertionError, "some matrix has no unit stride");
                             failbit = 1;
                  };
                }
              if (failbit) %(fail)s;
            }
        }
      break;
    case NPY_DOUBLE:
        {
          PyErr_SetString(PyExc_NotImplementedError,
                          "GemmBatched float64");
        }
      break;
    }

  }


}