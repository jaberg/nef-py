
static
PyArrayObject * %(name)s_as_positive_strided12(PyArrayObject *X, int elemsize)
{
  npy_intp* S = PyArray_STRIDES(X);
  if (   (S[1] < 1) || (S[2] < 1)
      || (S[1] %% elemsize) || (S[2] %% elemsize)
      || ((S[1] != elemsize) && (S[2] != elemsize)))
    {
      PyArrayObject * _x_copy = (PyArrayObject *) PyArray_Copy(X);
      if (!_x_copy)
        {
          PyErr_SetString(PyExc_MemoryError,
                          "Failed to copy ndarray");
          return NULL;
        }
      Py_XDECREF(X);
      return _x_copy;
    }
  else
    {
      return X;
    }
}

static
int encode_strides_for_gemm(PyArrayObject * X, int elemsize, int d0, int d1)
{
  npy_intp * S = PyArray_STRIDES(X);
  return ((S[d1] == elemsize) ? 0x0 : (S[d0] == elemsize) ?  0x1 : 0x2);
}

static
void extract_lda(PyArrayObject * X, int elemsize, int d0, int d1,
                 int * lda0, int * lda1)
{
  npy_intp * Nx = PyArray_DIMS(X);
  npy_intp * Sx = PyArray_STRIDES(X);
  *lda0 = (Nx[d0] > 1) ? Sx[d0] / elemsize : (Nx[d1] + 1);
  *lda1 = (Nx[d1] > 1) ? Sx[d1] / elemsize : (Nx[d0] + 1);
}
