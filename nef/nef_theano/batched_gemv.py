import StringIO
import numpy as np
from theano import Op, Apply
from theano import tensor
from theano.tensor.blas_c import BaseBLAS



class DoubleBatchedGemv(BaseBLAS, Op):
    _use_c_code = False
    # Currently need N to identify which version of C code to use
    # In order to use one version of C code, it's necessary to remove the
    # varags calling convention of make_node.
    def c_code(self, node, name, inp, out, sub):
        if not self._use_c_code:
            return Op.c_code(self, node, name, inp, out, sub)
        code = dbgemv_c_code(inp, out, sub['fail'])
        return code

    def c_code_cache_version(self):
        if not self._use_c_code:
            return Op.c_code_cache_version(self)
        return ()
        return (10, blas_header_version())


def dbgemv_c_code(inp, out, fail):
    aa = inp[0]
    zz, = out
    ii = 1
    tups = []
    while ii + 4 <= len(inp):
        tups.append(inp[ii:ii + 4])
        ii += 4
    sio = StringIO.StringIO()
    # XXX only works for float32, no error checking!
    print >> sio, """

    int elemsize = 4;
    float fbeta1 = 1.0;
    float fbeta0 = 0.0;
    float falpha1 = 1.0;

    if (PyArray_NDIM(%(aa)s) != 1)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(aa) != 1");
        %(fail)s;
    }
    """ % locals()

    for ust, vst, U, V in tups:
        for xx in U, V:
            print >> sio, """
            if (PyArray_NDIM(%(xx)s) != 2)
            {
                PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(xx) != 2");
                %(fail)s;
            }

            """ % locals()

    # TODO: check dtypes
    # TODO: check sizes

    # allocate output if necessary
    print >> sio, """
    {
        if ((NULL == %(zz)s)
            || (PyArray_DIMS(%(zz)s)[0] != PyArray_DIMS(%(aa)s)[0]))
        {
            if (%(zz)s) Py_XDECREF(%(zz)s);
            %(zz)s = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(%(aa)s), type_num_%(aa)s);
            if(!%(zz)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemv output");
                %(fail)s
            }
        }
    }
    """ % locals()

    # loop setup
    print >> sio, """
    {
        int NA0 = PyArray_DIMS(%(aa)s)[0];
        int NZ0 = PyArray_DIMS(%(zz)s)[0];
        assert (NA0 == NZ0);
        float *Adata = (float*)(PyArray_DATA(%(aa)s));
        float *Zdata = (float*)(PyArray_DATA(%(zz)s));

        // -- TODO handle non-contiguous case
        assert (PyArray_STRIDES(%(aa)s)[0] == elemsize);
        assert (PyArray_STRIDES(%(zz)s)[0] == elemsize);

        int S1 = 1;
        // -- initialize Z <- 0, in case out the range
        // of outputs doesn't cover all of Z
        for (int ii = 0; ii < NZ0; ++ii) Zdata[ii] = 0;

        char TRANS = 'T';
        char NOTRANS = 'N';

        float fbuffer[100];
    """ % locals()

    for ust, vst, U, V in tups:
        print >> sio, """
        {
            assert (PyArray_DESCR(%(U)s)->type_num == NPY_FLOAT);
            assert (PyArray_DESCR(%(V)s)->type_num == NPY_FLOAT);

            int NU0 = PyArray_DIMS(%(U)s)[0];
            int NU1 = PyArray_DIMS(%(U)s)[1];
            int NV0 = PyArray_DIMS(%(V)s)[0];
            int NV1 = PyArray_DIMS(%(V)s)[1];
            assert (NU1 == NV1);
            assert (NU1 <= 100); // -- or else raise fbuffer size

            /* This formula is needed in the case where U is actually a row or
             * column matrix, because BLAS sometimes insists that the strides:
             *  - are not smaller than the number of elements in the array
             *  - are not 0.
             */
            int SU0 = (NU0 > 1) ? (PyArray_STRIDES(%(U)s)[0] / elemsize) : (NU1 + 1);
            int SU1 = (NU1 > 1) ? (PyArray_STRIDES(%(U)s)[1] / elemsize) : (NU0 + 1);
            int SV0 = (NV0 > 1) ? (PyArray_STRIDES(%(V)s)[0] / elemsize) : (NV1 + 1);
            int SV1 = (NV1 > 1) ? (PyArray_STRIDES(%(V)s)[1] / elemsize) : (NV0 + 1);

            float *Udata = (float*)(PyArray_DATA(%(U)s));
            float *Vdata = (float*)(PyArray_DATA(%(V)s));

            int ust = ((dtype_%(ust)s*)PyArray_DATA(%(ust)s))[0];
            int vst = ((dtype_%(vst)s*)PyArray_DATA(%(vst)s))[0];

            // -- if first stride is elemsize, then NOTRANS
            if (0) {
                fprintf(stderr, "ust=%%i vst=%%i\\n", ust, vst);
                fprintf(stderr, "NV0=%%i NV1=%%i\\n", NV0, NV1);
                fprintf(stderr, "U strides=%%i %%i\\n",
                    PyArray_STRIDES(%(U)s)[0],
                    PyArray_STRIDES(%(U)s)[1]);
                fprintf(stderr, "SU strides=%%i %%i\\n", SU0, SU1);
                fprintf(stderr, "V strides=%%i %%i\\n",
                    PyArray_STRIDES(%(V)s)[0],
                    PyArray_STRIDES(%(V)s)[1]);
                    }

            assert (PyArray_STRIDES(%(U)s)[0] == elemsize);
            assert (PyArray_STRIDES(%(V)s)[0] == elemsize);
            assert (NA0 >= ust + NU0);
            assert (NZ0 >= vst + NV0);

            sgemv_(&TRANS, &NU0, &NU1,
                &falpha1,
                Udata, &SU1,
                Adata + ust, &S1,
                &fbeta0,
                fbuffer, &S1);
            sgemv_(&NOTRANS, &NV0, &NV1,
                &falpha1,
                Vdata, &SV1,
                fbuffer, &S1,
                &fbeta0,
                Zdata + vst, &S1);

        }
        """ % locals()

    # -- end the loop over tups
    print >> sio, """
    }
    """

    rval = sio.getvalue()
    #print [c for c in rval if c == "{"]
    #print [c for c in rval if c == "}"]
    return rval
