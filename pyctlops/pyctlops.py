"""
Provides a Linear Operator for the Curvelet transform to interface with PyLops.
"""

import pyct as ct
import numpy as np
from pylops import LinearOperator
from itertools import product


def _fdct_docs(dimension):
    if dimension == 2:
        doc = "2D"
    elif dimension == 3:
        doc = "3D"
    else:
        doc = "2D/3D"
    return f"""{doc} dimensional Curvelet operator.
        Apply {doc} Curvelet Transform along two directions ``dirs`` of a
        multi-dimensional array of size ``dims``.
        The Curvelet operator is an overload of the PyCurvelab (``pyct``)
        package, which is in turn based on the CurveLab.
        Please see https://github.com/slimgroup/PyCurvelab and http://curvelet.org/
        for more information.

        Parameters
        ----------
        dims : :obj:`tuple`
            Number of samples for each dimension
        dirs : :obj:`tuple`, optional
            Direction along which FDCT is applied
        nbscales : :obj:`int`, optional
            Number of scales (including the coarsest level);
            Defaults to ceil(log2(min(input_dims)) - 3).
        nbangles_coarse : :obj:`int`, optional
            Number of angles at 2nd coarsest scale
        allcurvelets : :obj:`bool`, optional
            Use curvelets at all scales, including coarsest scale.
            If ``False``, a wavelet transform will be used for the
            coarsest scale.
        dtype : :obj:`str`, optional
            Type of the transform

        Attributes
        ----------
        shape : :obj:`tuple`
            Operator shape
        explicit : :obj:`bool`
            Operator contains a matrix that can be solved explicitly
            (True) or not (False)
        """


class FDCT(LinearOperator):
    __doc__ = _fdct_docs(0)

    def __init__(self, dims, dirs, nbscales=None, nbangles_coarse=16,
                 allcurvelets=True, dtype='complex128'):
        ndim = len(dims)
        dirs = [np.core.multiarray.normalize_axis_index(d, ndim) for d in dirs]

        input_shape = list(dims[d] for d in dirs)
        if nbscales is None:
            nbscales = int(np.ceil(np.log2(min(input_shape)) - 3))

        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            cpx = True
        else:
            cpx = False

        if len(dirs) == 2:
            ctfdct = ct.fdct2
        elif len(dirs) == 3:
            ctfdct = ct.fdct3
        else:
            raise NotImplementedError("FDCT is only implemented in 2D or 3D")

        self.FDCT = ctfdct(list(input_shape),
                           nbscales, nbangles_coarse, allcurvelets,
                           norm=False, cpx=cpx)
        self._iterable_axes = [
            False if i in dirs else True for i in range(ndim)]
        self.dims = dims
        self.dirs = dirs
        self.input_shape = input_shape
        self.out_len = sum(np.prod(j) for i in self.FDCT.sizes for j in i)
        self.iterable_dims = np.prod(np.array(self.dims)[self._iterable_axes])

        self.nbscales = nbscales
        self.nbangles_coarse = nbangles_coarse
        self.allcurvelets = allcurvelets

        self.shape = [self.iterable_dims * self.out_len, np.prod(dims)]
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        iterator = product(
            *(range(self.dims[ax]) if doiter else[slice(None)] for ax,
              doiter in enumerate(self._iterable_axes)))
        fwd_out = np.zeros(self.shape[0], dtype=self.dtype)

        iterator = product(*(range(self.dims[ax]) if doiter else[slice(None)]
                             for ax, doiter in enumerate(self._iterable_axes)))
        x_reshape = x.reshape(self.dims)
        for i, index in enumerate(iterator):
            fwd_out[i:i+self.out_len] = self.FDCT.fwd(x_reshape[index]).ravel()
        return fwd_out

    def _rmatvec(self, x):
        iterator = product(*(range(self.dims[ax]) if doiter else[slice(None)]
                             for ax, doiter in enumerate(self._iterable_axes)))
        inv_out = np.zeros(self.dims, dtype=np.dtype)
        for i, index in enumerate(iterator):
            inv_out[index] = self.FDCT.inv(
                x[i:i+self.out_len]).reshape(self.input_shape)

        return inv_out.ravel()

    def struct(self, v):
        return self.FDCT.struct(v)

    def vect(self, v):
        return self.FDCT.vect(v)


class FDCT2D(FDCT):
    __doc__ = _fdct_docs(2)

    def __init__(
            self, dims, dirs=(0, 1),
            nbscales=None, nbangles_coarse=16, allcurvelets=True,
            dtype='complex128'):
        super().__init__(dims, dirs, nbscales, nbangles_coarse, allcurvelets, dtype)


class FDCT3D(FDCT):
    __doc__ = _fdct_docs(3)

    def __init__(
            self, dims, dirs=(0, 1, 2),
            nbscales=None, nbangles_coarse=16, allcurvelets=True,
            dtype='complex128'):
        super().__init__(dims, dirs, nbscales, nbangles_coarse, allcurvelets, dtype)
