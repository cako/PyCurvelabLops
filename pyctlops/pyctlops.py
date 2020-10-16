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
        # Check dimension
        if len(dirs) == 2:
            ctfdct = ct.fdct2
        elif len(dirs) == 3:
            ctfdct = ct.fdct3
        else:
            raise NotImplementedError("FDCT is only implemented in 2D or 3D")

        ndim = len(dims)

        # Ensure directions are between 0, ndim-1
        dirs = [np.core.multiarray.normalize_axis_index(d, ndim) for d in dirs]

        # If input is shaped (100, 200, 300) and dirs = (0, 2)
        # then input_shape will be (100, 300)
        self._input_shape_2d = list(dims[d] for d in dirs)
        if nbscales is None:
            nbscales = int(np.ceil(np.log2(min(self._input_shape_2d)) - 3))

        # Complex operator is required to handle complex input
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            cpx = True
        else:
            cpx = False

        # We have enough info to create the operator
        self.FDCT = ctfdct(list(self._input_shape_2d),
                           nbscales, nbangles_coarse, allcurvelets,
                           norm=False, cpx=cpx)

        # Now we need to build the iterator which will only iterate along
        # the required directions. Following the example above,
        # iterable_axes = [ False, True, False ]
        iterable_axes = [False if i in dirs else True for i in range(ndim)]
        ndim_iterable = np.prod(np.array(dims)[iterable_axes])

        # Build the iterator itself. In our example, the slices
        # would be [:, i, :] for i in range(200)
        # We use slice(None) is the colon operator
        self._iterator = list(product(
            *(range(dims[ax]) if doiter else [slice(None)]
                for ax, doiter in enumerate(iterable_axes))))

        # For a single 2d/3d input, the length of the vector will be given by
        # the shapes in FDCT.sizes
        self._output_len = sum(np.prod(j) for i in self.FDCT.sizes for j in i)

        # Save some useful properties
        self.dims = dims
        self.dirs = dirs
        self.nbscales = nbscales
        self.nbangles_coarse = nbangles_coarse
        self.allcurvelets = allcurvelets
        self.cpx = cpx

        # Required by PyLops
        self.shape = (ndim_iterable * self._output_len, np.prod(dims))
        self.dtype = dtype
        self.explicit = False

    def _matvec(self, x):
        n = self._output_len
        fwd_out = np.zeros(self.shape[0], dtype=self.dtype)
        for i, index in enumerate(self._iterator):
            fwd_out[i*n:(i+1)*n] = self.FDCT.fwd(x.reshape(self.dims)[index])
        return fwd_out

    def _rmatvec(self, x):
        n = self._output_len
        inv_out = np.zeros(self.dims, dtype=self.dtype)
        for i, index in enumerate(self._iterator):
            inv_out[index] = self.FDCT.inv(
                x[i*n:(i+1)*n]).reshape(self._input_shape_2d)

        return inv_out.ravel()

    def inverse(self, x):
        return self._rmatvec(x)

    def struct(self, x):
        return self.FDCT.struct(x)

    def vect(self, x):
        return self.FDCT.vect(x)


class FDCT2D(FDCT):
    __doc__ = _fdct_docs(2)

    def __init__(self, dims, dirs=(-2, -1),
                 nbscales=None, nbangles_coarse=16, allcurvelets=True,
                 dtype='complex128'):
        if len(dirs) != 2:
            raise ValueError(
                "FDCT2D must be called with exactly two directions")
        super().__init__(dims, dirs, nbscales, nbangles_coarse, allcurvelets, dtype)


class FDCT3D(FDCT):
    __doc__ = _fdct_docs(3)

    def __init__(self, dims, dirs=(-3, -2, -1),
                 nbscales=None, nbangles_coarse=16, allcurvelets=True,
                 dtype='complex128'):
        if len(dirs) != 3:
            raise ValueError(
                "FDCT3D must be called with exactly three directions")
        super().__init__(dims, dirs, nbscales, nbangles_coarse, allcurvelets, dtype)
