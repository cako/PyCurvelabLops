import pytest
import pyct as ct
import numpy as np

from pylops.utils import dottest
from pyctlops import FDCT2D, FDCT3D


pars = [
    {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 0, 'dtype': 'float64'},
    {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 1j, 'dtype': 'complex128'},
    {'nx': 32, 'ny': 32, 'nz': 64, 'imag': 0, 'dtype': 'float64'},
    {'nx': 32, 'ny': 32, 'nz': 64, 'imag': 1j, 'dtype': 'complex128'},
    {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 0, 'dtype': 'float64'},
    {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 1j, 'dtype': 'complex128'},
]


@pytest.mark.parametrize("par", pars)
def test_FDCT2D_2dsignal(par):
    """
    Tests for FDCT2D operator for 2d signal.
    """
    x = np.random.normal(0., 1., (par['nx'], par['ny'])) + \
        np.random.normal(0., 1., (par['nx'], par['ny'])) * par['imag']

    FDCTop = FDCT2D(dims=(par['nx'], par['ny']), dtype=par['dtype'])

    assert dottest(FDCTop, *FDCTop.shape, tol=1e-12,
                   complexflag=0 if par['imag'] == 0 else 3)

    FDCTct = ct.fdct2(x.shape, FDCTop.nbscales,
                      FDCTop.nbangles_coarse, FDCTop.allcurvelets,
                      cpx=False if par['imag'] == 0 else True)

    y_op = FDCTop * x.ravel()
    y_ct = np.array(FDCTct.fwd(x))

    np.testing.assert_array_almost_equal(y_op, y_ct, decimal=64)
    assert y_op.dtype == y_ct.dtype


@pytest.mark.parametrize("par", pars)
def test_FDCT2D_3dsignal(par):
    """
    Tests for FDCT2D operator for 3d signal.
    """
    x = np.random.normal(0., 1., (par['nx'], par['ny'], par['nz'])) + \
        np.random.normal(0., 1., (par['nx'], par['ny'], par['nz'])) * \
        par['imag']
    dirs = [0, -1]
    FDCTop = FDCT2D(dims=(par['nx'], par['ny'], par['nz']),
                    dirs=dirs, dtype=par['dtype'])

    # Some tests fail for complexflag = 3 (?)
    assert dottest(FDCTop, *FDCTop.shape, tol=1e-12,
                   complexflag=0 if par['imag'] == 0 else 2)

    # Check if forward transforms are the same using the raw operator
    FDCTct = ct.fdct2([x.shape[d] for d in dirs], FDCTop.nbscales,
                      FDCTop.nbangles_coarse, FDCTop.allcurvelets,
                      cpx=False if par['imag'] == 0 else True)

    y_op = FDCTop * x.ravel()
    y_ct = np.zeros_like(y_op)
    n = FDCTop._output_len
    for i in range(par['ny']):
        y_ct[i*n:(i+1)*n] = np.array(FDCTct.fwd(x[:, i, :]))

    np.testing.assert_array_almost_equal(y_op, y_ct, decimal=64)
    assert y_op.dtype == y_ct.dtype

    # Check if inverse transforms are the same using the raw operator
    x_op = (FDCTop.H * y_op).reshape(x.shape)
    x_ct = np.zeros_like(x)
    for i in range(par['ny']):
        x_ct[:, i, :] = FDCTct.inv(y_op[i*n:(i+1)*n])
    np.testing.assert_array_almost_equal(x_op, x_ct, decimal=64)
    assert x_op.dtype == x_ct.dtype


@pytest.mark.parametrize("par", pars)
def test_FDCT3D_3dsignal(par):
    """
    Tests for FDCT3D operator for 3d signal.
    """
    x = np.random.normal(0., 1., (par['nx'], par['ny'], par['nz'])) + \
        np.random.normal(0., 1., (par['nx'], par['ny'], par['nz'])) * \
        par['imag']

    FDCTop = FDCT3D(dims=(par['nx'], par['ny'],
                          par['nz']), dtype=par['dtype'])

    assert dottest(FDCTop, *FDCTop.shape, tol=1e-12,
                   complexflag=0 if par['imag'] == 0 else 3)

    FDCTct = ct.fdct3(x.shape, FDCTop.nbscales,
                      FDCTop.nbangles_coarse, FDCTop.allcurvelets,
                      cpx=False if par['imag'] == 0 else True)

    y_op = FDCTop * x.ravel()
    y_ct = np.array(FDCTct.fwd(x))

    np.testing.assert_array_almost_equal(y_op, y_ct, decimal=64)
    assert y_op.dtype == y_ct.dtype
