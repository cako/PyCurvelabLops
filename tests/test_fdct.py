import pytest
import pyct as ct
import numpy as np

from pylops.utils import dottest
from pyctlops import FDCT2D, FDCT3D


pars = [
    {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 0., 'dtype': 'float32'},
    {'nx': 32, 'ny': 32, 'nz': 32, 'imag': 1j, 'dtype': 'complex64'},
    {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 0., 'dtype': 'float32'},
    {'nx': 100, 'ny': 50, 'nz': 20, 'imag': 1j, 'dtype': 'complex64'},
]


@pytest.mark.parametrize("par", pars)
def test_FDCT2D_2dsignal(par):
    """
    Tests for FDCT2D operator for 2d signal.
    """
    x = np.random.normal(0., 1., (par['nx'], par['ny'])) + \
        np.random.normal(0., 1., (par['nx'], par['ny'])) * par['imag']

    FDCTop = FDCT2D(dims=(par['nx'], par['ny']), dtype=par['dtype'])

    assert dottest(FDCTop, *FDCTop.shape,
                   complexflag=0 if par['imag'] == 0 else 3)

    FDCTct = ct.fdct2(
        x.shape, FDCTop.nbscales, FDCTop.nbangles_coarse, FDCTop.allcurvelets,
        cpx=False if par['imag'] == 0 else True)

    y_op = FDCTop(x.ravel())
    y_ct = FDCTct.fwd(x)

    np.testing.assert_array_almost_equal(y_op, y_ct, decimal=8)


@pytest.mark.parametrize("par", pars)
def test_FDCT3D_3dsignal(par):
    """
    Tests for FDCT2D operator for 2d signal.
    """
    x = np.random.normal(0., 1., (par['nx'], par['ny'], par['nz'])) + \
        np.random.normal(0., 1., (par['nx'], par['ny'], par['nz'])) * \
        par['imag']

    FDCTop = FDCT3D(dims=(par['nx'], par['ny'],
                          par['nz']), dtype=par['dtype'])

    assert dottest(FDCTop, *FDCTop.shape,
                   complexflag=0 if par['imag'] == 0 else 3)

    FDCTct = ct.fdct3(
        x.shape, FDCTop.nbscales, FDCTop.nbangles_coarse, FDCTop.allcurvelets,
        cpx=False if par['imag'] == 0 else True)

    y_op = FDCTop(x.ravel())
    y_ct = FDCTct.fwd(x)

    np.testing.assert_array_almost_equal(y_op, y_ct, decimal=8)
