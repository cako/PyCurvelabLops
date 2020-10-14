# PyCurvelabLops

Thin [PyLops](https://pylops.readthedocs.io/) wrapper for [PyCurvelab](https://github.com/slimgroup/PyCurvelab)

## Installation

Installing PyCurvelabLops requires the following components:

- [FFTW](http://www.fftw.org/download.html) 2.1.5
- [CurveLab](http://curvelet.org/software.html) >= 2.0.2
- [SWIG](http://www.swig.org/) >= 1.3
- [PyCurvelab](https://github.com/slimgroup/PyCurvelab)

After these are installed, you may install PyCurvelabLops directly from GitHub:

```bash
pip install git+https://github.com/cako/PyCurvelabLops
```

or you may download and install from the unpackaged repo:

```bash
pip install ./PyCurvelabLops/
```

Add the flag `-e` to install in developer mode.

## Get Started

Start with:

```python
import pyctlops
```

An excellent place to see how to use the library is the `examples/` folder. `Demo_Single_Curvelet` for example contains a `pyctlops` version of the CurveLab Matlab demo.
![Demo](https://github.com/cako/PyCurvelabLops/raw/main/docs/source/static/demo.png)

## Tips and Tricks for Dependencies

### FFTW

For FFTW 2.1.5, you must compile with position-independent code support. Do that with

```bash
./configure --with-pic --prefix=/home/user/opt/fftw-2.1.5 --with-gcc=/usr/bin/gcc
```

The `--prefix` and `--with-gcc` are optional and determine where it will install FFTW and where to find the GCC compiler, respectively.

### CurveLab

In the file `makefile.opt` set `FFTW_DIR`, `CC` and `CXX` variables as required in the instructions. To keep things consistent, set `FFTW_DIR=/home/user/opt/fftw-2.1.5` (or whatever directory was used in the `--prefix` option). For the others, use the same compiler which was used to compile FFTW.

### PyCurvelab

You must ensure that `FFTW` and `FDCT` enviroment variables are set. In Bash, this is done with

```bash
export FFTW=/home/user/opt/fftw-2.1.5
export FDCT=/home/cdacosta/opt/CurveLab-2.1.3
```

Then enter the PyCurvelab directory and run

```bash
python3 setup.py build install --record files.txt
```

This installation has been tested with Python 3.6 and 3.7.
To ensure that it has been installed, open you python interpreter and try to `import pyct`. If you get the following error

```bash
ModuleNotFoundError: No module named '_fdct2_wrapper'
```

try

```bash
export PYTHONPATH=/path/to/python3
```

In my experience this is only required in virtual environments and only for the first import but YMMV.
