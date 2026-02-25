"""I/O interface for PyIMCOM."""

import sys; sys.path.append("..")

import numpy as np

from pyimcom.config import Config
from pyimcom.coadd import InImage, Block
from .psfutil import PSFModel
from .io_general import InSlice, OutSlice


class FConfig(Config):

    def configure_fast_imcom(self) -> None:
        self()  # Calculate or update derived quantities.

        assert self.inpsf_format == "L2_2506", \
            'INPSF: Fast IMCOM only supports format "L2_2506".'
        PSFModel.NPIX = 128
        PSFModel.SAMP = 6
        PSFModel.NTOT = PSFModel.NPIX * PSFModel.SAMP
        PSFModel.YXCTR = (PSFModel.NTOT-1) / 2
        InSlice.NLAYER = self.n_inframe

        assert self.pad_sides in ["all", "none"], \
            'PADSIDES: Fast IMCOM only supports "all" or "none".'
        OutSlice.NSUB, OutSlice.NPIX_SUB, OutSlice.CDELT =\
            self.n1P//2, self.n2*2, self.dtheta
        OutSlice.NPIX_TOT = OutSlice.NSUB * OutSlice.NPIX_SUB

        assert self.n_out == 1, "NOUT: Fast IMCOM only supports 1."
        assert self.outpsf == "GAUSSIAN", \
            'OUTPSF: Fast IMCOM only supports "GAUSSIAN".'
        OutSlice.SIGMA = self.sigmatarget
        OutSlice.SAVE_ALL = False


class PyPSFModel(PSFModel):

    def __call__(self, x: float = -np.inf, y: float = -np.inf) -> np.ndarray:
        lpoly = InImage.LPolyArr(1, (x-2043.5)/2044.0, (y-2043.5)/2044.0)
        # pixels are in C/Python convention since pixloc was set this way
        return np.einsum('a,aij->ij', lpoly, self.psfdata)
        # Not calling InImage.smooth_and_pad because of PSFModel.pixelate_psf.
