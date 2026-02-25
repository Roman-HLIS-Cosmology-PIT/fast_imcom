"""I/O interface for PyIMCOM."""

import sys; sys.path.append("..")

from .psfutil import PSFModel, SubSlice
from .io_general import InSlice, OutSlice
from pyimcom.config import Config
from pyimcom.coadd import Block


class FConfig(Config):

    def configure_fast_imcom(self) -> None:
        if self.inpsf_format == "L2_2506":
            PSFModel.NPIX = 128
            PSFModel.SAMP = 6
            PSFModel.NTOT = PSFModel.NPIX * PSFModel.SAMP
            PSFModel.YXCTR = (PSFModel.NTOT-1) / 2
        InSlice.NLAYER = self.n_inframe

        assert self.pad_sides in ["all", "none"],\
            'PADSIDES: Fast IMCOM only supports "all" or "none".'
        OutSlice.NSUB, OutSlice.NPIX_SUB, OutSlice.CDELT =\
            self.n1P//2, self.n2*2, self.dtheta
        OutSlice.NPIX_TOT = OutSlice.NSUB * OutSlice.NPIX_SUB

        assert self.n_out == 1, "NOUT: Fast IMCOM only supports 1."
        assert self.outpsf == "GAUSSIAN",\
            'OUTPSF: Fast IMCOM only supports "GAUSSIAN".'
        OutSlice.SIGMA = self.sigmatarget
        OutSlice.SAVE_ALL = False
