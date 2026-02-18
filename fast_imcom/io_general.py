"""General I/O interface.

Classes
-------
InSlice : Input image slice, like InImage in PyIMCOM.
OutSlice : Output image slice, like Block in PyIMCOM.

"""


from time import perf_counter

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import units as u

from .psfutil import PSFModel, SubSlice


class InSlice:

    def __init__(self, filename: str, loaddata: bool = True,
                 psfmodel: PSFModel = None) -> None:
        self.filename = filename
        with fits.open(filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
            self.data = f["WFI01"].data.astype(np.float32) if loaddata else None
        self.psfmodel = psfmodel

    def outpix2world2inpix(self, outwcs: wcs.WCS, outxys: np.ndarray) -> np.ndarray:
        return self.wcs.all_world2pix(outwcs.all_pix2world(outxys, 0), 0)

    def inpix2world2outpix(self, outwcs: wcs.WCS, inxys: np.ndarray) -> np.ndarray:
        return outwcs.all_world2pix(self.wcs.all_pix2world(inxys, 0), 0)

    def assess_overlap(self) -> bool:
        ACCEPT = SubSlice.ACCEPT  # Shortcuts.
        NSUB, NPIX_SUB = OutSlice.NSUB, OutSlice.NPIX_SUB
        outxys_sp = np.moveaxis(np.array(np.meshgrid(
            np.arange(NSUB+1)*NPIX_SUB, np.arange(NSUB+1)*NPIX_SUB)), 0, -1).reshape(-1, 2)
        inxys_sp = self.outpix2world2inpix(self.outslice.wcs, outxys_sp)
        # self.inx_min, self.iny_min = np.rint(np.min(inxys_sp, axis=0)).astype(int) - ACCEPT
        # self.inx_max, self.iny_max = np.rint(np.max(inxys_sp, axis=0)).astype(int) + ACCEPT

        subsize = NPIX_SUB * self.outslice.wcs.wcs.cdelt[1] /\
            (0.11 * u.arcsec.to("degree"))  # Subslice size in input pixels.
        mask_sp = np.all((inxys_sp >= -0.5-subsize) &
                         (inxys_sp <= 4087.5+subsize), axis=1).reshape(NSUB+1, NSUB+1)
        self.mask_sp = mask_sp
        self.mask_out = np.zeros((OutSlice.NPIX_TOT, OutSlice.NPIX_TOT), dtype=bool)

        for X in range(NSUB):
            for Y in range(NSUB):
                if not np.any(mask_sp[Y:min(Y+2, NSUB+1), X:min(X+2, NSUB+1)]): continue

                outxys = np.moveaxis(np.array(np.meshgrid(
                    np.arange(NPIX_SUB) + X*NPIX_SUB,
                    np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)
                inxys = self.outpix2world2inpix(self.outslice.wcs, outxys)

                self.mask_out[Y*NPIX_SUB:(Y+1)*NPIX_SUB,
                              X*NPIX_SUB:(X+1)*NPIX_SUB] =\
                    np.all((inxys >= -0.5+ACCEPT) & (inxys <= 4087.5-ACCEPT), axis=1).\
                    reshape(NPIX_SUB, NPIX_SUB)


class OutSlice:

    NSUB = 73  # Output slice size in subslices, similar to n1 in PyIMCOM.
    NPIX_SUB = 56  # Subslice size in pixels, similar to n2 in PyIMCOM.
    NPIX_TOT = NSUB * NPIX_SUB  # Output slice size in pixels.

    @staticmethod
    def get_outwcs(outcrval: np.ndarray, outcrpix: list[float] = [2044.0, 2044.0],
                   outcdelt: float = 0.11 * u.arcsec.to("degree") / 2.0) -> wcs.WCS:
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = outcrval
        outwcs.wcs.crpix = outcrpix
        outwcs.wcs.cdelt = [-outcdelt, outcdelt]
        return outwcs

    def __init__(self, wcs: wcs.WCS, inslices: list[InSlice]) -> None:
        self.wcs = wcs
        self.inslices = inslices
        for inslice in self.inslices:
            inslice.outslice = self
        self.data = np.zeros((self.NPIX_TOT, self.NPIX_TOT), dtype=np.float32)

    def __call__(self, filename: str = None,
                 timing: bool = False) -> None:
        if timing: tstart = perf_counter()
        # for X in range(0, self.NPIX_TOT, self.NPIX_SUB):
        for X in range(self.NSUB):
            if timing: print(f"Processing subslices ({X}, *)",
                             f"@ t = {perf_counter() - tstart:.6f} s")
            # for Y in range(0, self.NPIX_TOT, self.NPIX_SUB):
            for Y in range(self.NSUB):
                SubSlice(self, X, Y)()
        self.data /= len(self.inslices)

        if filename is not None:
            self.writeto(filename)

    def writeto(self, filename: str) -> None:
        hdu = fits.PrimaryHDU(self.data)
        hdu.header.update(self.wcs.to_header())
        hdu.writeto(filename, overwrite=True)
