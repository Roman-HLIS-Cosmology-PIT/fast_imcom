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

from .psfutil import SubSlice


class InSlice:

    def __init__(self, filename: str) -> None:
        self.filename = filename
        with fits.open(filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
            self.data = f["WFI01"].data.astype(np.float32)


class OutSlice:

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
        self.data = np.zeros((4088, 4088), dtype=np.float32)

    def __call__(self, filename: str = None,
                 timing: bool = False) -> None:
        if timing: tstart = perf_counter()
        for X in range(0, 4088, 56):
            if timing: print(f"Processing subslices ({X // 56}, *)...", perf_counter() - tstart)
            for Y in range(0, 4088, 56):
                SubSlice(self, X, Y)()
        self.data /= len(self.inslices)

        if filename is not None:
            self.writeto(filename)

    def writeto(self, filename: str) -> None:
        hdu = fits.PrimaryHDU(self.data)
        hdu.header.update(self.wcs.to_header())
        hdu.writeto(filename, overwrite=True)
