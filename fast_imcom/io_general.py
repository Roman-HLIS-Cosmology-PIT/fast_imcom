"""General I/O interface.

Classes
-------
InSlice : Input image slice, like InImage in PyIMCOM.
OutSlice : Output image slice, like Block in PyIMCOM.

"""


import numpy as np
from astropy.io import fits
from astropy import wcs


class InSlice:

    def __init__(self, filename: str) -> None:
        self.filename = filename
        with fits.open(filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
            self.data = f["WFI01"].data


class OutSlice:

    def __init__(self, wcs = wcs.WCS) -> None:
        self.wcs = wcs
        self.data = np.zeros((4088, 4088), dtype=">f4")

    def writeto(self, filename: str) -> None:
        hdu = fits.PrimaryHDU(self.data)
        hdu.header.update(self.wcs.to_header())
        hdu.writeto(filename, overwrite=True)
