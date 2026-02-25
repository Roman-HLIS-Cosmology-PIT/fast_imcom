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

    NSIDE = 4088
    NLAYER = 1  # Number of input layers.

    def __init__(self, filename: str, psfmodel: PSFModel = None,
                 loaddata: bool = True, paddata: bool = True) -> None:
        self.filename = filename
        self.psfmodel = psfmodel

        self.inx_min = self.iny_min = 0
        if loaddata:
            self.load_data_and_mask()
            if paddata: self.pad_data_and_mask()
        else:
            self.wcs = self.scale = None
            self.data = self.mask = None

    def load_data_and_mask(self) -> None:
        self.data = np.zeros((InSlice.NLAYER, InSlice.NSIDE, InSlice.NSIDE), dtype=np.float32)
        with fits.open(self.filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
            self.data[0] = f["WFI01"].data.astype(np.float32)
        self.scale = np.abs(self.wcs.wcs.cd[0, 0])  # Pixel scale in degrees.
        self.mask = np.ones((InSlice.NSIDE, InSlice.NSIDE), dtype=bool)

    def pad_data_and_mask(self) -> None:
        ACCEPT = SubSlice.ACCEPT  # Shortcut.
        self.data = np.pad(self.data, ((0,)*2,) + ((ACCEPT,)*2,)*2,
                           mode="constant", constant_values=0)
        self.mask = np.pad(self.mask, ACCEPT, mode="constant", constant_values=False)

        self.NSIDE = InSlice.NSIDE + ACCEPT*2
        self.inx_min = self.iny_min = -ACCEPT
        self.wcs.wcs.crpix += np.array([ACCEPT, ACCEPT])

    def outpix2world2inpix(self, outwcs: wcs.WCS, outxys: np.ndarray) -> np.ndarray:
        return self.wcs.all_world2pix(outwcs.all_pix2world(outxys, 0), 0)

    def inpix2world2outpix(self, outwcs: wcs.WCS, inxys: np.ndarray) -> np.ndarray:
        return outwcs.all_world2pix(self.wcs.all_pix2world(inxys, 0), 0)

    def assess_overlap(self, shrink: bool = True) -> None:
        ACCEPT = SubSlice.ACCEPT  # Shortcuts.
        NSUB, NPIX_SUB = OutSlice.NSUB, OutSlice.NPIX_SUB
        outxys_sp = np.moveaxis(np.array(np.meshgrid(
            np.arange(NSUB+1)*NPIX_SUB, np.arange(NSUB+1)*NPIX_SUB)), 0, -1).reshape(-1, 2)
        inxys_sp = self.outpix2world2inpix(self.outslice.wcs, outxys_sp)
        if shrink:
            out_inx_min, out_iny_min = np.rint(np.min(inxys_sp, axis=0)).astype(int)
            out_inx_max, out_iny_max = np.rint(np.max(inxys_sp, axis=0)).astype(int)

        subsize = NPIX_SUB * self.outslice.wcs.wcs.cdelt[1] /\
            (0.11 * u.arcsec.to("degree"))  # Subslice size in input pixels.
        mask_sp = np.all((inxys_sp >=           -0.5-subsize) &
                         (inxys_sp <= self.NSIDE-0.5+subsize), axis=1).reshape(NSUB+1, NSUB+1)
        self.mask_out = np.zeros((OutSlice.NPIX_TOT, OutSlice.NPIX_TOT), dtype=bool)

        for X in range(NSUB):
            for Y in range(NSUB):
                if not np.any(mask_sp[Y:min(Y+2, NSUB+1), X:min(X+2, NSUB+1)]): continue

                outxys = np.moveaxis(np.array(np.meshgrid(
                    np.arange(NPIX_SUB) + X*NPIX_SUB,
                    np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)
                inxys = self.outpix2world2inpix(self.outslice.wcs, outxys)

                if shrink:
                    inx_min, iny_min = np.rint(np.min(inxys, axis=0)).astype(int)
                    inx_max, iny_max = np.rint(np.max(inxys, axis=0)).astype(int)
                    out_inx_min = min(out_inx_min, inx_min)
                    out_iny_min = min(out_iny_min, iny_min)
                    out_inx_max = max(out_inx_max, inx_max)
                    out_iny_max = max(out_iny_max, iny_max)

                self.mask_out[Y*NPIX_SUB:(Y+1)*NPIX_SUB,
                              X*NPIX_SUB:(X+1)*NPIX_SUB] =\
                    np.all((inxys >=           -0.5+ACCEPT) &
                           (inxys <= self.NSIDE-0.5-ACCEPT), axis=1).\
                    reshape(NPIX_SUB, NPIX_SUB)

        self.is_relevant = np.any(self.mask_out)
        if not self.is_relevant: return

        if shrink:
            out_inx_min = max(out_inx_min - ACCEPT*3, 0)
            out_iny_min = max(out_iny_min - ACCEPT*3, 0)
            out_inx_max = min(out_inx_max + ACCEPT*3, self.NSIDE-1)
            out_iny_max = min(out_iny_max + ACCEPT*3, self.NSIDE-1)

            self.inx_min += out_inx_min; self.iny_min += out_iny_min
            self.wcs.wcs.crpix -= np.array([out_inx_min, out_iny_min])
            self.data = self.data[:, out_iny_min:out_iny_max+1, out_inx_min:out_inx_max+1].copy()
            self.mask = self.mask[   out_iny_min:out_iny_max+1, out_inx_min:out_inx_max+1].copy()

    def get_psf(self, x: float = -np.inf, y: float = -np.inf) -> np.ndarray:
        return self.psfmodel(x + self.inx_min, y + self.iny_min)

    def get_data_and_mask(self, x_min, x_max, y_min, y_max) -> tuple[np.ndarray, np.ndarray]:
        return (self.data[:, y_min:y_max+1, x_min:x_max+1].copy(),
                self.mask[   y_min:y_max+1, x_min:x_max+1].copy())


class OutSlice:

    NSUB = 73  # Output slice size in subslices, similar to n1 in PyIMCOM.
    NPIX_SUB = 56  # Subslice size in pixels, similar to n2 in PyIMCOM.
    NPIX_TOT = NSUB * NPIX_SUB  # Output slice size in pixels.
    CDELT = 0.11 * u.arcsec.to("degree") / 2.0  # Output pixel scale in degrees.
    SIGMA = PSFModel.SIGMA["H158"] * 1.5  # Target output PSF width in native pixels.
    SAVE_ALL = True  # Whether to save individual regridded images.

    @classmethod
    def get_outwcs(cls, outcrval: np.ndarray, outcrpix: list[float, float] = None,
                   outcdelt: list[float, float] = None) -> wcs.WCS:
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = outcrval
        outwcs.wcs.crpix = outcrpix if outcrpix is not None else [cls.NPIX_TOT/2]*2
        outwcs.wcs.cdelt = outcdelt if outcdelt is not None else [-cls.CDELT, cls.CDELT]
        return outwcs

    def __init__(self, wcs: wcs.WCS, inslices: list[InSlice], timing: bool = False) -> None:
        self.wcs = wcs
        self.scale = np.abs(wcs.wcs.cdelt[0])  # Pixel scale in degrees.
        self.inslices = inslices

        if timing: tstart = perf_counter()
        for inslice in self.inslices:
            if timing: print(f"Assessing inslice {inslice.filename}",
                             f"@ t = {perf_counter() - tstart:.6f} s")
            inslice.outslice = self
            inslice.assess_overlap()
        if timing: print("Finished assessing inslices",
                         f"@ t = {perf_counter() - tstart:.6f} s", end="\n\n")

        self.inslices = [inslice for inslice in self.inslices if inslice.is_relevant]
        self.ninslice = len(self.inslices)
        if self.SAVE_ALL:
            self.data = np.zeros((InSlice.NLAYER, self.ninslice,
                                  self.NPIX_TOT, self.NPIX_TOT), dtype=np.float32)
        else:
            self.data = np.zeros((InSlice.NLAYER, self.NPIX_TOT, self.NPIX_TOT), dtype=np.float32)

    def __call__(self, filename: str = None, timing: bool = False, stop: int = np.inf) -> None:
        if timing: tstart = perf_counter()
        for X in range(self.NSUB):
            if stop > 0 and timing:
                print(f"Processing subslices ({X}, *)",
                      f"@ t = {perf_counter() - tstart:.6f} s")
            for Y in range(self.NSUB):
                if stop > 0:
                    SubSlice(self, X, Y)(sigma=self.SIGMA)
                stop -= 1
        if timing: print("Finished processing subslices",
                         f"@ t = {perf_counter() - tstart:.6f} s", end="\n\n")

        self.mask = np.stack([inslice.mask_out for inslice in self.inslices])
        if not self.SAVE_ALL: self.data /= self.mask.sum(axis=0)
        if InSlice.NLAYER == 1: self.data = self.data[0]
        if filename is not None: self.writeto(filename)

    def writeto(self, filename: str) -> None:
        datahdu = fits.PrimaryHDU(self.data, header=self.wcs.to_header())
        inputhdu = fits.TableHDU.from_columns([fits.Column(name="filename", \
            array=[inslice.filename for inslice in self.inslices], format="A512", ascii=True)])
        inputhdu.name = "INPUT"
        maskhdu = fits.ImageHDU(self.mask.astype(np.uint8), name="MASK")
        fits.HDUList([datahdu, inputhdu, maskhdu]).writeto(filename, overwrite=True)
