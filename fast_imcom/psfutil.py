"""Obsolete docstring, to be updated.

PSF utilities in two dimensions.

Functions
---------
psf_gaussian_2d : Generate a 2D Gaussian PSF.
pixelate_psf_2d : Pixelate a 2D (input) PSF.

"""

import numpy as np

from .routine import bandlimited_rfft2, bandlimited_irfft2
from .routine import compute_weights, adjust_weights, apply_weights


class PSFModel:

    NPIX = 48  # PSF array size in native pixels.
    SAMP = 4  # Oversampling rate of PSF arrays.
    NTOT = NPIX * SAMP  # PSF array size in oversampled pixels.
    YXCTR = NTOT // 2  # PSF array center in oversampled pixels.

    SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
    SIGMA = {
        "Y106": 2.0 / SIGMA_TO_FWHM,
        "J129": 2.1 / SIGMA_TO_FWHM,
        "H158": 2.2 / SIGMA_TO_FWHM,
        "F184": 2.3 / SIGMA_TO_FWHM,
        "K213": 2.4 / SIGMA_TO_FWHM,
    }

    @classmethod
    def psf_gaussian_2d(cls, sigma: float) -> np.ndarray:
        """Generate a 2D Gaussian PSF."""
        x, y = np.meshgrid(np.arange(-cls.NTOT//2, cls.NTOT//2) / (sigma*cls.SAMP),
                           np.arange(-cls.NTOT//2, cls.NTOT//2) / (sigma*cls.SAMP))
        return np.exp(-0.5 * (np.square(x) + np.square(y)))\
                / (2.0*np.pi * sigma**2)

    @classmethod
    def pixelate_psf_2d(cls, psf: np.ndarray) -> np.ndarray:
        """Pixelate a 2D (input) PSF."""
        k = np.linspace(0, 1, cls.NTOT, endpoint=False)
        k[-(cls.NTOT//2):] -= 1; k *= cls.SAMP
        return np.fft.irfft2(np.fft.rfft2(psf) *\
            np.sinc(k[None, :cls.NTOT//2+1]) * np.sinc(k[:, None]), s=(cls.NTOT, cls.NTOT))

    @classmethod
    def get_weight_field(cls, psf_in: np.ndarray, psf_out: np.ndarray) -> np.ndarray:
        psf_inp = cls.pixelate_psf_2d(psf_in)
        psf_inp_tbl = bandlimited_rfft2(psf_inp[None], cls.NPIX-1)[0]
        psf_out_tbl = bandlimited_rfft2(psf_out[None], cls.NPIX-1)[0]

        weight_tbl = psf_out_tbl / psf_inp_tbl; weight_tbl.imag = 0
        weight = np.fft.ifftshift(bandlimited_irfft2(weight_tbl[None], cls.NTOT, cls.NTOT))[0]
        weight *= cls.SAMP**2

        return weight

    def __init__(self, psfdata: np.ndarray) -> None:
        self.psfdata = psfdata

    def __call__(self, x: float = -np.inf, y: float = -np.inf) -> np.ndarray:
        assert self.psfdata.ndim == 2, "PSFModel: The base class only supports 2D data."
        return self.psfdata


class SubSlice:

    ACCEPT = 8  # ACCEPTance radius
    LOSS_THR = 0.001  # Threshold for total lost weight.

    def __init__(self, outslice, X: int, Y: int) -> None:
        self.outslice = outslice
        self.X, self.Y = X, Y
        NPIX_SUB = self.outslice.NPIX_SUB  # Shortcut.
        self.outxys = np.moveaxis(np.array(np.meshgrid(
            np.arange(NPIX_SUB) + X*NPIX_SUB,
            np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)

    def __call__(self, sigma: float = PSFModel.SIGMA["H158"] * 1.5) -> None:
        NPIX_SUB = self.outslice.NPIX_SUB  # Shortcut.

        for i_sl, inslice in enumerate(self.outslice.inslices):
            psf_in = inslice.get_psf()
            psf_out = PSFModel.psf_gaussian_2d(sigma)
            weight = PSFModel.get_weight_field(psf_in, psf_out)

            inxys = inslice.outpix2world2inpix(self.outslice.wcs, self.outxys)
            inxys_frac, inxys_int = np.modf(inxys); inxys_int = inxys_int.astype(int)
            x_min, y_min = np.min(inxys_int, axis=0) - self.ACCEPT
            x_max, y_max = np.max(inxys_int, axis=0) + self.ACCEPT
            indata, inmask = inslice.get_data_and_mask(x_min, x_max, y_min, y_max)
            mask_out = inslice.mask_out[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                        self.X*NPIX_SUB:(self.X+1)*NPIX_SUB].copy()
            inxys_int -= np.array([x_min, y_min])

            weights = np.zeros((NPIX_SUB**2, self.ACCEPT*2, self.ACCEPT*2))
            compute_weights(weights, mask_out.ravel(), weight, inxys_frac,
                            PSFModel.YXCTR, PSFModel.SAMP, self.ACCEPT)
            adjust_weights(weights, mask_out.ravel(), inmask, inxys_int, self.ACCEPT, self.LOSS_THR)
            inslice.mask_out[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                             self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] = mask_out

            outdata = np.zeros((NPIX_SUB, NPIX_SUB))
            apply_weights(weights, mask_out.ravel(), outdata.ravel(), indata, inxys_int, self.ACCEPT)
            if self.outslice.SAVE_ALL:
                self.outslice.data[i_sl, self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                         self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] = outdata
            else:
                self.outslice.data[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                   self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] += outdata
