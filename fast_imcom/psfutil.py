"""PSF utilities in two dimensions.

Functions
---------
psf_gaussian_2d : Generate a 2D Gaussian PSF.
pixelate_psf_2d : Pixelate a 2D (input) PSF.

"""

import numpy as np

from .routine import apply_weight_field


class PSFModel:

    NPIX = 48  # PSF array size in native pixels.
    SAMP = 4  # Oversampling rate of PSF arrays.
    NTOT = NPIX * SAMP  # PSF array size in oversampled pixels.

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
        # psf_in_t = np.fft.rfft2(np.fft.ifftshift(psf_in))
        psf_inp = cls.pixelate_psf_2d(psf_in)
        psf_inp_t = np.fft.rfft2(np.fft.ifftshift(psf_inp))
        psf_out_t = np.fft.rfft2(np.fft.ifftshift(psf_out))

        weight_t = psf_out_t / psf_inp_t
        weight_t[:, cls.NPIX:] = 0; weight_t[cls.NPIX:-cls.NPIX+1, :] = 0; weight_t.imag = 0
        weight = np.fft.ifftshift(np.fft.irfft2(weight_t, s=(cls.NTOT, cls.NTOT)))
        weight *= cls.SAMP**2

        return weight

    def __init__(self, psfdata: np.ndarray) -> None:
        self.psfdata = psfdata

    def __call__(self) -> float:
        assert self.psfdata.ndim == 2, "PSFModel: The base class only supports 2D data."
        return self.psfdata


class SubSlice:

    ACCEPT = 8  # ACCEPTance radius
    yxo = np.arange(PSFModel.NTOT//2 - (ACCEPT-1)*PSFModel.SAMP,
                    PSFModel.NTOT//2 + (ACCEPT+1)*PSFModel.SAMP, PSFModel.SAMP, dtype=float)

    def __init__(self, outslice, X: int, Y: int) -> None:
        self.outslice = outslice
        self.X, self.Y = X, Y
        self.outxys = np.moveaxis(np.array(np.meshgrid(
            np.arange(X, X+56), np.arange(Y, Y+56))), 0, -1).reshape(-1, 2)
        self.out_arr = np.zeros((self.ACCEPT*2, self.ACCEPT*2))

    def __call__(self, weight: np.ndarray) -> None:
        for inslice in self.outslice.inslices:
            # InImage.outpix2world2inpix
            inxys = inslice.wcs.all_world2pix(self.outslice.wcs.all_pix2world(self.outxys, 0), 0)
            inxys_frac, inxys_int = np.modf(inxys); inxys_int = inxys_int.astype(int)

            apply_weight_field(self.outxys, inxys_frac, inxys_int,
                               weight, self.outslice.data, inslice.data,
                               self.ACCEPT, self.yxo, PSFModel.SAMP)
