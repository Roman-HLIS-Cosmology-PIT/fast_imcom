import glob

import numpy as np
from astropy import wcs
from astropy import units as u
from astropy.io import fits

from fast_imcom.io_general import InSlice, OutSlice
from fast_imcom.psfutil import SAMP, NTOT, SIGMA
from fast_imcom.psfutil import psf_gaussian_2d, get_weight_field, SubSlice


inslices = [InSlice(name) for name in glob.glob("../test_imcom_stips/sim_*.fits")]
print(outcrval := np.mean([inslice.wcs.wcs.crval for inslice in inslices], axis=0))

outwcs = wcs.WCS(naxis=2)
outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
outwcs.wcs.crval = outcrval
outwcs.wcs.crpix = [2044.0, 2044.0]
cdelt = 0.11 * u.arcsec.to("degree") / 2.0
outwcs.wcs.cdelt = [-cdelt, cdelt]

outslice = OutSlice(outwcs)
outslice.inslices = inslices

psf_in = np.zeros((NTOT, NTOT))
with fits.open("../test_imcom_stips/psf_WFI_2.3_F158_wfi01.fits") as hdul:
    psf_in[8:-7, 8:-7] = hdul[0].data.mean(axis=0) * SAMP**2
# psf_in = psf_simple_airy(LDP["H158"])

psf_out = psf_gaussian_2d(SIGMA["H158"] * 1.5)
weight = get_weight_field(psf_in, psf_out)

for X in range(0, 4088, 56):
    print(f"Processing subslices ({X // 56}, *)...")
    for Y in range(0, 4088, 56):
        SubSlice(outslice, X, Y)(weight)

# outslice.data /= len(inslices)
outslice.writeto("../test_imcom_stips/test_imcom_stips_new.fits")

with fits.open("../test_imcom_stips/test_imcom_stips.fits") as a, \
    fits.open("../test_imcom_stips/test_imcom_stips_new.fits") as b:
    fd = fits.FITSDiff(a, b)
    print('identical:', fd.identical)
