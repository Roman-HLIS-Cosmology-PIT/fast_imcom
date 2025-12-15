"""PSF utilities in one dimension.

Functions
---------
psf_gaussian : Generates a Gaussian PSF.
psf_single_slit : Generates a single-slit PSF.
pixelate_psf : Pixelates a PSF.
visualize_psf : Visualizes a PSF.
format_and_show : Formats and displays a figure.
square_norm : Computes the square norm of a PSF.

psf_overlap : Computes the overlap between two PSFs.
show_matrix : Displays a matrix with a colorbar.
get_Asub : Computes an A submatrix for the IMCOM algorithm.
get_Amat : Computes the A matrix for the IMCOM algorithm.
get_Bsub : Computes a B submatrix for the IMCOM algorithm.
get_Bmat : Computes the B matrix for the IMCOM algorithm.

explore_case : Explores a case for the IMCOM algorithm.
visualize_case : Visualizes a case for the IMCOM algorithm.
get_meta_weights : Computes the meta weights for three shifts.
"""

from itertools import combinations_with_replacement

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


NPIX = 64  # PSF array size in native pixels.
SAMP = 32  # Oversampling rate of PSF arrays.
NTOT = NPIX * SAMP  # PSF array size in oversampled pixels.


SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
SIGMA = {
    "Y106": 2.0 / SIGMA_TO_FWHM,
    "J129": 2.1 / SIGMA_TO_FWHM,
    "H158": 2.2 / SIGMA_TO_FWHM,
    "F184": 2.3 / SIGMA_TO_FWHM,
    "K213": 2.4 / SIGMA_TO_FWHM,
}

def psf_gaussian(sigma: float) -> np.ndarray:
    """Generates a Gaussian PSF.

    Parameters
    ----------
    sigma : float
        The standard deviation of the Gaussian in native pixels.

    Returns
    -------
    np.ndarray
        The Gaussian PSF.
    """

    x = np.arange(-NTOT//2, NTOT//2) / (sigma*SAMP)
    return np.exp(-0.5 * np.square(x)) / (np.sqrt(2.0*np.pi) * sigma)


LDP = {"Y106": 0.834, "J129": 1.021, "H158": 1.250, "F184": 1.456, "K213": 1.685}

def psf_single_slit(ldp, obsc: float = 0.31):
    """Generates a single-slit PSF.

    Parameters
    ----------
    ldp : float
        The diffraction scale, i.e., the central wavelength
        divided by the aperture diameter, in native pixels.
    obsc : float, optional
        The linear obscuration factor (the default is 0.31).

    Returns
    -------
    np.ndarray
        The single-slit PSF.
    """

    x = np.arange(-NTOT//2, NTOT//2) / (ldp*SAMP)
    return np.square(np.sinc(x) - obsc * np.sinc(obsc*x)) / (ldp * (1-obsc))


def pixelate_psf(psf: np.ndarray) -> np.ndarray:
    """Pixelates a Point Spread Function (PSF).

    Parameters
    ----------
    psf : np.ndarray
        The PSF to pixelate.

    Returns
    -------
    np.ndarray
        The pixelated PSF.
    """

    return np.fft.irfft(np.fft.rfft(psf) *\
        np.sinc(np.linspace(0, 1/2, NTOT//2+1) * SAMP), n=NTOT)


def visualize_psf(axs: np.ndarray, psf: np.ndarray, label: str) -> None:
    """Visualizes a Point Spread Function (PSF).

    Parameters
    ----------
    axs : np.ndarray
        The array of axes to plot on.
    psf : np.ndarray
        The PSF to visualize.
    label : str
        The label for the PSF plot.

    Returns
    -------
    None
    """

    x = np.arange(-NTOT//2, NTOT//2) / SAMP
    axs[0].plot(x, psf, label=label)
    axs[0].set_ylabel('PSF')

    axs[1].plot(x, np.log10(psf))
    axs[1].set_ylabel('log(PSF)')
    axs[1].set_xlabel('$x$ [native pixel]')


def format_and_show(fig: mpl.figure.Figure, axs: np.ndarray) -> None:
    """Format and display the figure with the given axes.

    Parameters
    ----------
    fig : mpl.figure.Figure
        The figure to format and show.
    axs : np.ndarray
        The array of axes to format.

    Returns
    -------
    None
    """

    axs[0].legend()
    for ax in axs: ax.grid()
    fig.tight_layout()
    plt.show()


def square_norm(psf: np.ndarray) -> float:
    """Compute the square norm of a PSF.

    Parameters
    ----------
    psf : np.ndarray
        The PSF array.

    Returns
    -------
    float
        The square norm of the PSF.
    """

    return np.square(psf).sum()


def psf_overlap(psf1: np.ndarray, psf2: np.ndarray) -> np.ndarray:
    """Compute the overlap between two PSFs.

    Parameters
    ----------
    psf1 : np.ndarray
        The first PSF.
    psf2 : np.ndarray
        The second PSF.

    Returns
    -------
    np.ndarray
        The overlap between the two PSFs.
    """

    return np.fft.ifftshift(np.fft.irfft(
        np.fft.rfft(psf1).conj() * np.fft.rfft(psf2), n=NTOT))


def show_matrix(fig: mpl.figure.Figure, ax: mpl.axes.Axes,
                mat: np.ndarray, **kwargs) -> None:
    """Displays a matrix with a colorbar.

    Parameters
    ----------
    fig : mpl.figure.Figure
        The figure to plot on.
    ax : mpl.axes.Axes
        The axis to plot on.
    mat : np.ndarray
        The matrix to show.
    **kwargs : dict
        Additional keyword arguments for the colorbar.

    Returns
    -------
    None
    """

    vmin, vmax = mat.min(), mat.max()
    gain = max(abs(vmin), abs(vmax))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-gain, vmax=gain)
    cbar = fig.colorbar(im, ax=ax, **kwargs)
    if kwargs.get("orientation", "vertical") == "horizontal":
        cbar.ax.set_xlim((vmin, vmax))
    else:
        cbar.ax.set_ylim((vmin, vmax))


def get_Asub(ovl_ii: np.ndarray, shift: int = 0, reject: int = 0) -> np.ndarray:
    """Computes an A submatrix for the IMCOM algorithm.

    Parameters
    ----------
    ovl_ii : np.ndarray
        The overlap between two input PSFs.
    shift : int
        The shift of the input grid relative to the zeroth output pixel.
    reject : int
        The number of input pixels to reject on the left.

    Returns
    -------
    np.ndarray
        The A submatrix.
    """

    Asub = np.zeros((NPIX, NPIX))

    if shift == 0:
        for j in range(NPIX):
            Asub[j, j:min(j+NPIX//2, NPIX)] =\
                ovl_ii[NTOT//2::SAMP][:min(NPIX//2, NPIX-j)]
            Asub[j:, j] = Asub[j, j:]

    else:
        for j in range(NPIX):
            # Taking the absolute value of the shift difference is not correct in general,
            # but it works for the current setup since the two input PSFs are the same.
            Asub[j, max(j-(NPIX//2), 0):min(j+(NPIX//2), NPIX)] =\
                ovl_ii[abs(shift)::SAMP][max(NPIX//2-j, 0):min(NPIX*3//2-j, NPIX)]

    if shift <= 0:  # r_i <= r_j
        return Asub[reject:, reject:]
    else:  # r_i > r_j
        return Asub[reject:, reject:].T


def get_Amat(ovl_ii: np.ndarray, shifts: list[int] = [0],
             reject: int = 0) -> np.ndarray:
    """Computes the A matrix for the IMCOM algorithm.

    Parameters
    ----------
    ovl_ii : np.ndarray
        The overlap between two input PSFs.
    shifts : list[int]
        The shifts of the input grids relative to the zeroth output pixel.
    reject : int
        The number of input pixels to reject on the left.

    Returns
    -------
    np.ndarray
        The A matrix.
    """

    npix = NPIX - reject
    Amat = np.zeros((npix * len(shifts),)*2)

    for pair in combinations_with_replacement(range(len(shifts)), 2):
        Amat[pair[0]*npix:(pair[0]+1)*npix, pair[1]*npix:(pair[1]+1)*npix] =\
            get_Asub(ovl_ii, shift=shifts[pair[0]] - shifts[pair[1]], reject=reject)
        if pair[0] != pair[1]:
            Amat[pair[1]*npix:(pair[1]+1)*npix, pair[0]*npix:(pair[0]+1)*npix] =\
                Amat[pair[0]*npix:(pair[0]+1)*npix, pair[1]*npix:(pair[1]+1)*npix].T

    return Amat


def get_Bsub(ovl_oi: np.ndarray, nout: int = SAMP,
             shift: int = 0, reject: int = 0) -> np.ndarray:
    """Computes a B submatrix for the IMCOM algorithm.

    Parameters
    ----------
    ovl_oi : np.ndarray
        The overlap between the output PSF and an input PSF.
    nout : int
        The number of output pixels to consider.
    shift : int
        The shift of the input grid relative to the zeroth output pixel.
    reject : int
        The number of input pixels to reject on the left.

    Returns
    -------
    np.ndarray
        The B submatrix.
    """

    Bsub = np.zeros((nout, NPIX))

    for i in range(nout):
        Bsub[i] = ovl_oi[i+shift::SAMP]

    return Bsub[:, reject:]


def get_Bmat(ovl_oi: np.ndarray, shifts: list[int] = [0],
             reject: int = 0) -> np.ndarray:
    """Computes the B matrix for the IMCOM algorithm.

    Parameters
    ----------
    ovl_oi : np.ndarray
        The overlap between the output PSF and an input PSF.
    shifts : list[int]
        The shifts of the input grids relative to the zeroth output pixel.
    reject : int
        The number of input pixels to reject on the left.

    Returns
    -------
    np.ndarray
        The B matrix.
    """

    npix = NPIX - reject
    nout = SAMP - max(shifts)
    Bmat = np.zeros((nout, npix * len(shifts)))

    for i, shift in enumerate(shifts):
        Bmat[:, i*npix:(i+1)*npix] = get_Bsub(
            ovl_oi, nout=nout, shift=shift, reject=reject)

    return Bmat


def explore_case(ovl_ii: np.ndarray, ovl_oi: np.ndarray, shifts: list[int] = [0],
                 reject: int = 0, aspect: int = 0, figname: str = None) -> tuple[np.ndarray]:
    """Explores a case for the IMCOM algorithm.

    Parameters
    ----------
    ovl_ii : np.ndarray
        The overlap between two input PSFs.
    ovl_oi : np.ndarray
        The overlap between the output PSF and an input PSF.
    shifts : list[int], optional
        The shifts of the input grids relative to the zeroth output pixel.
    reject : int, optional
        The number of pixels to reject from each side of the input PSF.
    aspect : int, optional
        The aspect ratio for the B and T matrix colorbars.
        The default is 0, which means no visualization.

    Returns
    -------
    tuple[np.ndarray]
        The A, Ainv, B, and T matrices.
    """

    Amat = get_Amat(ovl_ii, shifts=shifts, reject=reject)
    Ainv = np.linalg.pinv(Amat)
    Bmat = get_Bmat(ovl_oi, shifts=shifts, reject=reject)
    Tmat = (Ainv @ Bmat.T).T

    if aspect:
        fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.0/33 * (20+aspect)), sharex=True,
                                sharey="row", height_ratios=[2, aspect/10])
        common = dict(orientation="vertical", pad=0.05)

        ax = axs[0, 0]; ax.set_title("$A$ matrix")
        show_matrix(fig, ax, Amat, **common)
        ax = axs[0, 1]; ax.set_title("$A^{-1}$ matrix")
        show_matrix(fig, ax, Ainv, **common)

        ax = axs[1, 0]; ax.set_title("$-B/2$ matrix")
        show_matrix(fig, ax, Bmat, aspect=aspect, **common)
        ax = axs[1, 1]; ax.set_title("$T$ matrix")
        show_matrix(fig, ax, Tmat, aspect=aspect, **common)

        axs[0, 0].set_ylabel("input pixel $j$")
        axs[1, 0].set_ylabel(r"output pixel $\alpha$")
        for ax in axs[1, :]: ax.set_xlabel("input pixel $i$")

        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            fig.savefig(f"plots/{figname}.pdf", bbox_inches="tight")
            plt.close(fig)

    return Amat, Ainv, Bmat, Tmat


def visualize_case(my_weightu_i: np.ndarray, my_weightu_f: np.ndarray,
                   psf_inp_t: np.ndarray, psf_out: np.ndarray,
                   label: str, options: list[int] = [3]) -> tuple[float]:
    """Visualizes a case for the IMCOM algorithm.

    Parameters
    ----------
    my_weightu_i : np.ndarray
        The IMCOM coaddition weights.
    my_weightu_f : np.ndarray
        The Fast IMCOM coaddition weights.
    psf_inp_t : np.ndarray
        The Fourier transform of the pixelated input PSF.
    psf_out : np.ndarray
        The target output PSF.
    label : str
        The label for the case.
    options : list[int]
        The options for visualization.

    Returns
    -------
    tuple(float)
        The PSF leakage for IMCOM and Fast IMCOM coaddition weights.
    """

    # Coordinates for visualizing coaddition weights.
    x = np.arange(-NTOT//2, NTOT//2) / SAMP  # Real space.
    u = np.fft.rfftfreq(NTOT, d=1/SAMP)  # Fourier space.
    C = square_norm(psf_out)  # L2-norm of the target PSF.

    if 1 in options:  # Option 1: coaddition weights.
        fig, ax = plt.subplots(1, 1, figsize=(9.6, 1.8))
        ax.plot(x, my_weightu_i, label="IMCOM")
        ax.plot(x, my_weightu_f, label="Fast IMCOM")
        ax.set_title(f"Coaddition weights, {label}")
        ax.set_xlabel('$x$ [native pixel]')
        ax.legend(); ax.grid(); plt.show()

    psf_outu_it = np.fft.rfft(np.fft.ifftshift(my_weightu_i))
    psf_outu_i = np.fft.ifftshift(np.fft.irfft(psf_outu_it * psf_inp_t, n=NTOT))
    psf_outu_ft = np.fft.rfft(np.fft.ifftshift(my_weightu_f))
    psf_outu_f = np.fft.ifftshift(np.fft.irfft(psf_outu_ft * psf_inp_t, n=NTOT))

    if 2 in options:  # Option 2: Fourier space.
        fig, axs = plt.subplots(2, 1, figsize=(9.6, 3.6), sharex=True)
        for i, part in enumerate(["real", "imag"]):
            for psf_outu_t, label in [(psf_outu_it, "IMCOM"),
                                      (psf_outu_ft, "Fast IMCOM")]:
                axs[i].plot(u[:2*SAMP], getattr(psf_outu_t, part)[:2*SAMP], label=label)
        axs[0].set_title(f"Reconstructed - Target PSF, {label}")
        axs[1].set_xlabel('$u$ [cycle per pixel]')
        format_and_show(fig, axs)

    uc_i = square_norm(psf_outu_i-psf_out)/C
    uc_f = square_norm(psf_outu_f-psf_out)/C

    if 3 in options:  # Option 3: real space.
        fig, axs = plt.subplots(2, 1, figsize=(9.6, 3.6), sharex=True)
        visualize_psf(axs, psf_outu_i-psf_out, f"IMCOM, $U/C$ = {uc_i:.4e}")
        visualize_psf(axs, psf_outu_f-psf_out, f"Fast IMCOM, $U/C$ = {uc_f:.4e}")
        axs[0].set_title(f"Reconstructed - Target PSF, {label}")
        format_and_show(fig, axs)

    return uc_i, uc_f


def get_meta_weights(shifts: list[int]) -> np.ndarray:
    """Computes the meta weights for the three shifts.

    Parameters
    ----------
    shifts : list[int]
        The list of shifts to compute the meta weights for.

    Returns
    -------
    np.ndarray
        The meta weights for the three shifts.
    """

    assert len(shifts) == 3, "Only support 3 shifts."
    if len(set(shifts)) == 1: return np.array([1/3, 1/3, 1/3])
    if len(set(shifts)) == 2:
        if shifts[0] == shifts[1]: return np.array([1/4, 1/4, 1/2])
        if shifts[0] == shifts[2]: return np.array([1/4, 1/2, 1/4])
        if shifts[1] == shifts[2]: return np.array([1/2, 1/4, 1/4])

    thetas = np.array(shifts) / SAMP * 2*np.pi
    A = np.empty((3, 3))
    A[0, :] = 1
    A[2, :] = np.cos(thetas)
    A[1, :] = np.sin(thetas)
    B = np.zeros(3); B[0] = 1
    return np.linalg.solve(A, B)

