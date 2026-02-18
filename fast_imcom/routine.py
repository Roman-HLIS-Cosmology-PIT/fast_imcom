'''
Numba version of pyimcom_croutines.c.

Slightly slower than C, used when furry-parakeet is not installed.

Functions
---------
iD5512C_getw : Interpolation code written by Python.
gridD5512C : iD5512C for rectangular grid.

'''

import numpy as np
from numba import njit


@njit
def iD5512C_getw(w: np.array, fh: float) -> None:
    '''
    Interpolation code written by Python.

    Parameters
    ----------
    w : np.array, shape : (10,)
        Interpolation weights in one direction.
    fh : float
        'xfh' and 'yfh' with 1/2 subtracted.

    Returns
    -------
    None.

    '''

    fh2 = fh * fh
    e_ =  (((+1.651881673372979740E-05*fh2 - 3.145538007199505447E-04)*fh2 +
          1.793518183780194427E-03)*fh2 - 2.904014557029917318E-03)*fh2 + 6.187591260980151433E-04
    o_ = ((((-3.486978652054735998E-06*fh2 + 6.753750285320532433E-05)*fh2 -
          3.871378836550175566E-04)*fh2 + 6.279918076641771273E-04)*fh2 - 1.338434614116611838E-04)*fh
    w[0] = e_ + o_
    w[9] = e_ - o_
    e_ =  (((-1.146756217210629335E-04*fh2 + 2.883845374976550142E-03)*fh2 -
          1.857047531896089884E-02)*fh2 + 3.147734488597204311E-02)*fh2 - 6.753293626461192439E-03
    o_ = ((((+3.121412120355294799E-05*fh2 - 8.040343683015897672E-04)*fh2 +
          5.209574765466357636E-03)*fh2 - 8.847326408846412429E-03)*fh2 + 1.898674086370833597E-03)*fh
    w[1] = e_ + o_
    w[8] = e_ - o_
    e_ =  (((+3.256838096371517067E-04*fh2 - 9.702063770653997568E-03)*fh2 +
          8.678848026470635524E-02)*fh2 - 1.659182651092198924E-01)*fh2 + 3.620560878249733799E-02
    o_ = ((((-1.243658986204533102E-04*fh2 + 3.804930695189636097E-03)*fh2 -
          3.434861846914529643E-02)*fh2 + 6.581033749134083954E-02)*fh2 - 1.436476114189205733E-02)*fh
    w[2] = e_ + o_
    w[7] = e_ - o_
    e_ =  (((-4.541830837949564726E-04*fh2 + 1.494862093737218955E-02)*fh2 -
          1.668775957435094937E-01)*fh2 + 5.879306056792649171E-01)*fh2 - 1.367845996704077915E-01
    o_ = ((((+2.894406669584551734E-04*fh2 - 9.794291009695265532E-03)*fh2 +
          1.104231510875857830E-01)*fh2 - 3.906954914039130755E-01)*fh2 + 9.092432925988773451E-02)*fh
    w[3] = e_ + o_
    w[6] = e_ - o_
    e_ =  (((+2.266560930061513573E-04*fh2 - 7.815848920941316502E-03)*fh2 +
          9.686607348538181506E-02)*fh2 - 4.505856722239036105E-01)*fh2 + 6.067135256905490381E-01
    o_ = ((((-4.336085507644610966E-04*fh2 + 1.537862263741893339E-02)*fh2 -
          1.925091434770601628E-01)*fh2 + 8.993141455798455697E-01)*fh2 - 1.213035309579723942E+00)*fh
    w[4] = e_ + o_
    w[5] = e_ - o_


@njit
def gridD5512C(infunc: np.array, xpos: np.array, ypos: np.array,
               fhatout: np.array) -> None:
    '''
    2D, 10x10 kernel interpolation for high accuracy

    this version works with output points on a rectangular grid so that the same
    weights in x and y can be used for many output points

    Notes:
    there are npi*nyo*nxo interpolations to be done in total
    but for each input pixel npi, there is an nyo x nxo grid of output points

    Parameters
    ----------
    infunc : np.array, shape : (ngy, ngx)
        Input function on some grid.
    xpos : np.array, shape : (npi, nxo)
        Input x values.
    ypos : np.array, shape : (npi, nyo)
        Input y values.
    fhatout : np.array, shape : (npi, nyo*nxo)
        Location to put the output values.

    Returns
    -------
    None.

    '''

    # extract dimensions
    ngy, ngx = infunc.shape
    npi, nxo = xpos.shape[:2]
    nyo = ypos.shape[1]

    wx_ar = np.zeros((nxo, 10))
    wy_ar = np.zeros((nyo, 10))
    xi = np.zeros((nxo,), dtype=np.int32)
    yi = np.zeros((nyo,), dtype=np.int32)

    # loop over points to interpolate
    for i_in in range(npi):
        # get the interpolation weights -- first in x, then in y.
        # do all the output points simultaneously to save time
        for ix in range(nxo):
            x = xpos[i_in, ix]
            xi[ix] = np.int32(x)

            # point off the grid, don't interpolate
            if xi[ix] < 4 or xi[ix] >= ngx-5:
                xi[ix] = 4
                wx_ar[ix] = 0.0
                continue
    
            iD5512C_getw(wx_ar[ix], x-xi[ix]-.5)

        # ... and now in y
        for iy in range(nyo):
            y = ypos[i_in, iy]
            yi[iy] = np.int32(y)

            # point off the grid, don't interpolate
            if yi[iy] < 4 or yi[iy] >= ngy-5:
                yi[iy] = 4
                wy_ar[iy] = 0.0
                continue
    
            iD5512C_getw(wy_ar[iy], y-yi[iy]-.5)

        # ... and now we can do the interpolation
        ipos = 0
        for iy in range(nyo):  # output pixel row
            for ix in range(nxo):  # output pixel column
                out = 0.0
                for i in range(10):
                    interp_vstrip = 0.0
                    for j in range(10):
                        interp_vstrip += wx_ar[ix, j] * infunc[yi[iy]-4+i, xi[ix]-4+j]
                    out += interp_vstrip * wy_ar[iy, i]
                fhatout[i_in, ipos] = out
                ipos += 1


# @njit
# def apply_weight_field(ACCEPT: int, YXO: np.ndarray, SAMP: int,
#                        weight: np.ndarray, inxys_frac: np.ndarray,
#                        indata: np.ndarray, inxys_int: np.ndarray,
#                        outdata: np.ndarray, outxys: np.ndarray) -> None:
#     out_arr = np.zeros((ACCEPT*2, ACCEPT*2))
#     for i in range(56**2):
#         gridD5512C(weight, YXO[None, :]-inxys_frac[i, 0]*SAMP, \
#             YXO[None, :]-inxys_frac[i, 1]*SAMP, out_arr.reshape((1, -1)))
#         outdata[outxys[i, 1], outxys[i, 0]] += np.sum(
#             out_arr * indata[inxys_int[i, 1]-ACCEPT:inxys_int[i, 1]+ACCEPT,
#                              inxys_int[i, 0]-ACCEPT:inxys_int[i, 0]+ACCEPT])

@njit
def compute_weights(weight: np.ndarray, YXO: np.ndarray, SAMP: int,
                    inxys_frac: np.ndarray, weights: np.ndarray) -> None:
    for i in range(inxys_frac.shape[0]):
        gridD5512C(weight, YXO[None, :]-inxys_frac[i, 0]*SAMP, \
            YXO[None, :]-inxys_frac[i, 1]*SAMP, weights[i].reshape((1, -1)))

# @njit
# def apply_weights(weights: np.ndarray, ACCEPT: int,
#                   indata: np.ndarray, inxys_int: np.ndarray,
#                   outdata: np.ndarray, outxys: np.ndarray) -> None:
#     for i in range(inxys_int.shape[0]):
#         outdata[outxys[i, 1], outxys[i, 0]] += np.sum(
#             weights[i] * indata[inxys_int[i, 1]-ACCEPT:inxys_int[i, 1]+ACCEPT,
#                                 inxys_int[i, 0]-ACCEPT:inxys_int[i, 0]+ACCEPT])

@njit
def apply_weights(weights: np.ndarray, ACCEPT: int,
                  indata: np.ndarray, inxys_int: np.ndarray,
                  outdata: np.ndarray) -> None:
    for i in range(inxys_int.shape[0]):
        outdata[i] += np.sum(weights[i] *\
            indata[inxys_int[i, 1]-ACCEPT:inxys_int[i, 1]+ACCEPT,
                   inxys_int[i, 0]-ACCEPT:inxys_int[i, 0]+ACCEPT])
