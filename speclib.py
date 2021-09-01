#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:00:36 2018

@author: wskang
"""

import numpy as np
import astropy.units as u
from astropy.modeling import models, fitting
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum, fit_lines, \
    find_lines_threshold, find_lines_derivative
from specutils.analysis import line_flux
from scipy.optimize import curve_fit, fmin, leastsq
# from scipy.ndimage import map_coordinates, gaussian_filter, minimum_position, shift
from scipy.interpolate import griddata, interp2d, interp1d
from scipy.special import wofz
import matplotlib.pyplot as plt

TDIR = './tmp/'


def read_params():
    f = open('tame.par', 'r')
    par = {}
    for line in f:
        tmps = line.split('#')[0].split()
        if len(tmps) < 1: continue
        keyword = tmps[0]
        contents = ''.join(tmps[1:])
        par.update({keyword: contents})
    return par

def find_elname(elnum):
    names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P',
             'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
             'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
             'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
             'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
             'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
             'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut',
             'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']
    name1 = names[int(elnum)-1]
    ionnum = round(np.modf(elnum)[0]*10)+1
    if ionnum < 4:
        name2 = ''.join(list('I')*ionnum)
    else:
        name2 = '*'
    return name1+name2


def iminterp1d(ox, x, y, method='linear'):
    x = np.array(x.flatten(), dtype=np.double)
    y = np.array(y.flatten(), dtype=np.double)
    f = interp1d(x, y, kind=method, fill_value='extrapolate')
    return f(ox)


def simplex(points, values, p0):
    # global val, pts
    val = values
    pts = points

    def fgrid(x):
        fval = griddata(pts, val, tuple(x), method='nearest')
        print(x, fval)
        return fval

    res = fmin(fgrid, p0, ftol=1e-12, xtol=10., disp=True)
    return res


def nderiv(x, y, smoother=5):
    """
    Find the numerical derivatives
    """
    if x.size != y.size:
        print('size of x, y is not the same')
        return
    tx = x.copy()
    ty = gaussian_convolve(y.copy(), smoother)
    dx = np.diff(tx)
    dy = np.diff(ty)

    dy1 = dy / dx
    dx1 = tx[:-1] + dx / 2.0

    result = np.interp(tx, dx1, dy1)

    return result


def gaussian_convolve(y, size):
    """
    Perform the Gaussian convolution for the spectrum
    """
    x = np.mgrid[-size:size + 1]
    g = np.exp(-(x ** 2 / np.float64(size)))
    kernal = g / g.sum()
    result = np.convolve(y, kernal, mode='same')
    return result


#############################################################################
# GAUSSAIN/VOIGT FITTING  
#############################################################################

def gaussian(x, *params):
    """
    Gaussian functions for normal fitting
    """
    i = 0
    peak = params[i]
    center = params[i + 1]
    width = params[i + 2]
    background = params[i + 3]
    y = peak * np.exp(-0.5 * ((x - center) / width) ** 2) + background
    return y


def fit_gaussian(x, y, p0=[0., 1., 0., 1.]):
    """
    Find the simple Gaussian fitting function
    """
    coeff, var_matrix = curve_fit(gaussian, x, y, p0=p0)
    return list(coeff)


def mgaussian(x, *params):
    """
    Multiple Gaussian functions for absorption lines
    """
    y = np.ones_like(x)
    for i in range(0, len(params), 3):
        peak = params[i]
        center = params[i + 1]
        width = params[i + 2]
        y = y - peak * np.exp(-0.5 * ((x - center) / width) ** 2)
    return y


def curve_fit_mgaussian(x, y, p0):
    """
    Find the simple Gaussian fitting function
    """
    coeff, var_matrix = curve_fit(mgaussian, x, y, p0=p0)
    return list(coeff)


def _voigt(x, y):
    # The Voigt function is also the real part of
    # w(z) = exp(-z^2) erfc(iz), the complex probability function,
    # which is also known as the Faddeeva function. Scipy has
    # implemented this function under the name wofz()
    z = x + 1j * y
    return wofz(z).real


def voigt(x, *param):  # alphaD, alphaL, nu_0, A):
    # The Voigt line shape in terms of its physical parameters
    background = 0.0
    peak, center, widthD, widthL = param
    # f = np.sqrt(np.log(2))
    x = (x - center) / widthD
    y = widthL / widthD
    return peak / (widthD * np.sqrt(np.pi)) * _voigt(x, y) + background


def mvoigt(x, *params):
    y = np.ones_like(x)
    for i in range(0, len(params), 4):
        peak = params[i]
        center = params[i + 1]
        widthD = params[i + 2]
        widthL = params[i + 3]
        v1 = (x - center) / widthD
        v2 = widthL / widthD
        y = y - peak / (widthD * np.sqrt(np.pi)) * _voigt(v1, v2)
    return y


def curve_fit_mvoigt(x, y, p0):
    """
    Find the simple Gaussian fitting function
    """
    coeff, var_matrix = curve_fit(mvoigt, x, y, p0=p0)
    return list(coeff)


def fit_mGaussian(x, y, cx0, cy0, fwhm=0.15):
    spec = Spectrum1D(flux=(y - 1) * u.flx, spectral_axis=x * u.Angstrom)
    fit_dx = fwhm / 10
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    fit_sig1, fit_sig2 = sigma * 0.8, sigma * 1.2
    # make fitting models with Gaussian 
    p_init = models.Const1D(amplitude=0, fixed={'amplitude': True})
    for ix, iy in zip(cx0, cy0):
        p_init += models.Gaussian1D(iy - 1, ix, sigma,
                                    bounds={'amplitude': (-0.01, iy - 1),
                                            'mean': (ix - fit_dx * (ix / 6000.), ix + fit_dx * (ix / 6000.)),
                                            'stddev': (fit_sig1 * (ix / 6000.), fit_sig2 * (ix / 6000.))})
    p = fit_lines(spec, p_init)
    yfit = p(x * u.Angstrom).value + 1.0
    return p, yfit


def calc_EW_Voigt(x_0, amplitude_L, fwhm_L, fwhm_G):
    p = models.Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G)
    x = np.linspace(x_0 - fwhm_G * 20, x_0 + fwhm_G * 20, 2000)
    spec = Spectrum1D(flux=p(x) * u.flx, spectral_axis=x * u.Angstrom)
    return 1000 * line_flux(spec).value


def fit_mVoigt(x, y, cx0, cy0, fwhm_G=0.15, fwhm_L=0.1):
    spec = Spectrum1D(flux=(y - 1) * u.flx, spectral_axis=x * u.Angstrom)
    fit_dx = fwhm_G / 10
    fit_fwhm_G1, fit_fwhm_G2 = fwhm_G * 0.8, fwhm_G * 1.5
    fit_fwhm_L1, fit_fwhm_L2 = fwhm_L * 0.8, fwhm_L * 1.5
    # make fitting models with Voigt
    p_init = models.Const1D(amplitude=0, fixed={'amplitude': True})
    for ix, iy in zip(cx0, cy0):
        p_init += models.Voigt1D(ix, (iy - 1), fwhm_L, fwhm_G,
                                 bounds={'amplitude_L': (-0.01, (iy - 1) * 2),
                                         'x_0': (ix - fit_dx * (ix / 6000.), ix + fit_dx * (ix / 6000.)),
                                         'fwhm_L': (fit_fwhm_L1 * (ix / 6000.), fit_fwhm_L2 * (ix / 6000.)),
                                         'fwhm_G': (fit_fwhm_G1 * (ix / 6000.), fit_fwhm_G2 * (ix / 6000.))})
    p = fit_lines(spec, p_init)
    yfit = p(x * u.Angstrom).value + 1
    return p, yfit


def calc_EW_Gaussian(mean, amplitude, stddev):
    p = models.Gaussian1D(amplitude, mean, stddev)
    x = np.linspace(mean - stddev * 20, mean + stddev * 20, 2000)
    spec = Spectrum1D(flux=p(x) * u.flx, spectral_axis=x * u.Angstrom)
    return 1000 * line_flux(spec).value


#############################################################################
# LOCAL CONTINUUM FITTING WITH POLYNOMIAL
#############################################################################
def fcont(x, *p):
    LIMIT1, LIMIT2 = 1e-3, 1e-5
    # LIMIT1, LIMIT2 = 1e-1, 1e-2
    y = np.zeros_like(x)
    for i, param in enumerate(p):
        if i == 1:
            if p[i] > LIMIT1: param = LIMIT1
            if p[i] < -LIMIT1: param = -LIMIT1
        elif i == 2:
            if p[i] > LIMIT2: param = LIMIT2
            if p[i] < -LIMIT2: param = -LIMIT2

        y = y + param * x ** i
    return y


def local_continuum(xp, yp, lower=0.9, upper=1.4, niter=15):
    x0 = xp.mean()
    xp = xp - x0
    coeff, _ = curve_fit(fcont, xp, yp, p0=[yp.mean(), 0, 0])
    yc = fcont(xp, *coeff)
    vv = np.where((yp > yc * lower) & (yp < yc * upper))[0]
    xv, yv = xp, yp
    for i in range(niter):
        # print len(vv), yv.mean(), coeff
        if len(xv) < 5: break
        if len(vv) == len(xv): break
        xv, yv = xv[vv], yv[vv]
        coeff, _ = curve_fit(fcont, xv, yv, p0=[yv.mean(), 0, 0])
        yc = fcont(xv, *coeff)
        vv = np.where((yv > yc * lower) & (yv < yc * upper))[0]

    return fcont(xp, *coeff)


#############################################################################
# Spectrum utilities ===========================================================
############################################################################

def smooth(x, width=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')
    if x.size < width:
        raise ValueError('Input vector needs to be bigger than window size.')
    if width < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'barlett', 'blackman']:
        raise ValueError("Window is one of 'flat','hanning','hamming','barlett','blackman'")
    s = np.r_[x[(width - 1):0:-1], x, x[-2:(-width - 1):-1]]
    if window == 'flat':
        w = np.ones(width, 'd')
    else:
        w = eval('np.' + window + '(width)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(width / 2):-int(width / 2)]


def find_absorption0(x, y, width=5, thres=0.99, name='test'):
    # CHECK if width is odd
    if (width % 2) == 0:
        width += 1
        # CALC. deriv
    dy = smooth((y[1:] - y[:-1]) / (x[1:] - x[:-1]), width=width)
    dx = (x[1:] + x[:-1]) / 2.0
    ddy = smooth((dy[1:] - dy[:-1]) / (dx[1:] - dx[:-1]), width=width)
    ddx = (dx[1:] + dx[:-1]) / 2.0
    dddy = smooth((ddy[1:] - ddy[:-1]) / (ddx[1:] - ddx[:-1]), width=width)
    dddx = (ddx[1:] + ddx[:-1]) / 2.0

    ddy_cut = max(ddy) / 1e2  # detect 1/100 of max
    npts = len(dddy)
    xcs = []
    for i in range(1, npts - 1):
        p1, p2 = dddy[i - 1], dddy[i]
        x1, x2 = dddx[i - 1], dddx[i]
        dp, dx = abs(p2 - p1), x2 - x1

        if (p1 > 0) & (p2 < 0):  # & (dp > 1):
            xcs.append(x1 + dx * abs(p1 / dp))
    xcs = np.array(xcs)
    ddycs = np.interp(xcs, ddx, ddy)
    ycs = np.interp(xcs, x, y)
    vv = np.where((ycs < thres) & (ddycs > ddy_cut))[0]

    # fig, (ax1, ax2, ax3) = plt.subplots(num=54, nrows=3, figsize=(8,12), sharex=True)
    # ax1.plot(x, y,'.-')
    # ax1.plot(xcs, ycs,'bo',ms=12,alpha=0.5)
    # ax1.plot(xcs[vv], ycs[vv],'ro',ms=12,alpha=0.5)
    # ax2.plot(ddx, ddy,'.-')
    # ax2.grid()
    # ax3.plot(dddx, dddy,'.-')
    # ax3.grid()
    # fig.savefig(TDIR+name+'_deriv.png')
    # fig.clf()

    return xcs[vv], ycs[vv]


def find_absorption1(x, y, thres=0.5, width=13):
    sawtooth = np.array([0, 0, 0, -1, -2, -1, 0, 1, 2, 1, 0, 0, 0])
    w0 = len(sawtooth) - 4
    if (width % 2) == 0:
        width += 1
    frac = float(w0) / width
    npts = len(x)
    print(frac, npts)
    if frac > 1:
        ifrac = int(frac) + 1
        ind = np.arange(npts)
        find = np.linspace(0, npts - 1, (npts - 1) * ifrac + 1)
        fx = np.interp(find, ind, x)
        fy = np.interp(fx, x, y)
    else:
        domain = np.arange(0, (w0 + 4) + frac, frac)
        sawtooth = np.interp(domain, range(w0 + 4), sawtooth)
        fx, fy = x, y

    NPIX = len(fy)
    # READ the middle column/row in the image 
    wsize = len(sawtooth)

    sfy = np.r_[fy[(wsize - 1):0:-1], fy, fy[-2:(-wsize - 1):-1]]
    pcov = np.convolve(sawtooth, sfy, 'valid')[(wsize / 2 - 1):-(wsize / 2)]
    print(NPIX, len(pcov), wsize)
    dind = 0  # int(width/4)
    cx, cy = [], []
    for i in range(1, NPIX - 1):
        p1, p2 = pcov[i - 1], pcov[i]
        py = fy[i - 1 + dind] + (fy[i + dind] - fy[i - 1 + dind]) * p1 / (p1 - p2)
        px = fx[i - 1 + dind] + (fx[i + dind] - fx[i - 1 + dind]) * p1 / (p1 - p2)
        if (p1 > 0) & (p2 < 0) & (py < thres):
            cx.append(px)
            cy.append(py)
    return np.array(cx), np.array(cy)
    # return fx, fy, pcov


def find_absorption2(x, y, thres=-0.01):
    spec = Spectrum1D(flux=(y - 1) * u.flx, spectral_axis=x * u.Angstrom)
    lines = find_lines_derivative(spec, thres)
    cc = lines[lines['line_type'] == 'absorption']['line_center_index'].value

    return x[cc], y[cc]
