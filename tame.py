# -*- coding: utf-8 -*-
"""
## 20160609 wskang
start TAME in python 
## 20180517 wskang 
change the line detection method. 
apply multi-Gaussian fitting method by astropy.modeling (from curve_fit in scipy) 
"""
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob
from speclib import find_elname, find_absorption2, find_absorption1, find_absorption0, \
        local_continuum, fit_mGaussian, fit_mVoigt, read_params, \
        calc_EW_Gaussian, calc_EW_Voigt
from matplotlib.backends.backend_pdf import PdfPages

par = read_params()
LINEFILE = par['LINEFILE']
SPECFILE = par['SPECFILE']
OUTPUT = par['OUTPUT']

N_MAX = int(par['NMAX'])
FWHMG = float(par['FWHMG'])
FWHML = float(par['FWHML'])
DWV = float(par['DWV'])
SNR = float(par['SNR'])
STRONGs = np.array(par['RVLINE'].split(','), float)

lineinfo = np.genfromtxt(LINEFILE, usecols=(0,1,2,3))
WAVs, ELEs, EPs, LGFs = lineinfo[:,0], lineinfo[:,1], lineinfo[:,2], lineinfo[:,3]
dat = np.genfromtxt(SPECFILE)
# IDENTIFY the aperture column
slam, sap, sint = dat[:, 0], dat[:, 1], dat[:, 2]
nrow, nset = len(sap), len(set(sap))
if nrow/nset < 10:
    sap, sint = sint, sap
apset = np.array(sorted(set(sap)))
aplam = [] 
for iap in apset:
    aa = np.where(sap == iap)[0]
    aplam.append(np.mean(slam[aa]))
aplam = np.array(aplam)

few = open(OUTPUT+'.ews','w')
pew = PdfPages(OUTPUT+'.pdf') 
xo = slam[:]
yo = sint[:]

# CHECK RV
RVs = [] 
for swv in STRONGs:
    mm = np.argmin(abs(aplam-swv))
    ap = apset[mm]
    rr = np.where((sap == ap) & (xo > swv-DWV) & (xo < swv+DWV))[0]
    xr, yr = xo[rr], yo[rr]
    cx, cy = find_absorption0(xr, yr, thres=(1-1/SNR))
    #print cx, cy
    if len(cx) == 0: continue
    mm = np.argmin(abs(cx-swv))
    print( swv, cx[mm])
    vr = (xr-swv)/swv * 2.9979e5
    cv = (cx-swv)/swv * 2.9979e5
    fig = plt.figure(num=1, figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(vr, yr, 'k.')
    ax.plot(cv, cy, 'bo', alpha=0.4)
    rv1 = cv[mm]
    ax.plot([0,0],[0,1], 'r-', lw=1)
    ax.plot([rv1,rv1],[0,1], 'b--', lw=3, alpha=0.4, label='RV=%.3f km/s' % (rv1,))
    ax.set_xlim(min(vr), max(vr))
    ax.set_ylim(0, 1.2)
    ax.grid()
    ax.legend()
    fig.savefig(f"{OUTPUT}_RV{swv:.0f}.png")
    fig.clf()
    RVs.append(rv1)
if len(RVs) > 0:  RV = np.mean(RVs)
else: RV = 0.0 
# Calibrate the RV ===========================
xo = xo * (1-(RV/2.9979e5))
print( RVs)
# LOOP for lines ========================================    
srv = []
for iline, lwv in enumerate(WAVs):
    # elements name 
    lel, lep, lloggf = ELEs[iline], EPs[iline], LGFs[iline]
    elname = find_elname(lel)
    print( '%s %.2f' % (elname, lwv))
    # CROP the region around line to be estimated ---------
    mm = np.argmin(abs(aplam-lwv))
    ap = apset[mm]
    rr = np.where((sap == ap) & (xo > lwv-DWV) & (xo < lwv+DWV))[0]
    #### SKIP for the case of NO SPECTRUM
    if len(rr) == 0: continue

    xr, yr = xo[rr], yo[rr]
    ymin1, ymax1 = min(yr), max(yr)
    ymin1, ymax1 = ymin1-(ymax1-ymin1)/10, ymax1+(ymax1-ymin1)/10
    
    # CORRECT local continuum -------------------------
    ycont = local_continuum(xr, yr, lower=(1-1.0/SNR), niter=N_MAX)
    yrc = yr/ycont
    
    # # CROP the region with lines ----------------------
    # ff = np.where((yrc > (1-1.0/SNR)) & (xr < lwv))[0]
    # fcut1 = max(list(ff)+[4,])
    # ff = np.where((yrc > (1-1.0/SNR)) & (xr > lwv))[0]
    # fcut2 = min(list(ff)+[len(xr)-4,])
    # #### SKIP TOO NARROW
    # #if (fcut2-fcut1) < WIDTH*5: continue 
    # xrf, yrf = xr[(fcut1-4):(fcut2+4)], yrc[(fcut1-4):(fcut2+4)]
    xrf, yrf = xr, yr
    ymin2, ymax2 = min(yrf), max(yrf)
    ymin2, ymax2 = ymin2-(ymax2-ymin2)/10, ymax2+(ymax2-ymin2)/10
    
    # FIND lines --------------------------------------
    cx0, cy0 = find_absorption0(xrf, yrf, thres=(1-1/SNR))
    NLINES = len(cx0)

    #### SKIP for the case of NO LINE
    if NLINES == 0: continue
    
    # FIT absorptions ---------------------------------
    if FWHML > 0:
        p, yfit = fit_mVoigt(xrf, yrf, cx0, cy0, fwhm_G=FWHMG, fwhm_L=FWHML)
        amplitude_ls, x_0s, fwhm_ls, fwhm_gs, ews = [], [], [], [], []
        for ifit in range(NLINES):
            x_0s.append(p.parameters[ifit*4+1])
            amplitude_ls.append(abs(p.parameters[ifit*4+2]))
            fwhm_ls.append(p.parameters[ifit*4+3])
            fwhm_gs.append(p.parameters[ifit*4+4])
            
            EW = calc_EW_Voigt(x_0s[-1], amplitude_ls[-1], fwhm_ls[-1], fwhm_gs[-1])
            ews.append(EW)
            print('%.2f %8.2f %8.2f %8.2f' % (x_0s[-1], EW, fwhm_ls[-1], fwhm_gs[-1]))
        means = np.array(x_0s)
        mm = np.argmin(abs(x_0s - lwv))
        lrv = (x_0s[mm]-lwv)/lwv*2.9979e5
        print(f'>> {x_0s[mm]:.2f} {ews[mm]:8.3f} {lrv:8.3f} {fwhm_gs[mm]:8.3f} {fwhm_ls[mm]:8.3f}\n')
        few.write(f'{lwv:10.3f} {lel:9.1f} {lep:9.3f} {lloggf:10.4f} {ews[mm]:29.2f} ' +
                  f'{means[mm]:10.3f} {lrv:10.3f} {fwhm_gs[mm]:10.3f} {fwhm_ls[mm]:10.3f}\n')

    else:
        p, yfit = fit_mGaussian(xrf, yrf, cx0, cy0, fwhm=FWHMG)
        amplitudes, means, stddevs, fwhms, ews = [], [], [], [], []
        for ifit in range(NLINES):
            amplitudes.append(abs(p.parameters[ifit*3+1]))
            stddevs.append(p.parameters[ifit*3+3])
            means.append(p.parameters[ifit*3+2])
            
            FWHM = (2*np.sqrt(2*np.log(2)))*stddevs[-1]
            EW1 = 1000*np.sqrt(2*np.pi)*amplitudes[-1]*stddevs[-1]
            EW = calc_EW_Gaussian(means[-1], amplitudes[-1], stddevs[-1])
            fwhms.append(FWHM)
            ews.append(EW)
            print(f'{means[-1]:.2f} {EW:8.2f} {FWHM:8.2f} {EW1:8.2f}')
        means = np.array(means)
        mm = np.argmin(abs(means-lwv))
        lrv = (means[mm]-lwv)/lwv*2.9979e5
        print(f'>> {means[mm]:.2f} {ews[mm]:8.3f} {lrv:8.3f} {fwhms[mm]:8.3f}\n')
        few.write(f'{lwv:10.3f} {lel:9.1f} {lep:9.3f} {lloggf:10.4f} {ews[mm]:29.2f} ' +
                  f'{means[mm]:10.3f} {lrv:10.3f} {fwhms[mm]:10.3f}\n')
    
    # PLOT the line 
    fig, (ax1, ax2) = plt.subplots(nrows=2, num=2, figsize=(6,8))
    ax1.plot(xr, yr, 'k.-')
    ax1.plot(xr, ycont, 'r-')
    ax1.plot([lwv, lwv], [ymin2, ymax2], 'k-', lw=2, alpha=0.4)
    ax2.plot(xr, yrc, 'k.-',alpha=0.6)
    ax2.plot([lwv,lwv], [ymin2, ymax2], 'k-', lw=2, alpha=0.4)
    for lidx, ix in enumerate(cx0):
        ax2.text(ix, ymin2, '%d' % (lidx+1,), color='g', fontsize=5)
        if FWHML > 0: 
            ax2.plot([ix,ix], [ymin2, ymax2], '--', lw=1, alpha=0.6,
                     label=f'{lidx + 1:d} {x_0s[lidx]:.2f} {ews[lidx]:7.2f} {fwhm_gs[lidx]:7.2f} {fwhm_ls[lidx]:7.2f}')
        else:
            ax2.plot([ix,ix], [ymin2, ymax2], '--', lw=1, alpha=0.6,
                     label=f'{lidx + 1:d} {means[lidx]:.2f} {ews[lidx]:7.2f} {fwhms[lidx]:7.2f}')
    ax2.plot(xrf, yfit, 'r-', lw=3, alpha=0.6)
    
    ax1.set_title('%s %.3f' % (elname,lwv))
    ax1.set_ylim(ymin1, ymax1)
    ax2.set_ylim(ymin2-(ymax2-ymin2)*0.2, ymax2)
    ax1.grid()
    ax2.grid()
    ax2.legend(loc='lower left', fontsize=4, numpoints=1, ncol=4)
    ax2.tick_params(axis='x', labelsize=7)
    pew.savefig(fig)
    plt.close(fig)

few.close()
pew.close()






