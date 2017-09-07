from __future__ import division
import numpy as np
import scipy.stats

import libstempo as t2
import libstempo.toasim as ts
import scipy.stats
import os

parfile = 'J1713+0747.par'
timfile = 'J1713+0747.tim'

def simulate_data(parfile, timfile, idx=0):

    pt = t2.tempopulsar(parfile, timfile)

    # draw error bars from log-normal distribution
    err = 10**(-7 + np.random.randn(len(pt.stoas)) * 0.2)*1e6

    # fake pulsar
    psr = ts.fakepulsar(parfile, pt.stoas[:], err)

    # red noise
    red = np.loadtxt('red.txt')
    psr.stoas[:] += red

    # outlier
    sigma_out = 1e-6
    theta = 0.05
    z = scipy.stats.binom.rvs(1, np.ones(len(psr.stoas))*theta)

    psr.stoas[:] += ((1-z) * err*1e-6 + z * sigma_out) * np.random.randn(len(psr.stoas)) / 86400
    psr.stoas[:][z.astype(bool)] -= red[z.astype(bool)]

    outdir = 'simulated_data/outliers/{}/'.format(idx)
    os.system('mkdir -p {}'.format(outdir))
    psr.savepar(outdir+'{}.par'.format(psr.name))
    psr.savetim(outdir+'{}.tim'.format(psr.name))
