from __future__ import division
import numpy as np
import os
import random

from gibbs import Gibbs
from simulate_data import simulate_data

from enterprise.pulsar import Pulsar
import enterprise.constants as const
from enterprise.signals import parameter
from enterprise.signals import utils
from enterprise.signals import prior
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import signal_base

# base par and tim file
parfile = 'J1713+0747.par'
timfile = 'J1713+0747.tim'

# simulate data
idx = random.getrandbits(32)
simulate_data(parfile, timfile, idx)

# grab simulated data
simparfile = 'simulated_data/outliers/{}/J1713+0747.par'.format(idx)
simtimfile = 'simulated_data/outliers/{}/J1713+0747.tim'.format(idx)
psr = Pulsar(simparfile, simtimfile)

## Set up enterprise model ##

# white noise
efac = parameter.Constant(1.0)
equad = parameter.Uniform(-10, -5)
ecorr = parameter.Uniform(-10, -5)

# backend selection
selection = selections.Selection(selections.no_selection)

ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)

# red noise
pl = utils.powerlaw(log10_A=parameter.Uniform(-18,-12), gamma=parameter.Uniform(1,7))
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

# timing model
tm = gp_signals.TimingModel()

# combined signal
s = ef + eq  + rn + tm

# PTA
pta = signal_base.PTA([s(psr)])

## Set up different outlier models ##
mdls = {}

# emulate Vallisneri and van Haasteren mixture model
gibbs = Gibbs(pta, model='vvh17', vary_df=False, theta_prior='uniform',
               vary_alpha=False, alpha=1e10, pspin=0.00457)
mdls['vvh17'] = gibbs

# uniform theta distribution
gibbs = Gibbs(pta, model='mixture', vary_df=True, theta_prior='uniform')
mdls['uniform'] = gibbs

# beta theta distribution
gibbs = Gibbs(pta, model='mixture', vary_df=True, theta_prior='beta')
mdls['beta'] = gibbs

# Gaussian
gibbs = Gibbs(pta, model='gaussian', vary_df=True, theta_prior='beta')
mdls['gaussian'] = gibbs

# t-distribution
gibbs = Gibbs(pta, model='t', vary_df=True, theta_prior='beta')
mdls['t'] = gibbs

# sample and ouput chains
for key, md in list(mdls.items()):
    params = np.array([p.sample() for p in md.params]).flatten()
    niter = 10000
    md.sample(params, niter=niter)
    os.system('mkdir -p output/{}/{}/'.format(key, idx))

    np.save('output/{}/{}/chain.npy'.format(key, idx), md.chain[100:,:])
    np.save('output/{}/{}/bchain.npy'.format(key, idx), md.bchain[100:,:])
    np.save('output/{}/{}/zchain.npy'.format(key, idx), md.zchain[100:,:])
    np.save('output/{}/{}/thetachain.npy'.format(key, idx), md.thetachain[100:])
    np.save('output/{}/{}/alphachain.npy'.format(key, idx), md.alphachain[100:,:])
    np.save('output/{}/{}/dfchain.npy'.format(key, idx), md.dfchain[100:])
