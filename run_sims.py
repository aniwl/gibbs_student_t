#!/usr/bin/env python

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


@signal_base.function
def svd_tm_basis(Mmat):
    u, s, v = np.linalg.svd(Mmat, full_matrices=False)
    return u, np.ones_like(s)

@signal_base.function
def tm_prior(weights):
    return weights * 10**40

# base par and tim file
parfile = 'J1713+0747.par'
timfile = 'J1713+0747.tim'

thetas = [0.05, 0.1, 0.15]
for theta in thetas:

    # simulate data
    idx = random.getrandbits(32)
    sigma_out = 1e-6
    simulate_data(parfile, timfile, theta=theta, idx=idx,
                  sigma_out=sigma_out)

    # grab simulated data
    simparfile = 'simulated_data/outlier/{}/{}/J1713+0747.par'.format(theta, idx)
    simtimfile = 'simulated_data/outlier/{}/{}/J1713+0747.tim'.format(theta, idx)
    psr = Pulsar(simparfile, simtimfile)

    simparfile = 'simulated_data/no_outlier/{}/{}/J1713+0747.par'.format(theta, idx)
    simtimfile = 'simulated_data/no_outlier/{}/{}/J1713+0747.tim'.format(theta, idx)
    psr2 = Pulsar(simparfile, simtimfile)

    psrs = [psr, psr2]
    ## Set up enterprise model ##

    # white noise
    efac = parameter.Constant(1.0)
    equad = parameter.Uniform(-10, -5)

    # backend selection
    selection = selections.Selection(selections.no_selection)

    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)

    # red noise
    pl = utils.powerlaw(log10_A=parameter.Uniform(-18,-12), gamma=parameter.Uniform(1,7))
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

    # timing model
    basis = svd_tm_basis()
    prior = tm_prior()
    tm = gp_signals.BasisGP(prior, basis)

    # combined signal
    s = ef + eq + rn + tm

    outdirs = ['output_outlier', 'output_no_outlier']

    for psr, outdir in zip(psrs, outdirs):

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
            out = '{}/{}/{}/{}/'.format(outdir, key, theta, idx)
            print out
            os.system('mkdir -p {}'.format(out))

            np.save('{}/chain.npy'.format(out), md.chain[100:,:])
            np.save('{}/bchain.npy'.format(out), md.bchain[100:,:])
            np.save('{}/zchain.npy'.format(out), md.zchain[100:,:])
            np.save('{}/poutchain.npy'.format(out), md.poutchain[100:,:])
            np.save('{}/thetachain.npy'.format(out), md.thetachain[100:])
            np.save('{}/alphachain.npy'.format(out), md.alphachain[100:,:])
            np.save('{}/dfchain.npy'.format(out), md.dfchain[100:])
