#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:22:11 2024

@author: a2853
"""

import numpy as np
from scipy.optimize import curve_fit

# variables
h = 6.62607004e-34     # m^2 kg s^-1    , planck constant
c = 2.99792458e8       # m s^-1         , speed of light
q = 1.6021766208e-19   # joule          , eV to joule
hc = 1239.84198        # eV nm          , planck constant times speed of light in eV


# laser_flux = 5.42e21   # photons m^-2 s^-1
kt_at_RT = 25.7


def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) / (2 * stddev))**2)


def pl_analysis_1(wavelength, intensity):
    energy = np.flip(hc/wavelength)
    fit_range_mask_2 = (wavelength >= 700) & (wavelength <= 980)  # change range according to pl spectra
    x_range = wavelength[fit_range_mask_2]
    y_range = intensity[fit_range_mask_2]
    intensity = np.flip(intensity)

    fit_range_mask = (energy >= 1.2) & (energy <= 1.75)
    x_fit = energy[fit_range_mask]
    y_fit = intensity[fit_range_mask]

    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[0.3, 1.4, 0.05])
    # y_fit_2 = gaussian(x_fit, *popt)

    quantum_yield = np.trapz(y_range, x_range)
    pl_peak = popt[1]
    voc = 0.932 * pl_peak + kt_at_RT * np.log(quantum_yield) - 0.167

    return pl_peak, quantum_yield, voc


def pl_analysis_2(wavelength, intensity):
    pl_peak, quantum_yield, voc = None, None, None
    return pl_peak, quantum_yield, voc


def bandgap_1(wavelength_transmission, intensity_transmission, wavelength_reflection, intensity_reflection):
    bandgap = None
    return bandgap
