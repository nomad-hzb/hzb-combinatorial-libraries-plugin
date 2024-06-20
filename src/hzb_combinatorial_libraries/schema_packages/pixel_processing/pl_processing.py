import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import math

h = 6.62607004e-34     # m^2 kg s^-1    , planck constant
c = 2.99792458e8       # m s^-1         , speed of light
q = 1.6021766208e-19   # joule          , eV to joule
hc = 1239.84198        # eV nm          , planck constant times speed of light in eV
#laser_flux = 5.42e21   # photons m^-2 s^-1
kt_at_RT = 25.7        # meV


# Define the Gaussian function
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) / (2 * stddev))**2)



class PL_funcs:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def gaussian(xg, amp, mean, stddev):
        return amp * np.exp(-((xg - mean) / (2 * stddev))**2)

    def fit_gaussian(self, x_data, y_data):
        # Restricting fit range between 1.2 and 1.5 eV
        fit_range_mask = (x_data >= 1.2) & (x_data <= 1.75)
        x_fit = x_data[fit_range_mask]
        y_fit = y_data[fit_range_mask]

        try:
            popt, pcov = curve_fit(self.gaussian, x_fit, y_fit, p0=[100, 1.45, 1])
            return popt
        except RuntimeError:
            return None

    def find_peak(self, x_data, y_data):
        popt = self.fit_gaussian(x_data, y_data)
        if popt is not None:
            peak_pos = popt[1]  # mean of the Gaussian is the peak position
            return peak_pos
        else:
            return None


def pl_data_processing(pl_data: dict):
    measurements = pl_data["measurements"]
    for measurement in measurements:
        measurement["data"] = np.array(measurement["data"]["intensity"])

    pl_df = pd.DataFrame(pl_data["measurements"])

    wavenumber = np.array(pl_data["wavelength"])
    energy=hc/np.array(wavenumber)

    pl_df['energy_data'] = pl_df['data'].apply(lambda data: hc * data / energy**2)

    # get pl integral
    for i in range(len(pl_df)):
        uv_data = pl_df.at[i, 'data']
        fit_range_mask = (wavenumber >= 700.0) & (wavenumber <= 980.0) # change range according to pl spectra
        x_range = wavenumber[fit_range_mask]
        y_range = uv_data[fit_range_mask]
        area = np.trapz(y_range, x_range)
        pl_df.at[i, 'integral_pl'] = area

    # get peak position and FWHM
    pl_analyzer = PL_funcs(pl_df)
    x_data = energy
    for i in range(len(pl_df)):
        y_data = pl_df.at[i, 'energy_data']
        # to find peak position
        popt = pl_analyzer.fit_gaussian(x_data, y_data)

        if popt is not None:
            A_fit, mu_fit, sigma_fit = popt # which should be used for max peak ? # a_fit hight, nu_fit mean, sigma_fit std deviation
            FWHM_fit = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
            pl_df.at[i, 'peak_pos'] = mu_fit
            pl_df.at[i, 'FWHM'] = FWHM_fit
        else:
            print(f"Error - curve_fit failed for row {i}.")
            pl_df.at[i, 'peak_pos'] = np.nan
            pl_df.at[i, 'FWHM'] = np.nan


    # get voc
    pl_df['voc'] = pl_df.apply(lambda row: 0.932 * row['peak_pos'] + kt_at_RT * math.log(row['integral_pl']) - 0.0167 if row['integral_pl'] > 0 else float('nan'), axis=1)

    # pl_df col names: [name	position_x	position_y	position_z	data	integral_pl	energy_data	peak_pos	FWHM	voc]
    return pl_df






