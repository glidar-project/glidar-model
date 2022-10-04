import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from scipy.integrate import BDF
from scipy import interpolate

from glidar_model.background_profile import BackgroundProfile, BackgroundProfileBuilder


def resampled_imet_backround(imet):

    # Only ascent
    imet = imet.iloc[:imet.altitude.idxmax()].copy()

    imet.sort_values('pressure', inplace=True, ascending=False)
    imet.reset_index(inplace=True)

    pressure = imet.pressure.values
    p = np.linspace(pressure[0], pressure[-1], pressure.size)

    altitude = np.interp(p[::-1], pressure[::-1],
                         imet.altitude.values[::-1])[::-1]

    temperature = np.interp(p[::-1], pressure[::-1],
                            imet.temperature.values[::-1])[::-1]

    dewpoint = np.interp(p[::-1], pressure[::-1],
                         imet.dewpoint.values[::-1])[::-1]

    pressure = p

    profile = BackgroundProfile(
        pressure * units.pascal,
        (temperature * units.celsius).to(units.degree_Kelvin),
        (dewpoint * units.celsius).to(units.degree_Kelvin),
        altitude * units.meter,
    )

    return profile

class iMetProfileBuilder(BackgroundProfileBuilder):

    def __init__(self, profile):

        self.imet_profile = resampled_imet_backround(profile)

    def get_profile(self, params) -> BackgroundProfile:
        
        return self.profile