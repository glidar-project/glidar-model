

import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from scipy.integrate import BDF
from scipy import interpolate

from glidar_model.moist_cache import MoistCache




def adjust_sounding_profile(surface_temperature, dew_point_temperature):

    # Calculate the base state parcel profile called t_bar
    t_bar = self.calc_temperature_profile(mp.surface_temperature, mp.dew_point_temperature)
    if use_sounding:
        idx = np.where(t_bar < tmp)[0]
        if idx.size > 0:
            idx = idx[0]
            t_bar = Utils.combine_pint_arrays(t_bar, tmp, idx)
        else:
            idx = None
    result.t_bar = t_bar

    # Calculate the base RH profile and the corresponding virtual potential temperature
    RH_bar = self.calculate_RH_profile(pre, t_bar, mpcalc.relative_humidity_from_dewpoint(
                                                mp.surface_temperature,
                                                mp.dew_point_temperature))
    if use_sounding:
        rh = mpcalc.relative_humidity_from_dewpoint(tmp, self.dewpoint)
        if idx is not None:
            # RH_bar[idx:] = rh[idx:]
            RH_bar = Utils.combine_pint_arrays(RH_bar, rh, idx)

        mx_obs = mpcalc.mixing_ratio_from_relative_humidity(pre, tmp, rh)
        mx_bar = mpcalc.mixing_ratio_from_relative_humidity(pre, t_bar, RH_bar)

        result.base_mixing_line = mx_bar

        water_obs = mpcalc.density(pre, tmp, mx_obs) * mx_obs
        water_bar = mpcalc.density(pre, t_bar, mx_bar) * mx_bar

        result.added_water = np.trapz(
            water_bar - water_obs,
            self.altitude
        )