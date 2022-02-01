import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from scipy.integrate import BDF
from scipy import interpolate

from glidar_model.background_profile import BackgroundProfile, BackgroundProfileBuilder
from glidar_model.moist_cache import MoistCache
from glidar_model.synthetic_profile import constant_mixing_RH_profile
from glidar_model.utils import combine_pint_arrays


def compute_added_water(sounding, adjusted):

    mx_obs = mpcalc.mixing_ratio_from_relative_humidity(
        sounding.pressure, sounding.temperature,
        mpcalc.relative_humidity_from_dewpoint(sounding.temperature,
                                               sounding.dewpoint))

    mx_bar = mpcalc.mixing_ratio_from_relative_humidity(
        adjusted.pressure, adjusted.temperature,
        mpcalc.relative_humidity_from_dewpoint(adjusted.temperature,
                                               adjusted.dewpoint))

    water_obs = mpcalc.density(sounding.pressure, sounding.temperature,
                               mx_obs) * mx_obs
    water_bar = mpcalc.density(sounding.pressure, adjusted.temperature,
                               mx_bar) * mx_bar

    added_water = np.trapz(water_bar - water_obs, sounding.altitude)

    return added_water


def adjust_sounding_profile_helper(sounding,
                                   surface_temperature,
                                   dew_point_temperature,
                                   get_parcel_profile=mpcalc.parcel_profile):

    t_bar = get_parcel_profile(sounding.pressure, surface_temperature,
                               dew_point_temperature)

    idx = np.where(t_bar < sounding.temperature)[0]
    if idx.size > 0:
        idx = idx[0]
        t_bar = combine_pint_arrays(t_bar, sounding.temperature, idx)
    else:
        idx = None

    # Calculate the base RH profile and the corresponding virtual
    # potential temperature
    RH_bar = constant_mixing_RH_profile(
        sounding.pressure, t_bar,
        mpcalc.relative_humidity_from_dewpoint(surface_temperature,
                                               dew_point_temperature))

    rh = mpcalc.relative_humidity_from_dewpoint(sounding.temperature,
                                                sounding.dewpoint)

    if idx is not None:
        RH_bar = combine_pint_arrays(RH_bar, rh, idx)

    dewpoint = mpcalc.dewpoint_from_relative_humidity(t_bar, RH_bar)

    return BackgroundProfile(sounding.pressure, t_bar, dewpoint,
                             sounding.altitude)


def adjust_sounding_profile(sounding, surface_temperature,
                            dew_point_temperature):

    return adjust_sounding_profile_helper(sounding, surface_temperature,
                                          dew_point_temperature)


class SoundingProfileBuilder(BackgroundProfileBuilder):

    def __init__(self, sounding, cache: MoistCache = None) -> None:

        self.sounding = sounding
        self.cache = cache

    def _get_parcel_profile(self, pressure, surface_temperature,
                           dew_point_temperature):

        if self.cache is None:
            return mpcalc.parcel_profile(pressure, surface_temperature,
                                         dew_point_temperature)

        # Here I assume that the cache has compatible pressure
        # range with the requested profile
        return self.cache.get_cached_profile(surface_temperature,
                                             dew_point_temperature)

    def get_profile(self, params) -> BackgroundProfile:

        return adjust_sounding_profile_helper(
            self.sounding,
            params.surface_temperature,
            params.dew_point_temperature,
            get_parcel_profile=self._get_parcel_profile)
