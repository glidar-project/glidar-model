import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from scipy.integrate import BDF
from scipy import interpolate

from glidar_model.moist_cache import MoistCache

from glidar_model.utils import combine_pint_arrays


class BackgrounProfile:

    def __init__(self, pressure, temperature, dew_point, altitude) -> None:

        self.pressure = pressure
        self.temperature = temperature
        self.dew_point = dew_point
        self.altitude = altitude


def synthetic_temperature_profile(pressure, surface_temperature,
                                  inversion_delta,
                                  inversion_height) -> np.array:
    """
    Fakes a temperature profile with inversion at the given hight and strength.
    The bottom profile is dry adiabatic.
    """

    p_inversion = mpcalc.height_to_pressure_std(inversion_height)

    potential_temperature = np.where(
        pressure > p_inversion, surface_temperature * np.ones_like(pressure),
        surface_temperature + inversion_delta * np.ones_like(pressure))

    temperature = mpcalc.temperature_from_potential_temperature(
        pressure, potential_temperature)

    return temperature


def constant_mixing_RH_profile(pre, tmp, rh_surface):
    """
    Calculates the relative humidity profile.
    It computes the mixing ratio at the surface level and 
    then extrapolateds the values for RH using a constant 
    mixing ratio. After condensation the RH is capped.
    """
    mx_rel_surface = mpcalc.mixing_ratio_from_relative_humidity(
        pre[0], tmp[0], rh_surface)
    RH = mpcalc.relative_humidity_from_mixing_ratio(pre, tmp, mx_rel_surface)
    RH[RH > 1] = 1

    return RH


def synthetic_dew_point_profile(pressure, temperature, surfcace_dew_point,
                                inversion_height, inversion_dew_point):
    """
    Fakes the dewpoint profile using constant mixing ratio.
    """

    p_inversion = mpcalc.height_to_pressure_std(inversion_height)

    rh_bottom = constant_mixing_RH_profile(
        pressure, temperature,
        mpcalc.relative_humidity_from_dewpoint(temperature[0],
                                               surfcace_dew_point))
    rh_top = constant_mixing_RH_profile(
        pressure, temperature,
        mpcalc.relative_humidity_from_dewpoint(temperature[0],
                                               inversion_dew_point))

    rh = np.where(pressure > p_inversion, rh_bottom, rh_top)

    dewpoint = mpcalc.dewpoint_from_relative_humidity(temperature, rh)

    return dewpoint


def synthetic_profile(surface_temperature, inversion_delta, inversion_height,
                      surface_dew_point,
                      inversion_dew_point) -> BackgrounProfile:

    pressure = np.linspace(1000, 800, 200) * units.hPa
    altitude = mpcalc.pressure_to_height_std(pressure)

    temperature = synthetic_temperature_profile(pressure, surface_temperature,
                                                inversion_delta,
                                                inversion_height)

    dew_point = synthetic_dew_point_profile(pressure, temperature,
                                            surface_dew_point,
                                            inversion_height,
                                            inversion_dew_point)

    result = BackgrounProfile(pressure, temperature, dew_point, altitude)
    return result


def adjust_sounding_profile_helper(sounding, surface_temperature,
                            dew_point_temperature, get_parcel_profile=mpcalc.parcel_profile):

    t_bar = get_parcel_profile(sounding.pressure, surface_temperature, dew_point_temperature)

    idx = np.where(t_bar < sounding.temperature)[0]
    if idx.size > 0:
        idx = idx[0]
        t_bar = combine_pint_arrays(t_bar, sounding.temperature, idx)
    else:
        idx = None

    # Calculate the base RH profile and the corresponding virtual potential temperature
    RH_bar = constant_mixing_RH_profile(
        sounding.pressure, t_bar,
        mpcalc.relative_humidity_from_dewpoint(surface_temperature,
                                               dew_point_temperature))

    rh = mpcalc.relative_humidity_from_dewpoint(sounding.pressure,
                                                sounding.dewpoint)

    if idx is not None:
        RH_bar = combine_pint_arrays(RH_bar, rh, idx)

    mx_obs = mpcalc.mixing_ratio_from_relative_humidity(
        sounding.pressure, sounding.temperature, rh)
    mx_bar = mpcalc.mixing_ratio_from_relative_humidity(
        sounding.pressure, t_bar, RH_bar)

    water_obs = mpcalc.density(sounding.pressure, sounding.temperature,
                               mx_obs) * mx_obs
    water_bar = mpcalc.density(sounding.pressure, t_bar, mx_bar) * mx_bar

    added_water = np.trapz(water_bar - water_obs, sounding.altitude)

    return BackgrounProfile(sounding.pressure, t_bar, )


def adjust_sounding_profile(sounding, surface_temperature,
                            dew_point_temperature):

    return adjust_sounding_profile_helper(sounding, surface_temperature,
                            dew_point_temperature)