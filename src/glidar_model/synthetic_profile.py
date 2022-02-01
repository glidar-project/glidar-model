from os import stat
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from glidar_model.background_profile import BackgroundProfile, BackgroundProfileBuilder


def synthetic_temperature_profile(pressure, surface_temperature,
                                  inversion_delta,
                                  inversion_height) -> np.array:
    """
    Fakes a temperature profile with in at the given hight and strength.
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


##############################################################################
#
#   Builder Class definition here
#
class SyntheticProfileParams:

    def __init__(self, surface_temperature, inversion_delta, inversion_height,
                 dew_point_temperature, inversion_dew_point) -> None:

        self.surface_temperature = surface_temperature
        self.inversion_delta = inversion_delta
        self.inversion_height = inversion_height
        self.dew_point_temperature = dew_point_temperature
        self.inversion_dew_point = inversion_dew_point


class SyntheticProfileBuilder(BackgroundProfileBuilder):

    def __init__(self, pressure=None) -> None:

        self.pressure = pressure
        if self.pressure is None:
            self.pressure = np.linspace(1000, 800, 200) * units.hPa

        self.altitude = mpcalc.pressure_to_height_std(self.pressure)

    def get_profile(self, params) -> BackgroundProfile:

        return self.synthetic_profile(self.pressure, self.altitude,
                                      params.surface_temperature,
                                      params.inversion_delta,
                                      params.inversion_height,
                                      params.dew_point_temperature,
                                      params.inversion_dew_point)

    @staticmethod
    def synthetic_profile(pressure, altitude, surface_temperature,
                          inversion_delta, inversion_height, surface_dew_point,
                          inversion_dew_point) -> BackgroundProfile:

        temperature = synthetic_temperature_profile(pressure,
                                                    surface_temperature,
                                                    inversion_delta,
                                                    inversion_height)

        dew_point = synthetic_dew_point_profile(pressure, temperature,
                                                surface_dew_point,
                                                inversion_height,
                                                inversion_dew_point)

        result = BackgroundProfile(pressure, temperature, dew_point, altitude)
        return result
