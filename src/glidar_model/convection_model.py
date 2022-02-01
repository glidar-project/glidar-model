from bdb import effective
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.units import units
from pytest import param
from scipy.integrate import BDF
from scipy import interpolate

# import time

from glidar_model.parcel_profile import ParcelProfile
from glidar_model.moist_cache import MoistCache


class AirProfile:
    """
    This is the class containing sounding data
    """

    def __init__(self, altitude, pressure, temperature, dewpoint):
        """
        The constructor taking the observation data

        :param altitude: the altitude
        :param pressure: the pressure
        :param temperature: air temperature
        """
        self.altitude = altitude
        self.pressure = pressure
        self.temperature = temperature
        self.dewpoint = dewpoint


class ModelParams:
    """
    The surface initial conditions. The units need to be included.
    """

    def __init__(self,
                 surface_temperature,
                 temperature_anomaly,
                 dew_point_temperature,
                 dew_point_anomaly,
                 thermal_altitude,
                 drag_coeff,
                 entrainment_coeff,
                 humidity_entrainment_coeff,
                 aspect_ratio=0,
                 quadratic_drag_coeff=0):
        self.surface_temperature = surface_temperature
        self.temperature_anomaly = temperature_anomaly
        self.dew_point_temperature = dew_point_temperature
        self.dew_point_anomaly = dew_point_anomaly
        self.thermal_altitude = thermal_altitude
        self.drag_coeff = drag_coeff
        self.quadratic_drag_coeff = quadratic_drag_coeff
        self.entrainment_coeff = entrainment_coeff
        self.humidity_entrainment_coeff = humidity_entrainment_coeff
        self.aspect_ratio = aspect_ratio


class ModelResult:
    """
    This is the class containing the result of a model run.

    All the shite should be including units.
    """

    def __init__(self):

        # Params that made this result
        self.background_profile = None
        self.parcel_profile = None
        self.params = None

        # Computed entrained profile
        self.entrained_profile = None

        # Vertical velocity profiles
        self.velocity_profile = None

        # Corresponding buoyancy profiles
        self.buoyancy = None


class ThermalModel:
    """
    This is the mighty class computing vertical velocity profiles from 
    temperature profiles.
    """
    g = 9.81  # the gravitational constant ms^-2

    def __init__(
        self,
        #  altitude,
        #  pressure,
        #  temperature,
        #  dewpoint,
        #  cached=True,
        ):
        """
        The constructor taking the observation data

        :param altitude: the altitude of the observations
        :param pressure: the pressure
        :param temperature: air temperature
        """
        # p = np.linspace(pressure[0], pressure[-1], pressure.size)

        # self.altitude = np.interp(p[::-1], pressure[::-1],
        #                           altitude[::-1])[::-1]
        # self.temperature = np.interp(p[::-1], pressure[::-1],
        #                              temperature[::-1])[::-1]
        # self.dewpoint = np.interp(p[::-1], pressure[::-1],
        #                           dewpoint[::-1])[::-1]
        # self.pressure = p

        # self.air_profile = AirProfile(self.altitude, self.pressure,
        #                               self.temperature, self.dewpoint)

        # self.moist_cache = None
        # if self.cached:
        #     self.moist_cache = MoistCache(self.pressure)

    @staticmethod
    def entrain_variable(background, parcel, altitude, coeff):

        ddp = parcel - background
        delta = ThermalModel.exp_decay(ddp.magnitude, altitude.magnitude,
                                       coeff) * ddp.units

        return background + delta

    @staticmethod
    def entrain_profile(background_profile, parcel_profile, params):

        temperature = ThermalModel.entrain_variable(
            background_profile.temperature, parcel_profile.temperature,
            background_profile.altitude, params.entrainment_coeff)

        dewpoint = ThermalModel.entrain_variable(
            background_profile.dewpoint, parcel_profile.dewpoint,
            background_profile.altitude, params.humidity_entrainment_coeff)

        return ParcelProfile(parcel_profile.pressure, temperature, dewpoint)

    def compute_theta(self, profile):

        return mpcalc.virtual_potential_temperature(
            profile.pressure, profile.temperature,
            mpcalc.mixing_ratio_from_specific_humidity(
                mpcalc.specific_humidity_from_dewpoint(profile.pressure,
                                                       profile.dewpoint)))

    @staticmethod
    def compute_buoyancy(theta_bar, theta):

        theta_prime = theta - theta_bar

        # The buoyancy computed directly from temperature perturbation
        buoyancy = ThermalModel.g * theta_prime / theta_bar
        return buoyancy

    def run_model(self, background_profile, parcel_profile,
                  params) -> ModelResult:
        """
        Calculating the vertical velocity profile
        """
        result = ModelResult()
        result.background_profile = background_profile
        result.parcel_profile = parcel_profile
        result.params = params

        entrained_profile = self.entrain_profile(background_profile,
                                                 parcel_profile, params)
        result.entrained_profile = entrained_profile

        theta_bar = self.compute_theta(background_profile)
        theta_ent = self.compute_theta(entrained_profile)

        buoyancy = self.compute_buoyancy(theta_bar, theta_ent)
        result.buoyancy = buoyancy

        effective_mass = self.calculate_back_pressure_term(params.aspect_ratio)
        effective_buoyancy = buoyancy * effective_mass

        w, z = self.riemann_calc_velocity_profile(
            effective_buoyancy,
            background_profile.altitude,
            params.thermal_altitude,
            alpha=params.drag_coeff * effective_mass,
            quadratic_drag=params.quadratic_drag_coeff * effective_mass,
        )
        result.velocity_profile = (w * units.meter / units.second,
                                   z * units.meter)

        return result

    @staticmethod
    def exp_decay(t_prime, z, gamma):
        """
        Approximates the entrainment of the temperature profile.
        """
        # z = self.altitude.magnitude

        dz = z[1:] - z[:-1]
        dt_prime = t_prime[1:] - t_prime[:-1]
        t = np.zeros_like(t_prime)
        t[0] = t_prime[0]
        for i, (dt, dz) in enumerate(zip(dt_prime, dz)):
            t[i + 1] = t[i] + dt - dz * gamma * t[i]
        return t

    @staticmethod
    def calculate_back_pressure_term(aspect_ratio):
        """
        Calculates the effective buoyancy according to Jeevanjee and Romps [1],
        assuming a cylindrical thermal with aspect ratio D/H.
        [1] https://doi.org/10.1002/qj.2683

        :param aspect_ratio: width devided by height of the thermal
        :return: effective buoyancy
        """
        effective_buoyancy = 1 / np.sqrt(1 + aspect_ratio**2)
        return effective_buoyancy

    @staticmethod
    def riemann_calc_velocity_profile(buoyancy,
                                      altitude,
                                      z_0,
                                      alpha=0.00,
                                      quadratic_drag=0):
        """
        Found a way how to simplify the integration

        :param buoyancy: with units
        :param altitude: with units
        :param z_0:
        :param alpha:
        :param debug:
        :return:
        """
        buoyancy = buoyancy.magnitude
        altitude = altitude.to('meter').magnitude
        z_0 = z_0.to('meter').magnitude

        dz = (altitude[1:] - altitude[:-1])
        w2 = np.zeros_like(altitude)

        i0 = 0
        iii = np.where(altitude > z_0)[0]
        if len(iii):
            i0 = iii[0]

        for i, dz in enumerate(dz):

            # skip the values below the starting altitude
            if i < i0:
                continue

            # Integrate the buoyancy contributions
            w = w2[i] + (buoyancy[i] - alpha * np.sqrt(w2[i]) -
                         quadratic_drag * w2[i]) * 2 * dz

            # Break if vertical velocity becomes negative
            if w < 0:
                break
            w2[i + 1] = w

        return np.sqrt(w2[i0:i + 1]), altitude[i0:i + 1]


class DryThermalModel(ThermalModel):

    def __init__(self):
        super().__init__()

    def compute_theta(self, profile):

        return mpcalc.potential_temperature(profile.pressure,
                                            profile.temperature)
