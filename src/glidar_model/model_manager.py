
import metpy.calc as mpcalc
from metpy.units import units

from glidar_model.background_profile import BackgrounProfile
from glidar_model.convection_model import ThermalModel
from glidar_model.moist_cache import MoistCache
from glidar_model.parcel_profile import ParcelProfile


class Model():

    def __init__(self, background_profile: BackgrounProfileBuilder, parcel_profile: ParcelProfileBuilder, cached=False) -> None:
        
        self.background_profile = background_profile
        self.parcel_profile = parcel_profile

        # Chache handling
        self.cached = cached
        self.moist_cache = None
        if self.cached:
            self.moist_cache = MoistCache(self.background_profile.pressure)

        self.thermal_model = ThermalModel

    def get_parcel_profile(self, pressure, temperature, dew_point):
        """
        Helper function calling MetPy to calculate temperature profiles,
        or returning cached results.

        :param temperature: surface temperature
        :param dew_point: surface dew point
        :return: temperature profile
        """
        if self.cached:
            return self.moist_cache.get_cached_profile(temperature, dew_point)

        res = mpcalc.parcel_profile(pressure,
                                    temperature,
                                    dew_point).to(units.kelvin)

        return res

    def run_model(self, params):

        background_profile = self.background_profile.get_profile(params)
        parcel_profile = self.parcel_profile.get_profile(params)

        self.thermal_model.run_model(background_profile, parcel_profile, params)





