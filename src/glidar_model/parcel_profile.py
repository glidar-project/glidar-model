import logging
import metpy.calc as mpcalc

from glidar_model.moist_cache import MoistCache
from glidar_model.synthetic_profile import constant_mixing_RH_profile


class ParcelProfile:

    def __init__(self, pressure, temperature, dewpoint) -> None:

        self.pressure = pressure
        self.temperature = temperature
        self.dewpoint = dewpoint


class ParcelProfileBuilder:

    def __init__(self) -> None:
        raise NotImplementedError('This is an abstract class.')

    def get_profile(self, params):
        raise NotImplementedError('This is an abstract class.')


class SimpleParcelProfile(ParcelProfileBuilder):

    def __init__(self, pressure) -> None:
        self.pressure = pressure

    def get_profile(self, params):

        t0 = params.surface_temperature + params.temperature_anomaly
        dp0 = params.dew_point_temperature + params.dew_point_anomaly

        temperature = self._get_parcel_profile(self.pressure, t0, dp0)

        dewpoint = mpcalc.dewpoint_from_relative_humidity(
            temperature,
            constant_mixing_RH_profile(
                self.pressure, temperature,
                mpcalc.relative_humidity_from_dewpoint(t0, dp0)))

        return ParcelProfile(self.pressure, temperature, dewpoint)

    def _get_parcel_profile(self, pressure, temperature, dewpoint):

        return mpcalc.parcel_profile(pressure, temperature, dewpoint)


class CachedParcelProfile(SimpleParcelProfile):

    def __init__(self, pressure, cache: MoistCache) -> None:
        super().__init__(pressure)
        self.cache = cache

    def _get_parcel_profile(self, pressure, temperature, dewpoint):

        # TODO: There might be an idea to resample the chached profile
        # to the given pressure. For now I just assume that the requested
        # and cahched pressure are the same array.
        return self.cache.get_cached_profile(temperature, dewpoint)


class ObservedParcelProfile(ParcelProfileBuilder):

    def __init__(self, pressure, temperature, dewpoint) -> None:

        self.pressure = pressure
        self.temperature = temperature
        self.dewpoint = dewpoint

    def get_profile(self, params):

        return self
