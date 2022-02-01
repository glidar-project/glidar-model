import pytest
import numpy as np

import metpy.calc as mpcalc

from metpy.units import units

from glidar_model.convection_model import ModelParams
from glidar_model.moist_cache import MoistCache
from glidar_model.parcel_profile import CachedParcelProfile, ParcelProfile, ParcelProfileBuilder, SimpleParcelProfile


@pytest.fixture
def params():

    return ModelParams(10 * units.celsius, 1 * units.delta_degree_Celsius,
                       0 * units.celsius, 0 * units.delta_degree_Celsius,
                       100 * units.meter, 0, 0, 0)


@pytest.fixture
def pressure():

    return np.linspace(1000, 800, 100) * units.hPa


@pytest.fixture
def temperature(pressure):

    return mpcalc.temperature_from_potential_temperature(
        pressure, 10 * np.ones_like(pressure) * units.celsius)


def test_abstract_class_invocation():

    with pytest.raises(NotImplementedError):
        ParcelProfileBuilder()


def test_parcel_profile(pressure, params):

    pp = SimpleParcelProfile(pressure)

    t = pp._get_parcel_profile(pressure, 10 * units.celsius, 5 * units.celsius)
    p = pp.get_profile(params)


def test_cached_profile(pressure, params):

    pp = SimpleParcelProfile(pressure)

    t = pp._get_parcel_profile(pressure, params.surface_temperature,
                              params.dew_point_temperature)
    p = pp.get_profile(params)

    cache = MoistCache(pressure)
    cpp = CachedParcelProfile(pressure, cache)

    profile = cpp.get_profile(params)

    assert np.allclose(p.temperature.magnitude, profile.temperature.magnitude)
    assert np.allclose(p.dewpoint.magnitude, profile.dewpoint.magnitude)
