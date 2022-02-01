

import pytest
import numpy as np

import metpy.calc as mpcalc

from metpy.units import units

from glidar_model.moist_cache import MoistCache


@pytest.fixture
def pressure():

    return np.linspace(1000, 800, 100) * units.hPa


@pytest.fixture
def cache(pressure):

    return MoistCache(pressure)


def test_zero_celsius(cache, pressure):

    temperature = 0 * units.celsius
    dewpoint = -5 * units.celsius


    profile = cache.get_cached_profile(temperature, dewpoint)
    mp_profile = mpcalc.parcel_profile(pressure, temperature, dewpoint).to(profile.units)
    
    assert np.max(np.abs(mp_profile.magnitude - profile.magnitude)) < 0.001


def test_moist_profile(cache, pressure):

    temperature = 10 * units.celsius
    dewpoint = 0 * units.celsius

    profile = cache.get_cached_profile(temperature, dewpoint)
    mp_profile = mpcalc.parcel_profile(pressure, temperature, dewpoint).to(profile.units)
    
    assert np.max(np.abs(mp_profile.magnitude - profile.magnitude)) < 0.001


def test_dry_profile(cache, pressure):

    temperature = 0 * units.celsius
    dewpoint = -100 * units.celsius

    profile = cache.get_cached_profile(temperature, dewpoint)
    mp_profile = mpcalc.parcel_profile(pressure, temperature, dewpoint).to(profile.units)
    
    assert np.max(np.abs(mp_profile.magnitude - profile.magnitude)) < 0.001


def test_kelvin_profile(cache, pressure):

    temperature = (10 * units.celsius).to(units.kelvin)
    dewpoint = 0 * units.celsius

    profile = cache.get_cached_profile(temperature, dewpoint)
    mp_profile = mpcalc.parcel_profile(pressure, temperature, dewpoint).to(profile.units)
    
    assert np.max(np.abs(mp_profile.magnitude - profile.magnitude)) < 0.001


def test_warm_moist_profile(cache, pressure):

    temperature = 5 * units.celsius
    dewpoint = 10 * units.celsius

    with pytest.raises(ValueError) as ex:
        profile = cache.get_cached_profile(temperature, dewpoint)
 
        assert profile is None
        assert 'dewpoint' in ex