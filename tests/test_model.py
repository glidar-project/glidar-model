import numpy as np
from metpy.units import units

import metpy.calc as mpcalc
import pytest

from glidar_model.convection_model import DryThermalModel, ModelParams
from glidar_model.parcel_profile import CachedParcelProfile, ParcelProfile, ParcelProfileBuilder, SimpleParcelProfile
from glidar_model.synthetic_profile import SyntheticProfileBuilder, SyntheticProfileParams
from glidar_model.sounding_profile import SoundingProfileBuilder

from glidar_model.model_manager import Model


@pytest.fixture
def params():

    return ModelParams(10 * units.celsius, 1 * units.delta_degree_Celsius,
                        1 * units.celsius, 0.5 * units.delta_degree_Celsius,
                        100 * units.meter, 0, 0, 0)

@pytest.fixture
def sounding():

    t = 9 * units.celsius
    dp = 0 * units.celsius
    h = 1000 * units.meter
    dpi = -15 * units.celsius
    ti = 3 * units.delta_degree_Celsius

    params = SyntheticProfileParams(t, ti, h, dp, dpi)

    builder = SyntheticProfileBuilder()
    profile =  builder.get_profile(params)
    
    return profile


def test_model(params, sounding):

    mp = params
    
    bckg = SoundingProfileBuilder(sounding)
    parcel = SimpleParcelProfile(sounding.pressure)

    model = Model(bckg, parcel)

    result = model.run_model(mp)

    assert False


def test_dry_model(sounding, params):

    bckg = SoundingProfileBuilder(sounding)
    parcel = SimpleParcelProfile(sounding.pressure)

    model = Model(bckg, parcel, DryThermalModel)

    result = model.run_model(params)

    assert False
    