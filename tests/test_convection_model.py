
import pytest

import matplotlib.pyplot as plt
import numpy as np

from metpy.units import units

import netCDF4 as nc
import xarray as xr
import time

from glidar_model.legacy_convection_model import ThermalModel, ModelParams, ModelResult


def get_observed_data(filename='data/sola_20180331-20180430.nc',
                      date='2018-04-29 11:06:37'):

    ds = nc.Dataset(filename)
    xds = xr.open_dataset(filename)
    df = xds.to_dataframe()
    data = df.loc[date].iloc[3:, :]

    d = data.altitude[data.altitude < 3000]
    index = data.altitude[data.altitude < 3000].index

    alt = data.altitude[index].values * units[ds.variables['altitude'].units]
    tmp = data.air_temperature[index].values * units[ds.variables['air_temperature'].units]
    pre = data.air_pressure[index].values * units[ds.variables['air_pressure'].units]
    dtp = data.dew_point_temperature[index].values * units[ds.variables['dew_point_temperature'].units]

    return alt, tmp, pre, dtp


###################################################################################################################
# Some testing functionality below ...
###################################################################################################################


def test_riemann_integration():

    folder = 'data/'
    sola_file = 'sola_20180331-20180430.nc'
    date = '2018-04-29 11:06:37'

    alt, tmp, pre, dtp = get_observed_data(folder + sola_file, date)

    deltaT = 0.5 * units.kelvin
    T_0 = (273.15 + 8) * units.kelvin
    Td_0 = (273.15 + 0) * units.kelvin

    params = ModelParams(T_0, deltaT, Td_0,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=100,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)

    result = ModelResult()
    start = time.time()

    m = ThermalModel(alt, pre, tmp, dtp)
    b, _ = m.calc_buoyancy_profile(params, result)
    w, z = m.calc_velocity_profile(b, alt, 300, debug=True)
    
    # TODO: Hacking units
    b *= units.meter / units.second / units.second

    for w0 in np.logspace(-1, -5, 10):
        ww, a = m.riemann_calc_velocity_profile(b, alt, 300, w0=w0)
        plt.plot(ww, a)

    end = time.time()

    plt.plot(w, z, 'x')
    plt.show()


def test_time_model_computation():

    from tqdm.cli import tqdm

    alt , tmp , pre , dtp = get_observed_data()

    params = ModelParams(surface_temperature=(273.15 + 8) * units.kelvin,
                         temperature_anomaly=0.5 * units.kelvin,
                         dew_point_temperature=(273.15 + 0) * units.kelvin,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=100,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)

    for flags in [(False, False),
                  (True, False),
                  (False, True),
                  (True, True)]:

        m = ThermalModel(alt, pre, tmp, dtp, *flags)
        start = time.time()
        for i in tqdm(range(100)):
            m.compute_fit(params)
        end = time.time()
        print('Cached: {}, Riemann: {}, Timed average over 100 runs: {}'.format(*flags, (end - start)/100))


def test_model_computation():

    folder = 'data/'
    sola_file = 'sola_20180331-20180430.nc'
    date = '2018-04-29 11:06:37'

    alt , tmp , pre , dtp = get_observed_data(folder + sola_file, date)

    deltaT = 0.5 * units.kelvin
    T_0 = (273.15 + 8) * units.kelvin
    Td_0 = (273.15 + 0) * units.kelvin

    params = ModelParams(T_0, deltaT, Td_0,
                         dew_point_anomaly=0 * units.kelvin,
                         thermal_altitude=100,
                         drag_coeff=0,
                         entrainment_coeff=0,
                         humidity_entrainment_coeff=0,
                         aspect_ratio=0)

    result = ModelResult()

    m = ThermalModel(alt, pre, tmp, dtp)

    # plt.plot(pre, alt, ',')
    # plt.plot(m.pressure, m.altitude, ',')
    # plt.show()


    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        w, z = m.calc_velocity_profile(b , alt, 300, debug=True)

    end = time.time()
    print('Timed average over 100 runs:', (end - start)/100)

    # Riemann
    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        #    def riemann_calc_velocity_profile(self,  buoyancy, altitude, z_0, alpha=0.00, quadratic_drag=0, w0=0.01):
        w, z = m.riemann_calc_velocity_profile(b , alt, 300)

    end = time.time()
    print('Timed Riemann average over 100 runs:', (end - start)/100)

    #
    # CACHED TIMES
    m = ThermalModel(alt, pre, tmp, dtp, cached=True)

    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        w, z = m.calc_velocity_profile(b , alt, 300, debug=True)

    end = time.time()
    print('Cached timed average over 100 runs:', (end - start)/100)

    # Riemann
    start = time.time()
    for i in range(100):
        b, _ = m.calc_buoyancy_profile(params, result)
        #    def riemann_calc_velocity_profile(self,  buoyancy, altitude, z_0, alpha=0.00, quadratic_drag=0, w0=0.01):
        w, z = m.riemann_calc_velocity_profile(b , alt, 300)

    end = time.time()
    print('Cached riemann timed average over 100 runs:', (end - start)/100)

    zz = np.linspace(0, 3000, 100000)

    # plt.title('Check spline interpolation')
    # plt.plot(bb.b(zz), zz)
    # plt.plot(b, alt, 'x')
    # plt.xlabel('Buoyancy')
    # plt.xlabel('altitude')
    # plt.show()
    #
    # plt.plot(w, z, 'x-')
    # plt.plot(b * 100, alt)
    # plt.plot(tmp.magnitude - 273.15, alt)
    # plt.show()


# def test_model_fitting():

#     folder = 'data/'
#     sola_file = 'sola_20180331-20180430.nc'
#     date = '2018-04-29 11:06:37'

#     alt , tmp , pre , dtp = get_observed_data(folder + sola_file, date)


#     deltaT = 0.5 * units.kelvin
#     p_0 = 1008 * units.hPa
#     T_0 = (273.15 + 8) * units.kelvin
#     Td_0 = (273.15 + 0) * units.kelvin

#     params = ModelParams(T_0, deltaT, Td_0,
#                          dew_point_anomaly=0 * units.kelvin,
#                          thermal_altitude=100,
#                          drag_coeff=0,
#                          entrainment_coeff=0,
#                          humidity_entrainment_coeff=0,
#                          aspect_ratio=0)

#     result = ModelResult()

#     m = ThermalModel(alt, pre, tmp, dtp)
#     b, _ = m.calc_buoyancy_profile(params, result)
#     w, z = m.calc_velocity_profile(b , alt, 300)

#     df = pd.read_csv('data/clusters.csv')

#     w, z, res = m.fit_thermal(df[df.labels == 178], [
#         T_0.magnitude, Td_0.magnitude, deltaT.magnitude, 0, 0, 500
#     ], [
#         (273.15, 293.15),       # T_0 = args[0]
#         (263.15, 283.15),       # Td_0 = args[1]
#         (0, 5),                 # deltaT = args[2]
#         (0, 0.1),               # a = args[3]
#         (0, 0.1),               # g = args[4]
#         (0, 1000),              # z0  = args[5]
#     ])
  
#     plt.plot(w, z)

#     params = ModelParams(
#         res.x[0] * units.kelvin, 
#         res.x[2] * units.kelvin, 
#         res.x[1] * units.kelvin,
#         dew_point_anomaly=0 * units.kelvin,
#         thermal_altitude=res.x[5],
#         drag_coeff=res.x[3],
#         entrainment_coeff=res.x[4],
#         humidity_entrainment_coeff=0,
#         aspect_ratio=0)

#     result = ModelResult()

#     b, _ = m.calc_buoyancy_profile(params, result)
#     w, z = m.calc_velocity_profile(b , alt, params.thermal_altitude)
#     plt.plot(w, z)
#     plt.show()


