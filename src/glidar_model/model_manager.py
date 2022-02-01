import metpy.calc as mpcalc
from metpy.units import units

from glidar_model.background_profile import BackgroundProfileBuilder
from glidar_model.convection_model import ModelResult, ThermalModel
from glidar_model.moist_cache import MoistCache
from glidar_model.parcel_profile import ParcelProfile, ParcelProfileBuilder, SimpleParcelProfile
from glidar_model.sounding_profile import SoundingProfileBuilder


class Model():

    def __init__(self,
                 background_profile: BackgroundProfileBuilder,
                 parcel_profile: ParcelProfileBuilder,
                 model: ThermalModel = ThermalModel,
                 cached=False) -> None:

        self.background_profile = background_profile
        self.parcel_profile = parcel_profile

        # Chache handling
        self.cached = cached
        self.moist_cache = None
        if self.cached:
            self.moist_cache = MoistCache(
                self.background_profile.get_profile().pressure)

        self.thermal_model = model()

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

        res = mpcalc.parcel_profile(pressure, temperature,
                                    dew_point).to(units.kelvin)

        return res

    def run_model(self, params) -> ModelResult:

        background_profile = self.background_profile.get_profile(params)
        parcel_profile = self.parcel_profile.get_profile(params)

        return self.thermal_model.run_model(background_profile, parcel_profile,
                                            params)


def compute_model_from_sounding(sounding, params):

    bckg = SoundingProfileBuilder(sounding)
    parcel = SimpleParcelProfile(sounding.pressure)
    model = Model(bckg, parcel)

    return model.run_model(params)

# def fit_thermal(self, df, x0, bounds):

#     w, z = df['vario'].to_numpy(), df['altitude'].to_numpy()

#     zz = np.linspace(0,2000, 200)
#     ww = np.zeros_like(zz)
#     idx = (z/10.).astype(np.int)

#     for i, j in enumerate(idx):
#         if ww[j] < w[i]:
#             ww[j] = w[i]

#     sub = np.nonzero(ww)

#     def fit_fn(args):

#         params = ModelParams(*args)
#         result = ModelResult()

#         b, _ = self.calc_buoyancy_profile()
#         w, z = self.calc_velocity_profile(b , self.altitude, z0, alpha=a)

#         return np.sum((ww - np.interp(zz,z,w)) **2)

#     from scipy.optimize import minimize

#     return ww, zz, minimize(fit_fn, x0, method='Nelder-Mead', bounds=bounds)
