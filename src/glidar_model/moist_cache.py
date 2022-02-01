import numpy as np
import metpy.calc as mpcalc

from metpy.units import units


class MoistCache:
    """
    Moist cache precomutes humid profiles for interactive
    acceleration of the model runs. Not neccessary for a single
    model run.
    """

    def __init__(self, pressure, T_min=-20, T_max=30, N=50) -> None:
        """
        Constructs an instance of the cache for a given pressure profile.
        The temperature range is given in degrees Celsius.
        """
        self.pressure = pressure
        self.T_min = T_min
        self.T_max = T_max
        self.N = N

        self.moist = None

        self.precompute_cache()

    def precompute_cache(self):

        self.moist = []
        # Precompute the moist adiabatic profiles
        temp = np.linspace(self.T_min, self.T_max, self.N)
        for t in temp:
            m = mpcalc.moist_lapse(self.pressure, t * units.celsius)
            self.moist.append(m)
        self.moist = np.array(
            self.moist) * units.celsius  # np.array strips the units..
        print('Cache computed.')

    def get_cached_profile(self, temperature, dew_point):

        if dew_point > temperature:
            raise ValueError('Dewpoint is higher than air temperature.',
                             temperature, dew_point)

        if self.moist is None:
            self.precompute_cache()
            print('Cache was not precomputed, computing now...')

        p0, t0 = mpcalc.lcl(self.pressure[0], temperature, dew_point)
        dry = mpcalc.dry_lapse(self.pressure, temperature)

        ip = None
        ips = np.where(self.pressure < p0)
        if len(ips[0]) > 0:
            ip = ips[0][0]
        # The lcl is above the end of the profile, dry profile is sufficient.
        else:
            return dry

        it = None
        its = np.where(self.moist.T[ip] > t0)
        if len(its[0] > 0):
            it = its[0][0]
            if it == 0:
                raise RuntimeError(
                    'Cannot get cached profile, temperature too low.', p0, t0,
                    temperature, dew_point)
            it = (it - 1, it)
        else:
            raise RuntimeError(
                'Cannot get cached profile, temperature too high.', p0, t0,
                temperature, dew_point)

        # Bi-linear interpolation
        pp = self.pressure[[ip - 1, ip]]
        ttt = self.moist[it[0]:it[1] + 1, [ip - 1, ip]]
        tt = (ttt[:, 0] * (pp[1] - p0) +
              (p0 - pp[0]) * ttt[:, 1]) / (pp[1] - pp[0])

        f = ((tt[1] - t0) * self.moist[it[0], :] +
             (t0 - tt[0]) * self.moist[it[1], :]) / (tt[1] - tt[0])

        return np.where(self.pressure > p0, dry, f)
