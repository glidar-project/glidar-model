import os
import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from metpy.units import units
import metpy.calc as mpcalc

from glidar_model.convection_model import ModelParams, ThermalModel
from glidar_model.parcel_profile import SimpleParcelProfile
from glidar_model.imet_profile import resampled_imet_backround


def modell_convection_from_imet(imet, args):

    profile = resampled_imet_backround(imet)
    parcel = SimpleParcelProfile(profile.pressure)

    params = ModelParams(profile.temperature[0],
                         args['delta_T'] * units.delta_degree_Celsius,
                         profile.dewpoint[0],
                         0 * units.delta_degree_Celsius,
                         profile.altitude[0],
                         drag_coeff=args['drag'],
                         entrainment_coeff=args['entrainment'],
                         humidity_entrainment_coeff=0)

    model = ThermalModel()
    result = model.run_model(profile, parcel.get_profile(params), params)

    return result


def plot_result(result, ofile=None):

    background_profile = result.background_profile
    parcel_profile = result.parcel_profile
    ent_profile = result.entrained_profile

    fig, ax = plt.subplots(1, 2, sharey=True)

    plt.suptitle(f'{os.path.basename(ofile)[:-4]} delta t: {result.params.temperature_anomaly}')

    ax[0].plot(*result.velocity_profile)


    ax[1].axhline(y = result.velocity_profile[1][-1], c='k', linestyle=':')

    ax[1].plot(background_profile.temperature,
               background_profile.altitude,
               'r-',
               label='temperature')
    ax[1].plot(background_profile.dewpoint,
               background_profile.altitude,
               'g-',
               label='dewpoint')
    ax[1].plot(parcel_profile.temperature,
               background_profile.altitude,
               'r:',
               label='temperature')
    ax[1].plot(parcel_profile.dewpoint,
               background_profile.altitude,
               'g:',
               label='dewpoint')
    ax[1].plot(ent_profile.temperature,
               background_profile.altitude,
               'r,',
               label='temperature')
    ax[1].plot(ent_profile.dewpoint,
               background_profile.altitude,
               'g,',
               label='dewpoint')

    # ax[1].invert_yaxis()
    ax[0].set_ylim(0, 4000)

    ax[0].set_xlim(0, 10)
    ax[1].set_xlim(250, 290)


    if ofile is None:
        plt.show()
    else:
        plt.savefig(ofile)
        plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'imet_profile',
        help='CSV file containing a single temperature and humidity profile')
    parser.add_argument('--delta_T',
                        help='Temperature anomaly parameter, default 0.5',
                        type=float,
                        default=0.5)
    parser.add_argument('--drag',
                        help='Adjustable drag parameter, default 0',
                        type=float,
                        default=0)
    parser.add_argument('--entrainment',
                        help='Entrainment parameter, default 0',
                        type=float,
                        default=0)

    args = parser.parse_args()

    imet = pd.read_csv(args.imet_profile)
    result = modell_convection_from_imet(imet, {
        'delta_T': args.delta_T,
        'drag': args.drag,
        'entrainment': args.entrainment
    })

    plot_result(result)


if __name__ == '__main__':

    main()
