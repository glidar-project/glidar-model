
import metpy.calc as mpcalc


class ParcelProfile():

    def __init__(self) -> None:
        pass



def get_parcel_profile(pressure, temperature, dewpoint):

    return mpcalc.parcel_profile(pressure, temperature, dewpoint)

