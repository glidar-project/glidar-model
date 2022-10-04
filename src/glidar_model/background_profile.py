class BackgroundProfile:

    def __init__(self, pressure, temperature, dewpoint, altitude) -> None:

        self.pressure = pressure
        self.temperature = temperature
        self.dewpoint = dewpoint
        self.altitude = altitude


class BackgroundProfileBuilder:

    def __init__(self) -> None:
        raise NotImplementedError('Invocation of abstract class.')

    def get_profile(self, params) -> BackgroundProfile:
        raise NotImplementedError('This method should be re-implemented.')


class StaticBackroundProfileBuilder(BackgroundProfileBuilder):

    def __init__(self, profile: BackgroundProfile) -> None:

        self.profile = profile

    def get_profile(self, params) -> BackgroundProfile:

        return self.profile