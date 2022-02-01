

class BackgroundProfile:

    def __init__(self, pressure, temperature, dewpoint, altitude) -> None:

        self.pressure = pressure
        self.temperature = temperature
        self.dewpoint = dewpoint
        self.altitude = altitude


class BackgroundProfileBuilder:

    def __init__(self) -> None:
        raise NotImplementedError('Invocation of abstract class.')

    def get_profile(params) -> BackgroundProfile:
        raise NotImplementedError('This method should be re-implemented.')

