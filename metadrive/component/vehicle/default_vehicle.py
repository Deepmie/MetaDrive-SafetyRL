from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace

class DefaultVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.DEFAULT_VEHICLE)
    # LENGTH = 4.51
    # WIDTH = 1.852
    # HEIGHT = 1.19
    TIRE_RADIUS = 0.313
    TIRE_WIDTH = 0.25
    MASS = 1100
    LATERAL_TIRE_TO_CENTER = 0.815
    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    path = ('ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0))  # asset path, scale, offset, HPR

    DEFAULT_LENGTH = 4.515  # meters
    DEFAULT_HEIGHT = 1.19  # meters
    DEFAULT_WIDTH = 1.852  # meters

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH