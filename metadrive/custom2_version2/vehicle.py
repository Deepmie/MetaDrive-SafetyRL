from metadrive.component.vehicle.default_vehicle import DefaultVehicle
from metadrive.custom2_version2.policy import Policy

class DpVehicle(DefaultVehicle):
    def __init__(self, *args, **kwargs):
        super(DpVehicle, self).__init__(*args, **kwargs)
        self.policy = Policy()
    