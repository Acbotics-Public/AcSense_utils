import struct
import numpy as np
from collections import namedtuple

from .generic_data import Generic_Data


class IMU_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
        self.accel_x = []
        self.accel_y = []
        self.accel_z = []
        self.roll = []
        self.pitch = []

    def _parse(self, header, raw_data):
        pass

        res = {}
        Imu = namedtuple(
            "Imu",
            "PitchNed_DegreesX100 RollNed_DegreesX100 Accel_X Accel_Y Accel_Z Gyro_X Gyro_Y Gyro_Z",
        )

        data = Imu._make(struct.unpack("<iihhhhhh", raw_data))
        self.timestamps.append(header["Header"].timestamp)
        self.gyro_x.append(data.Gyro_X)
        self.gyro_y.append(data.Gyro_Y)
        self.gyro_z.append(data.Gyro_Z)
        self.accel_x.append(data.Accel_X)
        self.accel_y.append(data.Accel_Y)
        self.accel_z.append(data.Accel_Z)
        self.roll.append(data.RollNed_DegreesX100)
        self.pitch.append(data.PitchNed_DegreesX100)

    def as_dict(self):
        dic = {
            "timestamp": self.timestamps,
            "PitchNed_DegreesX100": self.pitch,
            "RollNed_DegreesX100": self.roll,
            "Accel_X": self.accel_x,
            "Accel_Y": self.accel_y,
            "Accel_Z": self.accel_z,
            "Gyro_X": self.gyro_x,
            "Gyro_Y": self.gyro_y,
            "Gyro_Z": self.gyro_z,
        }
        return dic

    def get_name(self):
        return "IMU"


class Internal_PTS_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.pressure = []
        self.temperature = []

    def _parse(self, header, raw_data):
        raw_pressure = np.frombuffer(raw_data, count=1, dtype=np.uint32)[0]
        raw_temp = np.frombuffer(raw_data, count=1, offset=4, dtype=np.int32)[0]
        self.temperature.append(float(raw_temp) / 100.0)
        self.pressure.append(float(raw_pressure) / 100.0)
        self.timestamps.append(header["Header"].timestamp)

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "pressure_mbar": self.pressure,
            "temperature_c": self.temperature,
        }

    def get_name(self):
        return "Internal_PTS"
