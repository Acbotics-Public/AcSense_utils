import copy
import datetime
import logging
import struct
from collections import namedtuple

import numpy as np
from pynmeagps import NMEAReader  # type: ignore

from .generic_data import Generic_Data

logger = logging.getLogger(__name__)


class GPS_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.nmea = []
        self.lat = []
        self.lon = []
        self.epoch_date = []
        self.epoch_time = []
        self.time_str = []
        self.unix_time = []

        self.template_data = {
            "EpochTime": None,
            "EpochDate": None,
            "Lat": None,
            "Lon": None,
            "timestr": None,
            "raw_nmea": None,
            "timestamp": None,
            "UnixTime": None,
        }

    def _parse(self, header, raw_data):
        st = raw_data.decode(encoding="UTF-8").strip("\n\r\x00")
        data = copy.copy(self.template_data)
        data["timestamp"] = header["Header"].timestamp
        data["raw_nmea"] = st
        if "RMC" in st and "$" in st:
            try:
                msg = NMEAReader.parse(st)
            except Exception as e:
                logger.info("failed to parse " + repr(st) + " " + repr(e))
                return None
            data["EpochTime"] = msg.time
            data["EpochDate"] = msg.date
            data["Lat"] = msg.lat
            data["Lon"] = msg.lon
            data["timestr"] = str(data["EpochDate"]) + "_T" + str(data["EpochTime"])
            try:
                data["UnixTime"] = datetime.datetime.strptime(
                    data["timestr"] + "Z", "%Y-%m-%d_T%H:%M:%S.%f%z"
                ).timestamp()

            except Exception:
                data["UnixTime"] = datetime.datetime.strptime(
                    data["timestr"] + "Z", "%Y-%m-%d_T%H:%M:%S%z"
                ).timestamp()

        self.timestamps.append(data["timestamp"])
        self.nmea.append(data["raw_nmea"])
        self.lat.append(data["Lat"])
        self.lon.append(data["Lon"])
        self.epoch_date.append(data["EpochDate"])
        self.epoch_time.append(data["EpochTime"])
        self.time_str.append(data["timestr"])
        self.unix_time.append(data["UnixTime"])

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "raw_nmea": self.nmea,
            "lat": self.lat,
            "lon": self.lon,
            "epoch_date": self.epoch_date,
            "epoch_time": self.epoch_time,
            "timestr": self.time_str,
            "unix_time": self.unix_time,
        }

    def get_name(self):
        return "gps"


class Ping_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.distance = []
        self.confidence = []
        self.transmit_duration = []
        self.ping_number = []
        self.scan_start = []
        self.scan_length = []
        self.gain_setting = []
        self.profile_data_length = []
        self.profile_values = []

    def _parse(self, header, raw_data):
        Ping = namedtuple(
            "Ping",
            "distance confidence transmit_duration ping_number scan_start scan_length gain_setting profile_data_length profile_values",
        )

        data = Ping._make(struct.unpack("<IHHIIIIH200s", raw_data))
        self.timestamps.append(header["Header"].timestamp)
        self.distance.append(data.distance)
        self.confidence.append(data.confidence)
        self.transmit_duration.append(data.transmit_duration)
        self.ping_number.append(data.ping_number)
        self.scan_start.append(data.scan_start)
        self.scan_length.append(data.scan_length)
        self.gain_setting.append(data.gain_setting)
        self.profile_data_length.append(data.profile_data_length)
        self.profile_values.append(data.profile_values)

    def as_dict(self):
        dic = {
            "timestamp": self.timestamps,
            "distance": self.distance,
            "confidence": self.confidence,
            "transmit_duration": self.transmit_duration,
            "ping_number": self.ping_number,
            "scan_start": self.scan_start,
            "scan_length": self.scan_length,
            "gain_setting": self.gain_setting,
            "profile_data_length": self.profile_data_length,
            # "profile_data": [self.data.profile_values],
        }
        if len(self.profile_values) > 0:
            profile_data = np.frombuffer(self.profile_values[0], dtype=np.uint8)
            for i in range(len(profile_data)):
                dic["profile_data" + str(i)] = []

            for ind in range(len(self.profile_values)):
                for rng in range(len(profile_data)):
                    pd = np.frombuffer(self.profile_values[ind], dtype=np.uint8)
                    dic["profile_data" + str(rng)].append(pd[rng])

        return dic

    def get_name(self):
        return "ping"


class Magnetometer_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.MagScaleFactor_mG = []
        self.Mag_X = []
        self.Mag_Y = []
        self.Mag_Z = []
        self.Temperature = []

    def _parse(self, header, raw_data):
        MagScaleFactor_mG = np.frombuffer(raw_data, count=1, dtype=np.float64)[0]
        Mag_X = np.frombuffer(raw_data, count=1, offset=8, dtype=np.int32)[0]
        Mag_Y = np.frombuffer(raw_data, count=1, offset=12, dtype=np.int32)[0]
        Mag_Z = np.frombuffer(raw_data, count=1, offset=16, dtype=np.int32)[0]
        raw_temp = np.frombuffer(raw_data, count=1, offset=20, dtype=np.int16)[0]

        self.MagScaleFactor_mG.append(MagScaleFactor_mG)
        self.Mag_X.append(Mag_X)
        self.Mag_Y.append(Mag_Y)
        self.Mag_Z.append(Mag_Z)
        self.Temperature.append(float(raw_temp) / 100)

        self.timestamps.append(header["Header"].timestamp)

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "mag_x": self.Mag_X,
            "mag_y": self.Mag_Y,
            "mag_z": self.Mag_Z,
            "temperature": self.Temperature,
            "MagScaleFactor_mG": self.MagScaleFactor_mG,
        }

    def get_name(self):
        return "Magnetometer"


class External_PTS_Data_Bar100(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.pressure = []
        self.temperature = []

    def _parse(self, header, raw_data):
        raw_pressure = np.frombuffer(raw_data, count=1, dtype=np.int32)
        raw_temp = np.frombuffer(raw_data, count=1, offset=4, dtype=np.int32)
        self.pressure.append(float(raw_pressure) / 100.0)
        self.temperature.append(float(raw_temp) / 100.0)
        self.timestamps.append(header["Header"].timestamp)

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "pressure_bar": self.pressure,
            "temperature_c": self.temperature,
        }

    def get_name(self):
        return "External_PTS_Bar100"


class External_PTS_Data_Bar30(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.pressure = []
        self.temperature = []

    def _parse(self, header, raw_data):
        raw_pressure = np.frombuffer(raw_data, count=1, dtype=np.int32)
        raw_temp = np.frombuffer(raw_data, count=1, offset=4, dtype=np.int32)
        self.pressure.append(float(raw_pressure) / 10.0 / 1000)
        self.temperature.append(float(raw_temp) / 100.0)
        self.timestamps.append(header["Header"].timestamp)

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "pressure_bar": self.pressure,
            "temperature_c": self.temperature,
        }

    def get_name(self):
        return "External_PTS_Bar30"


class Image_Meta_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.file_numbers = []
        self.resolutions = []

    def _parse(self, header, raw_data):
        file_number = np.frombuffer(raw_data, count=1, dtype=np.uint32)
        resolution = np.frombuffer(raw_data, count=1, offset=4, dtype=np.uint8)
        # reserved = np.frombuffer(raw_data, count=3, offset=5, dtype=np.uint8)
        self.file_numbers.append(file_number[0])
        self.resolutions.append(resolution[0])
        self.timestamps.append(header["Header"].timestamp)

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "file_number": self.file_numbers,
            "resolution": self.resolutions,
        }

    def get_name(self):
        return "Image Meta"


class NAU7802_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.adc_data = []
        self.channels = []

    def _parse(self, header, raw_data):
        channel = np.frombuffer(raw_data, count=1, dtype=np.uint8)[0]
        adc3 = np.frombuffer(raw_data, count=1, offset=1, dtype=np.uint8)[0]
        adc2 = np.frombuffer(raw_data, count=1, offset=2, dtype=np.uint8)[0]
        adc1 = np.frombuffer(raw_data, count=1, offset=3, dtype=np.uint8)[0]
        adc_data = np.int32(
            (np.int32(adc1) << 24) + (np.int32(adc2) << 16) + (np.int32(adc3) << 8)
        )
        self.timestamps.append(header["Header"].timestamp)
        self.adc_data.append(adc_data)
        self.channels.append(channel)

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            "channel": self.channels,
            "adc_data": self.adc_data,
        }

    def get_name(self):
        return "NAU7802"


class Ctd_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        # self.data = []
        self.temperature = []
        self.conductivity = []
        self.salinity = []
        self.pressure = []
        self.temp_internal = []
        self.density = []
        self.sound_speed = []
        self.elapsed_time = []

    def _parse(self, header, raw_data):
        data = np.frombuffer(raw_data, count=8, dtype=np.float32)
        self.timestamps.append(header["Header"].timestamp)
        # self.data.append(data)
        self.temperature.append(data[0])
        self.conductivity.append(data[1])
        self.salinity.append(data[2])
        self.pressure.append(data[3])
        self.temp_internal.append(data[4])
        self.density.append(data[5])
        self.sound_speed.append(data[6])
        self.elapsed_time.append(data[7])

    def as_dict(self):
        return {
            "timestamp": self.timestamps,
            # "data": self.data,
            "temperature_c": self.temperature,
            "conductivity": self.conductivity,
            "salinity": self.salinity,
            "pressure_decibars": self.pressure,
            "temp_c_internal": self.temp_internal,
            "density": self.density,
            "sound_speed": self.sound_speed,
            "elapsed_time": self.elapsed_time,
        }

    def get_name(self):
        return "CTD"


class RTC_Data(Generic_Data):
    def __init__(self):
        self.timestamps = []
        self.data = []
        self.seconds = []
        self.minutes = []
        self.hours = []
        self.wday = []
        self.mday = []
        self.month = []
        self.year = []
        self.timestr = []

    def _parse(self, header, raw_data):
        RTC = namedtuple(
            "RTC",
            "sec_bcd min_bcd hour_bcd day date_bcd mon_bcd year_bcd junk",
        )

        data = RTC._make(struct.unpack("<bbbbbbbb", raw_data))
        self.timestamps.append(header["Header"].timestamp)
        self.seconds.append((data.sec_bcd & 0xF) + (((data.sec_bcd & 0xF0) >> 4) * 10))
        self.minutes.append((data.min_bcd & 0xF) + (((data.min_bcd & 0xF0) >> 4) * 10))
        self.hours.append((data.hour_bcd & 0xF) + (((data.hour_bcd & 0x30) >> 4) * 10))
        self.wday.append(data.day)
        self.mday.append((data.date_bcd & 0xF) + (((data.date_bcd & 0x30) >> 4) * 10))
        self.month.append((data.mon_bcd & 0xF) + (((data.mon_bcd & 0x10) >> 4) * 10))
        self.year.append((data.year_bcd & 0xF) + (((data.year_bcd & 0xF0) >> 4) * 10))
        self.timestr.append(
            "{0:04d}{1:02d}{2:02d}T{3:02d}{4:02d}{5:02d}".format(
                2000 + self.year[-1],
                self.month[-1],
                self.mday[-1],
                self.hours[-1],
                self.minutes[-1],
                self.seconds[-1],
            )
        )

    def as_dict(self):
        res = {}
        res["timestamp"] = self.timestamps
        res["seconds"] = self.seconds
        res["minutes"] = self.minutes
        res["hours"] = self.hours
        res["wday"] = self.wday
        res["mday"] = self.mday
        res["month"] = self.month
        res["year"] = self.year
        res["timestr"] = self.timestr
        return res

    def get_name(self):
        return "RTC"
