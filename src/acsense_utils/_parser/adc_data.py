import numpy as np
import pandas as pd
import logging
import os
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class SPI_ADC_Data:
    def __init__(self):
        self.timestamps = []
        self.data = []
        self.sample_count = []

        self.sample_rate = None
        self.channels = None
        self.bitsPerChannel = None
        self.bytesPerChannel = None
        # self.scale = None

    def _parse(self, header, raw_data, signed=True):
        num_channels = header["Header"].channels
        bytes_per_channel = header["Header"].bytesPerChannel
        num_records = header["Header"].dataRecordsPerBuffer
        if signed:
            dtype = np.int16
        else:
            dtype = np.uint16

        data = np.zeros((num_channels, num_records), dtype=dtype)
        offset_step = bytes_per_channel * num_channels
        # logger.debug(header)
        # logger.debug(data.shape)
        do_break = False
        for i in range(num_records):
            for j in range(num_channels):
                try:
                    data[j, i] = np.frombuffer(
                        raw_data,
                        dtype=dtype,
                        offset=i * offset_step + j * bytes_per_channel,
                    )[0]

                except Exception as e:
                    logger.error(
                        f"Exception encountered while parsing data:"
                        f"\n{e}"
                        f"\nHeader is:"
                        f"\n{header}"
                    )
                    do_break = True
                    break
            if do_break:
                break

        self.data.append(data)
        self.timestamps.append(header["Header"].timestamp)
        self.sample_count.append(header["Header"].sampleCount)

        self.sample_rate = header["Header"].sampleRate
        self.channels = header["Header"].channels
        self.bitsPerChannel = header["Header"].bitsPerChannel
        self.bytesPerChannel = header["Header"].bytesPerChannel
        # self.scale = header["Header"].scale

    def as_dict(self):
        dic = {}
        dic["timestamp"] = []
        dic["sample_count"] = []

        if len(self.data) > 0:
            for i in range(self.data[0].shape[0]):
                dic["channel_" + repr(i)] = []

            for ind in range(len(self.data)):
                for i in range(self.data[ind].shape[1]):
                    dic["timestamp"].append(
                        self.timestamps[ind] + i * 1.0e8 * (1.0 / self.sample_rate)
                    )
                    dic["sample_count"].append(self.sample_count[ind] + i)

                    for j in range(self.data[ind].shape[0]):
                        dic["channel_" + repr(j)].append(self.data[ind][j, i])
        return dic

    def write_csv(self, outfile):

        data_len = len(self.data)
        if data_len > 0:

            mult = 1.0e8 * (1.0 / self.sample_rate) if self.sample_rate else 1

            cols = []
            for i in range(self.data[0].shape[0]):
                cols.append("channel_" + repr(i))

            prog_bar = tqdm(
                desc=f"Writing {os.path.basename(outfile):40s}", total=data_len
            )
            ind = 0
            while ind < data_len:
                tt = self.timestamps[ind]

                cur_count = self.sample_count[ind]
                cur_len = self.data[ind].shape[1]
                cur_t_list = tt + np.linspace(
                    0, cur_len * mult, cur_len, endpoint=False
                )
                logger.debug(f"SPI_ADC cur_t_list : {cur_t_list}")
                cur_pd_array = pd.DataFrame(
                    data=np.transpose(self.data[ind]), index=cur_t_list, columns=cols
                )
                cur_pd_array.index.name = "timestamp"
                cur_pd_array["sample_count"] = cur_count + np.linspace(
                    0, cur_len, cur_len, endpoint=False
                ).astype("uint64")
                if ind == 0:
                    # create csv and write:
                    cur_pd_array.to_csv(outfile)
                else:
                    cur_pd_array.to_csv(outfile, mode="a", header=False)

                prog_bar.update(1)
                ind += 1
            prog_bar.refresh()
            prog_bar.close()

        return

    def get_name(self):
        return "spiadc"

    @classmethod
    def from_header(cls, header, raw_data):
        return cls(header, raw_data)


class Internal_ADC_Data:
    def __init__(self):
        self.timestamps = []
        self.data = []

        self.sample_rate = None
        self.channels = None
        self.bitsPerChannel = None
        self.bytesPerChannel = None
        # self.scale = None

    def _parse(self, header, raw_data, signed=True):
        num_channels = header["Header"].channels
        bytes_per_channel = header["Header"].bytesPerChannel
        num_records = header["Header"].dataRecordsPerBuffer
        if signed:
            dtype = np.int16
        else:
            dtype = np.uint16

        data = np.zeros((num_channels, num_records), dtype=dtype)
        offset_step = bytes_per_channel * num_channels
        # logger.debug(f"Header is :\n{header}")
        # logger.debug(f"Data shape is : {data.shape}")
        do_break = False
        for i in range(num_records):
            for j in range(num_channels):
                try:
                    data[j, i] = np.frombuffer(
                        raw_data,
                        dtype=dtype,
                        offset=i * offset_step + j * bytes_per_channel,
                    )[0]
                except Exception as e:
                    logger.error(
                        f"Exception encountered while parsing data:"
                        f"\n{e}"
                        f"\nHeader is:"
                        f"\n{header}"
                    )
                    do_break = True
                    break
            if do_break:
                break

        self.data.append(data)
        self.timestamps.append(header["Header"].timestamp)

        self.sample_rate = header["Header"].sampleRate
        self.channels = header["Header"].channels
        self.bitsPerChannel = header["Header"].bitsPerChannel
        self.bytesPerChannel = header["Header"].bytesPerChannel
        # self.scale = header["Header"].scale

    def as_dict(self):
        dic = {}
        dic["timestamp"] = []

        if len(self.data) > 0:
            for i in range(self.data[0].shape[0]):
                dic["channel_" + repr(i)] = []

            for ind in range(len(self.data)):
                for i in range(self.data[ind].shape[1]):
                    dic["timestamp"].append(
                        self.timestamps[ind] + i * 1.0e8 * (1.0 / self.sample_rate)
                    )

                    for j in range(self.data[ind].shape[0]):
                        dic["channel_" + repr(j)].append(self.data[ind][j, i])
        return dic

    def get_name(self):
        return "intadc"

    @classmethod
    def from_header(cls, header, raw_data):
        return cls(header, raw_data)
