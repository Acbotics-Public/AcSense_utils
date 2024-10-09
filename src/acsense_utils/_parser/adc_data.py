import json
import logging
import os

import numpy as np
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

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

        self.wrote_first = False
        self.outfile = None

    def _parse(
        self,
        header,
        raw_data,
        signed=True,
        export=False,
        output_dir=None,
        input_filename=None,
    ):
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

        if export:
            if not self.outfile:
                self.outfile = os.path.join(
                    output_dir,
                    os.path.basename(input_filename).replace(".dat", "")
                    + "_"
                    + self.get_name()
                    + ".csv",
                ).replace(" ", "_")

            if not self.wrote_first:
                # logger.info(f"Writing metadata for {self.outfile}")
                self.write_metadata()
                # logger.info(f"Writing to {self.outfile}")
                # logger.info(f"Header is :\n{header['Header']}")

            self.write_csv(self.outfile, progress=False)

            # For export mode, do not keep data buffer
            self.timestamps = []
            self.data = []
            self.sample_count = []

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

    def write_csv(self, outfile, progress=True, pbar_position=None):
        data_len = len(self.data)
        if data_len > 0:
            mult = 1.0e8 * (1.0 / self.sample_rate) if self.sample_rate else 1

            cols = []
            for i in range(self.data[0].shape[0]):
                cols.append("channel_" + repr(i))

            if progress:
                prog_bar = tqdm(
                    desc=f"Writing {os.path.basename(outfile):40s}",
                    total=data_len,
                    position=pbar_position,
                )
            ind = 0
            for ind in range(data_len):
                tt = self.timestamps[ind]

                cur_count = self.sample_count[ind]
                cur_len = self.data[ind].shape[1]
                cur_t_list = tt + np.linspace(
                    0, cur_len * mult, cur_len, endpoint=False
                )
                logger.debug(f"SPI_ADC cur_t_list : {cur_t_list}")

                _sample_count = cur_count + np.linspace(
                    0, cur_len, cur_len, endpoint=False
                ).astype("uint64")

                cur_pd_array = pd.DataFrame(
                    data=np.transpose(self.data[ind]), index=_sample_count, columns=cols
                )
                cur_pd_array.insert(0, "timestamp", cur_t_list)

                if not self.wrote_first:
                    # create csv and write:
                    cur_pd_array.to_csv(outfile, header=True)
                    self.wrote_first = True
                else:
                    cur_pd_array.to_csv(outfile, mode="a", header=False)

                if progress:
                    prog_bar.update(1)
                # ind += 1
            if progress:
                prog_bar.refresh()
                prog_bar.close()

        return

    def write_metadata(self):
        ac_meta = json.dumps(
            {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "bitsPerChannel": self.bitsPerChannel,
                "bytesPerChannel": self.bytesPerChannel,
            },
            indent=4,
            sort_keys=True,
        )
        with open(self.outfile.replace(".csv", "_meta.txt"), "w") as fn2:
            fn2.write(ac_meta)
            fn2.write("\n")

    def get_name(self):
        return "spiadc"

    @classmethod
    def from_header(cls, header, raw_data):
        return cls(header, raw_data)


class Internal_ADC_Data:
    def __init__(self):
        self.timestamps = []
        self.data = []
        self.sample_count = []

        self.sample_rate = None
        self.channels = None
        self.bitsPerChannel = None
        self.bytesPerChannel = None
        # self.scale = None

        self.wrote_first = False
        self.outfile = None

    def _parse(
        self,
        header,
        raw_data,
        signed=True,
        export=False,
        output_dir=None,
        input_filename=None,
    ):
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
        self.sample_count.append(header["Header"].sampleCount)

        self.sample_rate = header["Header"].sampleRate
        self.channels = header["Header"].channels
        self.bitsPerChannel = header["Header"].bitsPerChannel
        self.bytesPerChannel = header["Header"].bytesPerChannel
        # self.scale = header["Header"].scale

        if export:
            if not self.outfile:
                self.outfile = os.path.join(
                    output_dir,
                    os.path.basename(input_filename).replace(".dat", "")
                    + "_"
                    + self.get_name()
                    + ".csv",
                ).replace(" ", "_")

            if not self.wrote_first:
                # logger.info(f"Writing metadata for {self.outfile}")
                self.write_metadata()
                # logger.info(f"Writing to {self.outfile}")
                # logger.info(f"Header is :\n{header['Header']}")

            self.write_csv(self.outfile, progress=False)

            # For export mode, do not keep data buffer
            self.timestamps = []
            self.data = []
            self.sample_count = []

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

    def write_csv(self, outfile, progress=True, pbar_position=None):
        data_len = len(self.data)
        if data_len > 0:
            mult = 1.0e8 * (1.0 / self.sample_rate) if self.sample_rate else 1

            cols = []
            for i in range(self.data[0].shape[0]):
                cols.append("channel_" + repr(i))

            if progress:
                prog_bar = tqdm(
                    desc=f"Writing {os.path.basename(outfile):40s}",
                    total=data_len,
                    position=pbar_position,
                )
            ind = 0
            for ind in range(data_len):
                tt = self.timestamps[ind]

                cur_count = self.sample_count[ind]
                cur_len = self.data[ind].shape[1]
                cur_t_list = tt + np.linspace(
                    0, cur_len * mult, cur_len, endpoint=False
                )
                logger.debug(f"SPI_ADC cur_t_list : {cur_t_list}")

                _sample_count = cur_count + np.linspace(
                    0, cur_len, cur_len, endpoint=False
                ).astype("uint64")

                cur_pd_array = pd.DataFrame(
                    data=np.transpose(self.data[ind]), index=_sample_count, columns=cols
                )
                cur_pd_array.insert(0, "timestamp", cur_t_list)

                if not self.wrote_first:
                    # create csv and write:
                    cur_pd_array.to_csv(outfile, header=True)
                    self.wrote_first = True
                else:
                    cur_pd_array.to_csv(outfile, mode="a", header=False)

                if progress:
                    prog_bar.update(1)
                # ind += 1
            if progress:
                prog_bar.refresh()
                prog_bar.close()

        return

    def write_metadata(self):
        ac_meta = json.dumps(
            {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "bitsPerChannel": self.bitsPerChannel,
                "bytesPerChannel": self.bytesPerChannel,
            },
            indent=4,
            sort_keys=True,
        )
        with open(self.outfile.replace(".csv", "_meta.txt"), "w") as fn2:
            fn2.write(ac_meta)
            fn2.write("\n")

    def get_name(self):
        return "intadc"

    @classmethod
    def from_header(cls, header, raw_data):
        return cls(header, raw_data)
