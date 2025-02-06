import logging
import os

from tqdm.auto import tqdm  # type: ignore
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore

from .adc_data import Internal_ADC_Data, SPI_ADC_Data
from .external_sensor_data import (
    BNO_Data,
    Ctd_Data,
    External_PTS_Data_Bar30,
    External_PTS_Data_Bar100,
    GPS_Data,
    Image_Meta_Data,
    Magnetometer_Data,
    NAU7802_Data,
    Ping_Data,
    RDO_Data,
    RTC_Data,
)
from .generic_data import Generic_Data
from .headers import Generic_Header, Internal_ADC_Header, SPI_ADC_Header
from .internal_sensor_data import IMU_Data, Internal_PTS_Data

logger = logging.getLogger(__name__)

# TICK = 1e-9  # sample interval is s

MSG_INFO = {
    0x09: {"header": SPI_ADC_Header, "parser": SPI_ADC_Data},
    0x0F: {
        "header": Internal_ADC_Header,
        "parser": Internal_ADC_Data,
    },
    0x0B: {
        "header": Generic_Header,
        "parser": Internal_PTS_Data,
    },
    0x0C: {"header": Generic_Header, "parser": IMU_Data},
    0x0E: {"header": Generic_Header, "parser": RTC_Data},
    0x10: {"name": "external pts", "header": Generic_Header, "parser": Generic_Data},
    0x11: {"header": Generic_Header, "parser": GPS_Data},
    0x12: {"header": Generic_Header, "parser": Generic_Data},
    0x0D: {"header": Generic_Header, "parser": Ping_Data},
    0x16: {"header": Generic_Header, "parser": Image_Meta_Data},
    0x17: {"header": Generic_Header, "parser": Ctd_Data},
    0x1A: {"header": Generic_Header, "parser": RDO_Data},
    0x1E: {"header": Generic_Header, "parsed": BNO_Data},
}


class Parser:
    def __init__(self, block_size=512):
        gen_hdr = Generic_Header()
        int_adc_header = Internal_ADC_Header()
        spi_adc_header = SPI_ADC_Header()
        self.headers = {
            0x09: spi_adc_header,
            0x0B: gen_hdr,
            0x0C: gen_hdr,
            0x0E: gen_hdr,
            0x12: gen_hdr,
            0x0D: gen_hdr,
            0x0F: int_adc_header,
            0x11: gen_hdr,
            0x14: gen_hdr,
            0x16: gen_hdr,
            0x17: gen_hdr,
            0x18: gen_hdr,
            0x1A: gen_hdr,
            0x1B: gen_hdr,
            0x1E: gen_hdr,
        }

        self.parsers = [
            {
                "msg_id": 0x09,
                "header": spi_adc_header,
                "ID1": "A",
                "ID2": "D",
                "parser": SPI_ADC_Data(),
            },
            {
                "msg_id": 0x0B,
                "header": gen_hdr,
                "ID1": "P",
                "ID2": "T",
                "parser": Internal_PTS_Data(),
            },
            {
                "msg_id": 0x0C,
                "header": gen_hdr,
                "ID1": "I",
                "ID2": "M",
                "parser": IMU_Data(),
            },
            {
                "msg_id": 0x0E,
                "header": gen_hdr,
                "ID1": "R",
                "ID2": "T",
                "parser": RTC_Data(),
            },
            {
                "msg_id": 0x12,
                "header": gen_hdr,
                "ID1": "N",
                "ID2": "A",
                "parser": NAU7802_Data(),
            },
            {
                "msg_id": 0x12,
                "header": gen_hdr,
                "ID1": "E",
                "ID2": "P",
                "parser": External_PTS_Data_Bar30(),
            },
            {
                "msg_id": 0x12,
                "header": gen_hdr,
                "ID1": "E",
                "ID2": "D",
                "parser": External_PTS_Data_Bar100(),
            },
            {
                "msg_id": 0x14,
                "header": gen_hdr,
                "ID1": "M",
                "ID2": "G",
                "parser": Magnetometer_Data(),
            },
            {
                "msg_id": 0x0D,
                "header": gen_hdr,
                "ID1": "E",
                "ID2": "C",
                "parser": Ping_Data(),
            },
            {
                "msg_id": 0x0F,
                "header": int_adc_header,
                "ID1": "A",
                "ID2": "1",
                "parser": Internal_ADC_Data(),
            },
            {
                "msg_id": 0x11,
                "header": gen_hdr,
                "ID1": "G",
                "ID2": "P",
                "parser": GPS_Data(),
            },
            {
                "msg_id": 0x16,
                "header": gen_hdr,
                "ID1": "I",
                "ID2": "S",
                "parser": Image_Meta_Data(),
            },
            {
                "msg_id": 0x17,
                "header": gen_hdr,
                "ID1": "C",
                "ID2": "T",
                "parser": Ctd_Data(),
            },
            {
                "msg_id": 0x1A,
                "header": gen_hdr,
                "ID1": "R",
                "ID2": "D",
                "parser": RDO_Data(),
            },
            {
                "msg_id": 0x1B,
                "header": gen_hdr,
                "ID1": "B",
                "ID2": "N",
                "parser": BNO_Data(),
            },
        ]
        self.block_size = block_size
        self.sens_dict = {}
        pass

    def parse_sense_file(self, fn, pbar_position=None):
        self.sens_dict = {}

        file_size = os.stat(fn).st_size
        if file_size == 0:
            logger.warning(
                f"{os.path.basename(fn):40s} : file size is {file_size}! Not processing"
            )
            return self.parsers

        prog_bar = tqdm(
            desc=f"Parsing {os.path.basename(fn):40s}",
            total=file_size,
            position=pbar_position,
        )
        with logging_redirect_tqdm():
            with open(fn, "rb") as f:
                while True:
                    start_tell = f.tell()
                    self.read_block("INT", f)
                    prog_bar.update(f.tell() - start_tell)
                    if f.tell() >= file_size:
                        break

        prog_bar.refresh()
        prog_bar.close()
        return self.parsers

    def parse_ac_file(
        self, fn, use_int, export=False, output_dir=None, pbar_position=None
    ):
        file_size = os.stat(fn).st_size
        if file_size == 0:
            logger.warning(
                f"{os.path.basename(fn)} : file size is {file_size}! Not processing"
            )
            return self.parsers

        prog_bar = tqdm(
            desc=(
                f"Exporting to CSV from {os.path.basename(fn):26s}"
                if export
                else f"Parsing {os.path.basename(fn):40s}"
            ),
            total=file_size,
            position=pbar_position,
        )
        with logging_redirect_tqdm():
            with open(fn, "rb") as f:
                while True:
                    start_tell = f.tell()
                    self.read_block(
                        "INT" if use_int else "EXT",
                        f,
                        ac_file=True,
                        export=export,
                        output_dir=output_dir,
                        input_filename=fn,
                    )  # MAKE CONFIGURABLE AS INT OR EXT!!!!
                    prog_bar.update(f.tell() - start_tell)
                    if (
                        start_tell == f.tell()
                        or f.tell() >= os.fstat(f.fileno()).st_size
                    ):
                        break
        prog_bar.refresh()
        prog_bar.close()
        return self.parsers

    def read_block_header(self, f):
        num_entries = int.from_bytes(f.read(2), "little")
        fill_index = int.from_bytes(f.read(2), "little")
        size_fill = int.from_bytes(f.read(2), "little")
        return {
            "num_entries": num_entries,
            "fill_index": fill_index,
            "size_fill": size_fill,
        }

    def read_index(self, f):  # Used in SENS files
        return int.from_bytes(f.read(2), "little")

    def parse_record_ac(
        self,
        f,
        hydrophone_ADC="INT",
        timeonly=False,
        export=False,
        output_dir=None,
        input_filename=None,
    ):
        if hydrophone_ADC == "INT":
            msg_id = 0x0F
            msg_type = "InternalADC"
        else:
            msg_id = 0x09
            msg_type = "SPI_ADC"
        header = self.headers[msg_id]
        h = header.read_header(f)
        if h:
            data = f.read(h["payload_bytes"])
            for d in self.parsers:
                if d["msg_id"] == msg_id and h["Type"] == msg_type:
                    d["parser"]._parse(
                        h,
                        data,
                        export=export,
                        output_dir=output_dir,
                        input_filename=input_filename,
                    )
            # return {"record_size": h["payload_bytes"] + header.header_size_bytes}

    def parse_record(self, f, hydrophone_ADC="EXT", timeonly=False):
        # hydrophone_ADC = "EXT" if using 8ch or 16ch, "INT" if a mini system
        next_index = self.read_index(f)
        msg_id = int.from_bytes(f.read(2), "little")
        # logger.debug(f"next->{next_index}, tell->{f.tell()}, msg_id {msg_id}")
        header = self.headers[msg_id]
        h = header.read_header(f)
        data = f.read(h["payload_bytes"])
        for d in self.parsers:
            if d["msg_id"] == msg_id:
                if h["Type"] == "Generic":
                    for d in self.parsers:
                        if d["ID1"] == chr(h["Header"].id1) and d["ID2"] == chr(
                            h["Header"].id2
                        ):
                            d["parser"]._parse(h, data)
                elif h["Type"] == "InternalADC":
                    d["parser"]._parse(h, data)

        # if "parser" in .keys():
        #     res = msg_info["parser"].from_header(h, data)
        #     name = res.get_name()
        #     if not name in self.sens_dict.keys():
        #         self.sens_dict[name] = []
        #     self.sens_dict[name].append(res)

        # else:
        #     logger.info("No parse function implemented for " + repr(msg_info))
        return {"next_index": next_index}

    def read_block(
        self,
        hydrophone_ADC,
        f,
        timeonly=False,
        ac_file=False,
        export=False,
        output_dir=None,
        input_filename=None,
    ):
        block_start = f.tell()
        if not ac_file:
            header = self.read_block_header(f)
        else:
            header = {"num_entries": 1}
        already_told = 0
        # logger.info('header: ' + repr(header))

        if ac_file:
            self.parse_record_ac(
                f,
                hydrophone_ADC=hydrophone_ADC,
                timeonly=timeonly,
                export=export,
                output_dir=output_dir,
                input_filename=input_filename,
            )

        else:
            for i in range(header["num_entries"]):
                data = self.parse_record(
                    f, hydrophone_ADC=hydrophone_ADC, timeonly=timeonly
                )
                next_index = data["next_index"] + block_start
                if next_index < f.tell() and not already_told:
                    logger.info(
                        "Seeking backwards. Issue with parse? "
                        + repr(next_index - f.tell())
                    )
                    break
                    already_told = 1
                f.seek(next_index)
            curr_pos = f.tell()
            # start on next block
            offset = self.block_size - (curr_pos % self.block_size)
            offset = offset % self.block_size
            next_block_start = curr_pos + offset
            f.seek(next_block_start)
        return
