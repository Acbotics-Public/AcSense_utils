import logging
import os
import struct
from collections import namedtuple

logger = logging.getLogger(__name__)


class Internal_ADC_Header:
    def __init__(self, double_sample_rate=False):
        if (
            double_sample_rate
        ):  # handles older headers that used double rather than float.
            self.struct_format = "<BBBBBBHdIQd"
        else:
            self.struct_format = "<BBBBBBHfIQd"
        self.header_size_bytes = struct.calcsize(self.struct_format)
        self.header = namedtuple(
            "Header",
            "versionId channels bitsPerChannel bytesPerChannel unpackedShiftRight overflowCount dataRecordsPerBuffer sampleRate sampleCount timestamp scale",
        )

    def read_header(self, f):
        h = handle_adc_header_read_verify(self, f)
        if h:
            return {
                "Type": "InternalADC",
                "payload_bytes": h.bytesPerChannel
                * h.channels
                * h.dataRecordsPerBuffer,
                "Header": h,
            }


class SPI_ADC_Header:
    def __init__(self, use_int_sr=False):
        if use_int_sr:
            self.struct_format = "<BBBBBBHIIQd"
        else:
            self.struct_format = "<BBBBBBHfIQd"

        self.header_size_bytes = struct.calcsize(self.struct_format)
        self.header = namedtuple(
            "Header",
            "versionId channels bitsPerChannel bytesPerChannel unpackedShiftRight overflowCount dataRecordsPerBuffer sampleRate sampleCount timestamp scale",
        )

    def read_header(self, f):
        h = handle_adc_header_read_verify(self, f)
        if h:
            return {
                "Type": "SPI_ADC",
                "payload_bytes": h.bytesPerChannel
                * h.channels
                * h.dataRecordsPerBuffer,
                "Header": h,
            }


def handle_adc_header_read_verify(obj, f):
    """Shared file handler utility for SPI and Internal ADC Headers

    Seeks a known pattern in the header and adjusts offset if needed.

    Pattern to match:

    - Version ID == 1 => from firmware spec!
    - bitsPerChannel == [12,16] => from known ADC resolutions
    - bytesPerChannel == 2 => from repackaged 12-bit and 16-bit data into 16-bit form

    """
    curr_pos = f.tell()
    remainder = os.fstat(f.fileno()).st_size - curr_pos - obj.header_size_bytes
    if remainder < 0:
        logger.debug(f"Current pos : {curr_pos}, remainder after header : {remainder}")
        return

    for offset in range(remainder):
        f.seek(curr_pos + offset)

        h = obj.header._make(
            struct.unpack(obj.struct_format, f.read(obj.header_size_bytes))
        )
        if h.versionId == 1 and h.bitsPerChannel in [12, 16] and h.bytesPerChannel == 2:
            # logger.info(offset)
            break

    if not h.versionId == 1:
        logger.warning("Bad Version ID!")
    # elif offset != 0:
    else:
        logger.debug(
            f"\nInit is {curr_pos}"
            f"\nOffset is {offset}"
            f"\nStart of header is {curr_pos + offset}"
            f"\nEnd of header is {f.tell()}"
            f"\nPayload bytes is {h.bytesPerChannel * h.channels * h.dataRecordsPerBuffer}"
            f"\nEnd of data is "
            f"{f.tell() + h.bytesPerChannel * h.channels * h.dataRecordsPerBuffer}"
            f"\nHeader is {h}"
        )
    # else:
    # logger.info(f"Offset is {offset}")
    return h


class Generic_Header:
    def __init__(self):
        pass

    def read_header(self, f):
        struct_format = "<QBBH"
        Header = namedtuple(
            "Header",
            "timestamp id1 id2 num_bytes",
        )

        h = Header._make(
            struct.unpack(struct_format, f.read(struct.calcsize(struct_format)))
        )
        return {"Type": "Generic", "payload_bytes": h.num_bytes, "Header": h}
