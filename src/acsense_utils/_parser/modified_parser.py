'''MODIFIED_PARSER extends Parser and overwrites methods to exclude the GUI '''

import logging
import os
from acsense_utils._parser.parser import Parser  # correct
from tqdm import tqdm

logger = logging.getLogger(__name__)

interval = 10 #seconds

class ModParser(Parser):
    def __init__(self, block_size=512, double_sample_rate=False, use_int_sr=False): #same as parser
        super().__init__(block_size, double_sample_rate, use_int_sr) 

    def parse_sense_file(self, fn): 
        self.sens_dict = {}
        file_size = os.stat(fn).st_size
        if file_size == 0:
            print("file size is not printing!")
            return self.parsers

        with open(fn, "rb") as f, tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
            while f.tell() < file_size:
                pos = f.tell()
                self.read_block("INT", f)
                new_pos = f.tell()
                pbar.update(new_pos - pos)
                if new_pos == pos:
                    raise RuntimeError(
                        f"read_block() did not advance the file pointer at offset {pos}"
                    )
        return self.parsers

    def parse_ac_file(self, fn, use_int, export=False, output_dir=None):
        file_size = os.stat(fn).st_size
        if file_size == 0:
            return self.parsers

        with open(fn, "rb") as f:
            while True:
                start_tell = f.tell()
                self.read_block(
                    "INT" if use_int else "EXT",f, ac_file=True,export=export,output_dir=output_dir,input_filename=fn,
                )
                if start_tell == f.tell() or f.tell() >= os.fstat(f.fileno()).st_size:
                    break

        return self.parsers

