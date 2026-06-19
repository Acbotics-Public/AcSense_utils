'''MODIFIED_PARSER extends Parser and overwrites methods to exclude the GUI '''

import logging
import os
from acsense_utils._parser.parser import Parser  # correct

logger = logging.getLogger(__name__)

interval = 10 #seconds

class ModParser(Parser):
    def __init__(self, block_size=512, double_sample_rate=False, use_int_sr=False): #same as parser
        super().__init__(block_size, double_sample_rate, use_int_sr) 

    def parse_sense_file(self, fn): #got rid of prog_bar and tqdm depedencies
        self.sens_dict = {}
        file_size = os.stat(fn).st_size
        if file_size == 0:
            print("file size is not printing!")
            return self.parsers

        with open(fn, "rb") as f:
            while True:
                    start_tell = f.tell()
                    self.read_block("INT", f)
                    if f.tell() >= file_size: #i think this was causing the loop there was no break condition
                        break
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

