import logging
import os
from acsense_utils._parser.modified_parser import ModParser
import glob
import pandas as pd
import numpy as np
from acsense_utils._parser.headers import *
from acsense_utils._parser.external_sensor_data import *
from acsense_utils._parser.internal_sensor_data import *
from datetime import datetime, timezone
import math
import copy
import dill as pickle
'''
MOD_PARSER_SCRIPT parses through AC and SENS files or directories and returns CSV files 
with an appended epoch column. The exported files are in folder 'output' with headers 'ACO' and 'SENS'. '''

#unoptimized time: 6:13 minutes lol
# improvements: threading for the parsing and conversion to dataframe, vectorzing the epoch math, 
# --> and using data container/keeping less data loaded in 
def main():
    '''----------------------------------------------------------------------------------------------------'''
    '''Configurables here. Change to desired input full path name and interval of exports'''
    interval = 10 #seconds
    path_src="/home/acbotics2/AcSense_utils/PLUTOS_testdata-20260617T163459Z-3-001"
    '''----------------------------------------------------------------------------------------------------'''
        #with open("cached_data2.pkl", "rb") as file:
        #done = pickle.load(file)
        #print("loaded in data")

    use_int = False #assuming external only for the PLUTOS purposes
    parsed = {}
    if path_src is None:
        print(f"Input path {path_src} is not valid. Please re-run and re-enter a valid path. Check working directory if path name is valid.")
        return None
    #add in file parsing here 
    elif os.path.isdir(path_src): 
        files_to_process = sorted(
            glob.glob(os.path.join(path_src, "**/SENS*.dat"), recursive=True)
        )
        files_to_process += sorted(
            glob.glob(os.path.join(path_src, "**/AC*.dat"), recursive=True)
        )
        if len(files_to_process) == 0:
            return None
        #main loop here!!
        for fn in files_to_process:
            key = os.path.basename(fn).split("/")[-1]
            print(f"Parsing {key} ...")
            p = ModParser()
            try:
                base = os.path.basename(fn)
                if base.startswith("AC"):
                    parser_list = p.parse_ac_file(os.path.join(path_src, fn), use_int)
                    print(f"Finished parsing AC file")
                    for i in range(len(parser_list)):
                        parser_obj = parser_list[i]['header']
                        if type(parser_obj) is SPI_ADC_Header: #external
                            parser_dict = parser_list[i]['parser'].as_dict() #['timestamp', 'sample_count', 'channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7']
                            parser_df = pd.DataFrame(parser_dict)
                            break
                    print(f"Finished assembling data")
                    
                elif base.startswith("SENS"):
                    parser_list = p.parse_sense_file(os.path.join(path_src, fn))
                    print(f"Finished parsing SENS file")
                    for i in range(len(parser_list)):
                        parser_obj = parser_list[i]['header']
                        if type(parser_obj) is Generic_Header:
                            obj = (type(parser_list[i]['parser']).__name__)
                            parser_dict = parser_list[i]['parser'].as_dict()
                            parser_df = pd.DataFrame(parser_dict)
                            print(f"Finished assembling {obj}")
                elif base.endswith("JPG") or base.endswith(".jpg"):
                    continue
                else:
                    continue

            except Exception as e:
                print(f"Exception encountered while parsing file {fn} :\n{e}")




def export_to_csv_ac(fn, start_index, end_index, epoch, parsed_ac, base_dir="./output"):
    df = parsed_ac[fn]
    chunk = df.iloc[start_index:end_index].copy()
    chunk.insert(1, 'Epoch', epoch)
    formatted_time = datetime.fromtimestamp(epoch[0])
    base = os.path.splitext(os.path.basename(fn))[0]
    out_name = f"{base}_{formatted_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    out_path = get_output_path(base_dir, "ACO", filename=out_name)
    print(f"exporting {out_name}")
    chunk.to_csv(out_path, index=False)

def export_to_csv_sens(fn, start_index, end_index, epoch, parsed_sens, base_dir="./output"):
    df = parsed_sens[fn]
    chunk = df.iloc[start_index:end_index].copy()
    chunk.insert(1, 'Epoch', epoch)
    formatted_time = datetime.fromtimestamp(epoch[0])
    base = os.path.splitext(os.path.basename(fn))[0]
    out_name = f"{base}_{formatted_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    out_dir = os.path.join(base_dir, "SENS", fn)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    chunk.to_csv(out_path, index=False)

def get_output_path(base_dir, category, sensor_type=None, filename=None):
    if category == "ACO":
        out_dir = os.path.join(base_dir, "ACO")
    else:  # SENS
        out_dir = os.path.join(base_dir, "SENS", sensor_type)

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)
 
#takes in a path, ext vs int channel, export condition, output directory, sample rate, and mystery input var and outputs a directory with the file name as the key
#verified tested functionality of a single AC file, and AC directory, SENS works now! =^'.'^=
def parse(
        path_src=None,
        use_int=False, #EXT channel
        export=False,
        output_dir=None,
        use_double_sr=False,
        use_int_sr=False,
    ):
        # path_src = path_src or self.file_path
        parsed = {}
        if path_src is None:
            return None

        if os.path.isfile(path_src):
            p = ModParser(double_sample_rate=use_double_sr, use_int_sr=use_int_sr)
            fn = os.path.basename(path_src).split("/")[-1]
            if fn.startswith("AC"):
                parsed = {
                    fn: p.parse_ac_file(
                        path_src,
                        use_int,
                        export=export,
                        output_dir=output_dir,
                    )
                }
            elif fn.startswith("SENS"):
                parsed = {fn: p.parse_sense_file(path_src)}
        elif os.path.isdir(path_src):
            files_to_process = sorted(
                glob.glob(os.path.join(path_src, "**/SENS*.dat"), recursive=True)
            )
            files_to_process += sorted(
                glob.glob(os.path.join(path_src, "**/AC*.dat"), recursive=True)
            )
            if len(files_to_process) == 0:
                return None

            for fn in files_to_process:
                print(f"Parsing {fn} ...")
                p = ModParser(double_sample_rate=use_double_sr, use_int_sr=use_int_sr)
                try:
                    base = os.path.basename(fn)
                    if base.startswith("AC"):
                        parsed[fn] = copy.deepcopy(
                            p.parse_ac_file(os.path.join(path_src, fn), use_int)
                        )

                    elif base.startswith("SENS"):
                        parsed[fn] = copy.deepcopy(
                            p.parse_sense_file(os.path.join(path_src, fn))
                        )
                    elif base.endswith("JPG") or base.endswith(".jpg"):
                        continue
                    else:
                        continue

                except Exception as e:
                    print(f"Exception encountered while parsing file {fn} :\n{e}")


        return parsed


if __name__ == "__main__":
    main()

