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
PARSER_V2 parses through AC and SENS files or directories and returns CSV files 
with an appended epoch column. The exported files are in folder 'parsed_{path}' with headers 'AC' and 'SENS'. '''
 
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
    if path_src is None:
        print(f"Input path {path_src} is not valid. Please re-run and re-enter a valid path. Check working directory if path name is valid.")
        return None
    elif os.path.isdir(path_src): 
        files_to_process = sorted(
            glob.glob(os.path.join(path_src, "**/SENS*.dat"), recursive=True)
        )
        files_to_process += sorted(
            glob.glob(os.path.join(path_src, "**/AC*.dat"), recursive=True)
        )
        if len(files_to_process) == 0:
            return None
        path_rel = os.path.basename(path_src).split("/")[-1]
        #main loop here!!
        for fn in files_to_process:
            key = os.path.basename(fn).split("/")[-1]
            print(f"Parsing {key} ...")
            p = ModParser()
            try:
                base = os.path.basename(fn)
                if base.startswith("SENS"):
                    parser_list = p.parse_sense_file(os.path.join(path_src, fn))
                    print(f"Finished parsing SENS file")
                    for i in range(len(parser_list)): #grab rtc and gps data first
                        parser_obj = parser_list[i]['header']
                        obj = (type(parser_list[i]['parser']).__name__) 
                        if obj == "RTC_Data" or obj == "GPS_Data":
                            if type(parser_obj) is Generic_Header:
                                parser_dict = parser_list[i]['parser'].as_dict()
                                parser_df = pd.DataFrame(parser_dict)
                                print(f"Finished assembling {obj}")
                                if obj == "RTC_Data":
                                    rtc_data = parser_df
                                elif obj == "GPS_Data":
                                    gps_data = parser_df
                    for i in range(len(parser_list)): #add epochs to all sens data
                        parser_obj = parser_list[i]['header']
                        obj = (type(parser_list[i]['parser']).__name__) 
                        if type(parser_obj) is Generic_Header:
                            parser_dict = parser_list[i]['parser'].as_dict()
                            if any(parser_dict.values()):
                                parser_df = pd.DataFrame(parser_dict)
                                print(f"Finished assembling {obj}")
                                parser_mod = append_epoch(rtc_data, gps_data, parser_df)
                                print(f"Added Epoch Col")
                                out_path = get_output_path(f"./parsed_{path_rel}", "SENS", sensor_type=obj, filename=f"{obj}.csv")
                                parser_mod.to_csv(out_path, index=False)
                                print(f"Exported File")
                            else:
                                print(f"{obj} has no data. No output file will be made.")
                    
                elif base.startswith("AC"):
                    parser_list = p.parse_ac_file(os.path.join(path_src, fn), use_int)
                    print(f"Finished parsing AC file")
                    for i in range(len(parser_list)):
                        parser_obj = parser_list[i]['header']
                        if type(parser_obj) is SPI_ADC_Header: #external
                            parser_dict = parser_list[i]['parser'].as_dict() #['timestamp', 'sample_count', 'channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7']
                            parser_df = pd.DataFrame(parser_dict)
                            parser_mod = append_epoch(rtc_data, gps_data, parser_df)
                            print(f"Added Epoch Col")
                            out_path = get_output_path(f"./parsed_{path_rel}", "AC", filename=f"{base}.csv")
                            parser_mod.to_csv(out_path, index=False)
                            print(f"Exported File")
                            break                
                elif base.endswith("JPG") or base.endswith(".jpg"):
                    continue
                else:
                    continue

            except Exception as e:
                print(f"Exception encountered while parsing file {fn} :\n{e}")

'''APPEND EPOCH returns a AC or SENS dataframe with its corresponding epoch column from the inputted RTC data and unmodified parsed data'''
def append_epoch(rtc_data, gps_data, parser_df):
    tick = 1e-8 # 10 nanoseconds
    offset = rtc_data["timestamp"].iloc[0]
    rtc_start = rtc_data["timestr"].iloc[0]
    start_epoch = datetime.strptime(rtc_start, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    start_time = start_epoch.timestamp()
    time = parser_df['timestamp']
    epoch = start_time + (time-offset) * tick #theorically should be a n by 1 col vector
    parser_df.insert(1, 'epoch', epoch)
    return parser_df

def get_output_path(base_dir, category, sensor_type=None, filename=None):
    if category == "AC":
        out_dir = os.path.join(base_dir, "AC")
    else:  # SENS
        out_dir = os.path.join(base_dir, "SENS", sensor_type)

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)
 

if __name__ == "__main__":
    main()

