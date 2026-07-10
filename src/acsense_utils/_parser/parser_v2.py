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
from multiprocessing import Pool, cpu_count, current_process
'''
PARSER_V2 parses through AC and SENS files or directories and returns CSV files 
with an appended epoch column. The exported files are in folder 'parsed_{path}' with headers 'AC' and 'SENS'. '''
 
def main():
    '''----------------------------------------------------------------------------------------------------'''
    '''Configurables here. Change to desired input full path name and interval of exports'''
    interval = 10 #seconds
    path_src="/home/acbotics2/AcSense_utils/source_a_d7"
    '''----------------------------------------------------------------------------------------------------'''
        #with open("cached_data2.pkl", "rb") as file:
        #done = pickle.load(file)
        #print("loaded in data")

    use_int = False #assuming external only for the PLUTOS purposes
    if path_src is None:
        print(f"Input path {path_src} is not valid. Please re-run and re-enter a valid path. Check working directory if path name is valid.")
        return None
    if os.path.isdir(path_src): 
        files_to_process = sorted(
            glob.glob(os.path.join(path_src, "**/SENS*.dat"), recursive=True)
        )
        files_to_process += sorted(
            glob.glob(os.path.join(path_src, "**/AC*.dat"), recursive=True)
        )
        if len(files_to_process) == 0:
            return None
        
        sens_files = [f for f in files_to_process if os.path.basename(f).startswith("SENS")]
        ac_files = [f for f in files_to_process if os.path.basename(f).startswith("AC")]

        #proccess sens first
        for fn in sens_files:
            rtc_data, gps_data = process_sens_file(fn, path_src)

        num_workers = min(cpu_count(), len(ac_files))
        print(f"Processing {len(ac_files)} AC files with {num_workers} workers")
        with Pool(processes=num_workers) as pool:
            pool.map(process_ac_file, [(fn, path_src, use_int, rtc_data, gps_data) for fn in ac_files])
        print("Done!")


def process_sens_file(fn, path_src):
    p = ModParser()
    path_rel = os.path.basename(path_src).split("/")[-1]
    base = os.path.basename(fn)
    print(f"Parsing {base} ...")
    if base.startswith("SENS"):
        parser_list = p.parse_sense_file(os.path.join(path_src, fn))
        for i in range(len(parser_list)): #grab rtc and gps data first
            parser_obj = parser_list[i]['header']
            obj = (type(parser_list[i]['parser']).__name__) 
            if obj == "RTC_Data" or obj == "GPS_Data":
                if type(parser_obj) is Generic_Header:
                    parser_dict = parser_list[i]['parser'].as_dict()
                    parser_df = pd.DataFrame(parser_dict)
                    print(f"Loaded {obj}")
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
                    parser_mod = append_epoch(rtc_data, gps_data, parser_df)
                    out_path = get_output_path(f"./parsed_{path_rel}", "SENS", sensor_type=obj, filename=f"{obj}.csv")
                    parser_mod.to_csv(out_path, index=False)
                    print(f"Exported file {obj}.csv")
                else:
                    print(f"{obj} has no data. No output file will be made.")
    return rtc_data, gps_data


def process_ac_file(args):
    fn, path_src, use_int, rtc_data, gps_data = args    
    path_rel = os.path.basename(path_src).split("/")[-1]
    base = os.path.splitext(os.path.basename(fn))[0]
    print(f"Parsing {base} ...")
    p = ModParser()
    if base.startswith("AC"):
        parser_list = p.parse_ac_file(os.path.join(path_src, fn), use_int)
        for i in range(len(parser_list)):
            parser_obj = parser_list[i]['header']
            if type(parser_obj) is SPI_ADC_Header: #external
                parser_dict = parser_list[i]['parser'].as_dict() #['timestamp', 'sample_count', 'channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7']
                parser_df = pd.DataFrame(parser_dict)
                parser_mod = append_epoch(rtc_data, gps_data, parser_df)
                f_name = f"{base}.csv"
                out_path = get_output_path(f"./parsed_{path_rel}", "AC", filename=f_name)
                parser_mod.to_csv(out_path, index=False)
                print(f"Exported file {f_name}")
                break                


'''APPEND EPOCH returns a AC or SENS dataframe with its corresponding epoch column from the inputted RTC data and unmodified parsed data'''
def append_epoch(rtc_data, gps_data, parser_df):
    tick = 1e-8 # 10 nanoseconds
    print(gps_data.head())
    if gps_data.empty and not rtc_data.empty:
        offset = rtc_data["timestamp"].iloc[0]
        rtc_start = rtc_data["timestr"].iloc[0]
        start_epoch = datetime.strptime(rtc_start, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        start_time = start_epoch.timestamp()
        time = parser_df['timestamp']
        epoch = start_time + (time-offset) * tick #theorically should be a n by 1 col vector
        parser_df.insert(1, 'epoch', epoch) 
    elif not gps_data.empty and rtc_data.empty:
        print("gps data")
    elif not gps_data.empty and not rtc_data.empty:
        nmea = gps_data["raw_nmea"]
        for str in nmea:
            if str == "GPRMC":
                print("hlep")
    elif gps_data.empty and rtc_data.empty:
        print(f"RTC and GPS data not available. Epoch column will not be added.")
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

