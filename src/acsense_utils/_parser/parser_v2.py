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
from multiprocessing import Pool, cpu_count
import gc

'''
PARSER_V2 parses through AC and SENS files in directories and returns CSV files 
with an appended epoch column. The exported files are in folder 'parsed_{path}' with headers 'AC' and 'SENS'. '''
 
def main():
    interval = 10 #future development
    path_src=input("Enter path to input files or directory to be parsed: ")

    use_int = False #also future development (false is external ADC)

    start_time = 0
    offset = 0
    epoch_bool = False

    if path_src is None or not os.path.exists(path_src):
        print(f"Input path {path_src} is not valid. Please re-run and re-enter a valid path.")
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
            result = process_sens_file(fn, path_src)
            if result is None or result[0] is None:
                print(f"Skipping {fn} due to parse error. Check if SENS file is corrupted")
                continue
            start_time, offset, epoch_bool = result
    
        num_workers = min(cpu_count(), len(ac_files))
        print(f"Processing {len(ac_files)} AC files with {num_workers} workers")
        with Pool(processes=num_workers) as pool:
            pool.map(process_ac_file, [(fn, path_src, use_int, start_time, offset, epoch_bool) for fn in ac_files])
        print("Done!")
    else:
        if os.path.basename(path_src).startswith("SENS"):
            result = process_sens_file(path_src, path_src)
        elif os.path.basename(path_src).startswith("AC"):
            result = process_ac_file(args = (path_src, path_src, use_int, start_time, offset, epoch_bool))


def process_sens_file(fn, path_src):
    p = ModParser()
    path_rel = os.path.basename(path_src).split("/")[-1]
    base = os.path.basename(fn)
    print(f"Parsing {base} ...")
    if base.startswith("SENS"):
        try:
            parser_list = p.parse_sense_file(os.path.join(path_src, fn))
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError in {fn}: {e}, skipping file")
            return None, None, False
        except Exception as e:
            print(f"Exception in {fn}: {e}, skipping file")
            return None, None, False
        for i in range(len(parser_list)): #grab rtc and gps data first
            parser_obj = parser_list[i]['header']
            obj = (type(parser_list[i]['parser']).__name__) 
            if obj == "RTC_Data" or obj == "GPS_Data" or obj == "Generic_Serial_Data":
                if type(parser_obj) is Generic_Header:
                    parser_dict = parser_list[i]['parser'].as_dict()
                    parser_df = pd.DataFrame(parser_dict)
                    print(f"Loaded {obj}")
                    if obj == "RTC_Data":
                        rtc_data = parser_df
                    elif obj == "GPS_Data":
                        gps_data = parser_df
                    elif obj == "Generic_Serial_Data":
                        genser_data = parser_df
        start_time, offset, epoch_bool = get_epoch_vars(rtc_data,gps_data,genser_data)
        for i in range(len(parser_list)): #add epochs to all sens data
            parser_obj = parser_list[i]['header']
            obj = (type(parser_list[i]['parser']).__name__) 
            if type(parser_obj) is Generic_Header:
                parser_dict = parser_list[i]['parser'].as_dict()
                if any(parser_dict.values()):
                    parser_df = pd.DataFrame(parser_dict)
                    if epoch_bool:
                        parser_df = append_epoch(parser_df,start_time,offset)
                    out_path = get_output_path(f"./parsed_{path_rel}", "SENS", sensor_type=obj, filename=f"{obj}.csv")
                    parser_df.to_csv(out_path, index=False)
                    print(f"Exported file {obj}.csv")
                else:
                    print(f"{obj} has no data. No output file will be made.")
    return start_time, offset, epoch_bool


def process_ac_file(args):
    fn, path_src, use_int, start_time, offset, epoch_bool = args    
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
                if epoch_bool:
                    parser_df = append_epoch(parser_df, start_time, offset)
                f_name = f"{base}.csv"
                out_path = get_output_path(f"./parsed_{path_rel}", "AC", filename=f_name)
                parser_df.to_csv(out_path, index=False)
                print(f"Exported file {f_name}")
                break                


def get_epoch_vars(rtc_data, gps_data, genser_data):
    epoch_bool = True
    start_time = -1
    offset = -1
    if not gps_data.empty:
    #check for valid GPS fix first
        nmea = gps_data["raw_nmea"]
        for i, sentence in enumerate(nmea):
            if (sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC")) and sentence.split(",")[2] == "A":
                fields = sentence.split(",")
                time_str = fields[1]   
                date_str = fields[9]
                dt = datetime.strptime(date_str + time_str[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
                start_time = dt.timestamp()
                offset = gps_data["timestamp"].iloc[i] 
                print(f"Using GPS fix for epoch: {dt}")
                break
    elif not genser_data.empty:
        for i,format in enumerate(genser_data['format']):
            if format.startswith("NMEA") and genser_data['serial_string'].iloc[i].split(",")[2] == "A":
                fields = genser_data['serial_string'].iloc[i].split(",")
                time_str = fields[1]   
                date_str = fields[9]
                dt = datetime.strptime(date_str + time_str[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
                start_time = dt.timestamp()
                offset = genser_data["timestamp"].iloc[i] 
                print(f"Using GPS (garmin) fix for epoch: {dt}")
                break
    if not rtc_data.empty and offset == -1 and start_time == -1:
        #RTC if no valid GPS fix found
        print("No valid GPS fix found, falling back to RTC")
        offset = rtc_data["timestamp"].iloc[0]
        rtc_start = rtc_data["timestr"].iloc[0]
        start_epoch = datetime.strptime(rtc_start, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        start_time = start_epoch.timestamp()

    if rtc_data.empty and gps_data.empty and genser_data.empty:
        print("RTC and GPS data not available. Epoch column will not be added.")
        epoch_bool = False
    return start_time, offset, epoch_bool


def append_epoch(parser_df, start_time, offset):
    tick = 1e-8  
    time = parser_df['timestamp']
    epoch = start_time + (time - offset) * tick
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

