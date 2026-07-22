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
import bisect
import gc
import argparse

'''
PARSER_V2 parses through AC and SENS files in directories and returns CSV files 
with an appended epoch column from an interpolation from RTC and/or GPS. The exported 
files are in folder 'parsed_{path}' with headers 'AC' and 'SENS'. '''
DEFAULT_TICK_RATE = 1e-8  

def main():
    interval = 10 #future development
    parser = argparse.ArgumentParser(description="Inputs to parser")
    path_src=input("Enter path to input files or directory to be parsed: ")
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of cores")


    use_int = True #also future development (false is external ADC)
    rtc_data = []
    gps_data = []
    genser_data = []
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
            rtc_data, gps_data, genser_data = result
    
        num_workers = min(cpu_count(), len(ac_files))
        print(f"Processing {len(ac_files)} AC files with {num_workers} workers")
        with Pool(processes=num_workers) as pool:
            pool.map(process_ac_file, [(fn, path_src, use_int, rtc_data, gps_data, genser_data) for fn in ac_files])
        print("Done!")
    else:
        if os.path.basename(path_src).startswith("SENS"):
            result = process_sens_file(path_src, path_src)
        elif os.path.basename(path_src).startswith("AC"):
            result = process_ac_file(args = (path_src, path_src, use_int, rtc_data, gps_data, genser_data))


def process_sens_file(fn, path_src):
    p = ModParser()
    base_dir = os.path.join(path_src, f"parsed_{os.path.basename(path_src.rstrip('/'))}")
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
        for i in range(len(parser_list)): #add epochs to all sens data
            parser_obj = parser_list[i]['header']
            obj = (type(parser_list[i]['parser']).__name__) 
            if type(parser_obj) is Generic_Header:
                parser_dict = parser_list[i]['parser'].as_dict()
                if any(parser_dict.values()):
                    parser_df = pd.DataFrame(parser_dict)
                    if not gps_data.empty or not genser_data.empty:
                        print(f"Adding Epoch_GPS Col to {obj}")
                        parser_df = append_epoch_gps(parser_df, gps_data,genser_data)
                    if not rtc_data.empty:
                        print(f"Adding Epoch_RTC Col to {obj}")
                        parser_df = append_epoch_rtc(parser_df, rtc_data)
                    out_path = get_output_path(base_dir, "SENS", sensor_type=obj, filename=f"{obj}.csv")
                    parser_df.to_csv(out_path, index=False)
                    print(f"Exported file {obj}.csv")
                else:
                    print(f"{obj} has no data. No output file will be made.")
    return rtc_data, gps_data, genser_data


def process_ac_file(args):
    fn, path_src, use_int, rtc_data, gps_data, genser_data = args    
    base = os.path.splitext(os.path.basename(fn))[0]
    base_dir = os.path.join(path_src, f"parsed_{os.path.basename(path_src.rstrip('/'))}")
    print(f"Parsing {base} ...")
    p = ModParser()
    if base.startswith("AC"):
        parser_list = p.parse_ac_file(os.path.join(path_src, fn), use_int)
        for i in range(len(parser_list)):
            parser_obj = parser_list[i]['header']
            if (type(parser_obj) is SPI_ADC_Header and use_int == False) or (type(parser_obj) is Internal_ADC_Header and use_int == True):
                parser_dict = parser_list[i]['parser'].as_dict() #['timestamp', 'sample_count', 'channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7']
                print(f"My header is {type(parser_obj)} and use_int is {use_int}") 
                parser_df = pd.DataFrame(parser_dict)
                if not gps_data.empty or not genser_data.empty:
                    parser_df = append_epoch_gps(parser_df, gps_data,genser_data)
                if not rtc_data.empty:
                    parser_df = append_epoch_rtc(parser_df, rtc_data)
                f_name = f"{base}.csv"
                out_path = get_output_path(base_dir, "AC", filename=f_name)
                parser_df.to_csv(out_path, index=False)
                print(f"Exported file {f_name}")
                break 
            else: print(f"My header is {type(parser_obj)} and use_int {use_int}") 
                             



def gps_interp(ticks, gps_fixes):
    ticks = np.asarray(ticks, dtype=float)
    xp = gps_fixes['timestamp'].to_numpy(dtype=float)
    yp = gps_fixes['dt'].to_numpy(dtype=float)
    n = len(xp)
    #one fix
    if n == 1:
        return yp[0] + (ticks - xp[0]) * DEFAULT_TICK_RATE
    epoch = np.interp(ticks, xp, yp)
    ##ticks fall before xp[0]
    before = np.searchsorted(ticks, xp[0], side='left')
    if before > 0:
        slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
        epoch[:before] = yp[0] + (ticks[:before] - xp[0]) * slope
    #ticks fall after xp[-1]
    after = np.searchsorted(ticks, xp[-1], side='right')
    if after < len(ticks):
        slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        epoch[after:] = yp[-1] + (ticks[after:] - xp[-1]) * slope
    return epoch

def rtc_interp(ticks, rtc_fixes):
    ticks = np.asarray(ticks, dtype=float)
    rtc_fixes = rtc_fixes.sort_values('timestamp')
    xp = rtc_fixes['timestamp'].to_numpy(dtype=float)
    yp = rtc_fixes['dt'].to_numpy(dtype=float)
    n = len(xp)
    if n == 1:
        return yp[0] + (ticks - xp[0]) * DEFAULT_TICK_RATE
    epoch = np.interp(ticks, xp, yp)
    before = np.searchsorted(ticks, xp[0], side='left')
    if before > 0:
        slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
        epoch[:before] = yp[0] + (ticks[:before] - xp[0]) * slope
    after = np.searchsorted(ticks, xp[-1], side='right')
    if after < len(ticks):
        slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
        epoch[after:] = yp[-1] + (ticks[after:] - xp[-1]) * slope
    return epoch

def get_gps_data(gps_data, genser_data):
    rows = []
    first_fix = False
    if not gps_data.empty:
        nmea = gps_data["raw_nmea"]
        for i, sentence in enumerate(nmea):
            try:
                if (sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC")): # and sentence.split(",")[2] == "A":
                    if sentence.split(",")[2] == "A" and first_fix == False:
                        first_fix = True
                    if first_fix == True:
                        fields = sentence.split(",")
                        time_str = fields[1]
                        date_str = fields[9]
                        dt = datetime.strptime(date_str + time_str[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
                        dt_float = dt.timestamp()
                        rows.append({
                            'timestamp': gps_data["timestamp"].iloc[i],
                            'dt': dt_float
                        })
            except (IndexError, ValueError) as e:
                continue
    elif not genser_data.empty:
        for i, fmt in enumerate(genser_data['format']):
            try:
                if fmt.startswith("NMEA") and genser_data['serial_string'].iloc[i].split(",")[2] == "A":
                    fields = genser_data['serial_string'].iloc[i].split(",")
                    time_str = fields[1]
                    date_str = fields[9]
                    dt = datetime.strptime(date_str + time_str[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
                    dt_float = dt.timestamp()
                    rows.append({
                        'timestamp': genser_data["timestamp"].iloc[i],
                        'dt': dt_float
                    })
            except IndexError as e:
                continue
    gps_fixes = pd.DataFrame(rows, columns=['timestamp', 'dt'])
    return gps_fixes

#converts timestr to epochs 
def get_rtc_fixes(rtc_data):
    rows = []
    if not rtc_data.empty:
        for i in range(len(rtc_data)):
            try:
                rtc_start = rtc_data["timestr"].iloc[i]
                dt = datetime.strptime(rtc_start, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                dt_float = dt.timestamp()
                rows.append({
                    'timestamp': rtc_data["timestamp"].iloc[i],
                    'dt': dt_float
                })
            except (ValueError, IndexError):
                continue
    rtc_fixes = pd.DataFrame(rows, columns=['timestamp', 'dt'])
    return rtc_fixes

def append_epoch_gps(parser_df, gps_data, genser_data):
    gps_fixes = get_gps_data(gps_data, genser_data)
    if gps_fixes.empty:
        #print("No valid GPS fixes found. Epoch_GPS column will not be added.")
        return parser_df
    epoch = gps_interp(parser_df['timestamp'],gps_fixes)
    parser_df.insert(1, 'Epoch_GPS', epoch)
    return parser_df

def append_epoch_rtc(parser_df, rtc_data):
    rtc_fixes = get_rtc_fixes(rtc_data)

    if rtc_fixes.empty:
        print("No valid RTC data found. Epoch_RTC column will not be added.")
        return parser_df

    epoch = rtc_interp(parser_df['timestamp'], rtc_fixes)
    insert_at = 2 if 'Epoch_GPS' in parser_df.columns else 1
    parser_df.insert(insert_at, 'Epoch_RTC', epoch)
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

