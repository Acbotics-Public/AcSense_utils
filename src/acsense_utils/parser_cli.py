import os
from acsense_utils._parser.modified_parser import ModParser
import glob
import pandas as pd
import numpy as np
from acsense_utils._parser.headers import *
from acsense_utils._parser.external_sensor_data import *
from acsense_utils._parser.internal_sensor_data import *
from datetime import datetime, timezone
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import argparse
import time


'''
PARSER_V2 parses through AC and SENS files in directories and returns CSV files 
with an appended epoch column from an interpolation from RTC and/or GPS. The exported 
files are in folder 'parsed_{path}' with headers 'AC' and 'SENS'. '''
DEFAULT_TICK_RATE = 1e-8  

def run_parser_cli():
    parser = argparse.ArgumentParser(
        prog="AcSense Parser",
        description="Utility to load, parse and export AcSense data from device logs from CLI",
        epilog="Acbotics Research, LLC",
    )
    parser.add_argument(
        "-p", "--path_src", 
        nargs="?", 
        default=None, 
        help="Path to input files or directory")
    parser.add_argument(
        "-ie", "--int_ext",
        type = str,
        nargs="?", 
        help="String input: INT for Internal (1 channel) ADC or EXT for External (16 channel) ADC for AC data",
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=max(1,cpu_count()-2),
        help=f"Number of workers. Defaults to {max(1,cpu_count() - 2)}.",
    )
    parser.add_argument(
        "-o", "--output_directory",
        type=str,
        nargs="?",
        help=f"Output directory for parsed files. Defaults to path_src/parsed_(n)",
        )

    args = parser.parse_args()

    path_src = args.path_src or input("Enter path to input files or directory to be parsed: ")
    use_int = args.int_ext or input("Enter INT for internal ADC or EXT for external ADC: ")
    if use_int == "INT":
        use_int = True
    elif use_int == "EXT":
        use_int = False
    num_cores = args.count
    if args.output_directory:
        path_out = args.output_directory
    else:
        if os.path.isdir(path_src):
            path_out = path_src
        else:
            path_out = os.path.split(path_src)[0]

    
    output_dir = os.path.join(path_out, f"parsed_{base_number(path_out)}")

    print(f"Your output directory is {output_dir}")

    rtc_data = pd.DataFrame()
    gps_data = pd.DataFrame()
    genser_data = pd.DataFrame()
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
        ac_files_all = [f for f in files_to_process if os.path.basename(f).startswith("AC")]
        indicies = []
        i=0
        for fn in sens_files:
            base = os.path.splitext(os.path.basename(fn))[0]
            split_index = int(base.split("_")[-1])
            indicies.append(split_index-1)
        indicies.append(len(ac_files_all)-1)
        for fn in sens_files:
            ac_files = ac_files_all[indicies[i]: indicies[i+1]]
            i += 1
            result = process_sens_file(fn,output_dir)
            if result is None or result[0] is None:
                print(f"Skipping {fn} due to parse error. Check if SENS file is corrupted")
                continue
            rtc_data, gps_data, genser_data = result
    
            num_workers = min(num_cores, len(ac_files))

            args_list = [(fn, use_int, output_dir, rtc_data, gps_data, genser_data) for fn in ac_files]
            print(f"Parsing AC files with {num_workers} workers: ")
            s = time.perf_counter()
            with Pool(processes=num_workers) as pool:
                result = list(
                tqdm(
                    pool.imap_unordered(process_ac_file, args_list),
                    total=len(ac_files),
                )
                )
            e = time.perf_counter()
            print(f"Parsed {len(ac_files)} AC files in {(e-s):.3f} seconds.")
            if not os.path.isdir(os.path.join(output_dir,"AC")):
                print(f"AC files were not exported. Check if AC data in input path exists and type of AC data. use_int is set to {use_int}")
        print("Done")
    else:
        if os.path.basename(path_src).startswith("SENS"):
            result = process_sens_file(path_src,output_dir)
        elif os.path.basename(path_src).startswith("AC"):
            result = process_ac_file(args = (path_src, use_int, output_dir,rtc_data, gps_data, genser_data))


def process_sens_file(fn, output_dir):
    p = ModParser()
    base = os.path.basename(fn)
    base_file = os.path.splitext(os.path.basename(fn))[0]
    split_index = int(base_file.split("_")[-1])
    exported = []
    gps_bool = False
    rtc_bool = False
    print(f"Parsing {base}: ")
    if base.startswith("SENS"):
        try:
            parser_list = p.parse_sense_file(fn)
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError in {fn}: {e}")
        for i in range(len(parser_list)): #grab rtc and gps data first
            parser_obj = parser_list[i]['header']
            obj = (type(parser_list[i]['parser']).__name__) 
            if obj == "RTC_Data" or obj == "GPS_Data" or obj == "Generic_Serial_Data":
                if type(parser_obj) is Generic_Header:
                    if obj == "RTC_Data":
                        parser_dict = parser_list[i]['parser'].as_dict()
                        parser_df = pd.DataFrame(parser_dict)
                        rtc_data = get_rtc_fixes(parser_df)
                    elif obj == "GPS_Data":
                        parser_dict = parser_list[i]['parser'].as_dict()
                        parser_df = pd.DataFrame(parser_dict)
                        gps_data = get_gps_data(gps_data=parser_df)
                    elif obj == "Generic_Serial_Data":
                        parser_dict = parser_list[i]['parser'].as_dict()
                        parser_df = pd.DataFrame(parser_dict)
                        genser_data = get_gps_data(genser_data=parser_df)
        for i in range(len(parser_list)): #add epochs to all sens data
            parser_obj = parser_list[i]['header']
            obj = (type(parser_list[i]['parser']).__name__) 
            if type(parser_obj) is Generic_Header:
                parser_dict = parser_list[i]['parser'].as_dict()
                if any(parser_dict.values()):
                    parser_df = pd.DataFrame(parser_dict)
                    if not gps_data.empty:
                        parser_df = append_epoch_gps(parser_df, gps_data)
                        gps_bool = True
                    if not genser_data.empty:
                        parser_df = append_epoch_genser(parser_df, genser_data)
                        gps_bool=True
                    if not rtc_data.empty:
                        parser_df = append_epoch_rtc(parser_df, rtc_data)
                        rtc_bool = True
                    out_path = get_output_path(output_dir, "SENS", sensor_type=obj, filename=f"{obj}_{split_index}.csv")
                    parser_df.to_csv(out_path, index=False)
                    exported.append(f"{obj}_{split_index}.csv")
                #else:
                    #print(f"{obj} has no data. No output file will be made.")
    added = []
    if gps_bool:
        added.append("Added Epoch_GPS column")
    if rtc_bool:
        added.append("Added Epoch_RTC column.")
    msg = ", ".join(added) if added else "No Epoch column added."
    print(f"Exported {', '.join(exported)}. \n{msg}")
    return rtc_data, gps_data, genser_data

def process_ac_file(args):
    fn, use_int, output_dir, rtc_data, gps_data, genser_data= args    
    base = os.path.splitext(os.path.basename(fn))[0]
    p = ModParser()
    if base.startswith("AC"):
        #status_queue.put((base, "parsing"))
        #print(f"parsing {base}")
        s_time = time.perf_counter()
        parser_list = p.parse_ac_file(fn, use_int)
        e_time = time.perf_counter()
        elapsed_time = e_time - s_time
        #(f"opt parsed in {elapsed_time:.6f} seconds")
        for i in range(len(parser_list)):
            parser_obj = parser_list[i]['header']
            if (type(parser_obj) is SPI_ADC_Header and use_int == False) or (type(parser_obj) is Internal_ADC_Header and use_int == True):
                s_time = time.perf_counter()
                parser_dict = parser_list[i]['parser'].as_dict_opt()
                e_time = time.perf_counter()
                elapsed_time = e_time - s_time
                #print(f"To_dict in {elapsed_time:.6f} seconds")
                parser_df = pd.DataFrame(parser_dict)
                if not gps_data.empty or not genser_data.empty:
                    parser_df = append_epoch_gps(parser_df, gps_data,genser_data)
                if not rtc_data.empty:
                    parser_df = append_epoch_rtc(parser_df, rtc_data)
                f_name = f"{base}.csv"
                out_path = get_output_path(output_dir, "AC", filename=f_name)
                s_time = time.perf_counter()
                parser_df.to_csv(out_path, index=False)
                e_time = time.perf_counter()
                elapsed_time = e_time - s_time
                #print(f"To_csv in {elapsed_time:.6f} seconds")
                #print(f"exported {f_name}")
                return 

def base_number(dir, n=1):
    output_dir = os.path.join(dir, f"parsed_{n}")
    if os.path.isdir(output_dir):
        n += 1
        return base_number(dir, n)
    else:
        return n

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

def get_gps_data(gps_data=pd.DataFrame(), genser_data=pd.DataFrame()):
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
                        dt_float = float(gps_data['unix_time'])
                        rows.append({
                            'timestamp': gps_data["timestamp"].iloc[i],
                            'dt': dt_float
                        })
            except (IndexError, ValueError) as e:
                continue
        if first_fix == False: ## no valid fixes but may have been a time fix
            for i, sentence in enumerate(nmea):
                try:
                    if (sentence.startswith("$GPRMC") or sentence.startswith("$GNRMC")):
                        fields = sentence.split(",")
                        time_str = fields[1]
                        date_str = fields[9]
                        dt = datetime.strptime(date_str + time_str[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
                        dt_float = dt.timestamp()
                        now = time.time()
                        if now - 31536000 * 5 <= dt_float <= now + 31536000 * 5:
                            rows.append({
                                'timestamp': gps_data["timestamp"].iloc[i],
                                'dt': dt_float
                            })
                except (IndexError, ValueError) as e:
                    continue
            if not rows.empty: tqdm.write("No positional GPS fixes found, time fix may have been aquired, use judgement on Epoch_GPS")
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
            except (IndexError, ValueError) as e:
                continue
    gps_fixes = pd.DataFrame(rows, columns=['timestamp', 'dt'])
    return gps_fixes

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

def append_epoch_gps(parser_df, gps_data):
    epoch = gps_interp(parser_df['timestamp'],gps_data)
    parser_df.insert(1, 'Epoch_GPS', epoch)
    return parser_df

def append_epoch_genser(parser_df, genser):
    epoch = gps_interp(parser_df['timestamp'],genser)
    parser_df.insert(1, 'Epoch_GPS', epoch)
    return parser_df

def append_epoch_rtc(parser_df, rtc_data):
    epoch = rtc_interp(parser_df['timestamp'], rtc_data)
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
    run_parser_cli()
