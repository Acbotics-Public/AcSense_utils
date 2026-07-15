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
MOD_PARSER_SCRIPT parses through ACO and SENS files or directories and returns CSV files 
with an appended epoch column. The exported files are in folder 'output' with headers 'ACO' and 'SENS'. '''

#unoptimized time: 6:13 minutes
def main():
    '''----------------------------------------------------------------------------------------------------'''
    '''Configurables here. Change to desired input full path name and interval of exports'''
    interval = 10 #seconds
    path="/home/acbotics2/AcSense_utils/eng_test_data"
    '''----------------------------------------------------------------------------------------------------'''
        #with open("cached_data2.pkl", "rb") as file:
        #done = pickle.load(file)
        #print("loaded in data")

    results = parse(path)
    parsed_ac = {} #list of dataframes for each AC file
    parsed_sens = {} 
    for key in results: #iterate through files 
        parser_list = results[key]#iterate through the returned parsers
        fn = os.path.basename(key).split("/")[-1]
        print(f"Assembling {fn} ...")
        if fn.startswith("AC"):
            for i in range(len(parser_list)):
                parser_obj = parser_list[i]['header']
                if type(parser_obj) is SPI_ADC_Header: #is this external? do i need to check for internal too orrr
                    parser_dict = parser_list[i]['parser'].as_dict() #['timestamp', 'sample_count', 'channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7']
                    parser_df = pd.DataFrame(parser_dict)
                    parsed_ac[fn] = parser_df
                    break
        else:
            for i in range(len(parser_list)):
                parser_obj = parser_list[i]['header']
                if type(parser_obj) is Generic_Header:
                    obj = (type(parser_list[i]['parser']).__name__)
                    parser_dict = parser_list[i]['parser'].as_dict()
                    parser_df = pd.DataFrame(parser_dict)
                    parsed_sens[obj] = parser_df
    

    #parsed_ac is a dictionary with the file name as the key accessing it's dataframe
    #parsed_sens is parsed_ac['AC17_1.dat']a dictionary with the sensor name as the key accessing it's dataframe. 
    #Internal_PTS_Data, IMU_Data, RTC_Data, NAU7802_Data, External_PTS_Data_Bar30, External_PTS_Data_Bar100, Magnetometer_Data, Ping_Data, GPS_Data
    #Image_Meta_Data, Ctd_Data, RDO_Data, BNO_Data, BNR_Data, Atlas_Data, Turbidity_DF_Data, Generic_Serial_Data, Edge_Detect_Data
   #print(parsed_sens['RTC_Data'].columns) #'timestamp', 'seconds', 'minutes', 'hours', 'wday', 'mday', 'month','year', 'timestr

    tick = 1e-8 # 10 nanoseconds
    offset = parsed_sens["RTC_Data"]["timestamp"].iloc[0]
    rtc_start = parsed_sens["RTC_Data"]["timestr"].iloc[0]
    start_epoch = datetime.strptime(rtc_start, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    ac_start_epoch = start_epoch.timestamp()
    #print(ac_start_epoch) verified

    for fn in parsed_ac:
        ac_time = parsed_ac[fn]['timestamp']
        start_chunck = 0
        epoch = []
        end = len(ac_time)
        print(f"File being exported is {fn} and should export {math.ceil(((ac_time[end-1] - ac_time[0]) * tick)/10)} files")
        for i in range(0,len(ac_time)):
            epoch.append((ac_start_epoch + (ac_time.iloc[i]-offset) * tick).item())
            if ((ac_time[i]-ac_time[start_chunck])*tick >= interval and i != 0) or len(ac_time)-1 == i:
                #print(f"the length of the epoch: {len(epoch)} and start index: {start_chunck}")
                export_to_csv_ac(fn, start_chunck, i+1, epoch, parsed_ac)
                start_chunck = i+1#row
                epoch = []

    for sensor in parsed_sens:
        sens_time = parsed_sens[sensor]['timestamp']
        start_chunck = 0
        epoch = []
        end = len(sens_time)
        if end==0:
            print(f"{sensor} Data Empty, Skipping...")
        else:
            print(f"Sensor being exported is {sensor} and should export {math.ceil(((sens_time[end-1] - sens_time[0]) * tick)/10)} files")
            for i in range(end):
                epoch.append((ac_start_epoch + (sens_time.iloc[i]-offset) * tick).item())
                if ((sens_time[i]-sens_time[start_chunck])*tick >= interval and i != 0) or len(sens_time)-1 == i:
                    #print(f"the length of the epoch: {len(epoch)} and start index: {start_chunck}")
                    export_to_csv_sens(sensor, start_chunck, i+1, epoch, parsed_sens)
                    start_chunck = i+1#row
                    epoch = []



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
#verified tested functionality of a single AC file, and AC directory, SENS works now!
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
                # WHY ONLY HERE
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




