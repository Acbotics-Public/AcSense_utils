from acsense_utils._parser.mod_parser_script import parse
import os
import pandas as pd
from acsense_utils._parser.headers import *
from acsense_utils._parser.external_sensor_data import *
from acsense_utils._parser.internal_sensor_data import *
import dill as pickle

def main():
    path="/home/acbotics2/AcSense_utils/PLUTOS_testdata-20260526T164605Z-3-001/PLUTOS_testdata"
    results = parse(path)
    parsed_ac = {} #list of dataframes for each AC file
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
            parsed_sens = {} #dictionary of dataframes with outer having a key of the sensor type
            for i in range(len(parser_list)):
                parser_obj = parser_list[i]['header']
                if type(parser_obj) is Generic_Header:
                    obj = (type(parser_list[i]['parser']).__name__)
                    parser_dict = parser_list[i]['parser'].as_dict()
                    parser_df = pd.DataFrame(parser_dict)
                    parsed_sens[obj] = parser_df
    
    done = [parsed_ac, parsed_sens]
    with open("cached_data2.pkl", "wb") as file:
        pickle.dump(done, file)
    print("Parsed Data Saved to Hard Drive")


if __name__ == "__main__":
    main()

