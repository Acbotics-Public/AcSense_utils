#!/usr/bin/env python3

# example plotter of sensor data v. time!
import os
import pandas as pd
import argparse
import logging
import glob

from ._plotter.plotter_funcs import *

## Adjust mag cal as needed
magz_cal = -15000
magx_cal = -2800
magy_cal = -2800


FS = 30000
TICK = 1e-9  # sample interval for internal timestamp is s
acc_correction = 9.81 / 2059  # acceleration correction for m/s^2


parser = argparse.ArgumentParser(
    prog="AcSense Plotter",
    description="Plot pre-parsed data from AcSense data logs",
    epilog="Need additional support? Contact Acbotics Research LLC (support@acbotics.com)",
)
parser.add_argument(
    "--dir", type=str, default="/tmp/", help="Directory of parsed AcSense data"
)
parser.add_argument(
    "--plot_ac", action="store_true", help="Enable plotting of AC* file data"
)
parser.add_argument(
    "--plot_cam", action="store_true", help="Enable rendenring frames from camera data"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Show verbose logs (detailed)"
)
parser.add_argument("--debug", action="store_true", help="Show verbose debugging logs")

args, unknown = parser.parse_known_args()

if args.debug and not args.verbose:
    args.verbose = True

logging.basicConfig(
    # format="[%(asctime)s] %(name)s.%(funcName)s() : \n\t%(message)s",
    format=(
        "[%(asctime)s] %(levelname)s: %(filename)s:L%(lineno)d : %(message)s"
        if args.verbose
        else "[%(asctime)s] %(levelname)s: %(message)s"
    ),
    # format="[%(asctime)s] %(levelname)s: %(filename)s:L%(lineno)d : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # level=logging.DEBUG,
    level=logging.INFO,
    force=True,
)

logger = logging.getLogger(__name__)

if args.debug:
    logger.setLevel(logging.DEBUG)

if not os.path.exists(args.dir):
    logger.warning(f'Path "{args.dir}" does not exist! Please enter a valid path.')
    exit(1)


def locate_files(pattern):

    select_files = sorted(glob.glob(os.path.join(args.dir, f"{pattern}*.csv")))
    logger.info(f"Located {len(select_files)} {pattern} files to process")
    select_file_roots = sorted(set(["_".join(x.split("_")[0:4]) for x in select_files]))

    return select_files, select_file_roots


def run_plotter():
    logger.info(f"Processing data from : {args.dir}")
    _, sens_file_roots = locate_files("SENS")
    _, imu_file_roots = locate_files("IMU")

    old_data_dict = {}

    for file_root in sens_file_roots + imu_file_roots:
        data_dict = {}
        logger.info(f"Processing file root : {os.path.basename(file_root)}")

        sens_features = [
            # (key, file_pattern, loginfo_pattern),
            # core modules
            ("ipt", "PTS", "Internal PTS"),
            ("int_adc", "intadc", "Internal ADC"),
            ("imu", "IMU", "IMU"),
            # add-on modules
            ("rtc", "RTC", "RTC"),
            ("gps", "gps", "GPS"),
            ("ping", "ping", "Ping Echosounder"),
            ("mag_raw", "Magnetometer", "Magnetometer"),
            ("ept30", "External_PTS_Bar30", "External PTS (Bar30)"),
            ("ept100", "External_PTS_Bar100", "External PTS (Bar100)"),
            ("ctd", "CTD", "CTD"),
            ("cam", "Image_Meta", "Image Meta (times)"),
        ]

        for feat in sens_features:
            _key = feat[0]
            _file_pattern = feat[1]
            _loginfo_pattern = feat[2]

            _file = glob.glob(file_root + f"*{_file_pattern}.csv")

            if len(_file) > 1:
                logger.warning(
                    "Encountered multiple eligible files! "
                    "This indicates manual user intervention on file names."
                )

            if len(_file) > 0:
                data_dict[_key] = pd.read_csv(_file[0])
                logger.info(f"got SENS : {_loginfo_pattern} data")
            else:
                # data_dict[_key] = None
                logger.debug(f"no {_loginfo_pattern} data")

        if "imu" in data_dict:
            logger.info("Processing IMU correction")
            # process IMU to get corrected pitch/roll
            imu_data_corrected = process_IMU(data_dict["imu"])
            data_dict["imu"] = imu_data_corrected

            if "mag_raw" in data_dict:
                logger.info("Processing Magnetometer correction")
                # process mag to get heading
                mag_data_corrected = get_heading(
                    data_dict["mag_raw"], imu_data_corrected
                )
                data_dict["imu_mag"] = mag_data_corrected

        # handle timing reference from known data:
        RTC_data = None
        if "rtc" in data_dict:
            RTC_data = data_dict["rtc"]

        if "gps" in data_dict:
            gps_data_select = data_dict["gps"][~data_dict["gps"]["lat"].isnull()]
            RTC_data = get_GPS_RTC_data(data_dict["gps"])
            RTC_data = RTC_data[RTC_data["unix_time"] > 4e08]
            data_dict["gps"] = gps_data_select

        # handle internal ADC data in SENS logs
        intadc_data = None
        if "int_adc" in data_dict:
            intadc_data = data_dict["int_adc"]

        # next, get and plot the acoustic data:
        logger.info("plotting sens data")
        # Plot!
        outfilename = file_root + "_sensFilePlot.png"

        plot_data_dict(data_dict, RTC_data, outfilename, intadc_data)

        ac_data = None

        if args.plot_ac:

            acoustic_files = sorted(glob.glob(os.path.join(args.dir, "AC*spiadc.csv")))
            acoust_type = "EXT"
            if len(acoustic_files) == 0:
                acoustic_files = sorted(
                    glob.glob(os.path.join(args.dir, "AC*intadc.csv"))
                )
                acoust_type = "INT"

            # logger.info(acoustic_files)
            for acoustic_file in acoustic_files:
                ac_data = pd.read_csv(acoustic_file)
                #        logger.info(ac_data)

                # create a plot:
                plot_name = acoustic_file.split(".dat")[0] + ".png"
                if acoust_type == "EXT":
                    plot_multichannel_acoustics(
                        get_xaxis(ac_data, RTC_data), ac_data, plot_name
                    )
                else:

                    plot_data_dict(data_dict, RTC_data, plot_name, ac_data)
                del ac_data
                plt.close("all")

        if args.plot_cam:
            # plot cam
            plot_cam_frame_points(data_dict, RTC_data, plot_name, intadc_data=ac_data)


if __name__ == "__main__":
    run_plotter()
