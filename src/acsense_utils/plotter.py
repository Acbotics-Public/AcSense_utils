#!/usr/bin/env python3

"""AcSense Plotter : plotter utility

This file contains the top-level logic to prepare data for plotting,
as well as the subsequent calls to the requisite plotting routines.
It can be used via the run_plotter() entrypoint from another Python
program, or via CLI using the acsense-plot entrypoint provided by
the package installation.
"""

import argparse
import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

from ._plotter.plotter_core import (  # type: ignore
    get_GPS_RTC_data,
    get_xaxis,
    plot_cam_frame_points,
    plot_data_dict,
    plot_multichannel_acoustics,
    process_IMU,
)

logger = logging.getLogger(__name__)


def get_file_root(fn):
    basename = os.path.basename(fn)
    dirname = os.path.dirname(fn)

    idx = 3
    spc = "_"
    if basename.startswith("AC"):
        idx = 2
        spc = ""
    elif basename.startswith("IMG"):
        idx = 2

    file_root = os.path.join(dirname, spc.join(basename.split("_")[0:idx]))
    return file_root


def locate_files(pattern, dir=None):
    if dir:
        select_files = sorted(glob.glob(os.path.join(dir, f"{pattern}*.csv")))
        logger.info(f"Located {len(select_files)} {pattern} files to process")
        select_file_roots = sorted(set([get_file_root(x) for x in select_files]))

        return select_file_roots


def run_plotter(parsed_dir=None, plot_ac=False, plot_cam=False, img_dir=None):
    if plot_cam:
        if not (img_dir and os.path.isdir(img_dir)):
            logger.error(f"Image directory {img_dir} is not a valid directory!")
        img_files = sorted(
            glob.glob(os.path.join(img_dir, "**/IMG*.jpg"), recursive=True)
        )
        if len(img_files) == 0:
            logger.error(
                "Trying to run plotter with image logs, but no valid image "
                "files located! Please check your top-level directory (or "
                "a subdirectory therein) contains valid image files "
                "(IMG*.jpg) matching the parsed data logs."
            )
            return

    logger.info(f"Processing data from : {parsed_dir}")
    sens_file_roots = locate_files("SENS", dir=parsed_dir)
    imu_file_roots = locate_files("IMU", dir=parsed_dir)

    for sens_root in sens_file_roots + imu_file_roots:
        data_dict = {}
        logger.info(f"Processing SENS file root : {os.path.basename(sens_root)}")

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

            _file = glob.glob(sens_root + f"*{_file_pattern}.csv")

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

            # if "mag_raw" in data_dict:
            #     logger.info("Processing Magnetometer correction")
            #     # process mag to get heading
            #     mag_data_corrected = get_heading(
            #         data_dict["mag_raw"], imu_data_corrected
            #     )
            #     data_dict["imu_mag"] = mag_data_corrected

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
        intadc_meta = None
        if "int_adc" in data_dict:
            intadc_data = data_dict["int_adc"]
            intadc_meta_file = sens_root + "_intadc_meta.txt"
            try:
                intadc_meta = json.load(open(intadc_meta_file, "r"))
            except Exception:
                logger.warning(
                    f"Could not locate metadata for {os.path.basename(intadc_meta_file)}"
                )

        # next, get and plot the acoustic data:
        logger.info(f"plotting sens data for {os.path.basename(sens_root)}")
        # Plot!
        outfilename = sens_root + "_sensFilePlot.png"

        plot_data_dict(data_dict, RTC_data, outfilename, intadc_data, intadc_meta)

        ac_data = None

        if plot_ac:
            ac_root = get_file_root(sens_root.replace("SENS", "AC"))
            logger.info(f"Processing AC file root : {os.path.basename(ac_root)}")

            acoustic_files = sorted(glob.glob(ac_root + "*spiadc.csv"))
            acoustic_files += sorted(glob.glob(ac_root + "*intadc.csv"))

            # logger.info(acoustic_files)
            for acoustic_file in acoustic_files:
                logger.info(f"plotting ac data : {os.path.basename(acoustic_file)}")
                ac_data = pd.read_csv(acoustic_file)

                ac_meta = None
                try:
                    ac_meta = json.load(
                        open(acoustic_file.replace(".csv", "_meta.txt"), "r")
                    )
                except Exception:
                    logger.warning(
                        f"Could not locate metadata for {os.path.basename(acoustic_file)}"
                    )

                # create a plot:
                plot_name = acoustic_file.replace(".csv", ".png")
                if acoustic_file.endswith("spiadc.csv"):
                    plot_multichannel_acoustics(
                        get_xaxis(ac_data, RTC_data), ac_data, ac_meta, plot_name
                    )
                else:
                    plot_data_dict(data_dict, RTC_data, plot_name, ac_data, ac_meta)
                del ac_data
                plt.close("all")

        if plot_cam:
            cam_root = os.path.basename(sens_root.replace("SENS_", "IMG"))
            cam_root = [x for x in img_files if cam_root in x]
            if len(cam_root) > 0:
                cam_root = get_file_root(cam_root[0])

                logger.info(f"Processing IMG file root : {os.path.basename(cam_root)}")
                plot_cam_frame_points(
                    data_dict,
                    RTC_data,
                    img_dir,
                    os.path.basename(cam_root),
                    os.path.dirname(outfilename),
                    intadc_data=intadc_data,
                )

    logger.info("Done!")


def run_plotter_cli():
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
        "--plot_cam",
        action="store_true",
        help="Enable rendenring frames from camera data",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Directory containing images for parsed AcSense data (required when using --plot_cam)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose logs (detailed)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show verbose debugging logs"
    )

    args, unknown = parser.parse_known_args()

    if args.debug and not args.verbose:
        args.verbose = True

    # Expand dir/path to absolute
    args.dir = os.path.abspath(os.path.expanduser(args.dir))

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

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(args.dir):
        logger.warning(f'Path "{args.dir}" does not exist! Please enter a valid path.')
        exit(1)

    run_plotter(
        parsed_dir=args.dir,
        plot_ac=args.plot_ac,
        plot_cam=args.plot_cam,
        img_dir=args.img_dir,
    )


if __name__ == "__main__":
    run_plotter_cli()
