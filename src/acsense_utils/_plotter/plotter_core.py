"""AcSense Plotter : core plotting functions

This file contains the rendering logic for high-level data structures.
It is intended as an implementation reference for the data structures
prepared by the top-level `plotter.py` file.
"""

import datetime
import glob
import logging
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy  # type: ignore
from matplotlib import use as mpl_use

from . import plotter_funcs as pf

mpl_use("agg")

if "seaborn-darkgrid" in plt.style.available:
    plt.style.use("seaborn-darkgrid")
elif "seaborn-v0_8-darkgrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-darkgrid")

logger = logging.getLogger(__name__)

# TICK = 1e-9  # sample interval for internal timestamp is s
TICK = 1e-8  # sample interval for internal timestamp is s
acc_correction = 9.81 / 2059  # acceleration correction for m/s^2


def get_xaxis(sensor_data, RTC_data=None):
    offset = 0
    if RTC_data is not None:
        # use RTC data to look up timestamps:
        # logger.info(RTC_data[0])
        try:
            # rtc_start = RTC_data.iloc[0]["timestr"]
            rtc_start = datetime.datetime.fromtimestamp(
                RTC_data.iloc[0]["unix_time"], datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            offset = RTC_data.iloc[0]["timestamp"]
        except Exception:
            rtc_start = "start"
            offset = 0

        xlabeln = f"Time [s] since {rtc_start}"

    else:
        logger.info("RTC data is None?")
        xlabeln = "Time [s]"
    # logger.info(sensor_data)
    if (
        type(sensor_data["timestamp"]) is list
        or type(sensor_data["timestamp"]) is np.ndarray
    ):
        tstamp_array = np.array(sensor_data["timestamp"]) - offset
    # elif isinstance(sensor_data["timestamp"],int) or isinstance(sensor_data["timestamp"],float):

    else:
        try:
            tstamp_array = sensor_data["timestamp"].to_numpy() - offset
        except Exception:
            tstamp_array = np.array([sensor_data["timestamp"]]) - offset
    tstamps = (
        tstamp_array * TICK
    )  # (np.array([x["timestamp"] for x in sensor_data]) - offset) * TICK
    # tstamps = tstamps - tstamps[0]
    return [tstamps, xlabeln]


def process_IMU(IMU_data):
    rot_info = {}
    rot_info["gyroz_rot"] = np.array([])
    rot_info["accx_rot"] = []
    rot_info["accy_rot"] = []
    rot_info["accz_rot"] = []
    rot_info["timestamp"] = []
    rot_info["gyro_total"] = []
    rot_info["acc_total"] = []
    rot_info["Pitch"] = []
    rot_info["Roll"] = []
    # if pitchcalc:
    # estimate pitch v. time:
    raw_X = IMU_data["Accel_X"].to_numpy() * acc_correction
    raw_Y = IMU_data["Accel_Y"].to_numpy() * acc_correction
    raw_Z = IMU_data["Accel_Z"].to_numpy() * acc_correction

    roll = 180 * np.arctan2(raw_Y, raw_Z) / np.pi
    pitch = 180 * np.arctan2(-raw_X, np.sqrt(raw_Z**2 + raw_Y**2)) / np.pi

    for ll in IMU_data.index:
        cur_data = IMU_data.loc[ll]

        Pitch = pitch[ll]
        Roll = roll[ll]
        rot_info = rotate_IMU_data(rot_info, cur_data, Pitch, Roll)

    # plot_IMU_data(IMU_data, rot_info)
    return rot_info


def rotate_IMU_data(rot_info, cur_data, Pitch, Roll):
    myrot = scipy.spatial.transform.Rotation.from_euler(
        "YX", np.array([Pitch, Roll]), degrees=True
    )
    gyro_z = (
        myrot.apply([cur_data["Gyro_X"], cur_data["Gyro_Y"], cur_data["Gyro_Z"]])[2]
        / 100
    )

    # rot_info["time"].append(cur_data["timestamp"] * AcSense_parser_utils.TICK)
    rot_info["gyroz_rot"] = np.append(rot_info["gyroz_rot"], gyro_z)
    rot_info["gyro_total"].append(
        np.sqrt(
            (cur_data["Gyro_X"] / 100) ** 2
            + (cur_data["Gyro_Y"] / 100) ** 2
            + (cur_data["Gyro_Z"] / 100) ** 2
        ),
    )

    acc_xy_rot = myrot.apply(
        [
            cur_data["Accel_X"] * acc_correction,
            cur_data["Accel_Y"] * acc_correction,
            cur_data["Accel_Z"] * acc_correction,
        ]
    )

    rot_info["timestamp"].append(cur_data["timestamp"])
    rot_info["Pitch"].append(Pitch)
    rot_info["Roll"].append(Roll)
    rot_info["accx_rot"].append(acc_xy_rot[0])
    rot_info["accy_rot"].append(acc_xy_rot[1])
    rot_info["accz_rot"].append(acc_xy_rot[2])
    rot_info["acc_total"].append(
        np.sqrt(acc_xy_rot[0] ** 2 + acc_xy_rot[1] ** 2 + acc_xy_rot[2] ** 2)
    )

    return rot_info


def get_GPS_RTC_data(GPS_data):
    GPS_select = GPS_data[["RMC" in str(x) for x in GPS_data["raw_nmea"].to_numpy()]]

    GPS_select2 = GPS_select[[",A," in x for x in GPS_select["raw_nmea"].to_numpy()]]

    if len(GPS_select2["raw_nmea"].to_numpy()) > 0:
        # use good fix to set time!!
        return GPS_select2
    else:
        # logger.info(GPS_select)

        return GPS_select


def get_active_features(data_dict):
    sens_features = {
        # key: plot_func,
        # core modules
        # "int_adc": plot_intadc,
        "imu": pf.plot_IMU_accelerations,
        "ipt": pf.plot_PTSInt,
        # add-on modules
        # "rtc": plot,
        "ept30": pf.plot_PTS,
        "ept100": pf.plot_PTS,
        "gps": pf.plot_lat_lon,
        "mag_raw": pf.plot_raw_mag,
        "ping": pf.plot_ping,
        "ctd": pf.plot_CTD,
        "cam": pf.plot_image_capture_times,
        # derivative fields (i.e. imu_mag)
        "imu_mag": pf.plot_pitchrollyaw,
    }

    active_feats = {
        _key: _fcn
        for _key, _fcn in sens_features.items()
        if _key in data_dict and len(data_dict[_key]["timestamp"]) > 0
    }

    return active_feats


def plot_data_dict(
    data_dict,
    RTC_data,
    outfile=None,
    intadc_data=None,
    intadc_meta=None,
    fig=False,
    tick=None,
):
    # Get active features to determine number of plots
    active_feats = get_active_features(data_dict)
    plot_rows = len(active_feats)

    # Handle feature-specific alterations to figure
    if intadc_data is not None:
        if intadc_data.shape[1] == 5:
            plot_rows += 1
        else:
            plot_rows += 2

    if not fig:
        fig = plt.figure(
            figsize=(14 if "gps" in active_feats else 7, 10), constrained_layout=True
        )
        pass_fig = False
    else:
        pass_fig = True

    fig_plots = fig
    if "gps" in active_feats:
        fig_plots, fig_map = fig.subfigures(1, 2)
        pf.plot_lat_lon_2d(
            get_xaxis(data_dict["gps"], RTC_data), data_dict["gps"], fig_map, tick=tick
        )

    axs = fig_plots.subplots(
        plot_rows,
        1,
        sharex=True,
    )
    if type(axs) is not np.ndarray:
        axs = np.array([axs])

    ax_ind = 0
    ax_start = 0
    ax_end = 0
    if intadc_data is not None:
        intadc_axis = get_xaxis(intadc_data, RTC_data)
        # logger.info(intadc_data.shape)
        # blarg
        if intadc_data.shape[1] == 5:
            # assume geophone!
            ax_start = intadc_axis[0][0]
            ax_end = intadc_axis[0][-1]
            pf.plot_ADC_geophone(intadc_axis, intadc_data, axs[ax_ind])
            axs[ax_ind].set_xlim([ax_start, ax_end])
            ax_ind = ax_ind + 1
            # plt.show()
        else:
            ax_start = intadc_axis[0][0]
            ax_end = intadc_axis[0][-1]
            pf.plot_ADC_hydrophone(intadc_axis, intadc_data, axs[ax_ind])
            axs[ax_ind].set_xlim([ax_start, ax_end])
            ax_ind = ax_ind + 1
            pf.plot_ADC_hydrophone_specgram(
                intadc_axis, intadc_data, intadc_meta, axs[ax_ind]
            )
            axs[ax_ind].set_xlim([ax_start, ax_end])

            fig2, ax2 = plt.subplots(2, 1, sharex=True)
            pf.plot_ADC_hydrophone(intadc_axis, intadc_data, ax2[0])
            pf.plot_ADC_hydrophone_specgram(
                intadc_axis, intadc_data, intadc_meta, ax2[1]
            )

            fig2.tight_layout()
            fig2.savefig(outfile.split(".")[0] + "_acoust.png")

            ax_ind = ax_ind + 1

    for _key, _fcn in active_feats.items():
        _fcn(get_xaxis(data_dict[_key], RTC_data), data_dict[_key], axs[ax_ind])
        if ax_end != 0 and not _key == "gps":
            axs[ax_ind].set_xlim([ax_start, ax_end])
        ax_ind = ax_ind + 1

    if tick:
        for ax in axs:
            ax.axvline(x=tick, color="r")

    if pass_fig:
        return

    fig.savefig(outfile)
    plt.close()


def plot_multichannel_acoustics(xaxis, acoustic_data, acoustic_meta, fname_out):
    x_axis = xaxis[0] - xaxis[0][0]

    FS = int(np.round(acoustic_meta["sample_rate"]))

    ch_cols = [x for x in acoustic_data.columns if "channel_" in x]
    logger.debug(f"available channels : {len(ch_cols)} ch")

    # plot time-series data:
    fig, axs = plt.subplots(
        len(ch_cols),
        1,
        sharex=True,
        sharey=True,
        figsize=(20, 10),
        constrained_layout=True,
    )
    if len(ch_cols) == 1:
        axs = np.array([axs])

    for ii, ch in enumerate(ch_cols[:]):
        cur_data = acoustic_data[ch].to_numpy()
        axs[ii].plot(x_axis, cur_data - np.mean(cur_data))
        axs[ii].set_ylabel("ch " + ch.split("_")[-1])
    del cur_data

    fig.suptitle("Time-series data")
    axs[ii].set_xlabel("Time [s]")
    plt.savefig(fname_out)
    plt.close()

    # plot select channel spectrograms (first, last):
    fig, axs = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=True,
        figsize=(20, 10),
        constrained_layout=True,
    )
    axs[0].specgram(
        acoustic_data[ch_cols[0]].to_numpy(),
        Fs=FS,
        NFFT=int(np.floor(FS / 10)),
        noverlap=int(np.floor(FS / 12)),
        vmin=-65,
        vmax=0,
        cmap="viridis",
    )
    axs[0].grid(visible=False)
    # axs[0].grid(alpha=0.25)
    axs[0].set_title("ch 0", loc="right", y=1.0, pad=-14, backgroundcolor="#ffffffaa")

    axs[1].specgram(
        acoustic_data[ch].to_numpy(),
        Fs=FS,
        NFFT=int(np.floor(FS / 10)),
        noverlap=int(np.floor(FS / 12)),
        vmin=-65,
        vmax=0,
        cmap="viridis",
    )
    axs[1].grid(visible=False)
    # axs[1].grid(alpha=0.25)
    # axs[0].set_ylim([0, 6000])
    axs[1].set_title(
        "ch " + ch.split("_")[-1],
        loc="right",
        y=1.0,
        pad=-14,
        backgroundcolor="#ffffffaa",
    )
    fig.suptitle("Spectrograms")
    axs[0].set_ylabel("Freq [Hz]")
    axs[1].set_ylabel("Freq [Hz]")
    axs[1].set_xlabel("Time [s]")

    plt.savefig(fname_out.replace(".png", "_spec_firstlast.png"))
    plt.close()


def plot_cam_frame_points(
    data_dict, RTC_data, img_dir, img_pattern, outdir, intadc_data=None
):
    active_feats = get_active_features(data_dict)
    if "cam" in active_feats:
        if intadc_data is not None:
            cam_select = data_dict["cam"][
                (data_dict["cam"]["timestamp"] > intadc_data["timestamp"].iloc[0])
                & (data_dict["cam"]["timestamp"] < intadc_data["timestamp"].iloc[-1])
            ]
        else:
            cam_select = data_dict["cam"]

        for idx in range(0, cam_select.shape[0]):
            cur_cam = cam_select.iloc[idx]
            ff = cur_cam["file_number"]

            # get camera file:
            _img_pattern = "**/" + img_pattern + "*_" + str(ff) + ".jpg"
            _img_pattern = os.path.join(img_dir, _img_pattern)
            cur_file = sorted(glob.glob(_img_pattern, recursive=True))
            if len(cur_file) > 0:
                cur_file = cur_file[0]
                cur_tick = get_xaxis(cur_cam, RTC_data)[0][0]

                logger.info(f"Generating frame for {os.path.basename(cur_file)}")
                # logger.info(f"Current time tick : {cur_tick}")

                if "gps" in active_feats:
                    fig = plt.figure(constrained_layout=True, figsize=(28, 10))
                    subfigs = fig.subfigures(1, 2, width_ratios=[1, 1])
                else:
                    fig = plt.figure(constrained_layout=True, figsize=(21, 10))
                    subfigs = fig.subfigures(1, 2, width_ratios=[1, 2])

                plot_data_dict(
                    data_dict,
                    RTC_data,
                    intadc_data=intadc_data,
                    fig=subfigs[0],
                    tick=cur_tick,
                )

                ax = subfigs[1].add_subplot(1, 1, 1)
                imdata = mpimg.imread(cur_file)
                ax.imshow(imdata)
                ax.grid(visible=False)

                fig.savefig(
                    os.path.join(outdir, os.path.split(cur_file)[-1].split(".jpg")[0])
                    + "_frame.jpg"
                )

                plt.close()
            else:
                logger.warning(
                    f"Could not locate expected file matching {os.path.basename(_img_pattern)}!"
                )
