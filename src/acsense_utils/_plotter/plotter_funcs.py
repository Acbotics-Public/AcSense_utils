import os
import glob
import scipy
import numpy as np
import logging

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import use as mpl_use
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

mpl_use("agg")

logger = logging.getLogger(__name__)

# ## Adjust mag cal as needed
# magz_cal = -15000
# magx_cal = -2800
# magy_cal = -2800


FS = 52734
TICK = 1e-9  # sample interval for internal timestamp is s
acc_correction = 9.81 / 2059  # acceleration correction for m/s^2


# def get_heading(MAG_data, IMU_data):
#     # process magnetometer data using IMU data
#     # first, rotate mag data based on pitch/roll:
#     # id the min/max
#     # take the median of the top 5 values and the bottom 5 values:
#     sorted_z = np.array(sorted(MAG_data["mag_z"]))
#     sorted_y = np.array(sorted(MAG_data["mag_y"]))
#     sorted_x = np.array(sorted(MAG_data["mag_x"]))
#     cal_x = (np.median(sorted_x[-5:]) + np.median(sorted_x[0:5])) / 2
#     cal_y = (np.median(sorted_y[-5:]) + np.median(sorted_y[0:5])) / 2
#     cal_z = (np.median(sorted_z[-5:]) + np.median(sorted_z[0:5])) / 2
#     # logger.info(IMU_data.keys())
#     logger.info("(max+min)/2 of mag raw values:")
#     logger.info("magz_cal")
#     logger.info(cal_z)
#     logger.info("magx_cal")
#     logger.info(cal_x)
#     logger.info("magy_cal")
#     logger.info(cal_y)
#     # logger.info("if doing mag cal, use these values for magz_cal, magx_cal, magy_cal")
#     # logger.info(magx_cal)
#     # blarg
#     pitch_interp = scipy.interpolate.interp1d(
#         IMU_data["timestamp"], IMU_data["Pitch"], fill_value="extrapolate"
#     )
#     roll_interp = scipy.interpolate.interp1d(
#         IMU_data["timestamp"], IMU_data["Roll"], fill_value="extrapolate"
#     )
#     mag_pitch = pitch_interp(MAG_data["timestamp"])
#     mag_roll = roll_interp(MAG_data["timestamp"])

#     hdg2 = []
#     new_x = []
#     new_y = []
#     new_z = []
#     for ii in range(0, len(mag_pitch)):
#         cur_data = MAG_data.iloc[ii]
#         # logger.info(mag_pitch[ii])
#         # logger.info(mag_roll[ii])
#         myrot = scipy.spatial.transform.Rotation.from_euler(
#             "YX", np.array([mag_pitch[ii], mag_roll[ii]]), degrees=True
#         )
#         new_mag = myrot.apply(
#             [
#                 cur_data["mag_x"] - magx_cal,
#                 cur_data["mag_y"] - magy_cal,
#                 cur_data["mag_z"] - magz_cal,
#             ]
#         )
#         # logger.info(new_mag[2])
#         new_x.append(new_mag[0])
#         new_y.append(new_mag[1])
#         new_z.append(new_mag[2])
#         heading2 = -np.arctan2(new_mag[1], new_mag[0]) * 180 / np.pi
#         hdg2.append(heading2)
#     MAG_data["heading"] = (
#         np.arctan2(MAG_data["mag_z"] - magz_cal, MAG_data["mag_x"] - magx_cal)
#         * 180
#         / np.pi
#     )
#     MAG_data["heading2"] = hdg2
#     MAG_data["pitch"] = mag_pitch
#     MAG_data["roll"] = mag_roll

#     return MAG_data


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


def plot_lat_lon(caxis, gps_data, ax):
    ax.clear()

    ax.scatter(gps_data["lon"], gps_data["lat"], c=caxis[0], alpha=0.5)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_title("Color=" + caxis[1])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))


def plot_pitchrollyaw(xaxis, Mag_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    # first, pitch/roll:

    ax.plot(x_axis, Mag_data["pitch"], label="Pitch NED deg")
    ax.plot(x_axis, Mag_data["roll"], label="Roll NED deg")
    ax.plot(x_axis, Mag_data["heading"], label="Heading naive")
    ax.plot(x_axis, Mag_data["heading2"], label="heading pr corrected")
    ax.legend()
    ax.set_ylabel("Degrees")
    ax.set_title("Pitch/roll/yaw")
    ax.set_xlabel(xlabel)


def plot_image_capture_times(xaxis, Cam_Meta, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]
    ax.plot(x_axis, x_axis * 0, "rx")
    logger.info("diff cam median is: " + str(np.median(np.diff(x_axis))))
    # logger.info("min cam diff is: " + str(np.mean(np.min(x_axis))))
    # logger.info("max cam diff is: " + str(np.mean(np.max(x_axis))))


def plot_IMU_accelerations(xaxis, IMU_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    ax.plot(x_axis, IMU_data["accx_rot"], ".", label="accx")
    ax.plot(x_axis, IMU_data["accy_rot"], ".", label="accy")
    ax.plot(x_axis, IMU_data["accz_rot"], ".", label="accz")
    # ax.plot(x_axis, IMU_data["acc_total"])
    ax.legend()
    ax.set_ylabel("acc m/s")
    ax.set_title("IMU rotated accelerations")
    ax.set_xlabel(xlabel)


def plot_ADC_hydrophone_specgram(xaxis, ADC_data, ax):
    # logger.info(ADC_data)
    # plot geophone data
    x_axis = xaxis[0]
    adc_data_start = ADC_data.columns.get_loc("channel_0")
    xlabel = xaxis[1]
    adc_ch_vals = ADC_data.iloc[:, adc_data_start]
    # logger.info(adc_ch_vals.shape)
    ax.specgram(
        adc_ch_vals,
        NFFT=5000,
        Fs=FS,
        xextent=(x_axis[0], x_axis[-1]),
        vmin=-65,
        vmax=0,
    )
    # ax.set_ylim([0, 3.3])
    ax.set_title("Hydrophone data")

    ax.set_xlabel(xlabel)


def plot_ADC_hydrophone(xaxis, ADC_data, ax):
    # logger.info(ADC_data)
    # plot geophone data
    x_axis = xaxis[0]  # ADC_data["timestamp"] * TICK
    # ax.plot(np.diff(ADC_data["timestamp"]), ".")
    # plt.show()
    adc_data_start = ADC_data.columns.get_loc("channel_0")
    xlabel = xaxis[1]
    adc_ch_vals = ADC_data.iloc[:, adc_data_start].to_numpy()

    ax.plot(x_axis, adc_ch_vals)
    # ax.set_ylim([0, 3.3])
    ax.set_title("Hydrophone data")

    ax.set_xlabel(xlabel)


def plot_ADC_geophone(xaxis, ADC_data, ax):
    # logger.info(ADC_data)
    # plot geophone data
    x_axis = xaxis[0]  # ADC_data["timestamp"] * TICK
    # ax.plot(np.diff(ADC_data["timestamp"]), ".")
    # plt.show()
    CH0 = ADC_data.columns.get_loc("channel_0")
    CH1 = ADC_data.columns.get_loc("channel_1")
    CH2 = ADC_data.columns.get_loc("channel_2")
    xlabel = xaxis[1]

    ax.plot(x_axis, ADC_data.iloc[:, CH0].to_numpy())
    ax.plot(x_axis, ADC_data.iloc[:, CH1].to_numpy())
    ax.plot(x_axis, ADC_data.iloc[:, CH2].to_numpy())
    # ax.set_ylim([0, 3.3])
    ax.set_title("Geophone data")

    ax.set_xlabel(xlabel)


def plot_raw_mag(xaxis, mag_raw, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]
    ax.plot(x_axis, mag_raw["mag_x"], "b.", label="raw mag x")
    ax.plot(x_axis, mag_raw["mag_y"], "r.", label="raw mag y")
    ax.plot(x_axis, mag_raw["mag_z"], "g.", label="raw mag z")
    # ax.plot(x_axis, x_axis * 0 + magx_cal, "b--", label="mag x cal")
    # ax.plot(x_axis, x_axis * 0 + magy_cal, "r--", label="mag y cal")
    # ax.plot(x_axis, x_axis * 0 + magz_cal, "g--", label="mag z cal")
    ax.legend()
    ax.set_ylabel("Raw Mag")


def plot_PTS(xaxis, PTS_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    pvar = "pressure_bar"
    ax.set_title("External P/T")
    # plot PTS data!
    ax.plot(x_axis, PTS_data[pvar].to_numpy(), "g.")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Pressure (bar)")
    ax2 = ax.twinx()
    ax2.plot(x_axis, PTS_data["temperature_c"].to_numpy(), "b.")
    ax2.set_ylabel("Temp (deg C)")
    ax2.yaxis.label.set_color("blue")
    ax.set_xlabel(xlabel)


def plot_PTSInt(xaxis, PTS_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    pvar = "pressure_mbar"
    ax.set_title("Int P/T")
    # plot PTS data!
    ax.plot(x_axis, PTS_data[pvar].to_numpy(), "g.")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Pressure (bar)")
    ax2 = ax.twinx()
    ax2.plot(x_axis, PTS_data["temperature_c"].to_numpy(), "b.")
    ax2.set_ylabel("Temp (deg C)")
    ax2.yaxis.label.set_color("blue")
    ax.set_xlabel(xlabel)


def plot_CTD(xaxis, ctd_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    pvar = "data3"
    tvar = "data0"
    svar = "data2"
    ax.set_title("CTD data")
    # plot PTS data!
    ax.plot(x_axis, ctd_data[pvar].to_numpy(), "g.", label="depth")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Pressure (dbar)")
    ax.plot(x_axis, ctd_data[svar].to_numpy(), "r-", label="salinity")

    # ax2 = ax.twinx()
    ax.plot(x_axis, ctd_data[tvar].to_numpy(), "b-", label="temp")
    ax.set_ylabel("Temp (deg C)")
    ax.yaxis.label.set_color("blue")
    ax.legend()
    # ax.set_xlabel(xlabel)
    ax.set_ylim([0, 40])


def plot_ping(xaxis, PING_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]
    profile_data_start = PING_data.columns.get_loc("profile_data0")

    ping_profile_vals = PING_data.iloc[
        :, profile_data_start:
    ]  # np.array([np.array(x["profile_values"]) for x in PING_data])
    range_start = PING_data["scan_start"].to_numpy() / 1000
    range_length = PING_data["scan_length"].to_numpy() / 1000

    range_in_m = (np.array(list(range(0, 200))) * range_length[0] / 200) + range_start[
        0
    ]

    xmat, ymat = np.meshgrid(range(0, len(x_axis)), range_in_m)

    ax.pcolormesh(
        xmat, ymat, 20 * np.log10((np.transpose(ping_profile_vals) + 1) / 256)
    )
    ax.set_xlabel("time bin, " + xlabel)
    ax.set_ylabel("Range (m)")


def get_xaxis(sensor_data, RTC_data=None):
    offset = 0
    if RTC_data is not None:

        # use RTC data to look up timestamps:
        # logger.info(RTC_data[0])
        try:
            rtc_start = RTC_data.iloc[0]["timestr"]
            offset = RTC_data.iloc[0]["timestamp"]
        except:
            rtc_start = ""
            offset = 0

        xlabeln = "s since " + rtc_start

    else:
        logger.info("RTC data is None?")
        xlabeln = "s"
    # logger.info(sensor_data)
    if type(sensor_data["timestamp"]) == type([]) or type(
        sensor_data["timestamp"]
    ) == type(np.array([])):
        tstamp_array = np.array(sensor_data["timestamp"]) - offset
    # elif isinstance(sensor_data["timestamp"],int) or isinstance(sensor_data["timestamp"],float):

    else:
        try:
            tstamp_array = sensor_data["timestamp"].to_numpy() - offset
        except:
            tstamp_array = np.array([sensor_data["timestamp"]]) - offset
    tstamps = (
        tstamp_array * TICK
    )  # (np.array([x["timestamp"] for x in sensor_data]) - offset) * TICK
    # tstamps = tstamps - tstamps[0]
    return [tstamps, xlabeln]


def get_GPS_RTC_data(GPS_data):

    GPS_select = GPS_data[["RMC" in str(x) for x in GPS_data["raw_nmea"].to_numpy()]]

    GPS_select2 = GPS_select[[",A," in x for x in GPS_select["raw_nmea"].to_numpy()]]

    if len(GPS_select2["raw_nmea"].to_numpy()) > 0:
        # use good fix to set time!!
        return GPS_select2
    else:
        # logger.info(GPS_select)

        return GPS_select


def plot_cam_frame_points(data_dict, RTC_data, indir, outdir, intadc_data=None):
    if "cam" in data_dict.keys():
        rows = len(data_dict.keys())

        if "ept" in data_dict.keys() and "cam" in data_dict.keys():
            rows = rows - 1
        if intadc_data is not None:
            rows = rows + 2

        # fig, ax = plt.subplots(rows, cols, sharex=False, figsize=(20, 10))

        if intadc_data is not None:
            intadc_axis = get_xaxis(intadc_data, RTC_data)
            ax_start = intadc_axis[0][0]
            ax_end = intadc_axis[0][-1]
            cam_axis = get_xaxis(data_dict["cam"], RTC_data)[0]

            cam_select = data_dict["cam"][
                (data_dict["cam"]["timestamp"] > intadc_data["timestamp"].iloc[0])
                & ((data_dict["cam"]["timestamp"] < intadc_data["timestamp"].iloc[-1]))
            ]
        else:
            cam_select = data_dict["cam"]
        logger.info(cam_select)
        for idx in range(0, cam_select.shape[0]):
            fig = plt.figure(layout="constrained", figsize=(20, 10))
            gs = GridSpec(rows, 3, figure=fig)
            cur_cam = cam_select.iloc[idx]
            logger.info(cur_cam)
            ff = cur_cam["file_number"]
            cur_tstamp = cur_cam["timestamp"]
            # get camera file:
            logger.info(indir)
            cur_file = sorted(
                glob.glob(
                    os.path.join(
                        os.path.join(os.path.join(indir, ".."), "*"),
                        "IMG*_" + str(ff) + ".jpg",
                    )
                )
            )[0]
            cur_tick = get_xaxis(cur_cam, RTC_data)[0][0]
            logger.info(cur_tick)
            # create plot:
            ax = fig.add_subplot(gs[:, 2])
            # put file in ax:
            imdata = mpimg.imread(cur_file)
            ax.imshow(imdata)

            ax_ind = 0
            ax_start = 0
            ax_end = 0
            if intadc_data is not None:
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                intadc_axis = get_xaxis(intadc_data, RTC_data)
                cur_idx = np.argmin(np.abs(intadc_data["timestamp"] - cur_tstamp))
                ax_start = intadc_axis[0][0]
                ax_end = intadc_axis[0][-1]
                plot_ADC_hydrophone(intadc_axis, intadc_data, ax)
                ax.plot(cur_tick, 0, "ro")
                ax.set_xlim([ax_start, ax_end])
                ax_ind = ax_ind + 1
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_ADC_hydrophone_specgram(intadc_axis, intadc_data, ax)
                ax.plot(cur_tick, 0, "kx")
                ax.set_xlim([ax_start, ax_end])
                ax_ind = ax_ind + 1
            if "gps" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_lat_lon(
                    get_xaxis(data_dict["gps"], RTC_data), data_dict["gps"], ax
                )
                if len(data_dict["gps"]["timestamp"].to_numpy()) > 0:

                    gps_select = data_dict["gps"].iloc[
                        np.argmin(data_dict["gps"]["timestamp"].to_numpy() - cur_tick)
                    ]
                    plt.plot(gps_select["lat"], gps_select["lon"], "kx")
                ax_ind = ax_ind + 1
            if "imu_mag" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_pitchrollyaw(
                    get_xaxis(data_dict["imu_mag"], RTC_data),
                    data_dict["imu_mag"],
                    ax,
                )  # plot pitch/roll/yaw
                if ax_end != 0:
                    ax.set_xlim([ax_start, ax_end])
                ax.plot(cur_tick, 0, "kx")
                ax_ind = ax_ind + 1
            if "ctd" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_CTD(
                    get_xaxis(data_dict["ctd"], RTC_data),
                    data_dict["ctd"],
                    ax,
                )  # plot pitch/roll/yaw
                if ax_end != 0:
                    ax.set_xlim([ax_start, ax_end])
                ax.plot(cur_tick, 0, "kx")
                ax_ind = ax_ind + 1
            if "ept" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_PTS(
                    get_xaxis(data_dict["ept"], RTC_data),
                    data_dict["ept"],
                    "ext",
                    ax,
                )
                if "cam" in data_dict.keys():
                    plot_image_capture_times(get_xaxis(data_dict["cam"], RTC_data), ax)
                    ax.legend(["PT", "Image Captured"])
                if ax_end != 0:
                    ax.set_xlim([ax_start, ax_end])
                ax.plot(cur_tick, 0, "kx")
                ax_ind = ax_ind + 1
            else:
                if "cam" in data_dict.keys():
                    ax = fig.add_subplot(gs[ax_ind, 0:2])
                    plot_image_capture_times(get_xaxis(data_dict["cam"], RTC_data), ax)
                    if ax_end != 0:
                        ax.set_xlim([ax_start, ax_end])
                    ax.plot(cur_tick, 0, "kx")
                    ax_ind = ax_ind + 1

            if "imu" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_IMU_accelerations(
                    get_xaxis(data_dict["imu"], RTC_data), data_dict["imu"], ax
                )
                if ax_end != 0:
                    ax.set_xlim([ax_start, ax_end])
                ax.plot(cur_tick, 0, "kx")
                ax_ind = ax_ind + 1
            if "ping" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_ping(
                    get_xaxis(data_dict["ping"], RTC_data),
                    data_dict["ping"],
                    ax,
                )
                if ax_end != 0:
                    ax.set_xlim([ax_start, ax_end])
                ax.plot(cur_tick, 0, "kx")
                ax_ind = ax_ind + 1

            if "mag_raw" in data_dict.keys():
                ax = fig.add_subplot(gs[ax_ind, 0:2])
                plot_raw_mag(
                    get_xaxis(data_dict["mag_raw"], RTC_data),
                    data_dict["mag_raw"],
                    ax,
                )
                if ax_end != 0:
                    ax.set_xlim([ax_start, ax_end])
                ax.plot(cur_tick, 0, "kx")
                ax_ind = ax_ind + 1

            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, os.path.split(cur_file)[-1].split(".jpg")[0])
                + "_frame.jpg"
            )
            # plt.show()

            plt.close()

        # plt.show()
