"""AcSense Plotter : module-specific plotting logic

This file contains the rendering logic for specific data structures.
It is intended as an implementation reference for higher-level
plot-generation methods in the `plotter_core.py` file.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as mpl_use
from matplotlib.ticker import FormatStrFormatter

mpl_use("agg")
plt.style.use("seaborn-v0_8-darkgrid")


logger = logging.getLogger(__name__)

# ## Adjust mag cal as needed
# magz_cal = -15000
# magx_cal = -2800
# magy_cal = -2800


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


def plot_lat_lon_2d(caxis, gps_data, fig, tick=None):
    ax = fig.add_subplot(1, 1, 1)

    ss = ax.scatter(gps_data["lon"], gps_data["lat"], c=caxis[0], cmap="viridis")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_title("GPS Track")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    if tick:
        idx = np.argmin(np.abs(caxis[0] - tick))
        ax.plot(
            gps_data["lon"].to_numpy()[idx],
            gps_data["lat"].to_numpy()[idx],
            "dr",
            ms=10,
            label="current",
        )
        ax.legend()

    fig.colorbar(ss, label=caxis[1])


def plot_lat_lon(xaxis, gps_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    ax.clear()

    ax.set_title("Lat/Lon")

    ax.plot(x_axis, gps_data["lon"], "g.")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Lon")

    ax2 = ax.twinx()
    ax2.plot(x_axis, gps_data["lat"], "b.")
    ax2.yaxis.label.set_color("blue")
    ax2.set_ylabel("Lat")

    ax.set_xlabel(xlabel)


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
    ax.plot(x_axis, ["Capture"] * len(x_axis), "rx")
    logger.debug(
        f"camera : median( diff( timestamps ))  is: "
        f"{np.median(np.diff(x_axis)):.3} seconds"
    )
    # logger.info("min cam diff is: " + str(np.mean(np.min(x_axis))))
    # logger.info("max cam diff is: " + str(np.mean(np.max(x_axis))))
    ax.set
    ax.set_title("Image capture times")
    ax.set_xlabel(xlabel)


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


def plot_ADC_hydrophone_specgram(xaxis, ADC_data, ADC_meta, ax):
    FS = int(np.round(ADC_meta["sample_rate"]))

    # logger.info(ADC_data)
    # plot geophone data
    x_axis = xaxis[0]
    adc_data_start = ADC_data.columns.get_loc("channel_0")
    xlabel = xaxis[1]
    adc_ch_vals = ADC_data.iloc[:, adc_data_start]
    # logger.info(adc_ch_vals.shape)
    ax.specgram(
        adc_ch_vals,
        NFFT=4096,
        Fs=FS,
        xextent=(x_axis[0], x_axis[-1]),
        vmin=-65,
        vmax=0,
        cmap="viridis",
    )
    # ax.set_ylim([0, 3.3])
    ax.set_title("Hydrophone data")

    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.1)


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
    # xlabel = xaxis[1]
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
    ax.set_xlabel(xlabel)
    ax.grid(color="g", alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(x_axis, PTS_data["temperature_c"].to_numpy(), "b.")
    ax2.set_ylabel("Temp (deg C)")
    ax2.yaxis.label.set_color("blue")
    ax2.grid(color="b", alpha=0.2)


def plot_PTSInt(xaxis, PTS_data, ax):
    x_axis = xaxis[0]
    xlabel = xaxis[1]

    pvar = "pressure_mbar"
    ax.set_title("Int P/T")
    # plot PTS data!
    ax.plot(x_axis, PTS_data[pvar].to_numpy(), "g.")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Pressure (bar)")
    ax.set_xlabel(xlabel)
    ax.grid(color="g", alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(x_axis, PTS_data["temperature_c"].to_numpy(), "b.")
    ax2.set_ylabel("Temp (deg C)")
    ax2.yaxis.label.set_color("blue")
    ax2.grid(color="b", alpha=0.2)


def plot_CTD(xaxis, ctd_data, ax):
    x_axis = xaxis[0]
    # xlabel = xaxis[1]
    if "data3" in ctd_data.keys():
        pvar = "data3"
        tvar = "data0"
        svar = "data2"
    elif "pressure_decibars" in ctd_data.keys():
        pvar = "pressure_decibars"
        svar = "salinity"
        tvar = "temperature_c"
    else:
        return
    ax.set_title("CTD data")

    # plot PTS data!
    ax.plot(x_axis, ctd_data[pvar].to_numpy(), "g.")
    ax.yaxis.label.set_color("green")
    ax.set_ylabel("Pressure (dbar)")
    ax.yaxis.label.set_color("g")
    ax.grid(color="g", alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(x_axis, ctd_data[svar].to_numpy(), "r-")
    ax2.yaxis.label.set_color("r")
    ax2.set_ylabel("Salinity (ppt)")
    ax2.yaxis.label.set_color("r")
    ax2.set_ylim([0, 40])
    ax2.grid(color="r", alpha=0.2)

    ax3 = ax.twinx()
    ax3.plot(x_axis, ctd_data[tvar].to_numpy(), "b-")
    ax3.set_ylabel("Temp (deg C)")
    ax3.yaxis.label.set_color("blue")
    ax3.grid(color="b", alpha=0.2)

    ax3.spines["right"].set_position(("axes", 1.15))
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_color("k")


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
        xmat,
        ymat,
        20 * np.log10((np.transpose(ping_profile_vals) + 1) / 256),
        cmap="viridis",
    )
    ax.set_xlabel("time bin, " + xlabel)
    ax.set_ylabel("Range (m)")
