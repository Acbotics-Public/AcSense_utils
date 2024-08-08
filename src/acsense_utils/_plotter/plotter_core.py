import os
import glob
import numpy as np
import logging

import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
import matplotlib.image as mpimg

from . import plotter_funcs as pf

# FIXME : GET FS FROM LOGS!!
from .plotter_funcs import FS

mpl_use("agg")

logger = logging.getLogger(__name__)


def plot_data_dict(data_dict, RTC_data, outfile=None, intadc_data=None, fig=False):

    sens_features = [
        # (key, plot_func),
        # core modules
        ("ipt", pf.plot_PTSInt),
        # ("int_adc", plot),
        ("imu", pf.plot_IMU_accelerations),
        # add-on modules
        # ("rtc", plot),
        ("gps", pf.plot_lat_lon),
        ("ping", pf.plot_ping),
        ("mag_raw", pf.plot_raw_mag),
        ("ept30", pf.plot_PTS),
        ("ept100", pf.plot_PTS),
        ("ctd", pf.plot_CTD),
        ("cam", pf.plot_image_capture_times),
        # derivative fields (i.e. imu_mag)
        ("imu_mag", pf.plot_pitchrollyaw),
    ]

    rows = np.sum([feat[0] in data_dict for feat in sens_features])
    cols = 1

    if intadc_data is not None:
        if intadc_data.shape[1] == 5:
            rows += 1
        else:
            rows += 2

    if not fig:
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        pass_fig = False
    else:
        pass_fig = True

    ax = fig.subplots(
        rows,
        cols,
        sharex=False,
    )

    ax_ind = 0
    ax_start = 0
    ax_end = 0
    if intadc_data is not None:
        intadc_axis = pf.get_xaxis(intadc_data, RTC_data)
        # logger.info(intadc_data.shape)
        # blarg
        if intadc_data.shape[1] == 5:
            # assume geophone!
            ax_start = intadc_axis[0][0]
            ax_end = intadc_axis[0][-1]
            pf.plot_ADC_geophone(intadc_axis, intadc_data, ax[ax_ind])
            ax[ax_ind].set_xlim([ax_start, ax_end])
            ax_ind = ax_ind + 1
            # plt.show()
        else:
            ax_start = intadc_axis[0][0]
            ax_end = intadc_axis[0][-1]
            pf.plot_ADC_hydrophone(intadc_axis, intadc_data, ax[ax_ind])
            ax[ax_ind].set_xlim([ax_start, ax_end])
            ax_ind = ax_ind + 1
            pf.plot_ADC_hydrophone_specgram(intadc_axis, intadc_data, ax[ax_ind])
            ax[ax_ind].set_xlim([ax_start, ax_end])

            fig2, ax2 = plt.subplots(2, 1, sharex=True)
            pf.plot_ADC_hydrophone(intadc_axis, intadc_data, ax2[0])
            pf.plot_ADC_hydrophone_specgram(intadc_axis, intadc_data, ax2[1])

            fig2.tight_layout()
            fig2.savefig(outfile.split(".")[0] + "_acoust.png")

            ax_ind = ax_ind + 1

    for feat in sens_features:
        _key = feat[0]
        _fcn = feat[1]

        if _key in data_dict:
            _fcn(pf.get_xaxis(data_dict[_key], RTC_data), data_dict[_key], ax[ax_ind])
            if ax_end != 0 and not _key == "gps":
                ax[ax_ind].set_xlim([ax_start, ax_end])
            ax_ind = ax_ind + 1

    if pass_fig:
        return

    fig.savefig(outfile)
    plt.close()


def plot_multichannel_acoustics(xaxis, acoustic_data, fname_out):
    x_axis = xaxis[0] - xaxis[0][0]

    ch_cols = [x for x in acoustic_data.columns if "channel_" in x]
    logger.info(f"available channels : {len(ch_cols)} ch")

    # plot time-series data:
    fig, ax = plt.subplots(
        len(ch_cols),
        1,
        sharex=True,
        sharey=True,
        figsize=(20, 10),
        constrained_layout=True,
    )
    for ii, ch in enumerate(ch_cols[:]):
        cur_data = acoustic_data[ch].to_numpy()
        ax[ii].plot(x_axis, cur_data - np.mean(cur_data))
        ax[ii].set_ylabel("ch " + ch.split("_")[-1])
    del cur_data

    fig.suptitle("Time-series data")
    ax[ii].set_xlabel("Time [s]")
    plt.savefig(fname_out)
    plt.close()

    # plot select channel spectrograms (first, last):
    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        sharey=True,
        figsize=(20, 10),
        constrained_layout=True,
    )
    ax[0].specgram(
        acoustic_data[ch_cols[0]].to_numpy(),
        Fs=FS,
        NFFT=int(np.floor(FS / 10)),
        noverlap=int(np.floor(FS / 12)),
        vmin=-65,
        vmax=0,
    )
    ax[0].set_title("ch 0", loc="right", y=1.0, pad=-14, backgroundcolor="#ffffffaa")
    ax[1].specgram(
        acoustic_data[ch].to_numpy(),
        Fs=FS,
        NFFT=int(np.floor(FS / 10)),
        noverlap=int(np.floor(FS / 12)),
        vmin=-65,
        vmax=0,
    )
    # ax[0].set_ylim([0, 6000])
    ax[1].set_title(
        "ch " + ch.split("_")[-1],
        loc="right",
        y=1.0,
        pad=-14,
        backgroundcolor="#ffffffaa",
    )
    fig.suptitle("Spectrograms")
    ax[0].set_ylabel("Freq [Hz]")
    ax[1].set_ylabel("Freq [Hz]")
    ax[1].set_xlabel("Time [s]")
    plt.savefig(fname_out.replace(".png", "_spec_firstlast.png"))
    plt.close()


def plot_cam_frame_points(
    data_dict, RTC_data, img_dir, img_pattern, outdir, intadc_data=None
):
    if True and "cam" in data_dict:

        if intadc_data is not None:
            cam_select = data_dict["cam"][
                (data_dict["cam"]["timestamp"] > intadc_data["timestamp"].iloc[0])
                & ((data_dict["cam"]["timestamp"] < intadc_data["timestamp"].iloc[-1]))
            ]
        else:
            cam_select = data_dict["cam"]

        for idx in range(0, cam_select.shape[0]):
            cur_cam = cam_select.iloc[idx]
            ff = cur_cam["file_number"]

            # get camera file:
            _img_pattern = "**/" + img_pattern + "*" + str(ff) + ".jpg"
            _img_pattern = os.path.join(img_dir, _img_pattern)
            cur_file = sorted(glob.glob(_img_pattern, recursive=True))[0]
            cur_tick = pf.get_xaxis(cur_cam, RTC_data)[0][0]

            logger.info(f"Generating frame for {os.path.basename(cur_file)}")
            # logger.info(f"Current time tick : {cur_tick}")

            fig = plt.figure(constrained_layout=True, figsize=(20, 10))
            subfigs = fig.subfigures(1, 2)

            plot_data_dict(data_dict, RTC_data, intadc_data=intadc_data, fig=subfigs[0])

            ax = subfigs[1].add_subplot(1, 1, 1)
            imdata = mpimg.imread(cur_file)
            ax.imshow(imdata)

            for ax in subfigs[0].axes:
                ax.axvline(x=cur_tick, color="#bbbbbbbb", linestyle="dashed")

            fig.savefig(
                os.path.join(outdir, os.path.split(cur_file)[-1].split(".jpg")[0])
                + "_frame.jpg"
            )

            plt.close()
