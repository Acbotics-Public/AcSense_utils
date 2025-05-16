import argparse
import copy
import glob
import json
import logging
import os
import tkinter as tk
from multiprocessing import Pool, cpu_count, current_process
from shutil import get_terminal_size
from tkinter import filedialog

import numpy as np
import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from ._parser.parser import Parser  # type: ignore
from .plotter import run_plotter  # type: ignore

logger = logging.getLogger(__name__)


class MenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)
        self.parent = parent

        menu_file = tk.Menu(self, tearoff=0)
        menu_parse = tk.Menu(self, tearoff=0)
        menu_export = tk.Menu(self, tearoff=0)
        menu_plot = tk.Menu(self, tearoff=0)

        self.add_cascade(label="File", menu=menu_file)
        self.add_cascade(label="Parse", menu=menu_parse)
        self.add_cascade(label="Export", menu=menu_export)
        self.add_cascade(label="Plot", menu=menu_plot)

        menu_file_open = tk.Menu(self, tearoff=False)
        menu_file.add_cascade(label="Open", menu=menu_file_open)
        menu_file_open.add_command(label="Open file", command=self.open_file)
        menu_file_open.add_command(label="Open directory", command=self.open_dir)
        menu_file_open.add_command(
            label="Open, parse & export directory",
            command=self.open_parse_dir,
        )

        menu_file.add_command(label="Quit", command=parent.quit_application)

        menu_parse.add_command(
            label="Parse loaded file or directory",
            command=self.parse_path,
        )

        menu_export.add_command(label="Export data to CSV", command=self.export_to_csv)

        menu_plot.add_command(label="Plot parsed SENS data", command=self.plot_parsed)
        menu_plot.add_command(
            label="Plot parsed SENS+AC data",
            command=lambda: self.plot_parsed(plot_ac=True, plot_cam=False),
        )
        menu_plot.add_command(
            label="Plot parsed SENS+Camera data",
            command=lambda: self.plot_parsed(plot_ac=False, plot_cam=True),
        )
        menu_plot.add_command(
            label="Plot parsed data (ALL)",
            command=lambda: self.plot_parsed(plot_ac=True, plot_cam=True),
        )

    def open_file(self):
        _file_path = filedialog.askopenfilename(
            filetypes=(("DAT Files", "*.dat"), ("All files", "*.*"))
        )
        self.parent.file_path = _file_path if _file_path not in [(), ""] else None
        if self.parent.file_path:
            logger.info(f"Selected file : {self.parent.file_path}")

    def open_dir(self):
        _file_path = filedialog.askdirectory(title="Choose Log Directory")
        self.parent.file_path = _file_path if _file_path not in [(), ""] else None
        if self.parent.file_path:
            logger.info(f"Selected path : {self.parent.file_path}")

    def open_parse_dir(self):
        _file_path = filedialog.askdirectory(title="Choose Log Directory")
        self.parent.file_path = _file_path if _file_path not in [(), ""] else None
        if self.parent.file_path:
            logger.info(f"Selected source path : {self.parent.file_path}")
        else:
            return

        output_dir = filedialog.askdirectory(title="Choose Output Directory")
        output_dir = output_dir if output_dir not in [(), ""] else None
        if output_dir:
            logger.info(f"Selected base path : {output_dir}")
            output_dir = self.get_parser_outdir(output_dir)
            logger.info(f"Output path will be : {output_dir}")

            self.parent.parse_export_callback(
                path_src=self.parent.file_path, output_dir=output_dir
            )

    def parse_path(self):
        if self.parent.file_path:
            logger.info(f"Processing path : {self.parent.file_path}")
            self.parent.parse_callback(self.parent.file_path)
        else:
            logger.warning(f"Invalid path! Currently set to : {self.parent.file_path}")
            logger.warning("Please select a valid path before parsing")

    def export_to_csv(self):
        output_dir = filedialog.askdirectory(title="Choose Output Directory")
        output_dir = output_dir if output_dir not in [(), ""] else None
        if output_dir:
            logger.info(f"Selected base path : {output_dir}")
            output_dir = self.get_parser_outdir(output_dir)
            logger.info(f"Output path will be : {output_dir}")

            self.parent.save_csv_callback(self.parent.parsed, output_dir)

    def plot_parsed(self, plot_ac=False, plot_cam=False):
        parsed_dir = filedialog.askdirectory(title="Choose Parsed Data Directory")
        parsed_dir = parsed_dir if parsed_dir not in [(), ""] else None
        if parsed_dir:
            logger.info(f"Selected data path : {parsed_dir}")
        else:
            return

        # verify valid files are available in selected path:
        files = (
            sorted(glob.glob(os.path.join(parsed_dir, "SENS*.csv")))
            + sorted(glob.glob(os.path.join(parsed_dir, "IMU*.csv")))
            + sorted(glob.glob(os.path.join(parsed_dir, "AC*.csv")))
        )

        img_dir = None
        if plot_cam:
            img_dir = filedialog.askdirectory(title="Choose Matching Image Directory")
            img_files = sorted(
                glob.glob(os.path.join(img_dir, "**/IMG*.jpg"), recursive=True)
            )
            if len(img_files) == 0:
                logger.error(
                    "No valid image files located! Please check your top-level "
                    "directory (or a subdirectory therein) contains valid "
                    "image files (IMG*.jpg) matching the parsed data logs."
                )
                return

        if parsed_dir != "" and len(files) > 0:
            run_plotter(
                parsed_dir=parsed_dir,
                plot_ac=plot_ac,
                plot_cam=plot_cam,
                img_dir=img_dir,
            )
        else:
            logger.error(
                "No valid data files located! Please check your path contains"
                "valid CSV files (SENS*, IMU*, AC*) produced by the parser."
            )

    @staticmethod
    def get_parser_outdir(base_dir):
        parse_ind = 1
        output_dir = base_dir
        while True:
            try:
                output_dir = os.path.join(base_dir, "parsed" + repr(parse_ind))
                os.mkdir(output_dir)
                break
            except FileExistsError:
                parse_ind += 1
                continue
        return output_dir


class Parser_GUI_Tk(tk.Tk):
    def __init__(self, parser_args):
        tk.Tk.__init__(self)

        self.args = parser_args

        dpi = self.winfo_fpixels("1i")
        scaling = 1
        if dpi > 100:
            logger.info(f"Detected high DPI screen : {int(dpi)}")
            scaling = 1.5

        self.title("AcSense Parser")
        self.minsize(int(400 * scaling), int(300 * scaling))

        main_frame = tk.Frame(self)
        main_frame.pack_propagate(0)
        main_frame.pack(fill="both", expand=True)

        self.adc_mode = tk.BooleanVar(self, value=self.args.use_int)

        adc_mode_opts = {
            "External ADC": False,
            "Internal ADC": True,
        }

        ii = 0
        adc_mode_buttons = []
        for desc, val in adc_mode_opts.items():
            bb = tk.Radiobutton(
                main_frame,
                text=desc,
                variable=self.adc_mode,
                value=val,
                command=self.set_adc_mode,
            )
            bb.grid(row=0, column=ii)
            adc_mode_buttons.append(bb)
            ii += 1

        report_frame = tk.Frame(main_frame)
        report_frame.grid(row=1, column=0, columnspan=2, pady=20, sticky="nsew")

        l1 = tk.Label(report_frame, text="Channel Name")
        l1.grid(column=0, row=0, padx=5)
        l2 = tk.Label(report_frame, text="Number of Messages")
        l2.grid(column=1, row=0, padx=5)
        l3 = tk.Label(report_frame, text="--")
        l3.grid(column=0, row=1, padx=5)
        l4 = tk.Label(report_frame, text="--")
        l4.grid(column=1, row=1, padx=5)

        self.report_info = [l1, l2, l3, l4]

        self.file_path = None
        self.active_sense_dict = None

        self.parsed = {}

        self.report_frame = report_frame

        report_frame.tkraise()

        menubar = MenuBar(self)
        tk.Tk.config(self, menu=menubar)

    def quit_application(self):
        self.destroy()

    @staticmethod
    def _parse_callback(
        path_src=None,
        use_int=False,
        export=False,
        output_dir=None,
        pbar_position=None,
        use_double_sr=False,
    ):
        # path_src = path_src or self.file_path
        parsed = {}

        if path_src is None:
            return None

        if os.path.isfile(path_src):
            p = Parser(double_sample_rate=use_double_sr)
            fn = os.path.basename(path_src).split("/")[-1]
            if fn.startswith("AC"):
                parsed = {
                    fn: p.parse_ac_file(
                        path_src,
                        use_int,
                        export=export,
                        output_dir=output_dir,
                        pbar_position=pbar_position,
                    )
                }
            elif fn.startswith("SENS"):
                parsed = {fn: p.parse_sense_file(path_src, pbar_position=pbar_position)}
                if export:
                    Parser_GUI_Tk.save_csv_callback(
                        parsed, output_dir=output_dir, pbar_position=pbar_position
                    )
            else:
                parsed = {}
            if export:
                return None
        elif os.path.isdir(path_src):
            parsed = {}

            files_to_process = sorted(
                glob.glob(os.path.join(path_src, "**/SENS*.dat"), recursive=True)
            )
            files_to_process += sorted(
                glob.glob(os.path.join(path_src, "**/AC*.dat"), recursive=True)
            )

            if len(files_to_process) == 0:
                logger.warning(
                    "Could not locate SENS*.dat or AC*.dat files to process!"
                )
                return None

            for fn in files_to_process:
                logger.info(f"Loading {fn}")
                p = Parser(double_sample_rate=use_double_sr)
                try:
                    if fn.startswith("AC"):
                        parsed[fn] = copy.deepcopy(
                            p.parse_ac_file(os.path.join(path_src, fn), use_int)
                        )

                    elif fn.startswith("SENS"):
                        parsed[fn] = copy.deepcopy(
                            p.parse_sense_file(os.path.join(path_src, fn))
                        )
                    elif fn.endswith("JPG") or fn.endswith(".jpg"):
                        continue
                    else:
                        continue

                except Exception as e:
                    logger.info(f"Exception encountered while parsing file {fn} :\n{e}")

        return parsed

    def parse_callback(self, path_src=None, redraw=True):
        # path_src = path_src or self.file_path
        self.parsed = self._parse_callback(
            path_src=path_src,
            use_int=self.args.use_int,
            use_double_sr=self.args.use_double_sr,
        )
        if redraw:
            self.redraw_channels()

    @staticmethod
    def _parse_export_callback(args):
        idx, use_int, fn, output_dir, use_double_sr = args

        Parser_GUI_Tk._parse_callback(
            path_src=fn,
            use_int=use_int,
            export=True,
            output_dir=output_dir,
            pbar_position=current_process()._identity[0] - 1,
            use_double_sr=use_double_sr,
        )

    def parse_export_callback(self, path_src=None, output_dir=None):
        # path_src = path_src or self.file_path

        files_to_process = sorted(
            glob.glob(os.path.join(self.file_path, "**/SENS*.dat"), recursive=True)
        )
        files_to_process += sorted(
            glob.glob(os.path.join(self.file_path, "**/AC*.dat"), recursive=True)
        )

        basenames = set([os.path.basename(x) for x in files_to_process])
        if len(basenames) != len(files_to_process):
            logger.warning(
                "You have selected data from multiple missions! "
                "Output files for matching source file names may get overwritten."
            )

        _files_to_process = []
        for ii, fn in enumerate(files_to_process):
            _files_to_process.append(
                (ii, self.args.use_int, fn, output_dir, self.args.use_double_sr)
            )

        if len(_files_to_process) == 0:
            return

        _n_sens = sum([x.startswith("SENS") for x in basenames])
        _n_aco = sum([x.startswith("AC") for x in basenames])
        logger.info(f"Exporting files to CSV : {_n_sens} SENS & {_n_aco} AC")

        nproc = max([1, cpu_count() - 4])
        nproc = min([nproc, len(_files_to_process)])
        # nproc = min([max([1, cpu_count() - 4]), 8])

        # We must initialize progress bars here to prevent conflicting allocation
        # from forked processes; otherwise mp.Pool will allocate excess space
        # and some progress bars will *appear* orphaned due to printing in higher
        # than indended index, then relocating to desired position
        # >> prog_bar_alloc = tqdm(position=nproc)

        # Since we are alllocating here anyway, let's track overall progress
        prog_bar = tqdm(
            desc="OVERALL PROGRESS", position=nproc, total=len(_files_to_process)
        )

        with Pool(nproc) as p:
            for result in p.imap_unordered(
                self._parse_export_callback, _files_to_process, chunksize=1
            ):
                prog_bar.update()
                prog_bar.refresh()

        prog_bar.close()
        print((" " * get_terminal_size()[0] + "\n") * nproc)
        logger.info("Done!")

    def redraw_channels(self):
        logger.info("redrawing")
        for it in self.report_info:
            it.destroy()

        self.report_info = []
        names = []
        data = []

        if self.parsed != {}:
            # determine channels
            data_multi = {}
            for fn, parse in self.parsed.items():
                for p in parse:
                    d = p["parser"].as_dict()
                    name = p["parser"].get_name()
                    # names.append(name)
                    if name not in data_multi.keys():
                        data_multi[name] = [d]
                    else:
                        data_multi[name].append(d)
            for name, dlist in data_multi.items():
                names.append(name)
                data.append(sum([len(d["timestamp"]) for d in dlist]))

        l1 = tk.Label(self.report_frame, text="Channel Name")
        l1.grid(column=0, row=0, padx=5)
        l2 = tk.Label(self.report_frame, text="Number of Messages")
        l2.grid(column=1, row=0, padx=5)
        self.report_info = [l1, l2]

        logger.info(f"Data collected includes: {names}")

        if len(names) > 0:
            ind = 1
            for d, name in zip(data, names):
                l1 = tk.Label(self.report_frame, text=name)
                l1.grid(column=0, row=ind, padx=5)
                l2 = tk.Label(self.report_frame, text=str(d))
                l2.grid(column=1, row=ind, padx=5)

                self.report_info.extend([l1, l2])

                ind += 1

        else:
            l1 = tk.Label(self.report_frame, text="--")
            l1.grid(column=0, row=1, padx=5)
            l2 = tk.Label(self.report_frame, text="--")
            l2.grid(column=1, row=1, padx=5)
            self.report_info.extend([l1, l2])

    @staticmethod
    def save_csv_callback(parsed, output_dir, pbar_position=None):
        if parsed != {} and output_dir != "":
            for fn, parser_set in parsed.items():
                for p in parser_set:
                    fn1 = os.path.join(
                        output_dir,
                        fn.replace(".dat", "") + "_" + p["parser"].get_name() + ".csv",
                    ).replace(" ", "_")

                    if fn.startswith("AC") and p["parser"].get_name() in [
                        # "INTADC",
                        "spiadc",
                    ]:  # LATER: add Intadc config!
                        p["parser"].write_csv(fn1, pbar_position=pbar_position)

                        if os.path.isfile(fn1):
                            ac_meta = json.dumps(
                                {
                                    "sample_rate": p["parser"].sample_rate,
                                    "channels": p["parser"].channels,
                                    "bitsPerChannel": p["parser"].bitsPerChannel,
                                    "bytesPerChannel": p["parser"].bytesPerChannel,
                                },
                                indent=4,
                                sort_keys=True,
                            )
                            with open(fn1.replace(".csv", "_meta.txt"), "w") as fn2:
                                fn2.write(ac_meta)
                                fn2.write("\n")

                    elif p["parser"].get_name() not in [
                        # "INTADC",
                        "spiadc",
                    ]:
                        d = p["parser"].as_dict()

                        if len(d["timestamp"]) > 0:
                            for k in d.keys():
                                print(repr(k) + " " + repr(len(d[k])))
                            data = pd.DataFrame(d)
                            data.index.name = "index"

                            chunks = np.array_split(
                                data.index, 100
                            )  # split into 100 chunks

                            for chunk, subset in enumerate(
                                tqdm(
                                    chunks,
                                    desc=f"Writing {os.path.basename(fn1):40s}",
                                    position=pbar_position,
                                )
                            ):
                                if chunk == 0:  # first row
                                    data.loc[subset].to_csv(fn1, mode="w", index=True)
                                else:
                                    data.loc[subset].to_csv(
                                        fn1, header=None, mode="a", index=True
                                    )

                        if p["parser"].get_name() == "intadc" and os.path.isfile(fn1):
                            ac_meta = json.dumps(
                                {
                                    "sample_rate": p["parser"].sample_rate,
                                    "channels": p["parser"].channels,
                                    "bitsPerChannel": p["parser"].bitsPerChannel,
                                    "bytesPerChannel": p["parser"].bytesPerChannel,
                                },
                                indent=4,
                                sort_keys=True,
                            )
                            with open(fn1.replace(".csv", "_meta.txt"), "w") as fn2:
                                fn2.write(ac_meta)
                                fn2.write("\n")

    def set_adc_mode(self):
        self.args.use_int = self.adc_mode.get()


Parser_GUI = Parser_GUI_Tk


def run_parser_gui():
    parser = argparse.ArgumentParser(
        prog="AcSense Parser",
        description="Utility to load, parse and export AcSense data from device logs",
        epilog="Acbotics Research, LLC",
    )

    parser.add_argument(
        "--use_int",
        action="store_true",
        help="Use INT (Internal ADC) Mode on launch (can be switched on GUI)",
    )
    parser.add_argument(
        "--use_double_sr",
        action="store_true",
        help="Use double precision sample rate on int adc for use with older headers",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose logs (detailed)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show verbose debugging logs"
    )

    args = parser.parse_args()

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

    if args.debug:
        logger.setLevel(logging.DEBUG)

    gui = Parser_GUI(args)
    gui.mainloop()


if __name__ == "__main__":
    run_parser_gui()
