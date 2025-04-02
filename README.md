# AcSense_utils

AcSense data processing utilities:

- AcSense Parser GUI
- AcSense Data Plotter

# Installation

To use the AcSense Parser GUI, navigate to the top of this repo and run:

```bash
python3 -m pip install .
```

To install in editable mode, use the `-e` flag:

```bash
python3 -m pip install -e .
```

It is also possible to use the GUI without installing this package. For more
information on this, see [No-install execution](#no-install-execution).

# Execution

The standard installation procedure will provide a CLI command to start the
AcSense Parser GUI:

```bash
acsense-parse
```

The GUI provides access to the various sensor-specific parser modules, which
unpack the different data types in the AcSense logs. Data can be read from a
single log file or a directory of log files, at the user's convenience. The
data that has been loaded onto memory by the parsers can then be exported to
CSV files with the appropriate column headers. While parsing or exporting data,
the program will show progress bars on the terminal, to provide the user with
feedback on the current working state; the GUI may become unresponsive while
parsing large amounts of data.

When working with a large amount of data in a log directory, the user may prefer
to use the provided shortcut for parsing and exporting data from a batch of log
files in a given directory in one run. This prevents loading all the data onto
memory at once, opting instead to handle the files independently.

The GUI also provides easy access to the plotting utilities via the "Plot" menu.
This allows the user to generate visualizations of the parsed data. When using
a camera-enabled AcSense unit, the plotter utility can also generate reference
frames with time-stamped cursor markers for the sensor data shown alongside each
image; the program will require the paths for both the parsed directory with the
CSV files, and the original log directory which contains the individual image files.

Additionally, the plotting utility can also be called directly via its own CLI
entrypoint, with the command:

```bash
acsense-plot
```

To see more information on the CLI interface and accepted arguments, use the
`-h/--help` argument:

```bash
acsense-plot -h
```


## No-install execution

The AcSense Parser GUI can also be launched without installation. The easiest
way to do so is to navigate into the `./src/` directory and then run:

```bash
python3 -c "from acsense_utils import parser_gui; parser_gui.run_parser_gui()"
```

# Example user flow

To process a single file or a directory with a few small files (small enough to
load all data onto memory):

1. File > Open > Open file / Open directory
1. Select the target file or a directory containing the log files
1. Parse > Parse loaded file or directory
1. Export > Export data to CSV
1. Select the destination path where the resulting `parsed*` output should be written

To process a directory with a large amount of data:

1. File > Open > Open, parse & export directory
1. Select the target directory containing the log files
1. Select the destination path where the resulting `parsed*` output should be written

To plot sensor data (SENS* files) or sensor and acoustic data (SENS+AC):

1. Plot > Plot parsed \[SENS/SENS+AC\] data
1. Select the target directory containing the parsed data as CSV files

To plot camera frames alongside data:

1. Plot > Plot parsed SENS+Camera data / Plot parsed data (ALL)
1. Select the target directory containing the parsed data as CSV files
1. Select the original log directory containing the captured image files


# Troubleshooting

## Command not found : path issues

If your system path is not correctly set to include the location of installed Python binaries, the `acsense-parse` command may produce an error such as:

```text
acsense-parse: command not found
```

**RECOMMENDED** : This issue can be solved by using a virtual environment (venv); make sure the venv is active before installing the `AcSense_utils` package. The venv activation will augment your path to include any custom entrypoints and binaries installed within the venv. For an introduction to Python virtual environments, check out the built-in `venv` module at [https://docs.python.org/3/library/venv.html](https://docs.python.org/3/library/venv.html).

**ALTERNATIVE** : The issue can also be solved by augmenting your path to include the location of the target entrypoints. Typical locations for Python entrypoints include:

- Ubuntu: `~/.local/bin`
- MacOS: `~/Library/Python/{VERSION}/bin`
