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

The GUI provides access to the parser, which unpacks the data logs. The GUI
also provides easy access to the plotting utilities via the "Plot" menu.

The plotting utility can also be called directly via CLI with the command:

```bash
acsense-plot
```


## No-install execution

The AcSense Parser GUI can also be launched without installation. The easiest
way to do so is to navigate into the `./src/` directory and then run:

```bash
python3 -c "from acsense_utils import parser_gui; parser_gui.run_parser_gui()"
```
