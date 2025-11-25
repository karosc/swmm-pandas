"""Top-level package for swmm.pandas"""

import importlib.metadata

from swmm.pandas.constants import example_out_path, example_rpt_path
from swmm.pandas.input import Input, InputFile
from swmm.pandas.output import Output
from swmm.pandas.report import Report

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
