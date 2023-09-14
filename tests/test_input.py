"""Tests for `swmm-pandas` input class."""


import pathlib
from swmm.pandas import Input

_HERE = pathlib.Path(__file__).parent
test_inp_path = str(_HERE / "data" / "Model.inp")


# @pytest.fixture(scope="module")
def inpfile():
    inp = Input(test_inp_path)
    return inp


if __name__ == "__main__":
    inp = inpfile()
