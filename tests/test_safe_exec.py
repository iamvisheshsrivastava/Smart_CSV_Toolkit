import pandas as pd
import pytest
from safe_exec import safe_exec


def run(code):
    lv = {"df": pd.DataFrame({"a": [1, 2]})}
    safe_exec(code, {"pd": pd}, lv)
    return lv["df"]


def test_allows_normal_cleaning():
    assert run("import numpy as np\ndf['b'] = df['a'] * 2").shape == (2, 2)


@pytest.mark.parametrize("code", [
    "import os",
    "open('x')",
    "__import__('os')",
    "df.__class__.__mro__",
    "pd.read_csv('x')",
    "df.to_csv('x')",
    "eval('1')",
])
def test_blocks_dangerous(code):
    with pytest.raises(Exception):
        run(code)
