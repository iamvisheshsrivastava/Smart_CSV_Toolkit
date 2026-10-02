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
    # issue #19: str.format()/format_map() can traverse dunder attributes
    # (e.g. "{0.__class__.__init__.__globals__}".format(df)) as plain string
    # content, invisible to the AST attribute/name checks.
    "'{0.__class__.__mro__}'.format(df)",
    "'{0.__class__}'.format_map({0: df})",
    "df.query('1==1')",
    "df.eval('1==1')",
])
def test_blocks_dangerous(code):
    with pytest.raises(Exception):
        run(code)
