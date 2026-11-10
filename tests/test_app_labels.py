"""
Regression tests for pipeline step widget labels (issue #26).

The fix lives in app.py, which is a Streamlit script that imports heavy
optional dependencies (dtale, openai, wordcloud, ...). To keep these tests
hermetic and fast we do NOT import app.py. Instead we:

  1. Extract the pure ``format_option_label`` helper from app.py via AST and
     assert it produces human-readable labels;
  2. Statically scan the widget-call sites (``st.selectbox`` /
     ``st.number_input`` / ``st.text_input``) to guarantee the config loop
     never uses the raw ``param_key`` as the visible label and that every
     call keeps ``key=param_key`` for unique widget state.
"""

import ast
import unittest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
WIDGET_NAMES = ("selectbox", "number_input", "text_input")


def _load_tree():
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def _extract_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in app.py")


def _widget_calls(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr in WIDGET_NAMES
        ):
            yield node


class FormatOptionLabelTests(unittest.TestCase):
    """The display helper must turn snake_case keys into readable labels."""

    def setUp(self):
        tree = _load_tree()
        func = _extract_function(tree, "format_option_label")
        ns = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), "<app>", "exec"), ns)
        self.format_option_label = ns["format_option_label"]

    def test_snake_case_converted_to_title_case(self):
        cases = {
            "iqr_multiplier": "Iqr Multiplier",
            "fill_value": "Fill Value",
            "size_multiplier": "Size Multiplier",
            "sampling_strategy": "Sampling Strategy",
            "nan_threshold": "Nan Threshold",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.format_option_label(raw), expected)

    def test_short_single_word_kept(self):
        self.assertEqual(self.format_option_label("threshold"), "Threshold")
        self.assertEqual(self.format_option_label("code"), "Code")


class WidgetLabelRegressionTests(unittest.TestCase):
    """The config loop must show readable labels while keeping param_key."""

    def test_no_widget_uses_param_key_as_visible_label(self):
        offenders = []
        for call in _widget_calls(_load_tree()):
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Name) and first.id == "param_key":
                offenders.append(call.lineno)
            for kw in call.keywords:
                if kw.arg == "label" and isinstance(kw.value, ast.Name) and kw.value.id == "param_key":
                    offenders.append(call.lineno)
        self.assertEqual(offenders, [], f"widgets must not render param_key as label at lines {offenders}")

    def test_config_loop_widgets_pass_readable_label_and_unique_key(self):
        tree = _load_tree()
        seen_key = False
        seen_label = False
        for call in _widget_calls(tree):
            has_key = any(kw.arg == "key" and isinstance(kw.value, ast.Name) and kw.value.id == "param_key" for kw in call.keywords)
            if not has_key:
                continue
            seen_key = True
            first = call.args[0] if call.args else None
            if isinstance(first, ast.Name) and first.id == "label":
                seen_label = True
            for kw in call.keywords:
                if kw.arg == "label" and isinstance(kw.value, ast.Name) and kw.value.id == "label":
                    seen_label = True
        self.assertTrue(seen_key, "config-loop widgets must keep key=param_key")
        self.assertTrue(seen_label, "config-loop widgets must pass the readable label")

    def test_help_text_passed_to_widgets(self):
        has_help = False
        for call in _widget_calls(_load_tree()):
            if any(kw.arg == "help" for kw in call.keywords):
                has_help = True
                break
        self.assertTrue(has_help, "config-loop widgets should surface option descriptions as help text")


if __name__ == "__main__":
    unittest.main()
