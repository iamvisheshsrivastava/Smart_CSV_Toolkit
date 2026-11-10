"""
Regression tests for pipeline step option descriptions (issue #27).

The "Dataset Multiplier" step's `size_multiplier` option help text was
copy-pasted from the Oversample/Undersample `sampling_strategy` blurb.
`dataset_multiplier()` in pipeline_logic.py does not apply any sampling
strategy: it stacks copies of the full DataFrame. These tests pin the
corrected description and guard against the copy-paste pattern recurring.
"""

import json
import unittest
from pathlib import Path

TABLE_STEPS_PATH = Path(__file__).resolve().parents[1] / "table_steps.json"

EXPECTED_SIZE_MULTIPLIER_DESCRIPTION = (
    "Number of times to duplicate the full dataset (e.g. '2x' doubles the row count)."
)
COPY_PASTED_BLURB = "Strategy for sampling: factor by which the dataset will be augmented."


def _load_steps():
    with TABLE_STEPS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_step(node, name):
    """Recursively locate a step dict with the given name in the JSON tree."""
    if isinstance(node, list):
        for item in node:
            found = _find_step(item, name)
            if found is not None:
                return found
    elif isinstance(node, dict):
        if node.get("name") == name:
            return node
        for value in node.values():
            found = _find_step(value, name)
            if found is not None:
                return found
    return None


def _require_step(steps, name):
    step = _find_step(steps, name)
    if step is None:
        raise AssertionError(f"step {name!r} not found in table_steps.json")
    return step


class DatasetMultiplierDescriptionTests(unittest.TestCase):
    """The size_multiplier help text must describe duplication, not sampling."""

    def setUp(self):
        self.steps = _load_steps()
        self.step = _require_step(self.steps, "Dataset Multiplier")

    def test_dataset_multiplier_step_has_size_multiplier_option(self):
        names = [opt.get("name") for opt in self.step.get("options", [])]
        self.assertIn("size_multiplier", names)

    def test_size_multiplier_description_is_corrected(self):
        options = {opt["name"]: opt for opt in self.step["options"]}
        description = options["size_multiplier"]["description"]
        self.assertEqual(description, EXPECTED_SIZE_MULTIPLIER_DESCRIPTION)

    def test_size_multiplier_description_no_longer_mentions_sampling(self):
        options = {opt["name"]: opt for opt in self.step["options"]}
        description = options["size_multiplier"]["description"].lower()
        self.assertNotIn("sampling", description)
        self.assertNotIn("over-sample", description)
        self.assertNotIn("under-sample", description)

    def test_corrected_description_appears_exactly_once(self):
        text = TABLE_STEPS_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count(EXPECTED_SIZE_MULTIPLIER_DESCRIPTION), 1)

    def test_copy_pasted_blurb_is_gone(self):
        text = TABLE_STEPS_PATH.read_text(encoding="utf-8")
        self.assertNotIn(COPY_PASTED_BLURB, text)


class SamplingStrategyDescriptionsStillIntactTests(unittest.TestCase):
    """Oversample/Undersample keep their legitimate sampling help text."""

    def setUp(self):
        self.steps = _load_steps()

    def _sampling_description(self, step_name):
        step = _require_step(self.steps, step_name)
        options = {opt["name"]: opt for opt in step["options"]}
        return options["sampling_strategy"]["description"].lower()

    def test_oversample_sampling_strategy_description_mentions_oversampling(self):
        self.assertIn("oversample", self._sampling_description("Oversample"))

    def test_undersample_sampling_strategy_description_mentions_undersampling(self):
        self.assertIn("undersample", self._sampling_description("Undersample"))


if __name__ == "__main__":
    unittest.main()
