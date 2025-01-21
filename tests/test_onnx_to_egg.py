import difflib
import logging
import re
from pathlib import Path

import onnx
from egglog import *

from converter import Converter


logger = logging.getLogger(__name__)


def normalize_expr(expr_str: str) -> str:
    return re.sub(r"Dropout\(([^,]+),\s*\d+,", r"Dropout(\1, 0,", expr_str).strip()


def test_onnx_to_egg():
    current_dir = Path(__file__).parent
    models_path = f"{current_dir}/data/models"
    eggs_path = f"{current_dir}/data/eggs"

    model_files = [f for f in Path(models_path).iterdir()]

    for model_file in model_files:
        model_name = Path(model_file).stem
        logger.info(f"Processing model: {model_name}")

        expected_egglog_expr = Path(f"{eggs_path}/{model_name}.egg").read_text()

        with open(model_file, "rb") as of:
            onnx_model = onnx.load_model(of)
            egglog_expr = Converter.to_egglog(onnx_model.graph)

            diffs = []
            for i, s in enumerate(
                difflib.ndiff(
                    normalize_expr(str(egglog_expr)),
                    normalize_expr(expected_egglog_expr),
                )
            ):
                match s[0]:
                    case "-":
                        diffs.append(f"Delete {s[-1]} from position {i}")
                    case "+":
                        diffs.append(f"Add {s[-1]} to position {i}")
                    case _:
                        continue

            assert not diffs
