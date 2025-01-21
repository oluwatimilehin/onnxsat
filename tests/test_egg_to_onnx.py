from test_util import TestUtil

from converter import Converter

import onnx

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def test_egg_to_onnx():
    current_dir = Path(__file__).parent
    eggs_path = f"{current_dir}/data/eggs"
    models_path = f"{current_dir}/data/models"

    egg_files = [f for f in Path(eggs_path).iterdir()]

    for egg_file in egg_files:
        egg_name = Path(egg_file).stem
        logger.info(f"Processing egg: {egg_name}")

        egg_expr = Path(egg_file).read_text()
        actual_nodes = Converter.to_onnx(egg_expr)

        expected_nodes = onnx.load_model(f"{models_path}/{egg_name}.onnx").graph.node

        TestUtil.compare_onnx_nodes(actual_nodes, expected_nodes)
