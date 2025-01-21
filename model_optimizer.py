import copy
import os
import sys

from pathlib import Path
from typing import List, Tuple


import onnx
from onnx import ModelProto, NodeProto, OperatorSetIdProto

import numpy as np
import onnx_graphsurgeon as gs
import onnx
import onnx_tool

from converter import Converter
import fused_nodes_profile

from eggie.rewrites import *
from egglog import *

from sparsify_initializers import convert_initializers


sys.setrecursionlimit(10000)


class ModelOptimizer:
    def __init__(
        self, model_path: Path, original_data_dir: str, new_data_dir: str
    ) -> None:
        self._model_name = model_path.stem
        self._model = onnx.load_model(model_path)

        self._updated_data_dir = new_data_dir
        self._original_eggs_dir = f"{original_data_dir}/eggs"

        profile_results_file = (
            f"{original_data_dir}/models/{self._model_name}-profile.txt"
        )
        if not os.path.exists(profile_results_file):
            onnx_tool.model_profile(
                self._model,
                dynamic_shapes={"input": np.zeros((1, 3, 224, 224))},
                save_profile=profile_results_file,
            )

        graph_file = f"{original_data_dir}/models/{self._model_name}.proto"
        if not os.path.exists(graph_file):
            with open(graph_file, "w+") as f:
                f.write(str(self._model.graph))

    def run(self, prune_only: bool = False, sparsity_ratio: float = 0.0) -> ModelProto:
        new_model = copy.deepcopy(self._model)
        updated_model_dir = f"{self._updated_data_dir}/{self._model_name}"
        if not os.path.exists(updated_model_dir):
            os.makedirs(updated_model_dir)

        new_model_files_prefix = f"{updated_model_dir}/{self._model_name}"

        if not prune_only:
            graph = new_model.graph
            egg_expr = Converter.to_egglog(graph)
            file_name = f"{self._original_eggs_dir}/{self._model_name}.egg"
            with open(file_name, "w") as f:
                f.write(str(egg_expr))

            egraph = EGraph(save_egglog_string=True)
            egg_expr = egraph.let("expr", egg_expr)

            egraph.run(1000, ruleset=fusion_ruleset)
            print(f"Extracting expression")
            extracted = egraph.extract(egg_expr)
            # egraph.display()
            print(f"Extracted expression")

            new_egg_file = f"{new_model_files_prefix}.egg"
            with open(new_egg_file, "w+") as f:
                f.write(str(extracted))

            converted_onnx_nodes = Converter.to_onnx(extracted)
            new_model = self._update(new_model, converted_onnx_nodes)

        new_model_file = f"{new_model_files_prefix}-preprocessed.onnx"
        onnx.save(new_model, new_model_file)
        new_graph_file = f"{new_model_files_prefix}-preprocessed.proto"
        with open(new_graph_file, "w+") as f:
            f.write(str(new_model.graph))

        if sparsity_ratio > 0:
            new_model = self._sparsify(new_model, sparsity_ratio)
            new_model = convert_initializers(
                new_model, sparsity_threshold=sparsity_ratio, tolerance=1e-5
            )
            # new_model = onnx.shape_inference.infer_shapes(new_model)

        graph: gs.Graph = gs.import_onnx(new_model)
        graph = self._process_graph(graph)
        graph.cleanup(remove_unused_graph_inputs=True, remove_unused_node_outputs=True)

        new_model = gs.export_onnx(graph)
        new_model_file = f"{new_model_files_prefix}.onnx"
        onnx.save(new_model, new_model_file)

        onnx.checker.check_model(new_model)

        # onnx_tool.model_profile(
        #     new_model,
        #     dynamic_shapes={"input": np.zeros((1, 3, 224, 224))},
        #     save_profile=f"{new_model_files_prefix}-profile.txt",
        # )

        new_model_file = f"{new_model_files_prefix}.onnx"
        onnx.save(new_model, new_model_file)

        new_graph_file = f"{new_model_files_prefix}.proto"
        with open(new_graph_file, "w+") as f:
            f.write(str(new_model.graph))

        return new_model_file

    def _update(
        self, updated_model: ModelProto, new_nodes: List[NodeProto]
    ) -> ModelProto:
        opset_import = OperatorSetIdProto()
        opset_import.domain = "com.microsoft"
        opset_import.version = 1
        updated_model.opset_import.append(opset_import)

        del updated_model.graph.node[:]

        updated_model.graph.node.extend(new_nodes)

        return updated_model

    def _process_graph(self, graph: gs.Graph) -> gs.Graph:
        nodes = []
        for node in graph.nodes:
            match node.op:
                case "FusedConvBatchNorm":
                    new_node = self._process_conv_bn(node)
                    self._update_dependents(graph, node, new_node)
                    nodes.append(new_node)
                case _:
                    nodes.append(node)

        graph.nodes.clear()
        graph.nodes.extend(nodes)

        return graph

    def _update_dependents(self, graph: gs.Graph, old_node: gs.Node, new_node: gs.Node):
        for node in graph.nodes:
            for i, input in enumerate(node.inputs):
                if input in old_node.outputs:
                    node.inputs[i] = new_node.outputs[0]

    def _sparsify(self, model: ModelProto, sparsity_fraction: float) -> ModelProto:
        initializer_by_name = {init.name: init for init in model.graph.initializer}

        for node in model.graph.node:
            if node.op_type == "Conv" or node.op_type == "FusedConv":
                weights_init = initializer_by_name.get(node.input[1])
                weights_tensor = onnx.numpy_helper.to_array(weights_init)
                orig_shape = weights_tensor.shape

                weights_tensor = weights_tensor.flatten()

                # Determine the dynamic threshold based on the sparsity fraction
                threshold = np.percentile(
                    np.abs(weights_tensor), sparsity_fraction * 100
                )
                weights_tensor[np.abs(weights_tensor) < threshold] = 0
                weights_tensor = weights_tensor.reshape(orig_shape)

                weights_init.CopyFrom(
                    onnx.numpy_helper.from_array(weights_tensor, weights_init.name)
                )

        return model

    def _process_conv_bn(self, node: gs.Node) -> gs.Node:
        inputs = node.inputs

        bn_scale = inputs[1].values
        bn_bias = inputs[2].values
        bn_input_mean = inputs[3].values
        bn_input_var = inputs[4].values
        epsilon = node.attrs.get("epsilon")

        conv_w = inputs[5].values
        conv_bias = inputs[6].values if len(inputs) == 7 else np.zeros_like(bn_scale)

        adjusted_scale = bn_scale / np.sqrt(bn_input_var + epsilon)

        new_w = conv_w * adjusted_scale[:, None, None, None]
        new_b = (conv_bias - bn_input_mean) * adjusted_scale + bn_bias

        new_attrs = node.attrs.copy()
        new_attrs.pop("epsilon", None)
        new_attrs.pop("momentum", None)
        new_attrs = {
            name: attr
            for name, attr in node.attrs.items()
            if name != "epsilon" and name != "momentum"
        }

        op_type = "FusedConv"
        domain = "com.microsoft"

        if not new_attrs.get("activation"):
            op_type = "Conv"
            domain = ""
            new_attrs.pop("activation", None)

        new_node = gs.Node(
            op=op_type,
            name=f"{node.name}_bn_fused",
            domain=domain,
            attrs=new_attrs,
            inputs=[
                inputs[0],
                gs.Constant(name=f"{node.name}_fused_weights", values=new_w),
                gs.Constant(name=f"{node.name}_fused_bias", values=new_b),
            ],
            outputs=[
                gs.Variable(
                    name=f"{node.outputs[0].name}_bn_fused",
                )
            ],
        )

        return new_node
