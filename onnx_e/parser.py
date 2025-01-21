import base64
import random
import sys
import traceback
import logging
from typing import List, Iterable


from egglog import Vec
from onnx.onnx_ml_pb2 import GraphProto, NodeProto, AttributeProto

from eggie.operators import *


logger = logging.getLogger(__name__)


class OnnxParser:
    def __init__(self, graph: GraphProto) -> None:
        self._graph = graph
        self._output_per_node = {}

    def parse(self) -> TensorId:
        results: List[TensorId] = []
        for node in self._graph.node:
            result = self._process_node(node)
            self._output_per_node[node.output[0]] = result
            results.append(result)

        return results[-1]

    def _process_node(self, node: NodeProto) -> TensorId:
        op_type = node.op_type
        try:
            match op_type:
                case "Add":
                    return self._convert_add(node)
                case "AveragePool":
                    return self._convert_average_pool(node)
                case "BatchNormalization":
                    return self._convert_batch_normalization(node)
                case "Cast":
                    return self._convert_cast(node)
                case "Clip":
                    return self._convert_clip(node)
                case "Conv":
                    return self._convert_conv(node)
                case "Concat":
                    return self._convert_concat(node)
                case "Constant":
                    return self._convert_constant(node)
                case "ConstantOfShape":
                    return self._convert_constant_of_shape(node)
                case "DequantizeLinear":
                    return self._convert_dequantize_linear(node)
                case "Div":
                    return self._convert_div(node)
                case "Dropout":
                    return self._convert_dropout(node)
                case "Equal":
                    return self._convert_equal(node)
                case "Erf":
                    return self._convert_erf(node)
                case "Expand":
                    return self._convert_expand(node)
                case "Flatten":
                    return self._convert_flatten(node)
                case "Gather":
                    return self._convert_gather(node)
                case "Gemm":
                    return self._convert_gemm(node)
                case "GlobalAveragePool":
                    return self._convert_global_average_pool(node)
                case "HardSigmoid":
                    return self._convert_hard_sigmoid(node)
                case "Identity":
                    return self._convert_identity(node)
                case "MatMul":
                    return self._convert_matmul(node)
                case "Mul":
                    return self._convert_mul(node)
                case "MaxPool":
                    return self._convert_max_pool(node)
                case "Pad":
                    return self._convert_pad(node)
                case "Pow":
                    return self._convert_pow(node)
                case "QuantizeLinear":
                    return self._convert_quantize_linear(node)
                case "QLinearAdd":
                    return self._convert_q_linear_add(node)
                case "QLinearConv":
                    return self._convert_conv(node, quantized=True)
                case "QLinearGlobalAveragePool":
                    return self._convert_q_linear_global_average_pool(node)
                case "QLinearMatMul":
                    return self._convert_q_linear_matmul(node)
                case "ReduceMean":
                    return self._convert_reduce_mean(node)
                case "Relu":
                    return self._convert_relu(node)
                case "Reshape":
                    return self._convert_reshape(node)
                case "Sigmoid":
                    return self._convert_sigmoid(node)
                case "Softmax":
                    return self._convert_softmax(node)
                case "Shape":
                    return self._convert_shape(node)
                case "Slice":
                    return self._convert_slice(node)
                case "Split":
                    return self._convert_split(node)
                case "Squeeze":
                    return self._convert_squeeze(node)
                case "Sqrt":
                    return self._convert_sqrt(node)
                case "Sub":
                    return self._convert_sub(node)
                case "Transpose":
                    return self._convert_transpose(node)
                case "Unsqueeze":
                    return self._convert_unsqueeze(node)
                case "Where":
                    return self._convert_where(node)
                case _:
                    raise ValueError(f"Unsupported operator: {op_type}")
        except Exception as e:
            print(f"Error converting node {node.name}: {e}")
            print(traceback.format_exc())
            raise e

    def _attr_by_name(
        self, attrs: Iterable[AttributeProto]
    ) -> dict[str, AttributeProto]:
        return {attr.name: attr for attr in attrs}

    def _to_tensor_id(self, id: str) -> TensorId:
        if id in self._output_per_node:
            return self._output_per_node[id]

        return TensorId(id)

    def _convert_add(self, node: NodeProto) -> TensorId:
        return Op.Add(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_average_pool(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        auto_pad = (
            attrs_by_name.get("auto_pad").s.decode()
            if "auto_pad" in attrs_by_name
            else "NOTSET"
        )

        ceil_mode = (
            attrs_by_name.get("ceil_mode").i if "ceil_mode" in attrs_by_name else 0
        )
        count_include_pad = (
            attrs_by_name.get("count_include_pad").i
            if "count_include_pad" in attrs_by_name
            else 0
        )

        kernel_shape = [i64(i) for i in attrs_by_name["kernel_shape"].ints]

        pads = (
            [i64(i) for i in attrs_by_name.get("pads").ints]
            if "pads" in attrs_by_name
            else [i64(0) for _ in range(len(kernel_shape) * 2)]
        )

        strides: List[i64] = (
            [i64(i) for i in attrs_by_name.get("strides").ints]
            if "strides" in attrs_by_name
            else [i64(1) for _ in range(len(kernel_shape))]
        )

        attrs = AveragePoolAttrs(
            auto_pad,
            ceil_mode,
            count_include_pad,
            Vec[i64](*kernel_shape),
            Vec[i64](*pads),
            Vec[i64](*strides),
        )

        return Op.AveragePool(
            self._to_tensor_id(node.name),
            attrs,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_batch_normalization(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)
        epsilon = (
            attrs_by_name.get("epsilon").f if "epsilon" in attrs_by_name else 1e-05
        )
        momentum = (
            attrs_by_name.get("momentum").f if "momentum" in attrs_by_name else 0.9
        )

        attrs = BatchNormAttrs(epsilon, momentum)

        inputs = [self._to_tensor_id(input) for input in node.input]
        outputs = [self._to_tensor_id(out) for out in node.output]
        return Op.BatchNormalization(
            self._to_tensor_id(node.name),
            attrs,
            *inputs,
            outputs[
                0
            ],  # Batchnorm can have multiple outputs, but 1 seems to be standard and I want to minimize using Vec objects to speed up extraction
        )

    def _convert_cast(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        to = i64(attrs_by_name.get("to").i)

        return Op.Cast(
            self._to_tensor_id(node.name),
            to,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_clip(self, node: NodeProto) -> TensorId:
        inputs = [self._to_tensor_id(input) for input in node.input]

        return Op.Clip(
            self._to_tensor_id(node.name),
            inputs[0],
            inputs[1],
            inputs[2],
            self._to_tensor_id(node.output[0]),
        )

    def _convert_concat(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axis = attrs_by_name["axis"].i
        inputs = [self._to_tensor_id(input) for input in node.input]
        return Op.Concat(
            self._to_tensor_id(node.name),
            axis,
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_conv(self, node: NodeProto, quantized=False) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        auto_pad = (
            attrs_by_name.get("auto_pad").s.decode()
            if "auto_pad" in attrs_by_name
            else "NOTSET"
        )

        group = attrs_by_name.get("group").i if "group" in attrs_by_name else i64(1)

        kernel_shape = [i64(i) for i in attrs_by_name["kernel_shape"].ints]
        dilations = (
            [i64(i) for i in attrs_by_name.get("dilations").ints]
            if "dilations" in attrs_by_name
            else [i64(1) for _ in range(len(kernel_shape))]
        )

        pads = (
            [i64(i) for i in attrs_by_name.get("pads").ints]
            if "pads" in attrs_by_name
            else [i64(0) for _ in range(len(kernel_shape) * 2)]
        )

        strides: List[i64] = (
            [i64(i) for i in attrs_by_name.get("strides").ints]
            if "strides" in attrs_by_name
            else [i64(1) for _ in range(len(kernel_shape))]
        )

        attrs = ConvAttrs(
            auto_pad,
            group,
            Vec[i64](*dilations),
            Vec[i64](*kernel_shape),
            Vec[i64](*pads),
            Vec[i64](*strides),
        )

        inputs = [self._to_tensor_id(input) for input in node.input]

        return (
            Op.QLinearConv(
                self._to_tensor_id(node.name),
                attrs,
                Vec[TensorId](*inputs),
                self._to_tensor_id(node.output[0]),
            )
            if quantized
            else Op.Conv(
                self._to_tensor_id(node.name),
                attrs,
                inputs[0],
                inputs[1],
                inputs[2] if len(inputs) == 3 else TensorId(""),
                self._to_tensor_id(node.output[0]),
            )
        )

    def _convert_constant(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)
        # TODO: Only supporting a tensor type atm; consider other types?
        value = attrs_by_name["value"]

        dims = [i64(dim) for dim in value.t.dims]

        tensor_type = TensorType(
            Vec[i64](*dims),
            value.t.data_type,
            base64.b64encode(value.t.raw_data).decode(),
        )

        return Op.Constant(
            self._to_tensor_id(node.name),
            tensor_type,
            self._to_tensor_id(node.output[0]),
        )

    def _convert_constant_of_shape(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)
        # TODO: Only supporting a tensor type atm; consider other types?
        value = attrs_by_name["value"]

        dims = [i64(dim) for dim in value.t.dims]

        tensor_type = TensorType(
            Vec[i64](*dims),
            value.t.data_type,
            base64.b64encode(value.t.raw_data).decode(),
        )

        return Op.ConstantOfShape(
            self._to_tensor_id(node.name),
            tensor_type,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_dropout(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        seed = (
            i64(attrs_by_name.get("seed").i)
            if "seed" in attrs_by_name
            else i64(random.randint(0, sys.maxsize))
        )

        inputs = [self._to_tensor_id(input) for input in node.input]
        outputs = [self._to_tensor_id(output) for output in node.output]

        return Op.Dropout(
            self._to_tensor_id(node.name),
            seed,
            Vec[TensorId](*inputs),
            Vec[TensorId](*outputs),
        )

    def _convert_dequantize_linear(self, node: NodeProto) -> TensorId:
        inputs = [self._to_tensor_id(input) for input in node.input]
        return Op.DequantizeLinear(
            self._to_tensor_id(node.name),
            inputs[0],
            inputs[1],
            inputs[2],
            self._to_tensor_id(node.output[0]),
        )

    def _convert_div(self, node: NodeProto) -> TensorId:
        return Op.Div(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_equal(self, node: NodeProto) -> TensorId:
        return Op.Equal(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_erf(self, node: NodeProto) -> TensorId:
        return Op.Erf(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_expand(self, node: NodeProto) -> TensorId:
        return Op.Expand(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_flatten(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axis = attrs_by_name.get("axis").i if "axis" in attrs_by_name else 1
        return Op.Flatten(
            self._to_tensor_id(node.name),
            axis,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_gather(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axis = attrs_by_name.get("axis").i if "axis" in attrs_by_name else -1
        return Op.Gather(
            self._to_tensor_id(node.name),
            axis,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_gemm(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        alpha = (
            f64(attrs_by_name.get("alpha").f) if "alpha" in attrs_by_name else f64(1.0)
        )

        beta = f64(attrs_by_name.get("beta").f) if "beta" in attrs_by_name else f64(1.0)

        transA = (
            i64(attrs_by_name.get("transA").i) if "transA" in attrs_by_name else i64(0)
        )

        transB = (
            i64(attrs_by_name.get("transB").i) if "transB" in attrs_by_name else i64(0)
        )

        attrs = GemmAttrs(alpha, beta, transA, transB)
        inputs = [self._to_tensor_id(input) for input in node.input]
        return Op.Gemm(
            self._to_tensor_id(node.name),
            attrs,
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_global_average_pool(self, node: NodeProto) -> TensorId:
        return Op.GlobalAveragePool(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_hard_sigmoid(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)
        alpha = f64(attrs_by_name.get("alpha").f) if "alpha" in attrs_by_name else 0.2
        beta = f64(attrs_by_name.get("beta").f) if "beta" in attrs_by_name else 0.5

        return Op.HardSigmoid(
            self._to_tensor_id(node.name),
            alpha,
            beta,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_identity(self, node: NodeProto) -> TensorId:
        return Op.Identity(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_matmul(self, node: NodeProto) -> TensorId:
        return Op.MatMul(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_max_pool(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        auto_pad = (
            attrs_by_name.get("auto_pad").s.decode()
            if "auto_pad" in attrs_by_name
            else "NOTSET"
        )

        ceil_mode = (
            attrs_by_name.get("ceil_mode").i if "ceil_mode" in attrs_by_name else i64(0)
        )

        storage_order = (
            attrs_by_name.get("storage_order").i
            if "storage_order" in attrs_by_name
            else i64(0)
        )

        kernel_shape = [i64(i) for i in attrs_by_name["kernel_shape"].ints]
        dilations = (
            [i64(i) for i in attrs_by_name.get("dilations").ints]
            if "dilations" in attrs_by_name
            else [i64(1) for _ in range(len(kernel_shape))]
        )

        pads = (
            [i64(i) for i in attrs_by_name.get("pads").ints]
            if "pads" in attrs_by_name
            else [i64(0) for _ in range(len(kernel_shape) * 2)]
        )

        strides: List[i64] = (
            [i64(i) for i in attrs_by_name.get("strides").ints]
            if "strides" in attrs_by_name
            else [i64(1) for _ in range(len(kernel_shape))]
        )

        attrs = MaxPoolAttrs(
            auto_pad,
            ceil_mode,
            storage_order,
            Vec[i64](*dilations),
            Vec[i64](*kernel_shape),
            Vec[i64](*pads),
            Vec[i64](*strides),
        )

        outputs = [self._to_tensor_id(output) for output in node.output]
        return Op.MaxPool(
            self._to_tensor_id(node.name),
            attrs,
            self._to_tensor_id(node.input[0]),
            Vec[TensorId](*outputs),
        )

    def _convert_mul(self, node: NodeProto) -> TensorId:
        return Op.Mul(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_pad(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        mode = (
            attrs_by_name.get("mode").s.decode()
            if "mode" in attrs_by_name
            else String("constant")
        )
        inputs = [self._to_tensor_id(input) for input in node.input]

        return Op.Pad(
            self._to_tensor_id(node.name),
            mode,
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_pow(self, node: NodeProto) -> TensorId:
        return Op.Pow(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_quantize_linear(self, node: NodeProto) -> TensorId:
        inputs = [self._to_tensor_id(input) for input in node.input]
        return Op.QuantizeLinear(
            self._to_tensor_id(node.name),
            inputs[0],
            inputs[1],
            inputs[2],
            self._to_tensor_id(node.output[0]),
        )

    def _convert_q_linear_add(self, node: NodeProto) -> TensorId:
        inputs = [self._to_tensor_id(input) for input in node.input]

        return Op.QLinearAdd(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.domain),
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_q_linear_global_average_pool(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        channels_last = attrs_by_name["channels_last"].i
        inputs = [self._to_tensor_id(input) for input in node.input]

        return Op.QLinearGlobalAveragePool(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.domain),
            channels_last,
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_q_linear_matmul(self, node: NodeProto) -> TensorId:
        inputs = [self._to_tensor_id(input) for input in node.input]

        return Op.QLinearMatMul(
            self._to_tensor_id(node.name),
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_reduce_mean(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axes = [i64(i) for i in attrs_by_name["axes"].ints]
        keep_dims = (
            attrs_by_name.get("keepdims").i if "keepdims" in attrs_by_name else i64(1)
        )

        return Op.ReduceMean(
            self._to_tensor_id(node.name),
            Vec[i64](*axes),
            keep_dims,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_relu(self, node: NodeProto) -> TensorId:
        return Op.Relu(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_reshape(self, node: NodeProto) -> TensorId:
        return Op.Reshape(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_sigmoid(self, node: NodeProto) -> TensorId:
        return Op.Sigmoid(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_softmax(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axis = attrs_by_name.get("axis").i if "axis" in attrs_by_name else -1
        return Op.Softmax(
            self._to_tensor_id(node.name),
            axis,
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_shape(self, node: NodeProto) -> TensorId:
        return Op.Shape(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_slice(self, node: NodeProto) -> TensorId:
        inputs = [self._to_tensor_id(input) for input in node.input]
        return Op.Slice(
            self._to_tensor_id(node.name),
            Vec[TensorId](*inputs),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_split(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axis = attrs_by_name.get("axis").i if "axis" in attrs_by_name else i64(0)
        split = [i64(i) for i in attrs_by_name["split"].ints]
        outputs = [self._to_tensor_id(out) for out in node.output]

        return Op.Split(
            self._to_tensor_id(node.name),
            axis,
            Vec[i64](*split),
            self._to_tensor_id(node.input[0]),
            Vec[TensorId](*outputs),
        )

    def _convert_squeeze(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        axes = [i64(i) for i in attrs_by_name["axes"].ints]
        return Op.Squeeze(
            self._to_tensor_id(node.name),
            Vec[i64](*axes),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_sqrt(self, node: NodeProto) -> TensorId:
        return Op.Sqrt(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_sub(self, node: NodeProto) -> TensorId:
        return Op.Sub(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_transpose(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)

        perms = [i64(i) for i in attrs_by_name["perm"].ints]
        return Op.Transpose(
            self._to_tensor_id(node.name),
            Vec[i64](*perms),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_unsqueeze(self, node: NodeProto) -> TensorId:
        attrs_by_name = self._attr_by_name(node.attribute)
        axes = [i64(i) for i in attrs_by_name["axes"].ints]
        return Op.Unsqueeze(
            self._to_tensor_id(node.name),
            Vec[i64](*axes),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.output[0]),
        )

    def _convert_where(self, node: NodeProto) -> TensorId:
        return Op.Where(
            self._to_tensor_id(node.name),
            self._to_tensor_id(node.input[0]),
            self._to_tensor_id(node.input[1]),
            self._to_tensor_id(node.input[2]),
            self._to_tensor_id(node.output[0]),
        )
