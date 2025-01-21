from __future__ import annotations
from egglog import *


class TensorId(Expr):
    @method()
    def __init__(self, name: String): ...

    @method()
    def empty(self) -> TensorId: ...


class TensorType(Expr):
    @method()
    def __init__(self, dims: Vec[i64], data_type: i64, raw_data: String): ...


class AveragePoolAttrs(Expr):
    @method()
    def __init__(
        self,
        auto_pad: String,
        ceil_mode: i64,
        count_include_pad: i64,
        kernel_shape: Vec[i64],
        pads: Vec[i64],
        strides: Vec[i64],
    ): ...


class BatchNormAttrs(Expr):
    @method()
    def __init__(self, epsilon: f64, momentum: f64): ...


class ConvAttrs(Expr):
    @method()
    def __init__(
        self,
        auto_pad: String,
        group: i64,
        dilations: Vec[i64],
        kernel_shape: Vec[i64],
        pads: Vec[i64],
        strides: Vec[i64],
    ): ...


class FusedConvAttrs(Expr):
    @method()
    def __init__(
        self,
        activation: String,
        activation_params: Vec[f64],
        auto_pad: String,
        group: i64,
        dilations: Vec[i64],
        kernel_shape: Vec[i64],
        pads: Vec[i64],
        strides: Vec[i64],
    ): ...


class FusedGemmAttrs(Expr):
    @method()
    def __init__(
        self,
        activation: String,
        activation_alpha: f64,
        activation_beta: f64,
        activation_gamma: f64,
        alpha: f64,
        beta: f64,
        transA: i64,
        transB: i64,
    ): ...


class GemmAttrs(Expr):
    @method()
    def __init__(self, alpha: f64, beta: f64, transA: i64, transB: i64): ...


class MaxPoolAttrs(Expr):
    @method()
    def __init__(
        self,
        auto_pad: String,
        ceil_mode: i64,
        storage_order: i64,
        dilations: Vec[i64],
        kernel_shape: Vec[i64],
        pads: Vec[i64],
        strides: Vec[i64],
    ): ...


class Op(Expr):
    """
    TODO: There is currently no conversion from egglog for the following operators:
        - ConstantOfShape
        - Cast
        - Slice
        - HardSigmoid
        - Div
        - Sub
        - Pow
        - Sqrt
        - Erf
        - Where
        - Identity
        - Equal
    """

    @method()
    def __init__(self, name: String): ...

    @method()
    @classmethod
    def Add(
        cls, name: TensorId, a: TensorId, b: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def AveragePool(
        cls, name: TensorId, attrs: AveragePoolAttrs, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method(cost=20)
    @classmethod
    def BatchNormalization(
        cls,
        name: TensorId,
        attrs: BatchNormAttrs,
        x: TensorId,
        scale: TensorId,
        bias: TensorId,
        input_mean: TensorId,
        input_var: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def Cast(
        cls,
        name: TensorId,
        to: i64,
        input: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method(cost=10)
    @classmethod
    def Clip(
        cls,
        name: TensorId,
        input: TensorId,
        min: TensorId,
        max: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def Concat(
        cls, name: TensorId, axis: i64, inputs: Vec[TensorId], output: TensorId
    ) -> TensorId: ...

    # TODO: Note: There are other possible Constant types but I'm only supporting tensor types
    @method()
    @classmethod
    def Constant(
        cls, name: TensorId, value: TensorType, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def ConstantOfShape(
        cls, name: TensorId, value: TensorType, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method(cost=100)
    @classmethod
    def Conv(
        cls,
        name: TensorId,
        attrs: ConvAttrs,
        x: TensorId,
        w: TensorId,
        b: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def DequantizeLinear(
        cls,
        name: TensorId,
        x: TensorId,
        x_scale: TensorId,
        x_zero_point: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def Div(
        cls, name: TensorId, a: TensorId, b: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Dropout(
        cls, name: TensorId, seed: i64, inputs: Vec[TensorId], outputs: Vec[TensorId]
    ) -> TensorId: ...

    @method()
    @classmethod
    def Equal(
        cls, name: TensorId, a: TensorId, b: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Expand(
        cls, name: TensorId, input: TensorId, shape: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Erf(cls, name: TensorId, input: TensorId, output: TensorId) -> TensorId: ...

    @method()
    @classmethod
    def Flatten(
        cls, name: TensorId, axis: i64, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method(cost=100)
    @classmethod
    def FusedConvActivation(
        cls,
        name: TensorId,
        domain: TensorId,
        attrs: FusedConvAttrs,
        x: TensorId,
        w: TensorId,
        b: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method(cost=100)
    @classmethod
    def FusedConvBatchNorm(
        cls,
        name: TensorId,
        convattrs: FusedConvAttrs,
        bn_attrs: BatchNormAttrs,
        x: TensorId,
        bn_scale: TensorId,
        bn_bias: TensorId,
        bn_input_mean: TensorId,
        bn_input_var: TensorId,
        conv_w: TensorId,
        conv_b: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method(cost=100)
    @classmethod
    def FusedGemm(
        cls,
        name: TensorId,
        domain: TensorId,
        attrs: FusedGemmAttrs,
        inputs: Vec[TensorId],
        out: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def Gather(
        cls,
        name: TensorId,
        axis: i64,
        data: TensorId,
        indices: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method(cost=100)
    @classmethod
    def Gemm(
        cls,
        name: TensorId,
        attrs: GemmAttrs,
        inputs: Vec[TensorId],
        out: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def GlobalAveragePool(
        cls, name: TensorId, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def HardSigmoid(
        cls, name: TensorId, alpha: f64, beta: f64, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Identity(
        cls, name: TensorId, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def MatMul(
        cls, name: TensorId, a: TensorId, b: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def MaxPool(
        cls,
        name: TensorId,
        attrs: MaxPoolAttrs,
        input: TensorId,
        outputs: Vec[TensorId],
    ) -> TensorId: ...

    @method()
    @classmethod
    def Mul(
        cls, name: TensorId, a: TensorId, b: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Pad(
        cls, name: TensorId, mode: String, inputs: Vec[TensorId], output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Pow(
        cls, name: TensorId, base: TensorId, exp: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def QuantizeLinear(
        cls,
        name: TensorId,
        x: TensorId,
        y_scale: TensorId,
        y_zero_point: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def QLinearAdd(
        cls, name: TensorId, domain: TensorId, inputs: Vec[TensorId], output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def QLinearConv(
        cls,
        name: TensorId,
        attrs: ConvAttrs,
        inputs: Vec[TensorId],
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def QLinearGlobalAveragePool(
        cls,
        name: TensorId,
        domain: TensorId,
        channels_last: i64,
        inputs: Vec[TensorId],
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def QLinearMatMul(
        cls,
        name: TensorId,
        inputs: Vec[TensorId],
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def ReduceMean(
        cls,
        name: TensorId,
        axes: Vec[i64],
        keepdims: i64,
        input: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method(cost=10)
    @classmethod
    def Relu(cls, name: TensorId, input: TensorId, output: TensorId) -> TensorId: ...

    @method()
    @classmethod
    def Reshape(
        cls,
        name: TensorId,
        data: TensorId,
        shape: TensorId,
        output: TensorId,
    ) -> TensorId: ...

    @method()
    @classmethod
    def Shape(cls, name: TensorId, input: TensorId, output: TensorId) -> TensorId: ...

    @method()
    @classmethod
    def Sigmoid(cls, name: TensorId, input: TensorId, output: TensorId) -> TensorId: ...

    @method()
    @classmethod
    def Slice(
        cls, name: TensorId, inputs: Vec[TensorId], output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Softmax(
        cls, name: TensorId, axis: i64, input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Split(
        cls,
        name: TensorId,
        axis: i64,
        split: Vec[i64],
        input: TensorId,
        outputs: Vec[TensorId],
    ) -> TensorId: ...

    @method()
    @classmethod
    def Squeeze(
        cls, name: TensorId, axes: Vec[i64], input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Sqrt(cls, name: TensorId, input: TensorId, output: TensorId) -> TensorId: ...

    @method()
    @classmethod
    def Sub(
        cls, name: TensorId, a: TensorId, b: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Transpose(
        cls, name: TensorId, perm: Vec[i64], input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Unsqueeze(
        cls, name: TensorId, axes: Vec[i64], input: TensorId, output: TensorId
    ) -> TensorId: ...

    @method()
    @classmethod
    def Where(
        cls,
        name: TensorId,
        condition: TensorId,
        a: TensorId,
        b: TensorId,
        output: TensorId,
    ) -> TensorId: ...
