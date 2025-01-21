from __future__ import annotations
from egglog import *
from eggie.operators import *

fusion_ruleset = ruleset(name="fusion_ruleset")


@fusion_ruleset.register
def fuse_conv_batchnorm(
    conv_name: TensorId,
    autopad: String,
    group: i64,
    dilations: Vec[i64],
    kernel_shape: Vec[i64],
    pads: Vec[i64],
    strides: Vec[i64],
    conv_x: TensorId,
    conv_w: TensorId,
    conv_b: TensorId,
    conv_out: TensorId,
    bn_name: TensorId,
    bn_attrs: BatchNormAttrs,
    bn_scale: TensorId,
    bn_bias: TensorId,
    bn_mean: TensorId,
    bn_var: TensorId,
    bn_out: TensorId,
):
    yield rewrite(
        Op.BatchNormalization(
            bn_name,
            bn_attrs,
            Op.Conv(
                conv_name,
                ConvAttrs(autopad, group, dilations, kernel_shape, pads, strides),
                conv_x,
                conv_w,
                conv_b,
                conv_out,
            ),
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            bn_out,
        )
    ).to(
        Op.FusedConvBatchNorm(
            conv_name,
            FusedConvAttrs(
                "",
                Vec[f64](*[]),
                autopad,
                group,
                dilations,
                kernel_shape,
                pads,
                strides,
            ),
            bn_attrs,
            conv_x,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            conv_w,
            conv_b,
            bn_out,
        )
    )


@fusion_ruleset.register
def fuse_conv_bn_relu(
    conv_name: TensorId,
    autopad: String,
    group: i64,
    dilations: Vec[i64],
    kernel_shape: Vec[i64],
    pads: Vec[i64],
    strides: Vec[i64],
    conv_x: TensorId,
    conv_w: TensorId,
    conv_b: TensorId,
    conv_out: TensorId,
    bn_name: TensorId,
    bn_attrs: BatchNormAttrs,
    bn_scale: TensorId,
    bn_bias: TensorId,
    bn_mean: TensorId,
    bn_var: TensorId,
    bn_out: TensorId,
    relu_name: TensorId,
    relu_out: TensorId,
):
    empty_activations = []

    yield rewrite(
        Op.Relu(
            relu_name,
            Op.BatchNormalization(
                bn_name,
                bn_attrs,
                Op.Conv(
                    conv_name,
                    ConvAttrs(autopad, group, dilations, kernel_shape, pads, strides),
                    conv_x,
                    conv_w,
                    conv_b,
                    conv_out,
                ),
                bn_scale,
                bn_bias,
                bn_mean,
                bn_var,
                bn_out,
            ),
            relu_out,
        )
    ).to(
        Op.FusedConvBatchNorm(
            conv_name,
            FusedConvAttrs(
                "Relu",
                Vec[f64](*empty_activations),
                autopad,
                group,
                dilations,
                kernel_shape,
                pads,
                strides,
            ),
            bn_attrs,
            conv_x,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            conv_w,
            conv_b,
            relu_out,
        )
    )


@fusion_ruleset.register
def fuse_conv_relu(
    conv_name: TensorId,
    conv_x: TensorId,
    conv_w: TensorId,
    conv_b: TensorId,
    autopad: String,
    group: i64,
    dilations: Vec[i64],
    kernel_shape: Vec[i64],
    pads: Vec[i64],
    strides: Vec[i64],
    conv_out: TensorId,
    relu_name: TensorId,
    relu_out: TensorId,
):
    empty_activations = []

    yield rewrite(
        Op.Relu(
            relu_name,
            Op.Conv(
                conv_name,
                ConvAttrs(autopad, group, dilations, kernel_shape, pads, strides),
                conv_x,
                conv_w,
                conv_b,
                conv_out,
            ),
            relu_out,
        )
    ).to(
        Op.FusedConvActivation(
            conv_name,
            TensorId("com.microsoft"),
            FusedConvAttrs(
                "Relu",
                Vec[f64](*empty_activations),
                autopad,
                group,
                dilations,
                kernel_shape,
                pads,
                strides,
            ),
            conv_x,
            conv_w,
            conv_b,
            relu_out,
        )
    )


@fusion_ruleset.register
def fuse_conv_clip(
    conv_name: TensorId,
    conv_x: TensorId,
    conv_w: TensorId,
    conv_b: TensorId,
    autopad: String,
    group: i64,
    dilations: Vec[i64],
    kernel_shape: Vec[i64],
    pads: Vec[i64],
    strides: Vec[i64],
    conv_out: TensorId,
    clip_name: TensorId,
    clip_out: TensorId,
    clip_min: TensorId,
    clip_max: TensorId,
):
    min_and_max: Vec[f64] = Vec(f64(0.0), f64(6.0))

    yield rewrite(
        Op.Clip(
            clip_name,
            Op.Conv(
                conv_name,
                ConvAttrs(autopad, group, dilations, kernel_shape, pads, strides),
                conv_x,
                conv_w,
                conv_b,
                conv_out,
            ),
            clip_min,
            clip_max,
            clip_out,
        )
    ).to(
        Op.FusedConvActivation(
            conv_name,
            TensorId("com.microsoft"),
            FusedConvAttrs(
                "Clip",
                min_and_max,
                autopad,
                group,
                dilations,
                kernel_shape,
                pads,
                strides,
            ),
            conv_x,
            conv_w,
            conv_b,
            clip_out,
        )
    )


@fusion_ruleset.register
def fuse_gemm_relu(
    gemm_name: TensorId,
    gemm_in: Vec[TensorId],
    alpha: f64,
    beta: f64,
    transA: i64,
    transB: i64,
    gemm_out: TensorId,
    relu_name: TensorId,
    relu_out: TensorId,
):
    yield rewrite(
        Op.Relu(
            relu_name,
            Op.Gemm(
                gemm_name,
                GemmAttrs(alpha, beta, transA, transB),
                gemm_in,
                gemm_out,
            ),
            relu_out,
        )
    ).to(
        Op.FusedGemm(
            gemm_name,
            TensorId("com.microsoft"),
            FusedGemmAttrs(
                "Relu", f64(0.0), f64(0.0), f64(0.0), alpha, beta, transA, transB
            ),
            gemm_in,
            relu_out,
        )
    )
