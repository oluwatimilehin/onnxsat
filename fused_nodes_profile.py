from typing import List

import onnx_tool
from onnx_tool.node import ConvNode, GemmNode
from onnx_tool.tensor import Tensor


@onnx_tool.NODE_REGISTRY.register()
class FusedConvNode(ConvNode):
    def __init__(self, n):
        super(FusedConvNode, self).__init__(n)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        super(FusedConvNode, self).shape_infer(intensors, outtensors)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return super(FusedConvNode, self).profile(intensors, outtensors)


@onnx_tool.NODE_REGISTRY.register()
class FusedGemmNode(GemmNode):
    def __init__(self, n):
        super(FusedGemmNode, self).__init__(n)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        super(FusedGemmNode, self).shape_infer(intensors, outtensors)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return super(FusedGemmNode, self).profile(intensors, outtensors)
