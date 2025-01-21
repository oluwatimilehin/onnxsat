from eggie.operators import *
from eggie.parser import EggParser
from egglog import Expr
from onnx.onnx_ml_pb2 import GraphProto, NodeProto
from typing import List

from onnx_e.parser import OnnxParser


class Converter:
    @classmethod
    def to_egglog(cls, graph: GraphProto) -> Expr:
        return OnnxParser(graph).parse()

    @classmethod
    def to_onnx(cls, egglog: Expr) -> List[NodeProto]:
        parser = EggParser(egglog)
        return parser.parse()
