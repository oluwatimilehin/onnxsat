from enum import Enum, auto
import logging
from collections import namedtuple


class EgglogTokenKind(Enum):
    AVG_POOL_ATTRS = auto()
    BATCH_NORM_ATTRS = auto()
    COMMA = auto()
    CONV_ATTRS = auto()
    DOT = auto()
    EQUALS = auto()
    EOF = auto()
    FLOAT_LITERAL = auto()
    FUSED_CONV_ATTRS = auto()
    FUSED_GEMM_ATTRS = auto()
    GEMM_ATTRS = auto()
    LEFT_PARENTHESIS = auto()
    LEFT_SQUARE_BRACKET = auto()
    MAX_POOL_ATTRS = auto()
    INTEGER_LITERAL = auto()
    OP = auto()
    QUANTIZE_LINEAR_ATTRS = auto()
    RIGHT_PARENTHESIS = auto()
    RIGHT_SQUARE_BRACKET = auto()
    STRING_LITERAL = auto()
    TENSOR_ID = auto()
    TENSOR_TYPE = auto()
    VARIABLE_NAME = auto()
    VEC = auto()


str_to_token = {
    "AveragePoolAttrs": EgglogTokenKind.AVG_POOL_ATTRS,
    "BatchNormAttrs": EgglogTokenKind.BATCH_NORM_ATTRS,
    "ConvAttrs": EgglogTokenKind.CONV_ATTRS,
    "FusedConvAttrs": EgglogTokenKind.FUSED_CONV_ATTRS,
    "FusedGemmAttrs": EgglogTokenKind.FUSED_GEMM_ATTRS,
    "GemmAttrs": EgglogTokenKind.GEMM_ATTRS,
    "MaxPoolAttrs": EgglogTokenKind.MAX_POOL_ATTRS,
    "Op": EgglogTokenKind.OP,
    "QuantizeLinearAttrs": EgglogTokenKind.QUANTIZE_LINEAR_ATTRS,
    "TensorId": EgglogTokenKind.TENSOR_ID,
    "TensorType": EgglogTokenKind.TENSOR_TYPE,
    "Vec": EgglogTokenKind.VEC,
    "=": EgglogTokenKind.EQUALS,
    "[": EgglogTokenKind.LEFT_SQUARE_BRACKET,
    "]": EgglogTokenKind.RIGHT_SQUARE_BRACKET,
    "(": EgglogTokenKind.LEFT_PARENTHESIS,
    ")": EgglogTokenKind.RIGHT_PARENTHESIS,
    ",": EgglogTokenKind.COMMA,
    ".": EgglogTokenKind.DOT,
}


EgglogToken = namedtuple("EgglogToken", ["kind", "text"])


class Lexer:
    def __init__(self, input: str):
        self.input = input
        self.index = 0

    def next_token(self) -> EgglogToken:
        if self.index >= len(self.input):
            return EgglogToken(EgglogTokenKind.EOF, "")

        if self.input[self.index] == ",":
            # simplify the parser by not returning commas
            self.index += 1

        char = self.input[self.index]

        while char.isspace():
            self.index += 1
            if self.index >= len(self.input):
                return EgglogToken(EgglogTokenKind.EOF, "")
            char = self.input[self.index]

        if char.isalnum() or char == "-":
            token: str = ""

            if char == "-":
                token += char
                self.index += 1
                char = self.input[self.index]

            while char.isalnum():
                token += char
                self.index += 1
                char = self.input[self.index]

            if token in str_to_token:
                return EgglogToken(str_to_token[token], token)

            if token.lstrip("-").isdigit():
                if self.input[self.index] == ".":
                    # parse float
                    token += self.input[self.index]
                    self.index += 1

                    char = self.input[self.index]
                    while char.isdigit() or char in {
                        "e",
                        "-",
                    }:  # rhs of condition matches small floats and negative numbers
                        token += char
                        self.index += 1
                        char = self.input[self.index]

                    return EgglogToken(EgglogTokenKind.FLOAT_LITERAL, token)

                return EgglogToken(EgglogTokenKind.INTEGER_LITERAL, token)

            return EgglogToken(EgglogTokenKind.STRING_LITERAL, token)

        if char == '"':
            token = ""
            self.index += 1

            char = self.input[self.index]
            while char != '"':
                token += char
                self.index += 1
                char = self.input[self.index]

            self.index += 1
            return EgglogToken(EgglogTokenKind.STRING_LITERAL, token)

        # Parse a variable name; they are not enclosed in double quotes
        if char == "_":
            token = char
            self.index += 1

            char = self.input[self.index]
            while char.isalnum() or char == "_":
                token += char
                self.index += 1
                char = self.input[self.index]

            return EgglogToken(EgglogTokenKind.VARIABLE_NAME, token)

        self.index += 1
        if char not in str_to_token:
            raise ValueError(f"Unknown character: {char}")

        return EgglogToken(str_to_token[char], char)
