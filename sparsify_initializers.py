# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This script opens an existing model in onnx format and attempts to
# move initializers from model.graph.initializer field to model.graph.sparse_initializer field
# and convert them into ONNX COO flat index format.

# Copied code from : https://github.com/microsoft/onnxruntime/blob/main/tools/python/sparsify_initializers.py

import logging
import sys
from typing import Tuple

import numpy as np
import onnx
from onnx import ModelProto, SparseTensorProto, TensorProto, numpy_helper

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(filename)20s: %(message)s"))
logging_level = logging.ERROR
log_handler.setLevel(logging_level)
logger.addHandler(log_handler)
logger.setLevel(logging_level)

real_types = {int(TensorProto.FLOAT), int(TensorProto.DOUBLE)}

def convert_tensor_to_sparse(tensor, sparsity_threshold, tolerance):  # type: (TensorProto, float, float) -> Tuple[SparseTensorProto, float]
    """returns a tuple of sparse_tensor and sparsity level"""
    values = []
    indices = []
    nnz_count = 0
    tensor_data = numpy_helper.to_array(tensor).flatten()
    data_len = len(tensor_data)
    if tensor_data.dtype in real_types:
        for index in range(data_len):
            el = tensor_data[index]
            if abs(el) <= tolerance:
                values.append(el)
                indices.append(index)
                nnz_count += 1
    else:
        for index in range(data_len):
            el = tensor_data[index]
            if el != 0:
                values.append(el)
                indices.append(index)
                nnz_count += 1

    sparsity = 1.0 - float(nnz_count) / data_len

    ind_data_type = TensorProto.INT64
    ind_dtype = np.int64
    ind_len = len(indices)
    max_indices_value = 0

    logger.debug(
        f"initializer={tensor.name}, dtype={tensor_data.dtype}, \
                 data_len={data_len}, nnz={nnz_count}, sparsity={sparsity}, \
                 max_indices_value={max_indices_value}, sparse_indices_type={ind_dtype}"
    )

    sparsity = np.round(sparsity, 2)
    if sparsity < sparsity_threshold:
        return (object(), sparsity)

    tensor_data_bytes = tensor_data.nbytes
    # create np array and cast data to the appropriate type
    np_values = np.array(values).astype(tensor_data.dtype)
    # create np array and cast data to the inferred index type
    np_indices = np.array(indices).astype(ind_dtype)
    total_sparse_bytes = np_values.nbytes + np_indices.nbytes

    logger.debug(
        f"initializer={tensor.name}, initializer_bytes={tensor_data_bytes}, \
                sparse_initializer_bytes={total_sparse_bytes}"
    )

    values_tensor = onnx.helper.make_tensor(
        tensor.name, tensor.data_type, [len(values)], np_values.tobytes(), raw=True
    )

    indicies_tensor = onnx.helper.make_tensor(
        tensor.name + "_indicies",
        ind_data_type,
        [ind_len],
        np_indices.tobytes(),
        raw=True,
    )

    sparse_tensor = onnx.helper.make_sparse_tensor(
        values_tensor, indicies_tensor, tensor.dims
    )
    return (sparse_tensor, sparsity)


def convert_initializers(model, sparsity_threshold, tolerance):  # type: (ModelProto, float, float) -> ModelProto
    graph = model.graph
    converted_sparse = []
    remaining_initializers = []
    for initializer in graph.initializer:
        if initializer.data_type == TensorProto.BOOL:
            logger.info(f"initializer={initializer.name} contains bool, not converted")
            remaining_initializers.append(initializer)
            continue
        sparse_tensor, sparsity = convert_tensor_to_sparse(
            initializer, sparsity_threshold, tolerance
        )

        if sparsity >= sparsity_threshold:
            logger.info(
                f"initializer={initializer.name} converted. sparsity={sparsity}"
            )
            converted_sparse.append(sparse_tensor)
        else:
            remaining_initializers.append(initializer)
            logger.info(
                f"initializer={initializer.name} is not converted. sparsity={sparsity}"
            )

    graph.sparse_initializer.extend(converted_sparse)
    del graph.initializer[:]
    graph.initializer.extend(remaining_initializers)
    return model
