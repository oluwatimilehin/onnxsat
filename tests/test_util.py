from onnx import NodeProto

from typing import List


class TestUtil:
    @classmethod
    def compare_onnx_nodes(
        cls, actual_nodes: List[NodeProto], expected_nodes: List[NodeProto]
    ):
        assert len(actual_nodes) == len(expected_nodes)

        actual_nodes_by_name = {node.name: node for node in actual_nodes}
        expected_nodes_by_name = {node.name: node for node in expected_nodes}

        for name, actual_node in actual_nodes_by_name.items():
            expected_node = expected_nodes_by_name[name]

            assert actual_node.name == expected_node.name
            assert actual_node.op_type == expected_node.op_type
            assert actual_node.domain == expected_node.domain
            assert actual_node.input == expected_node.input
            assert actual_node.output == expected_node.output
            cls._compare_attributes(actual_node, expected_node)

    @classmethod
    def _compare_attributes(cls, actual_node, expected_node):
        # Because we set default values in the egg, we need to compare the attributes by name
        # instead of by index

        actual_attrs_by_name = {attr.name: attr for attr in actual_node.attribute}
        expected_attrs_by_name = {attr.name: attr for attr in expected_node.attribute}

        for name, expected_attr in expected_attrs_by_name.items():
            actual_attr = actual_attrs_by_name.get(name)

            assert (
                actual_attr is not None
            ), f"Could not find attribute: {name} for {actual_node}"
            assert actual_attr.name == expected_attr.name
            assert actual_attr.type == expected_attr.type
            assert actual_attr.i == expected_attr.i
            assert actual_attr.f == expected_attr.f
            assert actual_attr.s == expected_attr.s

            if actual_attr.t is not None:
                assert actual_attr.t.dims == expected_attr.t.dims
                assert actual_attr.t.data_type == expected_attr.t.data_type
                assert actual_attr.t.raw_data == expected_attr.t.raw_data
