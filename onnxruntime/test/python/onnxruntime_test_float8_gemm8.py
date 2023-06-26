# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import gc
import unittest
import time
from itertools import product

import numpy as np
from numpy.testing import assert_allclose
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxruntime import InferenceSession


class TestFloat8Gemm8(unittest.TestCase):
    def get_model_gemm(
        self,
        float_name,
        alpha=1.0,
        beta=0.0,
        transA=1,
        transB=0,
        row_major=1,
        fast_accumulation_mode=0,
        compute_type="CUBLAS_COMPUTE_32F",
        domain="",
        dtype=TensorProto.FLOAT,
        bias=False,
    ):
        proto_type = getattr(TensorProto, float_name)
        use_f8 = proto_type in (TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2)

        a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        d = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

        inits = []
        kwargs = {}
        node_inputs = ["Af", "Bf"]
        inputs = [a, b]
        if bias:
            inputs.append(make_tensor_value_info("C", TensorProto.FLOAT, [None, None]))
            node_inputs = ["Af", "Bf", "Cf"]
            if use_f8:
                node_inputs.extends(["one"] * 3)
        elif use_f8:
            node_inputs.append("")
            node_inputs.extend(["one"] * 3)

        if use_f8:
            assert domain == "com.microsoft"
            inits.append(from_array(np.array([1], dtype=np.float32), name="one"))
            kwargs = dict(
                rowMajor=row_major,
                computeType=compute_type,
                fastAccumulationMode=fast_accumulation_mode,
                domain=domain,
                dtype=dtype,
            )
            op_name = "GemmFloat8"
        elif domain == "com.microsoft":
            op_name = "GemmFloat8"
            kwargs = dict(
                rowMajor=row_major,
                computeType=compute_type,
                fastAccumulationMode=fast_accumulation_mode,
                domain=domain,
                dtype=dtype,
            )
        else:
            op_name = "Gemm"
        nodes = [
            make_node("Cast", ["A"], ["Af"], to=proto_type),
            make_node("Cast", ["B"], ["Bf"], to=proto_type),
            make_node("Cast", ["C"], ["Cf"], to=proto_type) if bias else None,
            make_node(
                op_name,
                node_inputs,
                ["Yf"],
                transA=transA,
                transB=transB,
                alpha=alpha,
                beta=beta,
                **kwargs,
            ),
            make_node("Cast", ["Yf"], ["Y"], to=TensorProto.FLOAT),
        ]
        nodes = [n for n in nodes if n is not None]
        graph = make_graph(nodes, "gemm", inputs, [d], inits)
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)], ir_version=9)
        if domain != "com.microsoft":
            check_model(onnx_model)
        return onnx_model

    def common_test_model_gemm(self, float_type, mul=0.33, atol=0, rtol=0, bias=False, **kwargs):
        n = 16
        a = np.arange(n**2).reshape((-1, n)).astype(np.float32)
        b = (a * mul).astype(np.float32)

        if kwargs.get("row_major", 1):
            feeds_1 = {"A": a, "B": b}
            feeds_2 = feeds_1
            if bias:
                c = -a.copy()
                feeds_1["C"] = c
                feeds_2["C"] = c
                expected = a.T @ b + c
            else:
                expected = a.T @ b

        else:
            feeds_1 = {"A": a, "B": b}
            feeds_2 = {"A": a.T, "B": b.T}
            expected = (a.T @ b).T
            if bias:
                c = -a.copy()
                feeds_1["C"] = c.T
                feeds_2["C"] = c.T
                expected = a.T @ b + c
            else:
                expected = a.T @ b

        onnx_model = self.get_model_gemm("FLOAT", bias=bias, beta=1.0 if bias else 0.0)

        ref = InferenceSession(
            onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref.run(None, feeds_1)[0]
        if float_type in ("FLOAT", "FLOAT16"):
            assert_allclose(expected, y, atol=atol, rtol=rtol)
        self.assertEqual(expected.shape, y.shape)
        self.assertEqual(expected.dtype, y.dtype)

        onnx_model_f8 = self.get_model_gemm(
            float_type, bias=bias, beta=1.0 if bias else 0.0, domain="com.microsoft", **kwargs
        )
        try:
            ref8 = InferenceSession(
                onnx_model_f8.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception as e:
            if "CUDA < 12.0 does not support bias" in str(e):
                return
            raise AssertionError(f"Could not load model {onnx_model_f8}") from e
        try:
            y = ref8.run(None, feeds_2)[0]
        except Exception as e:
            raise AssertionError(f"Could not execute model {onnx_model_f8}") from e
        assert_allclose(expected, y, atol=atol, rtol=rtol)
        self.assertEqual(expected.shape, y.shape)
        self.assertEqual(expected.dtype, y.dtype)

    def test_model_gemm_float(self):
        self.common_test_model_gemm("FLOAT", row_major=1, rtol=1e-5)

    def test_model_gemm_float_bias(self):
        self.common_test_model_gemm("FLOAT", row_major=1, rtol=1e-5, bias=True)

    def test_model_gemm_float_col_major(self):
        self.common_test_model_gemm("FLOAT", row_major=0, rtol=1e-5)

    def test_model_gemm_float16(self):
        self.common_test_model_gemm(
            "FLOAT16",
            row_major=True,
            compute_type="CUBLAS_COMPUTE_32F",
            rtol=1e-3,
            dtype=TensorProto.FLOAT16,
            mul=0.0001,
        )

    def test_model_gemm_float8_e4m3(self):
        self.common_test_model_gemm(
            "FLOAT8E4M3FN", compute_type="CUBLAS_COMPUTE_32F", row_major=0, rtol=1e-5, dtype=TensorProto.FLOAT
        )


if __name__ == "__main__":
    TestFloat8Gemm8().test_model_gemm_float_bias()
    # TestFloat8Gemm8().test_model_gemm_float_col_major()
    # TestFloat8Gemm8().test_model_gemm_float16()
    # stop
    unittest.main(verbosity=2)
