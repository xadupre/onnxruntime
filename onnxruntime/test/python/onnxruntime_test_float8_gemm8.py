# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxruntime import InferenceSession


class TestFloat8Gemm8(unittest.TestCase):
    def get_model_gemm(
        self,
        float_name,
        alpha=1.0,
        beta=0.0,
        transA=0,
        transB=0,
        row_major=1,
        fast_accumulation_mode=0,
        compute_type="CUBLAS_COMPUTE_32F",
        domain="",
        dtype=TensorProto.FLOAT,
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
        bias = beta != 0
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

    def common_test_model_gemm(self, float_type, mul=0.33, atol=0, rtol=0, square=True, **kwargs):
        if square:
            a = (np.arange(256) * 0.01).astype(np.float32).reshape((-1, 16))
            b = (np.arange(256) * -0.01).astype(np.float32).reshape((-1, 16))
            c = (np.arange(256) * 0.03).astype(np.float32).reshape((-1, 16))
            b[:, 0] += 1
        else:
            a = (np.arange(256) / 256).astype(np.float32).reshape((32, -1))
            b = (np.arange(512) / 512).astype(np.float32).reshape((32, -1))
            c = (np.arange(128) / 128).astype(np.float32).reshape((8, 16))

        feeds = {"A": a, "B": b}

        expected = (a.T if kwargs.get("transA", 0) else a) @ (b.T if kwargs.get("transB", 0) else b)
        expected *= kwargs.get("alpha", 1.0)
        if kwargs.get("beta", 0) != 0:
            expected += kwargs["beta"] * c
            feeds["C"] = c

        onnx_model = self.get_model_gemm("FLOAT", **kwargs)

        ref = InferenceSession(
            onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref.run(None, feeds)[0]
        if float_type in ("FLOAT", "FLOAT16"):
            try:
                assert_allclose(expected, y, atol=atol, rtol=rtol)
            except Exception as e:

                def check(f):
                    try:
                        return f()[:2, :2]
                    except Exception as e:
                        return str(e)

                raise AssertionError(
                    f"Gemm ERROR len(inputs)={len(feeds)}"
                    f"\na@b=\n{check(lambda:a@b)}"
                    f"\na.T@b=\n{check(lambda:a.T@b)}"
                    f"\na@b.T=\n{check(lambda:a@b.T)}"
                    f"\na.T@b.T=\n{check(lambda:a.T@b.T)}"
                    f"\n----\nb@a=\n{check(lambda:b@a)}"
                    f"\nb.T@a=\n{check(lambda:b.T@a)}"
                    f"\nb@a.T=\n{check(lambda:b@a.T)}"
                    f"\nb.T@a.T=\n{check(lambda:b.T@a.T)}"
                    f"\n----\nexpected=\n{expected[:2,:2]}"
                    f"\n----\ngot=\n{y[:2,:2]}"
                    f"\nkwargs={kwargs}"
                ) from e

        self.assertEqual(expected.shape, y.shape)
        self.assertEqual(expected.dtype, y.dtype)

        onnx_model_f8 = self.get_model_gemm(float_type, domain="com.microsoft", **kwargs)
        try:
            ref8 = InferenceSession(
                onnx_model_f8.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception as e:
            if "CUDA < 12.0 does not support bias" in str(e):
                return
            raise AssertionError(f"Could not load model {onnx_model_f8}") from e
        try:
            y = ref8.run(None, feeds)[0]
        except Exception as e:
            if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
                # Skipping. This machine does not support float8.
                warnings.warn("unable to test with float8 on this machine.")
                return
            raise AssertionError(f"Could not execute model {onnx_model_f8}") from e
        try:
            assert_allclose(expected, y, atol=atol, rtol=rtol)
        except Exception as e:

            def check(f):
                try:
                    return f()[:2, :2]
                except Exception as e:
                    return str(e)

            raise AssertionError(
                f"Gemm ERROR len(inputs)={len(feeds)}"
                f"\na@b=\n{check(lambda:a@b)}"
                f"\na.T@b=\n{check(lambda:a.T@b)}"
                f"\na@b.T=\n{check(lambda:a@b.T)}"
                f"\na.T@b.T=\n{check(lambda:a.T@b.T)}"
                f"\n----\nb@a=\n{check(lambda:b@a)}"
                f"\nb.T@a=\n{check(lambda:b.T@a)}"
                f"\nb@a.T=\n{check(lambda:b@a.T)}"
                f"\nb.T@a.T=\n{check(lambda:b.T@a.T)}"
                f"\n----\nexpected=\n{expected[:2,:2]}"
                f"\n----\ngot=\n{y[:2,:2]}"
                f"\nkwargs={kwargs}"
            ) from e
        self.assertEqual(expected.shape, y.shape)
        self.assertEqual(expected.dtype, y.dtype)

    def test_model_gemm_float(self):
        self.common_test_model_gemm("FLOAT", transA=1, row_major=1, rtol=1e-4)

    def test_model_gemm_float_bias(self):
        self.common_test_model_gemm("FLOAT", transA=1, row_major=1, beta=1.0, rtol=1e-4)

    def test_model_gemm_float_col_major(self):
        self.common_test_model_gemm("FLOAT", transB=1, row_major=0, rtol=1e-4)

    def test_model_gemm_float_col_major_bias(self):
        self.common_test_model_gemm("FLOAT", transB=1, row_major=0, beta=1.0, rtol=1e-4)

    def test_model_gemm_float16(self):
        self.common_test_model_gemm(
            "FLOAT16",
            row_major=1,
            compute_type="CUBLAS_COMPUTE_32F",
            rtol=1e-2,
            dtype=TensorProto.FLOAT16,
            transB=1,
        )

    def test_model_gemm_float16_col_major(self):
        self.common_test_model_gemm(
            "FLOAT16",
            row_major=0,
            compute_type="CUBLAS_COMPUTE_32F",
            rtol=1e-2,
            dtype=TensorProto.FLOAT16,
            transB=1,
        )

    def test_model_gemm_float8_e4m3(self):
        self.common_test_model_gemm(
            "FLOAT8E4M3FN", compute_type="CUBLAS_COMPUTE_32F", row_major=0, rtol=1e-5, dtype=TensorProto.FLOAT
        )


if __name__ == "__main__":
    TestFloat8Gemm8().test_model_gemm_float()
    unittest.main(verbosity=2)
