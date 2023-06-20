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


class TestFloat8Gemm8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from onnxruntime import InferenceSession

        cls.InferenceSession = InferenceSession
        # cls.available_providers = [provider for provider in onnxruntime.get_available_providers()]

    def get_model_gemm(self, float_name, alpha=1.0, beta=0.0, transA=1, transB=0, add_bias=False):
        proto_type = getattr(TensorProto, float_name)

        a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        c = None if not add_bias else make_tensor_value_info("C", TensorProto.FLOAT, [None, None])
        d = make_tensor_value_info("D", TensorProto.FLOAT, [None, None])

        nodes = [
            make_node("Cast", ["A"], ["Af"], to=proto_type),
            make_node("Cast", ["B"], ["Bf"], to=proto_type),
            None if c is None else make_node("Cast", ["C"], ["Cf"], to=proto_type),
            make_node(
                "Gemm",
                ["Af", "Bf"] if c is None else ["Af", "Bf", "Cf"],
                ["Df"],
                transA=transA,
                transB=transB,
                alpha=alpha,
                beta=beta,
            ),
            make_node("Cast", ["Df"], ["D"], to=TensorProto.FLOAT),
        ]
        nodes = [n for n in nodes if n is not None]
        graph = make_graph(nodes, "gemm", [a, b] if c is None else [a, b, c], [d])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)], ir_version=9)
        check_model(onnx_model)
        return onnx_model

    def get_model_gemm_float8(
        self,
        float_types,
        alpha=1.0,
        beta=0.0,
        transA=1,
        transB=0,
        smCount=0,
        fastAccumulationMode=0,
        compute_type="CUBLAS_COMPUTE_32F",
        add_bias=False,
    ):
        proto_type = [getattr(TensorProto, float_name) for float_name in float_types]

        a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        c = None if not add_bias else make_tensor_value_info("C", TensorProto.FLOAT, [None, None])
        d = make_tensor_value_info("D", TensorProto.FLOAT, [None, None])
        zero = from_array(np.array([0], dtype=np.float32), name="zero")

        nodes = [
            make_node("Cast", ["zero"], ["zeros"], to=proto_type[3]),
            make_node("Cast", ["zero"], ["zerof"], to=proto_type[4]),
            make_node("Cast", ["A"], ["Af"], to=proto_type[0]),
            make_node("Cast", ["B"], ["Bf"], to=proto_type[1]),
            make_node("Cast", ["zero" if c is None else "C"], ["Cf"], to=proto_type[2]),
            make_node(
                "GemmFloat8",
                ["Af", "Bf", "Cf", "zeros", "zerof"],
                ["Df"],
                domain="com.microsoft",
                transA=transA,
                transB=transB,
                smCount=smCount,
                fastAccumulationMode=fastAccumulationMode,
                alpha=alpha,
                beta=beta,
                name="gemmf8",
                computeType=compute_type,
            ),
            make_node("Cast", ["Df"], ["D"], to=TensorProto.FLOAT),
        ]
        nodes = [n for n in nodes if n is not None]
        graph = make_graph(nodes, "gemmf8model", [a, b] if c is None else [a, b, c], [d], [zero])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 19), make_opsetid("com.microsoft", 1)], ir_version=9
        )
        check_model(onnx_model)
        return onnx_model

    def common_test_model_gemm(self, float_type, mul=1, atol=0, rtol=0, **kwargs):
        n = 768
        a = np.arange(n**2).reshape((-1, n)).astype(np.float32)
        b = (2 ** np.arange(n**2).reshape((-1, n)) * mul).astype(np.float32)
        expected = a.T @ b
        feeds = {"A": a, "B": b}

        onnx_model = self.get_model_gemm("FLOAT")
        if float_type == "FLOAT8E4M3FN":
            float_types = ["FLOAT8E4M3FN", "FLOAT8E4M3FN", "FLOAT16", "FLOAT", "FLOAT8E4M3FN"]
        elif float_type == "FLOAT8E4M3FN2":
            float_types = ["FLOAT8E4M3FN", "FLOAT8E4M3FN", "BFLOAT16", "FLOAT", "FLOAT"]
        elif float_type == "FLOAT8E5M2":
            float_types = ["FLOAT8E5M2", "FLOAT8E4M3FN", "FLOAT16", "FLOAT", "FLOAT8E4M3FN"]
        elif float_type == "FLOAT16":
            float_types = ["FLOAT16", "FLOAT16", "FLOAT16", "FLOAT16", "FLOAT16"]
        elif float_type == "BFLOAT16":
            float_types = ["BFLOAT16", "BFLOAT16", "BFLOAT16", "FLOAT", "BFLOAT16"]
        elif float_type == "FLOAT":
            float_types = ["FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT"]
        else:
            raise AssertionError(f"Unexpected float_type={float_type!r}.")

        ref = self.InferenceSession(
            onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref.run(None, feeds)[0]
        with self.subTest(name="Gemm"):
            assert_allclose(expected, y, atol=atol, rtol=atol)
            self.assertEqual(expected.shape, y.shape)
            self.assertEqual(expected.dtype, y.dtype)

        onnx_model_f8 = self.get_model_gemm_float8(float_types, **kwargs)
        ref8 = self.InferenceSession(
            onnx_model_f8.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref8.run(None, feeds)[0]
        with self.subTest(name="GemmFloat8", float_types=float_types):
            assert_allclose(expected, y, atol=atol, rtol=atol)
            self.assertEqual(expected.shape, y.shape)
            self.assertEqual(expected.dtype, y.dtype)

    def test_model_gemm_float(self):
        self.common_test_model_gemm("FLOAT", compute_type="CUBLAS_COMPUTE_32F")

    def test_model_gemm_float16(self):
        self.common_test_model_gemm("FLOAT16", compute_type="CUBLAS_COMPUTE_16F")

    def test_model_gemm_float16_ct32(self):
        self.common_test_model_gemm("FLOAT16", compute_type="CUBLAS_COMPUTE_32F")

    def test_model_gemm_bfloat16_ct32(self):
        self.common_test_model_gemm("BFLOAT16", compute_type="CUBLAS_COMPUTE_32F", mul=2 ** (-10), atol=1e-2)

    def test_model_gemm_float_ct16(self):
        self.common_test_model_gemm("FLOAT", compute_type="CUBLAS_COMPUTE_32F_FAST_16F")

    def test_model_gemm_e4m3(self):
        self.common_test_model_gemm("FLOAT8E4M3FN", compute_type="CUBLAS_COMPUTE_32F", fastAccumulationMode=0)

    def test_model_gemm_e4m3_b(self):
        self.common_test_model_gemm("FLOAT8E4M3FN2", compute_type="CUBLAS_COMPUTE_32F")

    def test_model_gemm_e5m2(self):
        self.common_test_model_gemm("FLOAT8E5M2", compute_type="CUBLAS_COMPUTE_32F")

    def get_model_gemm_options(
        self, dtypes, computeType, fastAccumulationMode, smCount, alpha=1.0, beta=0.0, transA=1, transB=0
    ):
        a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        c = make_tensor_value_info("C", TensorProto.FLOAT, [None, None])
        d = make_tensor_value_info("D", TensorProto.FLOAT, [None, None])
        e = make_tensor_value_info("E", TensorProto.FLOAT, [None, None])
        y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])

        nodes = [
            make_node("Cast", ["A"], ["Af"], to=dtypes[0]),
            make_node("Cast", ["B"], ["Bf"], to=dtypes[1]),
            make_node("Cast", ["C"], ["Cf"], to=dtypes[2]),
            make_node("Cast", ["D"], ["Df"], to=dtypes[3]),
            make_node("Cast", ["E"], ["Ef"], to=dtypes[4]),
            make_node(
                "GemmFloat8",
                ["Af", "Bf", "Cf", "Df", "Ef"],
                ["Yf"],
                transA=transA,
                transB=transB,
                alpha=alpha,
                beta=beta,
                smCount=smCount,
                computeType=computeType,
                fastAccumulationMode=fastAccumulationMode,
                domain="com.microsoft",
                name="gemm8",
            ),
            make_node("Cast", ["Yf"], ["Y"], to=TensorProto.FLOAT),
        ]
        graph = make_graph(nodes, "gemm", [a, b, c, d, e], [y])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 19), make_opsetid("com.microsoft", 1)], ir_version=9
        )
        check_model(onnx_model)
        return onnx_model

    def gemm_float8_float32_combinations(self, typesab, typec, types):
        beta = [0.0, 1.0]
        computeType = [
            "CUBLAS_COMPUTE_32F",
            "CUBLAS_COMPUTE_16F",
            "CUBLAS_COMPUTE_32F_FAST_16F",
            "CUBLAS_COMPUTE_32F_FAST_16BF",
            "CUBLAS_COMPUTE_32F_FAST_TF32",
        ]
        fastAccumulationMode = [1, 0]
        smCount = [0]

        inputs = [np.arange(256 * 256).reshape(-1, 256).astype(np.float32) for t in range(5)]
        feeds = dict(zip("abcde".upper(), inputs))
        print()

        options = list(product(beta, computeType, fastAccumulationMode, smCount, typesab, typesab, typec, types, types))
        success = []
        for i, opts in enumerate(options):
            if i % 16 == 0:
                gc.collect()
            flag = "-".join(map(str, opts))
            b, ct, fa, sm, t1, t2, t3, t4, t5 = opts
            onx = self.get_model_gemm_options(
                [t1, t2, t3, t4, t5], beta=b, computeType=ct, fastAccumulationMode=fa, smCount=sm
            )
            s = onx.SerializeToString()
            try:
                sess = self.InferenceSession(s, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as e:
                print(f"{len(success)}: {i + 1} / {len(options)}-FAIL CUDA LOAD {flag}: {e}")
                continue
            try:
                sess.run(None, feeds)
            except Exception as e:
                if "CUBLAS_STATUS_NOT_SUPPORTED" in str(e):
                    e = "CUBLAS_STATUS_NOT_SUPPORTED"
                elif "CUBLAS_STATUS_INVALID_VALUE" in str(e):
                    e = "CUBLAS_STATUS_INVALID_VALUE"
                else:
                    e = str(e)
                print(f"{len(success)}: {i + 1} / {len(options)}-FAIL CUDA EXE {flag}: {e}")
                continue

            cl = time.perf_counter()
            for _ in range(5):
                sess.run(None, feeds)
            n = 15
            for _ in range(n):
                sess.run(None, feeds)
            dur = (time.perf_counter() - cl) / n
            print(f"{len(success)}: {i + 1} / {len(options)}-SUCCESS {flag} - {dur}")
            success.append((dur, flag))
        return success

    def _test_gemm_float8_float32_combinations(self):
        types = [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16]
        success = self.gemm_float8_float32_combinations(types, types, types)
        print("\n".join(map(str, sorted(success))))
        self.assertTrue(len(success) > 0)
        """
        (0.02733936100048595, '0.0-CUBLAS_COMPUTE_32F-1-0-1-1-1-1-1')
        (0.02900479100026132, '0.0-CUBLAS_COMPUTE_32F-1-0-1-1-1-16-1')
        (0.03042422700127645, '0.0-CUBLAS_COMPUTE_32F-1-0-10-10-1-1-1')
        (0.030477725998935057, '0.0-CUBLAS_COMPUTE_32F-1-0-10-10-1-16-1')
        (0.030877209999744082, '0.0-CUBLAS_COMPUTE_32F-1-0-10-10-10-16-10')
        (0.03113630000007106, '0.0-CUBLAS_COMPUTE_32F-1-0-10-10-10-1-10')
        (0.03120729700094671, '0.0-CUBLAS_COMPUTE_32F-1-0-10-10-10-10-10')
        (0.03180767400044715, '0.0-CUBLAS_COMPUTE_32F-1-0-1-1-1-10-1')
        (0.03332431499984523, '0.0-CUBLAS_COMPUTE_32F-1-0-16-16-1-16-1')
        (0.03334041500056628, '0.0-CUBLAS_COMPUTE_32F-1-0-16-16-1-10-1')
        (0.034372076001091045, '0.0-CUBLAS_COMPUTE_32F-1-0-16-16-16-16-16')
        (0.0345421689999057, '0.0-CUBLAS_COMPUTE_32F-1-0-16-16-16-10-16')
        (0.03485685700070462, '0.0-CUBLAS_COMPUTE_32F-1-0-16-16-16-1-16')
        (0.03675158400073997, '0.0-CUBLAS_COMPUTE_32F-1-0-16-16-1-1-1')
        (0.03979366699968523, '0.0-CUBLAS_COMPUTE_32F-1-0-10-10-1-10-1')
        (0.04064548199858109, '0.0-CUBLAS_COMPUTE_16F-1-0-10-10-10-10-10')
        (0.04151784600071551, '0.0-CUBLAS_COMPUTE_16F-1-0-10-10-10-16-10')
        (0.042687604000093415, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-10-10-10-16-10')
        (0.04279070000120555, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-1-1-1-1-1')
        (0.04390416100068251, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-1-1-1-10-1')
        (0.04432224599986512, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-1-1-1-16-1')
        (0.04557998499876703, '0.0-CUBLAS_COMPUTE_16F-1-0-10-10-10-1-10')
        (0.047199945998727344, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-10-10-1-10-1')
        (0.047339340000689845, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-10-10-1-16-1')
        (0.04825640799936082, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-10-10-10-1-10')
        (0.04932762699900195, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-1-1-1')
        (0.05028643699915847, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-10-10-1-1-1')
        (0.05040978599936352, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-1-16-1')
        (0.05041488499955449, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-1-10-1')
        (0.05083320599987928, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-1-1-1-1-1')
        (0.051650489000167, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-10-10-10-10-10')
        (0.05259690300044895, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-16-10-16')
        (0.05311992099996132, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-1-1-1-16-1')
        (0.054345859000022756, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-1-1-1-10-1')
        (0.05476542100041115, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-16-16-16')
        (0.05580888300028164, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-10-10-1-1-1')
        (0.05600327399952221, '0.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-16-1-16')
        (0.05650505800076644, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-10-10-1-16-1')
        (0.056914127999334596, '1.0-CUBLAS_COMPUTE_32F-1-0-1-1-1-1-1')
        (0.05758341799992195, '1.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-16-16-1-1-1')
        (0.058957769000699045, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-10-10-10-1-10')
        (0.05918206099886447, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-10-10-10-10-10')
        (0.05919199499840033, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-1-1-1-10-1')
        (0.06052071299927775, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-10-10-1-10-1')
        (0.060623109000516706, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-16-16-1-10-1')
        (0.06094819700047083, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-16-16-16-1-16')
        (0.06199504299911496, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-1-1-1-16-1')
        (0.06232364799870993, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-16-16-16-10-16')
        (0.06273243299983733, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-16-16-1-1-1')
        (0.06348340499971528, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-16-16-16-16-16')
        (0.06465779700010899, '1.0-CUBLAS_COMPUTE_32F_FAST_16F-1-0-1-1-1-16-1')
        (0.06514164100008202, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-10-10-1-1-1')
        (0.0651611529992806, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-1-1-1-1-1')
        (0.06536243700065825, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-10-10-10-16-10')
        (0.06565842700001667, '0.0-CUBLAS_COMPUTE_32F_FAST_16BF-1-0-16-16-1-16-1')
        (0.06638419599948975, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-10-10-1-16-1')
        (0.06767204900097568, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-10-10-10-1-10')
        (0.06882783800028847, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-16-16-1-1-1')
        (0.06927989100040577, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-10-10-10-16-10')
        (0.06965061000119022, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-16-16-1-16-1')
        (0.0696632769995631, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-10-10-10-10-10')
        (0.07042934999844874, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-10-10-1-10-1')
        (0.07235372100149107, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-16-16-1-10-1')
        (0.07291310300024634, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-16-16-16-10-16')
        (0.07423536000169406, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-16-16-16-16-16')
        (0.07543202099986956, '0.0-CUBLAS_COMPUTE_32F_FAST_TF32-1-0-16-16-16-1-16')
        """

    def _test_gemm_float8_float32_combinations(self):
        short = [TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2]
        types = [
            TensorProto.FLOAT,
            # TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.FLOAT8E4M3FN,
            # TensorProto.FLOAT8E5M2,
        ]
        self.gemm_float8_float32_combinations(short, types, types)


if __name__ == "__main__":
    unittest.main(verbosity=2)