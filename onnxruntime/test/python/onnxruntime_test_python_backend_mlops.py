# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import sys
import numpy as np
from numpy.testing import assert_allclose
import onnxruntime as onnxrt
from onnxruntime import datasets
import onnxruntime.backend as backend
from onnxruntime.backend.backend import OnnxRuntimeBackend as ort_backend
from onnx import load
from helper import get_name


def check_list_of_map_to_float(testcase, expected_rows, actual_rows):
    """Validate two list<map<key, float>> instances match closely enough."""

    num_rows = len(expected_rows)
    sorted_keys = sorted(expected_rows[0].keys())
    testcase.assertEqual(num_rows, len(actual_rows))
    testcase.assertEqual(sorted_keys, sorted(actual_rows[0].keys()))

    for i in range(num_rows):
        # use np.testing.assert_allclose so we can specify the tolerance
        np.testing.assert_allclose([expected_rows[i][key] for key in sorted_keys],
                                   [actual_rows[i][key] for key in sorted_keys],
                                   rtol=1e-05,
                                   atol=1e-07)


class TestBackend(unittest.TestCase):

    def testRunModelNonTensor(self):
        name = get_name("pipeline_vectorize.onnx")
        rep = backend.prepare(name)
        x = {0: 25.0, 1: 5.13, 2: 0.0, 3: 0.453, 4: 5.966}
        res = rep.run(x)
        output_expected = np.array([[49.752754]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelProto(self):
        name = datasets.get_example("logreg_iris.onnx")
        model = load(name)

        rep = backend.prepare(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                      [7.0, 8.0]], dtype=np.float32).reshape((-1, 4))
        res = rep.run(x)
        output_expected = np.array([2, 2], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
        output_expected = [
            {0: 0.1022685095667839, 1: 0.01814863830804825,
             2: 0.8795828223228455},
            {0: 3.6572727810622874e-13, 1: 2.1704063546401642e-10, 2: 1.0}]
        check_list_of_map_to_float(self, output_expected, res[1])

    def testRunModelProtoApi(self):
        name = datasets.get_example("logreg_iris.onnx")
        model = load(name)

        inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                           [7.0, 8.0]], dtype=np.float32).reshape((-1, 4))
        outputs = ort_backend.run_model(model, inputs)

        output_expected = np.array([2, 2], dtype=np.float32)
        np.testing.assert_allclose(
            output_expected, outputs[0], rtol=1e-05, atol=1e-08)
        output_expected = [
            {0: 0.1022685095667839, 1: 0.01814863830804825,
             2: 0.8795828223228455},
            {0: 3.6572727810622874e-13, 1: 2.1704063546401642e-10, 2: 1.0}]
        check_list_of_map_to_float(self, output_expected, outputs[1])

if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
