# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import unittest
import onnx.backend.test

import onnx_tensorrt.backend as trt

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(trt, __name__)

# Exclude unsupported nodes.
backend_test.exclude(r'.*test_adagrad.*')
backend_test.exclude(r'.*test_adam.*')
backend_test.exclude(r'.*test_bernoulli.*')
backend_test.exclude(r'.*test_bitshit.*')
backend_test.exclude(r'.*test_bitwise.*')
backend_test.exclude(r'.*test_blackmanwindow.*')
backend_test.exclude(r'.*test_center_crop_pad.*')
backend_test.exclude(r'.*test_col2im.*')
backend_test.exclude(r'.*test_compress.*')
backend_test.exclude(r'.*test_convinteger.*')
backend_test.exclude(r'.*test_det.*')
backend_test.exclude(r'.*test_dft.*')
backend_test.exclude(r'.*test_dynamicquantizelinear.*')
backend_test.exclude(r'.*test_hammingwindow.*')
backend_test.exclude(r'.*test_hannwindow.*')
backend_test.exclude(r'.*test_matmulinteger.*')
backend_test.exclude(r'.*test_maxunpool.*')
backend_test.exclude(r'.*test_melweightmatrix.*')
backend_test.exclude(r'.*test_momentum.*')
backend_test.exclude(r'.*test_nesterov.*')
backend_test.exclude(r'.*test_optional.*')
backend_test.exclude(r'.*test_qlinearconv.*')
backend_test.exclude(r'.*test_qlinearmatmul.*')
backend_test.exclude(r'.*test_sce.*')
backend_test.exclude(r'.*test_sequence.*')
backend_test.exclude(r'.*test_softplus.*')
backend_test.exclude(r'.*test_stft.*')
backend_test.exclude(r'.*test_strnormalizer.*')
backend_test.exclude(r'.*test_tfidfvectorizer.*')
backend_test.exclude(r'.*test_training.*')
backend_test.exclude(r'.*test_unique.*')

# Exclude unsupported models:

backend_test.exclude(r'.*test_gradient.*')
backend_test.exclude(r'.*test_strnorm.*')

globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
