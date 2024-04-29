# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
from .tensorrt_engine import Engine
import tensorrt as trt
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
import onnx
from onnx import helper as onnx_helper
from onnx import numpy_helper
import numpy as np
import six

# POLYGRAPHY:

from polygraphy.logger import G_LOGGER

from polygraphy.backend.trt import EngineBytesFromNetwork, EngineFromBytes, NetworkFromOnnxBytes, TrtRunner, Profile, CreateConfig
from polygraphy.comparator import Comparator, DataLoader

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p
def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        if isinstance(error_string, bytes):
            error_string = error_string.decode("utf-8")
        raise RuntimeError("cudaSetDevice: " + error_string)

def count_trailing_ones(vals):
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count

class TensorRTBackendRep(BackendRep):
    def __init__(self, model, device, **kwargs):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device) # TODO: Update.
        # TODO: Update logging verbosity.
        self._logger = G_LOGGER
        
        # self.inputs_shape_tuple = [(input.name, False) for input in model.graph.input]
        
        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model
        
        # print(model) # Model str representation
        # print(model_str) # model serialized to bytes.
        
        # Polygraphy helper functions to parse and build TensorRT engines from ONNX.    
        self.poly_network = NetworkFromOnnxBytes(model_str)
        
        self.builder, self.network, self.parser = self.poly_network()
        
        self.inputs = []
        
        for i in range(self.network.num_inputs):
            inp = self.network.get_input(i)
            is_dynamic = -1 in inp.shape
            self.inputs.append((inp.name, is_dynamic))

    def _set_device(self, device):
        self.device = device
        assert(device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """

        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
            
        config = CreateConfig()
            
        for runtime_input, input_metadata in zip(inputs, self.inputs):
            if input_metadata[1]: # Dynamic = True
                shape = runtime_input.shape
                config.profiles[-1].add(input_metadata[0], min=shape, opt=shape, max=shape)

        build_engine = EngineBytesFromNetwork([self.builder, self.network], config=config)
        deserialize_engine = EngineFromBytes(build_engine)
        
        # Use TensorRT polygraphy runner.
        runners = [
            TrtRunner(deserialize_engine),
        ]

        # Runner Execution. results is a list of (RunnerName, DataResults) tuples
        results = Comparator.run(runners)
        
        assert len(results) == 1
        
        dict_results = results[0][1][0]
        
        print(dict_results)
        
        return dict_results
        #build_engine = EngineBytesFromNetwork(self.poly_network)  
        
        #outputs = self.engine.run(inputs)
        #output_names = [output.name for output in self.engine.outputs]

        # for i, (name, array) in enumerate(zip(output_names, outputs)):
        #     output_shape = self._output_shapes[name]
        #     # HACK WAR for unknown output shape in run_node
        #     if output_shape == (-99,):
        #         # WAR for TRT requiring at least 2 dims (NC)
        #         min_dims = 2
        #         if _tensorrt_version()[0] < 4:
        #             # WAR for TRT only supporting 4D (NCHW) tensors
        #             min_dims = 4
        #         if array.ndim == min_dims:
        #             npadding_dims = count_trailing_ones(array.shape)
        #             if npadding_dims > 0:
        #                 outputs[i] = array.reshape(
        #                     array.shape[:-npadding_dims])
        #     else:
        #         # HACK WAR replace fixed batch dim with variable
        #         if self._output_dtype[name] == onnx.TensorProto.INT64 and array.dtype == np.int32:
        #             casted_output = np.array(outputs[i], dtype=np.int64)
        #             if np.equal(outputs[i], casted_output).all():
        #                 outputs[i] = np.array(outputs[i], dtype=np.int64)
        #         if self._output_dtype[name] == onnx.TensorProto.DOUBLE and array.dtype == np.float32:
        #             casted_output = np.array(outputs[i], dtype=np.double)
        #             if np.equal(outputs[i], casted_output).all():
        #                 outputs[i] = np.array(outputs[i], dtype=np.double)
        
        return None

        # outputs_tuple = namedtupledict('Outputs', output_names)(*outputs)
        # return namedtupledict('Outputs', output_names)(*outputs)

class TensorRTBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CUDA:0', **kwargs):
        """Build an engine from the given model.
        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        """
        print("PREPARE!")
        super(TensorRTBackend, cls).prepare(model, device, **kwargs)
        return TensorRTBackendRep(model, device, **kwargs)
    @classmethod
    def run_model(cls, model, inputs, device='CUDA:0', **kwargs):
        """Build and run an engine from the given model.
        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        print("RUN MODEL!")
        return cls.prepare(model, device, **kwargs).run(inputs)
    @classmethod
    def run_node(cls, node, inputs, device='CUDA:0'):
        """Build and run an engine from the given node.
        node -- An ONNX node as a deserialized protobuf.
        Note: This function is intended for testing purposes only;
              use prepare() or run_model() for other purposes.
        """
        print("RUN NODE!")
        # super(TensorRTBackend, cls).run_node(node, inputs, device)
        # # HACK TODO: This is somewhat dodgy. We first try with weights for all
        # #            inputs but the first, then we try again with no weights if
        # #            the first try fails.
        # model = make_node_test_model(node, inputs, use_weights=True)
        # try: results = TensorRTBackend.prepare(model, device).run(inputs[:1])
        # except RuntimeError:
        #     model = make_node_test_model(node, inputs, use_weights=False)
        #     results = TensorRTBackend.prepare(model, device).run(inputs)
        # return results
    @classmethod
    def supports_device(cls, device_str):
        device = Device(device_str)
        return device.type == DeviceType.CUDA

prepare         = TensorRTBackend.prepare
run_node        = TensorRTBackend.run_node
run_model       = TensorRTBackend.run_model
supports_device = TensorRTBackend.supports_device
