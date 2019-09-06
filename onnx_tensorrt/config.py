import tensorrt as trt

class Config():
    def __init__(self):
        # If TensorRT major is >= 5, then we use new Python bindings
        _tensorrt_version = [int(n) for n in trt.__version__.split('.')]
        self.USE_PYBIND = _tensorrt_version[0] >= 5
    
    def USE_PYBIND(self):
        return self.USE_PYBIND