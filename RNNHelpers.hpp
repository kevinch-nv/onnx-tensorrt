#pragma once

#include <NvInfer.h>
#include <string>
#include <vector>

#include "TensorOrWeights.hpp"
#include "ImporterContext.hpp"

namespace onnx2trt
{

nvinfer1::ITensor* addRNNInput(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, std::vector<TensorOrWeights>& inputs, const std::string& direction);

} // namespace onnx2trt
