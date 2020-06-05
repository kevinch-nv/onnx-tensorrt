#include "RNNHelpers.hpp"
#include "LoopHelpers.hpp"
#include "onnx2trt_utils.hpp"
#include <array>

namespace onnx2trt
{

nvinfer1::ITensor* addRNNInput(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, std::vector<TensorOrWeights>& inputs, const std::string& direction)
{
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse
    // iterator must be concatenated.
    // Input dimensions: [1, B, E]
    nvinfer1::ITensor* iterationInput{nullptr};
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    if (direction == "forward")
    {
        iterationInput = unsqueezeTensor(ctx, node, *loop->addIterator(*input)->getOutput(0), std::vector<int>{0});
    }
    else if (direction == "reverse")
    {
        nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
        reverseIterator->setReverse(true);
        iterationInput = unsqueezeTensor(ctx, node, *reverseIterator->getOutput(0), std::vector<int>{0});
    }
    else if (direction == "bidirectional")
    {
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);
        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        std::array<nvinfer1::ITensor*, 2> tensors{{unsqueezeTensor(ctx, node, *forward->getOutput(0), std::vector<int>{0}),
            unsqueezeTensor(ctx, node, *reverse->getOutput(0), std::vector<int>{0})}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(tensors.data(), 2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    else
    {
        return nullptr;
    }
    LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());
    return iterationInput;
}

} // namespace onnx2trt
