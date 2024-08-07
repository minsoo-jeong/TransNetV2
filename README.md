# TransNet V2: Shot Boundary Detection Neural Network



> **&#9989; Update** 
> 
> This repository is a fork of the [TransNetV2 project](https://github.com/ORIGINAL-OWNER/TransNetV2), which contain code for [TransNet V2: An effective deep network architecture for fast shot transition detection](https://arxiv.org/abs/2008.04838).

## Modifications
This fork includes the following updates:
- Refactored the original code to PyTorch version.
- Added functionality to export the model to ONNX format for easier deployment.


### Convert Pytorch2ONNX: 
- `./inference-pytorch/convert_onnx.py` [pth](https://github.com/minsoo-jeong/TransNetV2/releases/download/onnx/transnetv2-pytorch-weights.pth) [onnx](https://github.com/minsoo-jeong/TransNetV2/releases/download/onnx/transnetv2.onnx)

- `./inference-pytorch/run_onnx.py` 
