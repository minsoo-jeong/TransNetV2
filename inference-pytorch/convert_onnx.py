import torch
import onnx
import importlib
import onnxruntime

from transnetv2_pytorch import TransNetV2
import numpy as np
import decord
import cv2


# transnetv2_pytorch = importlib.import_module('inference-pytorch.transnetv2_pytorch')


@torch.no_grad()
def export(model, input, filename="transnetv2.onnx"):
    model.eval()

    torch.onnx.export(model,
                      input,
                      f=filename,
                      export_params=True,
                      # !!! the ONNX version to export the model -> check dependency with tritonserver
                      opset_version=17,
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],
                      output_names=['output0', 'output1'],
                      dynamic_axes={'input': {0: 'batch_size', 1: 'frame_length'},  # variable length axes for the input
                                    'output0': {0: 'batch_size', 1: 'frame_length'},
                                    'output1': {0: 'batch_size', 1: 'frame_length'},

                                    },
                      verbose=True)


if __name__ == '__main__':
    filename = "transnetv2.onnx"

    model = TransNetV2()
    model.load_state_dict(torch.load('./transnetv2-pytorch-weights.pth'))
    model.eval()

    dummy_input = torch.rand((1, 50, 27, 48, 3)) * 255
    dummy_input = dummy_input.type(torch.uint8)

    export(model, dummy_input, filename)

    # check diff
    sample = 'sample.mp4'
    video = decord.VideoReader(sample)

    frames = np.concatenate([video[:50].asnumpy(), video[-50:].asnumpy()], axis=0)

    dummy_input = torch.Tensor([cv2.resize(f, (48, 27)) for f in frames]).unsqueeze(0)
    dummy_input = dummy_input.type(torch.uint8)

    # Load the ONNX model
    with torch.no_grad():
        torch_pred = model(dummy_input)
        print(torch_pred[0].shape, torch_pred[1].shape)

    ort_session = onnxruntime.InferenceSession(filename)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_pred = ort_session.run(None, ort_inputs)

    print(ort_pred[0].shape, ort_pred[1].shape)

    for p1, p2 in zip(torch_pred, ort_pred):
        p1 = p1.numpy()
        print(p1.shape, p2.shape)

        diff = np.abs(p1 - p2)

        print("Max difference:", diff.max(), diff.mean())
