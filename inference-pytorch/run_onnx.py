import numpy as np
from PIL import Image

from collections import deque
import decord
import onnxruntime as ort


class SBDClient(object):
    seq_length = 100
    buf_length = 25
    stride = seq_length // 2
    input_name = 'input'
    output_name = 'output0'

    def __init__(self, onnx_file='transnetv2.onnx'):

        so = ort.SessionOptions()
        providers = [('CUDAExecutionProvider', {
            'device_id': 1,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })]

        self.session = ort.InferenceSession(onnx_file,
                                            sess_options=so,
                                            providers=providers)

    def run(self, path):
        video = decord.VideoReader(path, width=48 * 2, height=27 * 2)

        preds = []
        for sequence in self.sequence_generator(video):
            preds.append(self.predict(sequence))

        preds = np.concatenate(preds)[:len(video)]
        scenes = self.predictions_to_scenes(preds).tolist()
        return scenes

    def predict(self, sequence):
        # sequence: [1,100,27,48,3]

        pred = self.session.run([self.output_name], {self.input_name: sequence})[0]
        pred = pred[0, self.buf_length:-self.buf_length, 0]

        return pred

    def predictions_to_scenes(self, predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def sequence_generator(self, video: decord.VideoReader):

        q = deque(maxlen=self.seq_length)

        for idx, frame in enumerate(video):
            frame = self.read_frame(frame)
            q.append(frame)
            if idx == 0:
                q.extend([frame] * self.buf_length)

            if idx == len(video) - 1:
                remain = q.maxlen - len(q)
                q.extend([frame] * remain)
                if remain < self.buf_length:
                    yield np.array(q)[np.newaxis]
                    q = deque(list(q)[self.stride:], maxlen=self.seq_length)
                    q.extend([frame] * (q.maxlen - len(q)))

            if len(q) == q.maxlen:
                yield np.array(q)[np.newaxis]
                q = deque(list(q)[self.stride:], maxlen=self.seq_length)

    def read_frame(self, frame):
        frame = Image.fromarray(frame.asnumpy())
        if frame.mode != 'RGB':
            raise ValueError('frame mode is not RGB')

        frame = np.array(frame.resize((48, 27)))
        return frame


if __name__ == '__main__':
    client = SBDClient()
    path = 'sample.mp4'
    shot = client.run(path)
    print(shot)
