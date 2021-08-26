import onnxruntime


class OrtInference(object):
    def __init__(self, onnx_model, input_feed) -> None:
        super().__init__()
        self.onnx_model = onnx_model
        self.input_feed = input_feed
        
    def ort_inference(self, save_path = None):
        pass
    # if save_path in none: No save
    # if save_path not none: save