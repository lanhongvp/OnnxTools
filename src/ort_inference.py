import onnxruntime
import onnx

class OrtInference(object):
    def __init__(self, onnx_model, input_feed) -> None:
        super().__init__()
        self.onnx_model = onnx_model
        self.input_feed = input_feed
        self.output_tensors = []
        
    def _save_output_model(self, save_path):
        # onnx.checker.check_model(self.onnx_model)
        onnx.save(self.onnx_model, save_path)
        
    def _print_marked_tensor_shapes(self, tensor_names, ort_outputs):
        for tensor_name in tensor_names:
            idx = tensor_names.index(tensor_name)
            self.output_tensors.append((tensor_name, ort_outputs[idx]))
            print("----- Marked tensor {}, Shape {} ------".format(tensor_name, ort_outputs[idx].shape))

    def ort_inference(self, save_path, tensor_names):
        self._save_output_model(save_path)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 4

        ort_session = onnxruntime.InferenceSession(save_path, sess_options)
        ort_outputs = ort_session.run(tensor_names, self.input_feed)

        self._print_marked_tensor_shapes(tensor_names, ort_outputs)
        return self.output_tensors
    