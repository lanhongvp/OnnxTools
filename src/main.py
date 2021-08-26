import input_collection
import mark_output
import ort_inference

import onnx

onnx_model_path = "D:\\src\\onnxruntime-extensions\\tutorials\\Inner_GPT2_OUTPUT_3.onnx"

if __name__ == "__main__":
    onnx_model = onnx.load(onnx_model_path)
    input_sym_values = {}
    tensor_names = []

    model_input_collection = input_collection(onnx_model, input_sym_values)
    input_feed = model_input_collection.get_random_data()

    mark_needed_output = mark_output(onnx_model, tensor_names)

    model_ort_inference = ort_inference(onnx_model, onnx_model, input_feed) 
