import input_collection
import mark_output
import ort_inference
import argparse
import onnx
import json

cmdline = argparse.ArgumentParser(description='Load a onnx model, inference its unkonwn shapes.')
cmdline.add_argument('--inputmodel', '-im', help='path to input tensor model checkpoint to load from.', required=True)
cmdline.add_argument('--outputmodel', '-om', help='path where the exported model will be saved at.', required=True)
cmdline.add_argument('--tensornames', '-tn', nargs='+', help='a list of tensor names which you want to know their shapes by actual data.', required=True)
cmdline.add_argument('--inputsymvals', '-iv', type=str, help='a dictionary of input tensors which key is symbolic shape name value is actual shape data.', required=False)
cmdline.add_argument('--randintmax', '-imax', help='random int max for random data', required=False)


class TensorValInfo(object):
    def __init__(self, input_model, input_symvals, input_names, output_model) -> None:
        super().__init__()
        self.input_model = input_model
        self.input_symvals = input_symvals
        self.input_names = input_names
        self.output_model = output_model

    def get_value_info(self):
        onnx_model = onnx.load(self.inputmodel)
        input_sym_values = self.input_symvals
        tensor_names = self.input_names

        model_input_collection = input_collection.InputCollection(onnx_model, input_sym_values)
        input_feed = model_input_collection.gen_random_data()

        mark_model_output = mark_output.MarkOutput(onnx_model, tensor_names)
        marked_onnx_model = mark_model_output.mark_output()

        model_ort_inference = ort_inference.OrtInference(marked_onnx_model, input_feed) 
        output_tensors = model_ort_inference.ort_inference(self.output_model, tensor_names)

        return output_tensors

if __name__ == "__main__":
    args = cmdline.parse_args()

    onnx_model = onnx.load(args.inputmodel)
    input_sym_values = args.inputsymvals
    tensor_names = args.tensornames

    model_input_collection = input_collection.InputCollection(onnx_model, input_sym_values)
    input_feed = model_input_collection.gen_random_data()

    mark_model_output = mark_output.MarkOutput(onnx_model, tensor_names)
    marked_onnx_model = mark_model_output.mark_output()

    model_ort_inference = ort_inference.OrtInference(marked_onnx_model, input_feed) 
    model_inference = model_ort_inference.ort_inference(args.outputmodel, tensor_names)