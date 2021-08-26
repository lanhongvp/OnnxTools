import numpy
import onnx

elem_type_to_numpy_type = {
    1: numpy.float32,
    6: numpy.int32,
    7: numpy.int64,
    9: bool
}

class InputCollection(object):
    def __init__(self, onnx_model, input_sym_values = None) -> None:
        super().__init__()
        self.onnx_model = onnx_model
        self.input_sym_values = self._convert2dict_(input_sym_values) if input_sym_values is not None else {}

    def _convert2dict_(self, input_sym_values):
        input_sym_val_dict = {}
        input_sym_items = input_sym_values.strip().split(",")

        for input_sym_item in input_sym_items:
            input_sym_key = input_sym_item.split('=')[0]
            input_sym_val = input_sym_item.split('=')[-1]

            if input_sym_key in input_sym_val_dict.keys():
                raise Exception("Key {} already existed, check your input".format(input_sym_key))
    
            input_sym_val_dict[input_sym_key] = int(input_sym_val)
        
        if len(input_sym_val_dict) != len(input_sym_items):
            raise Exception("Different count between dict len {} with input symbol len {}, please check your input".format(len(input_sym_val_dict), len(input_sym_items)))
        
        print("Parse done", input_sym_val_dict)
        return input_sym_val_dict
    
    def read_data_from_disk(self, input_file_list):
        pass

    def gen_random_data(self, randint_max=100):
        model_inputs = self.onnx_model.graph.input
        # get all initializer names
        model_initializer_names = [x.name for x in self.onnx_model.graph.initializer]
        input_feed = {}

        for model_input in model_inputs:
            input_tensor_shape = []
            if model_input.name in model_initializer_names:
                continue
            # get tensor elem type
            input_tensor_type = model_input.type.tensor_type.elem_type
            if input_tensor_type not in elem_type_to_numpy_type.keys():
                raise Exception("{} Not supported elem type".format(input_tensor_type))

            input_tensor_dims = model_input.type.tensor_type.shape.dim
            for input_tensor_dim in input_tensor_dims:
                if input_tensor_dim.HasField("dim_value"):
                    input_tensor_shape.append(input_tensor_dim.dim_value)
                elif input_tensor_dim.HasField("dim_param"):
                    if input_tensor_dim.dim_param in self.input_sym_values.keys():
                        input_tensor_shape.append(self.input_sym_values[input_tensor_dim.dim_param])
                    else:
                        raise Exception("{} Not found, please check input symbol values".format(input_tensor_dim.dim_param))
                else:
                    raise Exception("Not found any dim info in input {}".format(model_input.name))

                # gen random data
                input_tensor_numpy_type = elem_type_to_numpy_type[input_tensor_type]
                if input_tensor_numpy_type in [numpy.float32, numpy.float64]:
                    model_input_random_data = numpy.random.random(input_tensor_shape).astype(input_tensor_numpy_type)

                if input_tensor_numpy_type in [numpy.int32, numpy.int64]:
                    model_input_random_data = numpy.random.randint(0, randint_max, input_tensor_shape).astype(input_tensor_numpy_type)
                
                if input_tensor_numpy_type in [bool]:
                    model_input_random_data = numpy.random.random(input_tensor_shape) > 0.5
                
                input_feed[model_input.name] = model_input_random_data
        
        return input_feed

if __name__ == "__main__":
    onnx_model_path = "..\\models\\DeepThinkV5.onnx"
    onnx_model = onnx.load(onnx_model_path)

    input_collection = InputCollection(onnx_model)

    input_feed = input_collection.gen_random_data()
    print(input_feed)
        
