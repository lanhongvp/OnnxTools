import onnx
from onnx import helper

from copy import deepcopy

class MarkOutput(object):
    def __init__(self, onnx_model, marked_tensor_names) -> None:
        super().__init__()
        self.ori_model = onnx_model
        self.model = deepcopy(onnx_model)
        self.marked_tensor_names = marked_tensor_names

    def __check_marked_tensor_names_existed(self):
        # check all node's inputs and outputs
        all_tensor_names = set()

        for node in self.model.graph.node:
            for input in node.input:
                all_tensor_names.add(input)
            for output in node.output:
                all_tensor_names.add(output)

        for marked_tensor_name in self.marked_tensor_names:
            if marked_tensor_name not in all_tensor_names:
                raise Exception("{} not exist in onnx model".format(marked_tensor_name))

    def mark_output(self):
        self.__check_marked_tensor_names_existed()
        for marked_tensor_name in self.marked_tensor_names:
            if marked_tensor_name in [x.name for x in self.model.graph.output]:
                continue

            # if marked_tensor_name is graph.input
            if marked_tensor_name in [x.name for x in self.model.graph.input]:
                raise Exception("{} is onnx model input".format(marked_tensor_name))

            # if marked_tensor_name is graph initializer
            if marked_tensor_name in [x.name for x in self.model.graph.initializer]:
                raise Exception("{} is onnx model initializer".format(marked_tensor_name))

            for val_info in self.model.graph.value_info:
                if marked_tensor_name == val_info.name:
                    self.model.graph.output.append(val_info)
                    self.model.graph.value_info.remove(val_info)
                    break
            
            marked_tensor_val_info = helper.make_tensor_value_info(marked_tensor_name, 0, None)
            marked_tensor_val_info.ClearField("type")
            # marked_tensor_val_info.type.tensor_type.ClearField("elem_type")
                
            self.model.graph.output.append(marked_tensor_val_info)
        
        return self.model








