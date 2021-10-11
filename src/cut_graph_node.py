import argparse
import onnx
import numpy
import os
import custom_node
import get_val_info

from onnx import helper

numpy2elem = {
    numpy.dtype('float32') : 1,
    numpy.dtype('int32') : 6,
    numpy.dtype('int64') : 7,
    bool : 9
}

cmdline = argparse.ArgumentParser(description='Load a onnx model, cut its submodel thru node list.')
cmdline.add_argument('--inputmodel', '-im', help='path to input tensor model checkpoint to load from.', required=True)
cmdline.add_argument('--cutnodelist', '-cn', help='node list to cut.', required=True)
cmdline.add_argument('--cutmodel', '-cm', help='path where the exported model will be saved at.', required=True)

class CutGraphNode(object):
    def __init__(self, onnx_model_path, cut_node_list, cutmodel_path):
        self.raw_onnx_model = onnx.load(onnx_model_path)
        self.onnx_model_path = onnx_model_path
        self.onnx_graph = self.raw_onnx_model.graph
        self.node_list = cut_node_list
        self.cutmodel_path = cutmodel_path
        self.sub_graph_nodes = {}
        self.sub_graph_edges = {}
        self.sub_graph_inputs = []
        self.sub_graph_outputs = []
        self.sub_graph_initializers = []
        self.sub_graph_missing_val_info_names = []

    def _collect_subgraph_info(self):
        for node in self.onnx_graph.node:
            if node.name in self.node_list:
                if node.name in self.sub_graph_nodes.keys():
                    raise Exception("Duplicate keys in subgraph map")                
                self.sub_graph_nodes[node.name] = custom_node.CustomNode(node)

                for input_name in node.input:
                    if input_name not in self.sub_graph_edges.keys():
                        self.sub_graph_edges[input_name] = custom_node.CustomEdge()
                    self.sub_graph_edges[input_name].out_degree += 1
                    self.sub_graph_edges[input_name].edge_name = input_name

                for output_name in node.output:
                    if output_name not in self.sub_graph_edges.keys():
                        self.sub_graph_edges[output_name] = custom_node.CustomEdge()
                    self.sub_graph_edges[output_name].in_degree += 1
                    self.sub_graph_edges[output_name].edge_name = output_name

        # Get sub_graph input & output thru in/out-degree
        for edge_name in self.sub_graph_edges.keys():
            if self.sub_graph_edges[edge_name].in_degree == 0:
                self.sub_graph_inputs.append(edge_name)
            if self.sub_graph_edges[edge_name].out_degree == 0:
                self.sub_graph_outputs.append(edge_name)

        # Complete the tensor proto thru input/output/initializer.
        for val_info in self.onnx_graph.input:
            if val_info.name in self.sub_graph_edges.keys():
                self.sub_graph_edges[val_info.name].value_info = val_info

        for val_info in self.onnx_graph.output:
            if val_info.name in self.sub_graph_edges.keys():
                self.sub_graph_edges[val_info.name].value_info = val_info

        for val_info in self.onnx_graph.initializer:
            if val_info.name in self.sub_graph_edges.keys():
                self.sub_graph_edges[val_info.name].value_info = val_info
                self.sub_graph_initializers.append(val_info.name)
                if val_info.name in self.sub_graph_inputs:
                    self.sub_graph_inputs.remove(val_info.name)
        
        for val_info in self.onnx_graph.value_info:
            if val_info.name in self.sub_graph_edges.keys():
                self.sub_graph_edges[val_info.name].value_info = val_info

        # Invoke shape tool to get input/output tensor proto.
        for edge_name in self.sub_graph_inputs + self.sub_graph_outputs:
            if self.sub_graph_edges[edge_name].value_info is None:
                self.sub_graph_missing_val_info_names.append(edge_name)
        
        if len(self.sub_graph_missing_val_info_names) > 0:
            input_model = self.onnx_model_path
            input_sys_val = "None=1"
            input_names = self.sub_graph_missing_val_info_names
            output_model = "..\\models\\marked_model.onnx"
            get_miss_val_info = get_val_info.TensorValInfo(input_sys_val, input_names, output_model)
            missing_val_info = get_miss_val_info.get_value_info(input_model)
            self._fill_in_missing_val_info(missing_val_info)
    
    def _fill_in_missing_val_info(self, missing_val_info):
        for name, data in missing_val_info:
            if name in self.sub_graph_edges.keys():
                if data.dtype in numpy2elem.keys():
                    elem_type = numpy2elem[data.dtype]
                else:
                    raise Exception("{} Unsupported data type".format(data.dtype))

                self.sub_graph_edges[name].value_info = helper.make_tensor_value_info(name, elem_type, data.shape)

    def create_subgraph(self):
        self._collect_subgraph_info()
        # Get node list
        subgraph_nodes = [node.node for node in self.sub_graph_nodes.values()]
        # Get input value_info
        subgraph_inputs = [self.sub_graph_edges[input].value_info for input in self.sub_graph_inputs]
        # Get output value_info
        subgraph_outputs = [self.sub_graph_edges[output].value_info for output in self.sub_graph_outputs]
        # Get output value_info
        subgraph_initializers = [self.sub_graph_edges[ini].value_info for ini in self.sub_graph_initializers]

        cut_subgraph = helper.make_graph(subgraph_nodes, "cut_submodel", subgraph_inputs, subgraph_outputs, subgraph_initializers)
        cut_submodel = helper.make_model(cut_subgraph)

        self._check_valid_submodel(cut_submodel)

    def _check_valid_submodel(self, cut_submodel):
        onnx.checker.check_model(cut_submodel)
        self._save_cutmodel(cut_submodel)

    def _save_cutmodel(self, cut_submodel):
        onnx.save(cut_submodel, self.cutmodel_path)
        print("Cut model saved")


if __name__ == "__main__":
    onnx_model_path = "..\\models\\yolov2.onnx"
    cut_node_list = ["pooling", "convolution1"]
    cutmodel_path = "..\\models\\yolov2_cut.onnx"

    args =  cmdline.parse_args()
    cutgraph = CutGraphNode(args.inputmodel, args.cutnodelist, args.cutmodel)
    cutgraph.create_subgraph()
