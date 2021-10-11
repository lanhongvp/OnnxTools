
class CustomNode(object):
    def __init__(self, onnx_node):
        self.node = onnx_node
        self.extra_input = []

class CustomEdge(object):
    def __init__(self) -> None:
        super().__init__()
        self.value_info = None
        self.in_degree = 0
        self.out_degree = 0
        self.edge_name = None
