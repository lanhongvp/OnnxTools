# Onnx Tools Collection

## Inference ONNX model's tensor shapes on demand
### Usage
Take Yolov2 as an example, we can use this command to get any tensor's shape as you want.

- Singe tensor support
    - `python .\main.py -im "..\\models\\yolov2.onnx" -om "..\\models\\yolov2_marked.onnx" -tn convolution2d_1_output -iv "None=1"`

- Multiple tensors support
    - `python .\main.py -im "..\\models\\yolov2.onnx" -om "..\\models\\yolov2_marked.onnx" -tn convolution2d_1_output convolution2d_2_output -iv "None=1"`

### Todo
- [ ] Loop body shape inference support
- [ ] Support read data from disk to run the onnx model, then get related tensor shape. 