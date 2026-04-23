import onnx
from onnx import helper, TensorProto
import sys
import os

OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 60, None])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

pool = helper.make_node("GlobalAveragePool", inputs=["input"], outputs=["pooled"])
flatten = helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1)

weight = helper.make_tensor("W", TensorProto.FLOAT, [2, 60], [0.01] * 120)
bias = helper.make_tensor("b", TensorProto.FLOAT, [2], [0.5, 0.5])
gemm = helper.make_node("Gemm", inputs=["flat", "W", "b"], outputs=["output"])

graph = helper.make_graph(
    [pool, flatten, gemm],
    "voiceguard_dummy",
    [input_tensor],
    [output_tensor],
    initializer=[weight, bias],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.checker.check_model(model)
onnx.save(model, OUTPUT_PATH)

print(f"Dummy model saved to: {os.path.abspath(OUTPUT_PATH)}")
print("Input shape:  (1, 1, 60, dynamic_frames)")
print("Output shape: (1, 2)  [prediction, confidence]")
