import torch
import os

model_path = "/home/qubit/malof_lab/Project_1/forward_model/models/Transformer/adm_transformer_v1_nautilus/best_model_forward.pt"
onnx_path  = "/home/qubit/malof_lab/Project_1/forward_model/converted_models/transformer_v1.onnx"

os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

print("Loading model...")
model = torch.load(model_path, map_location='cpu')
model.eval()
print("Model loaded.")

# Batch size=1, 14 geometry inputs
dummy_input = torch.randn(1, 14, dtype=torch.float32)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['geometry_input'],
    output_names=['spectrum_output'],
    opset_version=12,
)

print(f"Exported to: {onnx_path}")
