import torch
import torch.nn as nn # Need this to check for GELU
import os
import sys
import urllib.request

# 1. URLs for the files we need from the repo
base_url = "https://raw.githubusercontent.com/yangdeng-EML/ML_MM_Benchmark/main/models/MIXER/"
files = {
    "MLP_MIXER.py": base_url + "MLP_MIXER.py",
    "helper.py": base_url + "helper.py"
}

# 2. Download the files if they don't already exist
for filename, url in files.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

# 3. Patch the broken relative import inside MLP_MIXER.py
with open("MLP_MIXER.py", "r") as f:
    content = f.read()

if "from . import helper" in content or "from .helper" in content:
    content = content.replace("from . import helper", "import helper")
    content = content.replace("from .helper import", "from helper import")
    with open("MLP_MIXER.py", "w") as f:
        f.write(content)

# 4. Force Python to look in your current directory first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 5. Define your file paths
model_path = "/home/qubit/malof_lab/Project_1/forward_model/pre_trained_models/Mixer/ADM.pth"
onnx_path = "/home/qubit/malof_lab/Project_1/forward_model/converted_models/github_model_mixer.onnx"

# 6. Load the Model
print("Loading model...")
my_model = torch.load(model_path, map_location='cpu')
my_model.eval()
print("Model loaded successfully!")

# ---------------------------------------------------------
# 6.5 THE FIX: Patch older GELU layers for newer PyTorch
# ---------------------------------------------------------
print("Patching GELU layers for PyTorch compatibility...")
for module in my_model.modules():
    if isinstance(module, nn.GELU):
        if not hasattr(module, 'approximate'):
            # 'none' is the default value in newer PyTorch versions
            module.approximate = 'none' 
# ---------------------------------------------------------

# 7. Export to ONNX
# Batch size = 1, Features = 14
dummy_input = torch.randn(1, 14, dtype=torch.float32)

print("Exporting to ONNX...")
torch.onnx.export(
    my_model,
    dummy_input,
    onnx_path,
    input_names=['geometry_input'],
    output_names=['spectrum_output'],
    opset_version=12
)

print(f"Model successfully exported to ONNX at: {onnx_path}")