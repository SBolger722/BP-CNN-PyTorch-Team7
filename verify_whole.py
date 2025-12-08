import torch
import numpy as np
import pickle
import os
import ConvNet  # Your new PyTorch file
import Configrations as cfg


load_dir = "verification_data_pt" # or wherever you saved it
with open(os.path.join(load_dir, 'tf_weights.pkl'), 'rb') as f:
    tf_weights = pickle.load(f)

print("Keys found in TF weights file:")
for key in sorted(tf_weights.keys()):
    print(key)

# --- CONFIGURATION ---
load_dir = "verification_data_pt"

# 1. Load Data
dummy_input = np.load(os.path.join(load_dir, 'input_data.npy'))
with open(os.path.join(load_dir, 'tf_weights.pkl'), 'rb') as f:
    tf_weights = pickle.load(f)

# 2. Instantiate PyTorch Model
top_config = cfg.TopConfig()
net_config = cfg.NetConfig(top_config)
model_pt = ConvNet._CNNModel(net_config)

# 3. Transplant Weights
print("Transplanting weights...")
layer_idx = 0
with torch.no_grad():
    for layer in model_pt.layers:
        if isinstance(layer, torch.nn.Conv2d):
            # Construct names based on your print output
            tf_w_name = f"conv_layer{layer_idx}:0"
            tf_b_name = f"b{layer_idx}:0"
            
            print(f"Loading {tf_w_name} into layer {layer_idx}...")
            
            # --- Load Weights ---
            if tf_w_name in tf_weights:
                tf_w = tf_weights[tf_w_name]
                # Transpose TF (Height, Width, In, Out) -> PyTorch (Out, In, Height, Width)
                pt_w = np.transpose(tf_w, (3, 2, 0, 1))
                layer.weight.data = torch.from_numpy(pt_w)
            else:
                print(f"ERROR: Key {tf_w_name} not found!")

            # --- Load Biases ---
            if tf_b_name in tf_weights:
                tf_b = tf_weights[tf_b_name]
                layer.bias.data = torch.from_numpy(tf_b)
            else:
                print(f"ERROR: Key {tf_b_name} not found!")
            
            layer_idx += 1

# 4. Run Inference
model_pt.eval()
pt_input = torch.from_numpy(dummy_input)
pt_output_tensor = model_pt(pt_input)
pt_output = pt_output_tensor.detach().numpy()

# 5. Save PT Output
np.save(os.path.join(load_dir, 'pt_output.npy'), pt_output)
print(f"PyTorch output saved to {load_dir}/pt_output.npy")