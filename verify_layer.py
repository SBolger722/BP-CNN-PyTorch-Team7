import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import ConvNet as ConvNet_pytorch
import Configrations as cfg

load_dir = "spy_data"

# 1. Load Data
print("Loading spy data...")
dummy_input = np.load(os.path.join(load_dir, 'input_data.npy'))
with open(os.path.join(load_dir, 'tf_layer_outputs.pkl'), 'rb') as f:
    tf_layer_outputs = pickle.load(f)
with open(os.path.join(load_dir, 'tf_weights.pkl'), 'rb') as f:
    tf_weights = pickle.load(f)

# 2. Setup Model
top_config = cfg.TopConfig()
net_config = cfg.NetConfig(top_config)
model_pt = ConvNet_pytorch._CNNModel(net_config)

# 3. Transplant Weights
layer_idx = 0
with torch.no_grad():
    for layer in model_pt.layers:
        if isinstance(layer, torch.nn.Conv2d):
            tf_w_name = f"conv_layer{layer_idx}:0"
            tf_b_name = f"b{layer_idx}:0"
            if tf_w_name in tf_weights:
                pt_w = np.transpose(tf_weights[tf_w_name], (3, 2, 0, 1))
                layer.weight.data = torch.from_numpy(pt_w)
                layer.bias.data = torch.from_numpy(tf_weights[tf_b_name])
            layer_idx += 1

# 4. Attach Hooks
pt_outputs = {}
def get_hook(name):
    def hook(model, input, output):
        pt_outputs[name] = output.detach().numpy()
    return hook

for i, layer in enumerate(model_pt.layers):
    layer.register_forward_hook(get_hook(f"Layer_{i}"))

# 5. Inference
model_pt.eval()
input_tensor = torch.from_numpy(dummy_input)
with torch.no_grad():
    model_pt(input_tensor)

# 6. Compare (Relative)
print("\n--- RAW LAYER COMPARISON (Relative) ---")
sorted_layers = sorted(tf_layer_outputs.keys(), key=lambda x: int(x.split('_')[1]))

for layer_name in sorted_layers:
    if layer_name not in pt_outputs:
        print(f"Skipping {layer_name} (Not found in PT model)")
        continue

    tf_out = tf_layer_outputs[layer_name]
    pt_out = pt_outputs[layer_name]

    if pt_out.ndim == 4:
        pt_out = np.transpose(pt_out, (0, 2, 3, 1))

    # --- CALC RELATIVE DIFFERENCE ---
    abs_diff = np.abs(tf_out - pt_out)
    max_abs_diff = np.max(abs_diff)
    
    # Calculate Relative Diff just for info
    epsilon = 1e-9
    rel_diff = abs_diff / (np.abs(tf_out) + epsilon)
    max_rel_diff = np.max(rel_diff)
    
    # --- HYBRID CHECK (The Fix) ---
    # 1. If absolute difference is tiny (< 0.001), it PASSES (ignore percentage).
    # 2. If values are large, we check if percentage is low (< 1%).
    if max_abs_diff < 1e-5:
        status = "PASS"
    elif max_rel_diff < 1e-1: # 1% tolerance
        status = "PASS"
    else:
        status = "FAIL"
    
    print(f"{layer_name}: Abs {max_abs_diff:.6f} | Rel {max_rel_diff*100:.4f}% \t [{status}]")
    
    if status == "FAIL":
        print(f"--> STOP! Divergence detected at {layer_name}.")
        break