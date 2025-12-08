import numpy as np
import os

data_dir = "verification_data_pt"

tf_out = np.load(os.path.join(data_dir, 'tf_output.npy'))
pt_out = np.load(os.path.join(data_dir, 'pt_output.npy'))

# Check Max Difference
diff = np.abs(tf_out - pt_out)
max_diff = np.max(diff)

print(f"Max Difference: {max_diff}")

if np.allclose(tf_out, pt_out, atol=1e-5):
    print("SUCCESS: Models are mathematically identical!")
else:
    print("Models diverge in absolute terms.")
    # Show first few mismatches
    print("\n\nTF:", tf_out.flatten()[:15])
    print("PT:", pt_out.flatten()[:15])

tf_out = np.load(os.path.join(data_dir, 'tf_output.npy'))

# Load PT output (adjust path if saved elsewhere)
pt_out = np.load(os.path.join(data_dir, 'pt_output.npy'))

# 1. Calculate Absolute Difference
abs_diff = np.abs(tf_out - pt_out)

# 2. Calculate Relative Difference (avoid divide by zero)
# Formula: |TF - PT| / (|TF| + epsilon)
epsilon = 1e-9
rel_diff = abs_diff / (np.abs(tf_out) + epsilon)
max_rel_diff = np.max(rel_diff)

print(f"Max Relative Difference: {max_rel_diff:.6f} ({max_rel_diff*100:.4f}%)")

if max_rel_diff < 0.01: # Less than 1% error
    print("SUCCESS: The models are functionally equivalent.")
else:
    print("WARNING: Divergence detected.")