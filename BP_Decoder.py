import torch
import torch.nn as nn
import numpy as np

class GetMatrixForBPNet:
    # Fully converted to PyTorch operations to fix numpy shape errors
    def __init__(self, test_H, loc_nzero_row):
        # Ensure inputs are tensors
        if not torch.is_tensor(test_H):
            self.H = torch.tensor(test_H, dtype=torch.float32)
        else:
            self.H = test_H
            
        self.m, self.n = self.H.shape
        
        # .sum(dim=0) works on tensors
        self.H_sum_line = self.H.sum(dim=0).long() 
        self.H_sum_row = self.H.sum(dim=1).long()
        self.loc_nzero_row = loc_nzero_row
        
        # Fix: Use shape[0] instead of .size to avoid ambiguity
        self.num_all_edges = self.loc_nzero_row[1, :].shape[0]

        # Calculate linear indices
        self.loc_nzero1 = self.loc_nzero_row[1, :] * self.n + self.loc_nzero_row[0, :]
        
        # Sort values
        self.loc_nzero2 = torch.sort(self.loc_nzero1).values
        
        # Stack to create (2, num_edges)
        row_indices = (self.loc_nzero2 % self.n).long()
        col_indices = (self.loc_nzero2 // self.n).long()
        self.loc_nzero_line = torch.stack([row_indices, col_indices])
        
        self.loc_nzero4 = self.loc_nzero_line[0, :] * self.n + self.loc_nzero_line[1, :]
        self.loc_nzero5 = torch.sort(self.loc_nzero4).values

    def get_Matrix_VC(self):
        # Create tensors on the same device as H
        device = self.H.device
        
        H_x_to_xe0 = torch.zeros([self.num_all_edges, self.n], dtype=torch.float32, device=device)
        H_sum_by_V_to_C = torch.zeros([self.num_all_edges, self.num_all_edges], dtype=torch.float32, device=device)
        H_xe_last_to_y = torch.zeros([self.n, self.num_all_edges], dtype=torch.float32, device=device)
        
        # Helper for finding indices
        Map_row_to_line = torch.zeros([self.num_all_edges], dtype=torch.long, device=device)

        # PyTorch equivalent of finding indices
        for i in range(self.num_all_edges):
            # torch.nonzero replaces np.where
            matches = torch.nonzero(self.loc_nzero1 == self.loc_nzero2[i], as_tuple=True)[0]
            Map_row_to_line[i] = matches[0]

        map_H_row_to_line = torch.zeros([self.num_all_edges, self.num_all_edges], dtype=torch.float32, device=device)

        for i in range(self.num_all_edges):
            map_H_row_to_line[i, Map_row_to_line[i]] = 1

        count = 0
        for i in range(self.n):
            count_int = int(count)
            degree = int(self.H_sum_line[i])
            temp = count_int + degree
            
            # Fill ranges
            H_sum_by_V_to_C[count_int:temp, count_int:temp] = 1
            H_xe_last_to_y[i, count_int:temp] = 1
            H_x_to_xe0[count_int:temp, i] = 1
            
            # Diagonal zeroing
            for j in range(degree):
                H_sum_by_V_to_C[count_int + j, count_int + j] = 0
            
            count = count + degree

        return H_x_to_xe0, torch.matmul(H_sum_by_V_to_C, map_H_row_to_line), torch.matmul(H_xe_last_to_y, map_H_row_to_line)

    def get_Matrix_CV(self):
        device = self.H.device
        H_sum_by_C_to_V = torch.zeros([self.num_all_edges, self.num_all_edges], dtype=torch.float32, device=device)
        Map_line_to_row = torch.zeros([self.num_all_edges], dtype=torch.long, device=device)

        for i in range(self.num_all_edges):
            # FIX: torch.nonzero returns a tensor, we explicitly take [0] to get the scalar index
            matches = torch.nonzero(self.loc_nzero4 == self.loc_nzero5[i], as_tuple=True)[0]
            Map_line_to_row[i] = matches[0]

        map_H_line_to_row = torch.zeros([self.num_all_edges, self.num_all_edges], dtype=torch.float32, device=device)

        for i in range(self.num_all_edges):
            map_H_line_to_row[i, Map_line_to_row[i]] = 1

        count = 0
        for i in range(self.m):
            count_int = int(count)
            degree = int(self.H_sum_row[i])
            temp = count_int + degree
            
            H_sum_by_C_to_V[count_int:temp, count_int:temp] = 1
            for j in range(degree):
                H_sum_by_C_to_V[count_int + j, count_int + j] = 0
            count = count + degree
            
        return torch.matmul(H_sum_by_C_to_V, map_H_line_to_row)


class BP_NetDecoder(nn.Module):
    def __init__(self, H, batch_size=None):
        super(BP_NetDecoder, self).__init__()
        
        # Convert incoming numpy H to Tensor immediately if it isn't one
        if not torch.is_tensor(H):
            self.H = torch.tensor(H, dtype=torch.float32)
        else:
            self.H = H.float()
            
        _, self.v_node_num = self.H.shape
        
        # Get nonzero indices using PyTorch
        ii, jj = torch.nonzero(self.H, as_tuple=True)
        loc_nzero_row = torch.stack([ii, jj])
        
        # Initialize the helper with Tensors
        gm1 = GetMatrixForBPNet(self.H, loc_nzero_row)
        
        # These are now Tensors already
        H_sumC_to_V = gm1.get_Matrix_CV()
        H_x_to_xe0, H_sumV_to_C, H_xe_v_sumc_to_y = gm1.get_Matrix_VC()
        
        # Register as buffers (constants)
        self.register_buffer('H_sumC_to_V', H_sumC_to_V)
        self.register_buffer('H_x_to_xe0', H_x_to_xe0)
        self.register_buffer('H_sumV_to_C', H_sumV_to_C)
        self.register_buffer('H_xe_v_sumc_to_y', H_xe_v_sumc_to_y)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x + 1e-10))

    def one_bp_iteration(self, xe_v2c_pre_iter, xe_0):
        # 1. Tanh
        xe_tanh = torch.tanh(xe_v2c_pre_iter.double() / 2.0).float()
        xe_tanh_temp = torch.sign(xe_tanh)
        
        # 2. Check Node Update
        # 1-sign returns 0 for pos, 2 for neg. /2 -> 0 for pos, 1 for neg.
        xe_sum_log_img = torch.matmul(self.H_sumC_to_V, (1 - xe_tanh_temp) / 2.0 * 3.1415926)
        xe_sum_log_real = torch.matmul(self.H_sumC_to_V, torch.log(1e-8 + torch.abs(xe_tanh)))
        
        xe_sum_log_complex = torch.complex(xe_sum_log_real, xe_sum_log_img)
        xe_product = torch.real(torch.exp(xe_sum_log_complex))
        
        # Avoid zero issues
        xe_product_temp = torch.sign(xe_product) * -2e-7
        xe_pd_modified = xe_product + xe_product_temp
        
        # 3. Variable Node Update
        xe_v_sumc = self.atanh(xe_pd_modified) * 2.0
        xe_c_sumv = xe_0 + torch.matmul(self.H_sumV_to_C, xe_v_sumc)
        
        return xe_v_sumc, xe_c_sumv

    def forward(self, llr_in, bp_iter_num):
        llr_transposed = llr_in.t()
        xe_0 = torch.matmul(self.H_x_to_xe0, llr_transposed)
        xe_v2c = xe_0.clone()
        
        for i in range(bp_iter_num - 1):
            xe_v_sumc, xe_c_sumv = self.one_bp_iteration(xe_v2c, xe_0)
            xe_v2c = xe_c_sumv
            
        # Final marginals
        if bp_iter_num == 1:
             xe_v_sumc, _ = self.one_bp_iteration(xe_v2c, xe_0)
        else:
             # Recompute v_sumc from the last c_sumv
             xe_v_sumc, _ = self.one_bp_iteration(xe_v2c, xe_0)

        bp_out_llr = llr_transposed + torch.matmul(self.H_xe_v_sumc_to_y, xe_v_sumc)
        dec_out = torch.floor_divide(1 - torch.sign(bp_out_llr).int(), 2)
        
        return dec_out.t()

    def decode(self, llr_in, bp_iter_num):
        # Wrapper to maintain compatibility with calling code
        self.eval() 
        with torch.no_grad():
            if isinstance(llr_in, np.ndarray):
                llr_tensor = torch.from_numpy(llr_in).float()
            else:
                llr_tensor = llr_in.float()
            
            device = self.H_sumC_to_V.device
            llr_tensor = llr_tensor.to(device)
            
            y_dec = self.forward(llr_tensor, bp_iter_num)
            
            return y_dec.cpu().numpy()