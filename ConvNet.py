import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import copy
import DataIO  

# Internal PyTorch Module to replace the graph construction
class _CNNModel(nn.Module):
    def __init__(self, net_config):
        super(_CNNModel, self).__init__()
        self.net_config = net_config
        self.layers = nn.ModuleList()
        
        # Construct layers dynamically based on config
        for layer_idx in range(self.net_config.total_layers):
            if layer_idx == 0:
                in_channels = 1
            else:
                in_channels = self.net_config.feature_map_nums[layer_idx - 1]
            
            out_channels = self.net_config.feature_map_nums[layer_idx]
            kernel_size = self.net_config.filter_sizes[layer_idx]
            
            # Replicating TF 'SAME' padding for kernel (k, 1). 
            # Assuming stride 1. Padding = (k-1)/2
            pad_h = kernel_size // 2
            pad_w = 0 
            
            # TF shape was [filter_height, 1, in, out]. 
            # We use Conv2d with kernel (h, 1)
            conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=(kernel_size, 1), 
                             stride=1, 
                             padding=(pad_h, pad_w))
            
            # Xavier Initialization (matches tf.contrib.layers.xavier_initializer)
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
            
            self.layers.append(conv)

    def forward(self, x):
        # Input x shape: (Batch, Feature_Length)
        # Reshape to NCHW match TF logic: 
        # TF was: (-1, feature_length, 1, 1) -> NHWC (Height=FeatureLen, Width=1, Chan=1)
        # PyTorch needs: (Batch, Channel, Height, Width) -> (Batch, 1, FeatureLen, 1)
        x = x.view(-1, 1, self.net_config.feature_length, 1)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU to all except the last layer
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                
        # Final Output Reshape
        # TF: (-1, label_length)
        x = x.view(-1, self.net_config.label_length)
        return x

class ConvNet:
    def __init__(self, net_config_in, train_config_in, net_id):
        self.net_config = net_config_in
        self.train_config = train_config_in
        self.net_id = net_id
        self.res_noise_power_dict = {}
        self.res_noise_pdf_dict = {}
        
        # Device configuration (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State holder for the best model found during training
        self.best_model_state = None

    def restore_network_with_model_id(self, model, restore_layers_in, model_id):
        # Logic to generate file path matches original
        model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str)-1)]
        model_folder = format("%snetid%d_model%s" % (self.net_config.model_folder, self.net_id, model_id_str))
        restore_model_name = format("%s/model.pth" % model_folder) # Changed .ckpt to .pth for PyTorch convention

        if os.path.exists(restore_model_name):
            print(f"Loading model from {restore_model_name}")
            checkpoint = torch.load(restore_model_name, map_location=self.device)
            
            if restore_layers_in > 0:
                # Partial loading logic
                model_dict = model.state_dict()
                # Filter out unnecessary keys if needed, or just load strictly
                # Here we assume structure compatibility. 
                # If you need to load only first N layers, we iterate keys.
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print("Restore layers from checkpoint.\n")
        else:
            print("No checkpoint found, starting fresh.")

    def save_network_temporarily(self, model):
        # In PyTorch, we keep the state dict in memory instead of running a TF assign op
        self.best_model_state = copy.deepcopy(model.state_dict())

    def save_network(self, model_id):
        if self.best_model_state is None:
            return

        model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str) - 1)]
        save_model_folder = format("%snetid%d_model%s" % (self.net_config.model_folder, self.net_id, model_id_str))

        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)
            
        save_model_name = format("%s/model.pth" % (save_model_folder))
        torch.save(self.best_model_state, save_model_name)
        print("Save %d layers.\n" % self.net_config.save_layers)

    def calc_normality_test(self, residual_noise, batch_size, batch_size_for_norm_test):
        # PyTorch implementation of the custom skewness/kurtosis loss
        groups = int(batch_size // batch_size_for_norm_test)
        
        # Reshape: [groups, label_len * batch_norm]
        residual_noise = residual_noise.view(groups, self.net_config.label_length * batch_size_for_norm_test)

        mean = torch.mean(residual_noise, dim=1, keepdim=True)
        # Variance
        variance = torch.mean(torch.square(residual_noise - mean), dim=1, keepdim=True)
        # 3rd Moment
        moment_3rd = torch.mean(torch.pow(residual_noise - mean, 3), dim=1, keepdim=True)
        # 4th Moment
        moment_4th = torch.mean(torch.pow(residual_noise - mean, 4), dim=1, keepdim=True)
        
        skewness = torch.div(moment_3rd, torch.pow(variance, 1.5) + 1e-10)
        kurtosis = torch.div(moment_4th, torch.square(variance) + 1e-10)
        
        norm_test = torch.mean(torch.square(skewness) + 0.25 * torch.square(kurtosis - 3))
        return norm_test

    def test_network_online(self, dataio, model, orig_loss_calc_fn, test_loss_calc_fn, calc_org_loss):
        remain_samples = self.train_config.test_sample_num
        load_batch_size = self.train_config.test_minibatch_size
        ave_loss_after_train = 0.0
        ave_org_loss = 0.0
        
        model.eval() # Set to eval mode
        
        with torch.no_grad(): # No gradient calculation for testing
            while remain_samples > 0:
                if remain_samples < self.train_config.test_minibatch_size:
                    load_batch_size = remain_samples

                batch_xs, batch_ys = dataio.load_batch_for_test(load_batch_size)
                
                # Convert to torch tensor and move to device
                x_in = torch.tensor(batch_xs, dtype=torch.float32).to(self.device)
                y_label = torch.tensor(batch_ys, dtype=torch.float32).to(self.device)
                
                y_out = model(x_in)

                if calc_org_loss:
                    loss_after_training_value = test_loss_calc_fn(y_out, y_label, x_in).item()
                    orig_loss_value = orig_loss_calc_fn(y_out, y_label, x_in).item()
                    ave_org_loss += orig_loss_value * load_batch_size
                else:
                    loss_after_training_value = test_loss_calc_fn(y_out, y_label, x_in).item()
                
                remain_samples -= load_batch_size
                ave_loss_after_train += loss_after_training_value * load_batch_size

        if calc_org_loss:
            ave_org_loss /= float(self.train_config.test_sample_num)
            
        ave_loss_after_train /= float(self.train_config.test_sample_num)
        
        if calc_org_loss:
            print("Test loss: %f, orig loss: %f" % (ave_loss_after_train, ave_org_loss))
        else:
            print(ave_loss_after_train)
            
        return ave_loss_after_train, ave_org_loss

    def train_network(self, model_id):
        start = datetime.datetime.now()
        
        # Instantiate Model and move to device
        model = _CNNModel(self.net_config).to(self.device)
        
        dataio = DataIO.TrainingDataIO(self.train_config.training_feature_file, self.train_config.training_label_file,
                                       self.train_config.training_sample_num, self.net_config.feature_length,
                                       self.net_config.label_length)
        dataio_test = DataIO.TestDataIO(self.train_config.test_feature_file, self.train_config.test_label_file,
                                   self.train_config.test_sample_num, self.net_config.feature_length,
                                   self.net_config.label_length)

        # Define Loss Helper Functions to mimic TF graph behavior
        def compute_loss(y_out, y_label, x_in, is_training_data=True):
            # FIX: Get batch size dynamically from the data, not the config
            current_batch_size = y_out.size(0) 
            
            if self.train_config.normality_test_enabled:
                # Use current_batch_size instead of fixed config batch_size
                norm_loss = self.calc_normality_test(y_label - y_out, current_batch_size, 1)
                mse_loss = torch.mean(torch.square(y_out - y_label))
                
                if self.train_config.normality_lambda != np.inf:
                    return mse_loss + norm_loss * self.train_config.normality_lambda
                return mse_loss 
            else:
                return torch.mean(torch.square(y_out - y_label))

        def compute_orig_loss(y_out, y_label, x_in, is_training_data=False):
            # FIX: Get batch size dynamically from the data
            current_batch_size = x_in.size(0)
            
            if self.train_config.normality_test_enabled:
                norm_loss = self.calc_normality_test(y_label - x_in, current_batch_size, 1)
                mse_loss = torch.mean(torch.square(y_label - x_in))
                if self.train_config.normality_lambda != np.inf:
                    return mse_loss + norm_loss * self.train_config.normality_lambda
            return torch.mean(torch.square(y_label - x_in))

        # Optimizer
        optimizer = optim.Adam(model.parameters())

        # Restore
        self.restore_network_with_model_id(model, self.net_config.restore_layers, model_id)

        # Initial Test
        min_loss, ave_org_loss = self.test_network_online(
            dataio_test, model, compute_orig_loss, compute_loss, True
        )
        
        self.save_network_temporarily(model)

        count = 0
        epoch = 0
        print('Iteration\tLoss')

        while epoch < self.train_config.epoch_num:
            epoch += 1
            model.train() # Switch to training mode
            
            batch_xs, batch_ys = dataio.load_next_mini_batch(self.train_config.training_minibatch_size)
            
            x_in = torch.tensor(batch_xs, dtype=torch.float32).to(self.device)
            y_label = torch.tensor(batch_ys, dtype=torch.float32).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            y_out = model(x_in)
            loss = compute_loss(y_out, y_label, x_in, is_training_data=True)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0 or epoch == self.train_config.epoch_num:
                print(epoch)
                ave_loss_after_train, _ = self.test_network_online(
                    dataio_test, model, compute_orig_loss, compute_loss, False
                )
                
                if ave_loss_after_train < min_loss:
                    min_loss = ave_loss_after_train
                    self.save_network_temporarily(model)
                    count = 0
                else:
                    count += 1
                    if count >= 8:  # no patience
                        break
        
        self.save_network(model_id)
        
        end = datetime.datetime.now()
        print('Final minimum loss: %f' % min_loss)
        print('Used time for training: %ds' % (end - start).seconds)

    # These helpers remain largely the same, just removed self-dependency where not needed
    def get_res_noise_power(self, model_id, SNRset=np.zeros(0)):
        if len(self.res_noise_power_dict) == 0:
            model_id_str = np.array2string(model_id[0:(self.net_id+1)], separator='_', formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str)-1)]
            residual_noise_power_file = format("%sresidual_noise_property_netid%d_model%s.txt" % (self.net_config.residual_noise_property_folder, self.net_id, model_id_str))
            data = np.loadtxt(residual_noise_power_file, dtype=np.float32)
            shape_data = np.shape(data)
            if np.size(shape_data) == 1:
                self.res_noise_power_dict[data[0]] = data[1:shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_power_dict[data[i, 0]] = data[i, 1:shape_data[1]]
        return self.res_noise_power_dict

    def get_res_noise_pdf(self, model_id):
        if len(self.res_noise_pdf_dict) == 0:
            model_id_str = np.array2string(model_id[0:(self.net_id+1)], separator='_', formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str)-1)]
            residual_noise_pdf_file = format("%sresidual_noise_property_netid%d_model%s.txt" % (self.net_config.residual_noise_property_folder, self.net_id, model_id_str))
            data = np.loadtxt(residual_noise_pdf_file, dtype=np.float32)
            shape_data = np.shape(data)
            if np.size(shape_data) == 1:
                self.res_noise_pdf_dict[data[0]] = data[1:shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_pdf_dict[data[i, 0]] = data[i, 1:shape_data[1]]
        return self.res_noise_pdf_dict


