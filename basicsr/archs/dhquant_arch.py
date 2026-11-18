"""DHQuant fake quantizers and SwinIR layer replacements.

HCG follows Section III-D: cluster channel extrema once, initialize group
bounds with DOBI, and retain fixed assignments during bound optimization.
The uniform quantizer, straight-through estimators, and DOBI derive from 2DQuant.
"""

from typing import Any
from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import torch
from torch import Tensor, FloatTensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
from functools import partial
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

calibrated_num = 0
total_num = 0

def quant(input:torch.Tensor, lb:float, ub:float, bit:int):
    input = input.clamp(lb, ub)
    s = (ub - lb) / (2 ** bit -1)
    input = (input - lb)/s
    input = input.round()
    input = input * s + lb
    return input

def cal_mse(input:torch.Tensor, lb:float, ub:float, bit:int):
    quant_input = quant(input, lb, ub, bit)
    res = float(torch.norm(input - quant_input))
    return res

def DOBI(input:torch.Tensor, bit:int, one_direction = False, num:int=100):
    min_value = torch.min(input)
    max_value = torch.max(input)
    
    diff = (max_value - min_value) / (2 * num)
    
    history_min = float('inf')
    # Search on the input device, including CPU for small regression checks.
    
    if one_direction:
        diff = (max_value - min_value) / num
        for i in range(num):
            lb = min_value
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    else:
        diff = (max_value - min_value) / (2 * num)
        for i in range(num):
            lb = min_value + diff * i
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    global calibrated_num
    global total_num
    
    calibrated_num += 1
    print(f'calibration:{calibrated_num}/{total_num}')
    
    return float(best_lb), float(best_ub)

class Differentiable_Round(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs):
        return grad_outputs


class Differentiable_Clip(Function):
    @staticmethod
    def forward(
        ctx: _ContextMethodMixin,
        input: Tensor,
        min_val: Tensor,
        max_val: Tensor,
    ) -> Any:
        ctx.save_for_backward(input, min_val, max_val)
        
        # Support both scalar and tensor bounds
        if min_val.numel() == 1 and max_val.numel() == 1:
             output = input.clamp(min_val.item(), max_val.item())
        else:
             # For per-channel or grouped quantization, bounds are tensors
             # Ensure bounds are broadcastable to input
             output = torch.min(torch.max(input, min_val), max_val)
             
        return output

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensors

        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0
        
        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        
        # Handle gradient shape for min_val
        if min_val.numel() == 1:
            grad_min = grad_min.sum().view(1)
        else:
            # Sum gradients over dimensions that were broadcasted
            # min_val shape: (C, 1, 1, 1) or (C, 1)
            # input shape: (Out, In, K, K) or (Out, In)
            # We need to sum over all dimensions EXCEPT the channel dimension
            # Assuming min_val has the same number of dimensions as input (after broadcasting/view)
            dims_to_sum = [i for i in range(input.dim()) if min_val.shape[i] == 1 and input.shape[i] > 1]
            if dims_to_sum:
                grad_min = grad_min.sum(dim=dims_to_sum, keepdim=True)
            # Ensure the gradient shape matches min_val shape exactly
            if grad_min.shape != min_val.shape:
                 grad_min = grad_min.view(min_val.shape)

        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        
        # Handle gradient shape for max_val
        if max_val.numel() == 1:
            grad_max = grad_max.sum().view(1)
        else:
            # Same logic as min_val
            dims_to_sum = [i for i in range(input.dim()) if max_val.shape[i] == 1 and input.shape[i] > 1]
            if dims_to_sum:
                grad_max = grad_max.sum(dim=dims_to_sum, keepdim=True)
            if grad_max.shape != max_val.shape:
                 grad_max = grad_max.view(max_val.shape)
                 
        return grad_input, grad_min, grad_max


class FakeQuantizerBase(Module):
    def __init__(self, int_quant: bool = True, bit:int=4) -> None:
        super().__init__()
        self.lower_bound = Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        self.upper_bound = Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        self.n_bit = Parameter(
            torch.randn((1,), dtype=torch.float32),
        )
        # self.n_bit = bit
        self.set_n_bit_manually(bit)
        
        self.bit2bound = {}
        self.use_bit2bound = False
        self.size_of_input = None
        
        self.int_quant = int_quant

        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply
        
        self.calibrated = False
        self.one_direction_search = False
        
        global total_num 
        total_num += 1
        

    def set_int_quant(self, enable: bool):
        self.int_quant = enable

    def set_require_grad(self, enable_lb: bool, enable_up: bool, enable_nbit: bool):
        self.lower_bound.requires_grad = enable_lb
        self.upper_bound.requires_grad = enable_up
        # self.n_bit.requires_grad = enable_nbit

    def set_params_manually(self, lb: Tensor, ub: Tensor, n_bit: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)
        # self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)
        
    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.device
        self.lower_bound.data = FloatTensor([lb]).data.clone().to(device)
    
    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.device
        self.upper_bound.data = FloatTensor([ub]).data.clone().to(device)

    def set_n_bit_manually(self, n_bit):
        device = self.n_bit.device
        self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)


class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self, bit=4, per_channel=False, channel_len=1, weight_dim=4) -> None:
        super(FakeQuantizerWeight, self).__init__(bit=bit)
        self.per_channel = per_channel
        self.channel_len = channel_len
        self.weight_dim = weight_dim
        
        if self.per_channel:
            # Initialize as vectors based on weight dimension
            # For Conv2d (Out, In, K, K), we want (Out, 1, 1, 1)
            # For Linear (Out, In), we want (Out, 1)
            if self.weight_dim == 4:
                shape = (channel_len, 1, 1, 1)
            elif self.weight_dim == 2:
                shape = (channel_len, 1)
            else:
                # Fallback to 4D or handle other cases
                shape = (channel_len,) + (1,) * (self.weight_dim - 1)
                
            self.lower_bound = Parameter(torch.randn(shape, dtype=torch.float32))
            self.upper_bound = Parameter(torch.randn(shape, dtype=torch.float32))

    def forward(self, x:torch.Tensor):
        if not self.calibrated:
            if self.per_channel:
                # Per-channel initialization (Min-Max)
                # Assuming x is (Out, In, K, K) or (Out, In)
                # Channel axis is 0
                n_channels = x.shape[0]
                x_flat = x.reshape(n_channels, -1)
                min_val = x_flat.min(dim=1)[0]
                max_val = x_flat.max(dim=1)[0]
                
                # Reshape for broadcasting
                view_shape = [n_channels] + [1] * (x.dim() - 1)
                
                # Resize parameters to match input dimension if necessary
                if self.lower_bound.dim() != x.dim():
                     # E.g. param is (C, 1, 1, 1) but input is (C, In)
                     new_shape = [n_channels] + [1] * (x.dim() - 1)
                     self.lower_bound.data = self.lower_bound.data.view(new_shape)
                     self.upper_bound.data = self.upper_bound.data.view(new_shape)

                self.lower_bound.data = min_val.view(view_shape).to(self.lower_bound.device)
                self.upper_bound.data = max_val.view(view_shape).to(self.upper_bound.device)
            else:
                lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(rb)
            
            self.calibrated = True
            return x
        
        if self.size_of_input is None:
            self.size_of_input = x.numel()
        
        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        
        if self.use_bit2bound and not self.per_channel:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # Ensure parameter shapes match input for broadcasting (handle Linear vs Conv case)
        if self.per_channel and self.lower_bound.dim() != x.dim():
             view_shape = [x.shape[0]] + [1] * (x.dim() - 1)
             lb = self.lower_bound.view(view_shape)
             ub = self.upper_bound.view(view_shape)
        else:
             lb = self.lower_bound
             ub = self.upper_bound

        # (u-l)/(2^n-1)
        s = (ub - lb) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, lb, ub)

        # int value \in [0,2^n-1]
        r = self.round((c - lb) / s)

        return s * r + lb


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self,bit=4) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super(FakeQuantizerAct, self).__init__(bit=bit)

        self.running_stat = False # initial boundary
        self.first_iter = False # initial boundary
        self.dynamic = False
        self.beta = 0.995
        self.identity = False

    def forward(self, x):
        if not self.calibrated:
            lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)
            self.calibrated = True
            return x

        if self.size_of_input is None:
            self.size_of_input = x.numel()  

        if self.identity:
            return x
        
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / (torch.pow(2, n_bits) - 1)

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')
            
        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound
    
class HCGActivationQuantizer(FakeQuantizerBase):
    """Homogeneous Channel Grouping with fixed labels and learnable bounds."""

    def __init__(self, bit=4, n_groups=32, channel_axis=-1, channel_len=None) -> None:
        super(HCGActivationQuantizer, self).__init__(bit=bit)
        self.n_groups = n_groups
        self.channel_len = channel_len
        self.channel_axis = channel_axis
        self.grouped = False
        self.labels = None
        
        # Register labels_buffer for saving/loading
        # Initialize with a dummy value. It will be overwritten by load_state_dict or group_channels
        self.register_buffer('labels_buffer', torch.zeros(1, dtype=torch.long))
        
        # Initialize with correct size if channel_len is provided
        if channel_len is not None:
            n_clusters = min(self.n_groups, channel_len)
            self.grouped_lower_bound = Parameter(torch.randn(n_clusters, dtype=torch.float32))
            self.grouped_upper_bound = Parameter(torch.randn(n_clusters, dtype=torch.float32))
        else:
            # Fallback to dummy initialization
            self.grouped_lower_bound = Parameter(torch.tensor([0.0])) 
            self.grouped_upper_bound = Parameter(torch.tensor([1.0]))

    def group_channels(self, x):
        # x: (B, ..., C, ...)
        # Move channel axis to last
        if self.channel_axis != -1 and self.channel_axis != x.dim() - 1:
            permute_dims = list(range(x.dim()))
            permute_dims.append(permute_dims.pop(self.channel_axis))
            x_p = x.permute(permute_dims)
        else:
            x_p = x
            
        # Flatten to (N, C)
        C = x_p.shape[-1]
        x_flat = x_p.reshape(-1, C)
        
        # Calculate per-channel min/max
        # We use min/max as features for clustering
        min_val = x_flat.min(dim=0)[0] # Keep on device
        max_val = x_flat.max(dim=0)[0] # Keep on device
        
        # Check if we should recover labels from existing bounds (e.g. loaded from checkpoint)
        # We check if loaded_from_checkpoint flag is set AND if the bounds have meaningful size (not dummy 1)
        is_loaded = getattr(self, 'loaded_from_checkpoint', False)
        if is_loaded and self.grouped_lower_bound.numel() > 1:
             # Recovery Mode: Assign channels to nearest existing group center
             # Existing centers
             centers_lb = self.grouped_lower_bound.detach() # (K,)
             centers_ub = self.grouped_upper_bound.detach() # (K,)
             
             # Features: (C, 2) -> (min, max)
             # Centers: (K, 2) -> (lb, ub)
             
             # Compute distance matrix (C, K)
             # dist = (min_c - lb_k)^2 + (max_c - ub_k)^2
             
             # Expand for broadcasting
             # min_val: (C, 1)
             # centers_lb: (1, K)
             d_lb = min_val.unsqueeze(1) - centers_lb.unsqueeze(0)
             d_ub = max_val.unsqueeze(1) - centers_ub.unsqueeze(0)
             
             dist = d_lb.pow(2) + d_ub.pow(2)
             
             # Find nearest center for each channel
             self.labels = torch.argmin(dist, dim=1) # (C,)
             self.grouped = True
             
             # Do NOT update bounds - keep the trained ones
             return

        min_val = min_val.cpu().numpy()
        max_val = max_val.cpu().numpy()
        
        # Features: [min, max]
        features = np.stack([min_val, max_val], axis=1)
        
        # KMeans
        n_clusters = min(self.n_groups, C)
        # Handle small channel count
        if n_clusters < 2:
             kmeans_labels = np.zeros(C, dtype=int)
             kmeans_centers = np.mean(features, axis=0, keepdims=True)
        else:
             kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features)
             kmeans_labels = kmeans.labels_
             kmeans_centers = kmeans.cluster_centers_
        
        self.labels = torch.tensor(kmeans_labels, device=x.device, dtype=torch.long)
        
        # --- MSE Refinement for Groups ---
        # Instead of directly using K-Means centers (Min/Max based),
        # run DOBI on each group's actual data to find MSE-optimal bounds.
        
        refined_lb = torch.zeros(n_clusters, device=x.device)
        refined_ub = torch.zeros(n_clusters, device=x.device)
        
        # x_p is permuted x, reshape to (N, C)
        # We need to extract data for each group
        x_flat_all = x_p.reshape(-1, C) 
        
        print(f"Refining groups with DOBI (MSE Search)... n_clusters={n_clusters}")
        
        for k in range(n_clusters):
            # Identifying channels belonging to group k
            mask = (self.labels == k)
            if not mask.any():
                continue
            
            # Extract data: (N_samples, n_channels_in_group) -> flatten
            group_data = x_flat_all[:, mask].reshape(-1)
            
            # Sampling for speed if data is too large
            if group_data.numel() > 20000: 
                # Random sampling 20k points is usually enough for distribution estimation
                indices = torch.randint(0, group_data.numel(), (20000,), device=x.device)
                group_data = group_data[indices]
                
            # Run DOBI
            # Note: DOBI expects input on CUDA presumably, group_data is on device
            best_lb, best_ub = DOBI(group_data, bit=self.n_bit, one_direction=self.one_direction_search)
            
            refined_lb[k] = best_lb
            refined_ub[k] = best_ub
            
        # Re-initialize grouped parameters with MSE-refined bounds
        self.grouped_lower_bound.data = refined_lb
        self.grouped_upper_bound.data = refined_ub
        
        self.grouped_lower_bound = Parameter(refined_lb)
        self.grouped_upper_bound = Parameter(refined_ub)
        
        # Update buffer
        self.labels_buffer = self.labels
        
        self.grouped = True

        # Observe the exact descriptors/labels used in this initialization.
        # Checkpoint recovery returns above and must not masquerade as new HCG.
        observer = getattr(self, '_group_init_observer', None)
        if observer is not None:
            observer(self, x.shape, min_val, max_val)

    def forward(self, x):
        # Restore labels from buffer if needed (e.g. after loading checkpoint where calibrated=True but labels=None)
        if self.labels is None and self.labels_buffer is not None and self.labels_buffer.numel() > 1:
             self.labels = self.labels_buffer
             self.grouped = True
             self.calibrated = True

        if not self.calibrated:
            # Priority 1: Use loaded labels if available (New Checkpoint)
            if self.labels_buffer is not None and self.labels_buffer.numel() > 1:
                 self.labels = self.labels_buffer
                 self.grouped = True
                 self.calibrated = True
                 # print(f"DEBUG: Loaded labels from buffer. Shape: {self.labels.shape}")
                 # Set size_of_input for logging purposes
                 if self.size_of_input is None:
                     self.size_of_input = x.numel()
                 # Fall through to quantization
            else:
                # Priority 2: Run grouping on the first batch (Calibration / Old Checkpoint Recovery)
                # print("DEBUG: Running group_channels (Calibration/Recovery)")
                self.group_channels(x)
                self.calibrated = True
                # Set size_of_input for logging purposes
                if self.size_of_input is None:
                    self.size_of_input = x.numel()
                return x
            
        if self.size_of_input is None:
            self.size_of_input = x.numel()

        if not self.grouped:
             return x

        # Gather parameters based on labels
        # labels: (C,)
        # grouped_lb: (K,)
        lb = self.grouped_lower_bound[self.labels] # (C,)
        ub = self.grouped_upper_bound[self.labels] # (C,)
        
        # Reshape lb/ub to match x for broadcasting
        # x: (B, ..., C, ...)
        shape = [1] * x.dim()
        # Handle negative axis
        axis = self.channel_axis if self.channel_axis >= 0 else x.dim() + self.channel_axis
        shape[axis] = -1
        lb = lb.view(shape)
        ub = ub.view(shape)
        
        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        
        # (u-l)/(2^n-1)
        s = (ub - lb) / (torch.pow(2, n_bits) - 1)

        # clip(x,l,u)
        c = self.clip(x, lb, ub)

        # int value \in [0,2^n-1]
        r = self.round((c - lb) / s)

        return s * r + lb

##########################################################
##########################################################


class FakeQuantizerWeightParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / self.n_bit

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class FakeQuantizerActParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super().__init__()

        self.running_stat = True # initial boundary
        self.first_iter = True # initial boundary
        self.dynamic = False
        self.beta = 0.995

    def forward(self, x):
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
            n_bits = self.n_bit
            lb = torch.min(x).detach()
            ub = torch.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / n_bits

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = torch.min(x).detach().clone()
                self.upper_bound.data = torch.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  torch.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  torch.max(x)
            return x


        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / n_bits

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class QuantBase(Module):
    def __init__(self,config):
        super().__init__()
        self.quant = True
        self.config = config
        self.bit = config['bit']
        # Defer initialization to set_param where we know dimensions
        self.weight_quantizer = None
        self.act_quantizer = None

    def init_quantizers(self, channel_len, channel_axis, weight_dim=4, input_channel_len=None):
        weight_per_channel = self.config.get('weight_per_channel', False)
        hcg = self.config.get('hcg', False)
        n_groups = self.config.get('n_groups', 32)
        
        self.weight_quantizer = FakeQuantizerWeight(self.bit, per_channel=weight_per_channel, channel_len=channel_len, weight_dim=weight_dim)
        
        if hcg:
            self.act_quantizer = HCGActivationQuantizer(self.bit, n_groups=n_groups, channel_axis=channel_axis,
                                                      channel_len=input_channel_len or channel_len)
        else:
            self.act_quantizer = FakeQuantizerAct(self.bit)

    def get_weight_quantizer(self):
        return self.weight_quantizer

    def get_act_quantizer(self):
        return self.act_quantizer

    def set_quant_flag(self, enable: bool):
        self.quant = enable

    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        if self.weight_quantizer:
            self.weight_quantizer.set_require_grad(enable,enable, enable)
        if self.act_quantizer:
            self.act_quantizer.set_require_grad(enable,enable, enable)

    def set_weight_bias_grad(self, enable: bool):
        self.weight.requires_grad = enable
        if self.bias:
            self.bias.requires_grad = enable

    def get_quant_weight_bias(self):
        # Recompute on every forward: bound optimization also runs in eval mode.
        # Reusing a detached result would lose bound gradients, retain stale
        # bounds, or keep the unquantized first calibration pass indefinitely.
        quant_weight = self.weight_quantizer(self.weight)

        return (quant_weight, self.bias)


class QuantLinear(QuantBase):
    def __init__(self,config):
        super().__init__(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.act_quantizer.set_params_lb_manually(min_value)
        self.act_quantizer.set_params_ub_manually(max_value)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.in_features
        self.out_feature = linear.out_features
        
        # Init quantizers
        self.init_quantizers(channel_len=self.out_feature, channel_axis=-1, weight_dim=2,
                             input_channel_len=self.in_feature)

        self.weight = Parameter(linear.weight.data.clone())

        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data.clone())
        else:
            self.bias = linear.bias

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)

        quant_act = self.act_quantizer(x)
        quant_weight, bias = self.get_quant_weight_bias()
        
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, bias)

class QuantLinearQKV(Module):
    def __init__(self,config):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)
        
    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)
        
    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)


    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)
    

    def get_quant_weight_bias(self):
        w_q,b_q = self.q.get_quant_weight_bias()
        w_k,b_k = self.k.get_quant_weight_bias()
        w_v,b_v = self.v.get_quant_weight_bias()
        
        quant_weight = torch.cat([w_q, w_k, w_v],dim=0)
        if b_q is not None:
            bias = torch.cat([b_q,b_k,b_v])

        return (quant_weight, bias)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.in_features 
        self.out_feature = linear.out_features // 3
        
        linear_q = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_k = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_v = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        
        
        linear_q.weight.data,linear_k.weight.data,linear_v.weight.data = linear.weight.data.clone().reshape(3,self.in_feature,self.out_feature)
        
        
        if linear.bias is not None:
            linear_q.bias.data, linear_k.bias.data, linear_v.bias.data = linear.bias.data.clone().reshape(3,self.out_feature)
    
        self.q.set_param(linear_q)
        self.k.set_param(linear_k)
        self.v.set_param(linear_v)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # print('qsize',q.size())
        # print('ksize',k.size())
        # print('vsize',v.size())
        return torch.cat([q,k,v], dim=-1)
    
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)


class QuantConv2d(QuantBase):
    def __init__(self,config):
        super().__init__(config)

    def set_param(self, conv: Conv2d):
        """
        must be called before forward.
        """

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.conv_kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        }
        
        # Init quantizers
        self.init_quantizers(channel_len=self.out_channels, channel_axis=1, weight_dim=4,
                             input_channel_len=self.in_channels)
        self.weight = Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data.clone())
        else:
            self.bias = conv.bias

    def forward(self, x):
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.conv_kwargs)

        quant_act = self.act_quantizer(x)
        quant_weight, bias = self.get_quant_weight_bias()
        
        # bias doesn't need to be quant

        return F.conv2d(quant_act, quant_weight, bias, **self.conv_kwargs)
