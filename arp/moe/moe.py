import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor,Callable
from dataclasses import dataclass

from timm.models.vision_transformer import Mlp

@dataclass
class MoEArgs():
    input_size:int
    head_size:int 
    num_experts:int=8
    k:int=2
    limit_k:int=0
    task_num:int=9
    gating_activation:Any=nn.GELU()
    moe_multiple_gate:bool=False
    bias:bool=True
    activation:Any=nn.Sequential(nn.GELU())
    moe_cfg:Dict=None
    
@dataclass 
class SDPMoEArgs(MoEArgs):
    w_MI:float=0
    w_H:float=0
    w_finetune_MI:float=0
    noisy_gating:bool=False
    

@dataclass 
class DeepSeekMoEArgs(MoEArgs):
    pass
    
@dataclass
class DeepSeekMoEBlockArgs(DeepSeekMoEArgs):
    num_moe_layers:int=2    
             
@dataclass
class Beta:
    max_step_size:int
    max:float=0.95
    min:float=0.9
    step_size:float=None # (max-min)/max_step_size
    window_diff_factor:float = 7.0 # the factor multiplied with windows diff
    window_diff_func:Callable[[torch.Tensor], torch.Tensor]=torch.tanh # the function applied on windows diff, tanh effectively maps diff to [0,1]
    value:float=None # the beta value, default to Beta.min
    
    def __post_init__(self):
        self.value = self.min
        if self.step_size == None:
            self.step_size = (self.max-self.min)/self.max_step_size
            
    
    def calculate(self, current_step:int=None, window_diff:float=None)->float:
        """Calculate the beta value according to step size

        Args:
            current_step (int): the current step of calculation, at range [1, self.max_step_size]
            window_diff (float): the normalized difference between two consequential windows, at range [0.0, 1.0]
        Returns:
            float: the beta weight at range [self.min, self.max]
        """      
        assert current_step is None or current_step > 0, f'current_step must be greater than zero, but current_step={current_step} is given.'
        
        if window_diff is not None and current_step:
            # at the case when the window step havent hit 20
            # and dynamic beta computation is required
            self.value= min(self.min + \
                self.step_size * self.window_diff_factor * self.window_diff_func(window_diff) * current_step, 
                self.max)  
        elif current_step:
            # at the case when the window step havent hit 20
            # and no dynamic beta computation required
            self.value= min(self.min + \
                self.step_size * current_step, 
                self.max)   
        elif window_diff is not None:
            # at the case when the window step hit 20 once 
            # and beta uses windows diff to compute beta dynamically            
            self.value= min(self.min + \
                self.step_size * self.window_diff_func(window_diff)* self.window_diff_factor, 
                self.max)   
        else:
            # at the case when the window step hit 20 once
            # no dynamic beta required
            self.value=self.max
            
        return self.value
                  
@dataclass
class Window:
    window_size:int
    apply_dynamic_beta_flag:bool = False
    beta:Beta = None
    __history:Tensor = None
    __window_step:int = 1
    __first_window_flag:bool = True
    __initialize_flag:bool = True
    
    
    def __post_init__(self):
        # After the object is initialized, we can safely use the window_size field
        self.beta = Beta(max_step_size=self.window_size * 2)


    def __get_current_window(self):
        if self.__first_window_flag:
            return self.__history[:self.__window_step]
        else:
            return self.__history[:self.window_size]
    
    def __get_previous_window(self):
        return self.__history[self.window_size:]

    def __iterate_window(self):
        if self.__window_step == self.window_size*2:
            # clear steps when the step is equal to the length of history. 
            # It measures when the elements in the history are full-filled/refreshed
            self.__clear_step()
        self.__window_step += 1
        
    def __clear_step(self):
        if self.__first_window_flag:
            self.__first_window_flag = False
        self.__window_step = 0
    
    def init_history(self, value_size:Union[list,tuple,torch.Size], device:torch.device=None):
        """Initialize container for values, given the size of a single value.
           This function will exit if history is already initialized.

        Args:
            value_size (list or tuple or torch.size): The size of the value to be stored
        """        
        if self.__initialize_flag:
            self.__initialize_flag = False
            if isinstance(value_size, (torch.Size, tuple)):
                value_size = list(value_size)
            value_size.insert(0, self.window_size*2)
            self.__history = torch.zeros(size=value_size, device=device)
        
    def step(self, value:Tensor)->Tensor:
        """Compute the moving beta-weighted average value at history based on:
               output = beta * average window value + (1 - beta) * current value

        Args:
            value (Tensor): The value to recorded into history and windowed

        Returns:
            Tensor: a weighted moving windowed value, same shape as the input value.
        """
        # Add dimension if ndim is one. The operation is done for the torch.cat
        if value.ndim == 1:
            value = value.unsqueeze(0)
            
        # roll the history by insert the new value at first and remove the oldest value
        self.__history = torch.cat((value, self.__history[:-1]), dim=0)
        
        # get current window of history
        current_window = self.__get_current_window()
        
        # whether or not to parse current window step to beta calculator
        beta_step = self.__window_step if self.__first_window_flag else self.window_size
        
        # the following decides whether and when to parse windows difference to beta calculator
        get_diff_flag = self.__window_step == self.window_size*2 and self.apply_dynamic_beta_flag
        
        # Get norm of windows difference
        frobenius_norm = torch.norm(current_window-self.__get_previous_window(), p='fro') if get_diff_flag else None
        
        # Calculate beta value
        self.beta.calculate(beta_step, frobenius_norm)
        
        # Compute a moving pi with beta 
        weighted_average_value = (self.beta.value * current_window.mean(0) + (1 - self.beta.value) * value).flatten()
        
        # Iterate or reset steps. 
        # Note: call this after obtaining the step value 
        # because it will be updated after the function
        self.__iterate_window()
        
        return weighted_average_value
             
class ParallelLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd #(cast_inputs=torch.float32)
    def forward(ctx, input, expert_size, weight, bias=None):
        output = ParallelLinear.forward_scriptable(input, expert_size, weight, bias)
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, expert_size, weight, bias)
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor, expert_size: Tensor,
                           weight: Tensor, bias: Optional[Tensor]):
        output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
                                         device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        expert_size_list: List[int] = expert_size.tolist()
        # print('expert_size: ', expert_size)
        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                 input: Tensor, expert_size: Tensor,
                 weight: Tensor, bias: Optional[Tensor]):
        num_linears = weight.size(0)
        expert_size_list: List[int] = expert_size.tolist()
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias

@torch.jit.script
def compute_gating(k: int, scores: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """
    Compute gating for expert selection.

    Args:
        k (int): Number of top experts to select.
        scores (Tensor): Gating probabilities for experts. 
                    output shape: topk_weight.shape=(batch_size * length, 2 * num_experts) if noisy 
                    output shape: topk_weight.shape=(batch_size * length, num_experts) if clean 
        top_k_gates (Tensor): Top-k gating values. (The probabilities of the gates). shape of [batch_size, k]
        top_k_indices (Tensor): Indices of the top-k experts. shape of [batch_size, k]

    Returns:
        Tuple: Batch gates, batch indices, expert sizes, and sorted indices for top experts.
    """
    # Initialize a zero matrix
    zeros = torch.zeros_like(scores)
    
    # Obtain gate tensor 
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    
    # Flatten tensors 
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    
    # Get the Non-Zero top_k_expert from top_k_indices (where top_k_indices means the top_k_expert)
    # knowing that, top_k_gates and top_k_indices have the same indexing, 
    # therefore an non-zero index for top_k_gates and top_k_indices are the same
    top_k_experts_nonzero = top_k_experts[nonzeros]
    
    # Sort experts according to the expert ID, get the hash index of sorting
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    
    # Count active experts
    expert_size = (gates > 0).long().sum(0)
    
    # Get the sorted the nonzero top_k_experts indices
    # index_sorted_experts: contains the sorted indices of non-zero gating values, corresponding to active experts.
    index_sorted_experts = nonzeros[_index_sorted_experts]
    
    # Compute Batch Indices
    # index_sorted_experts.div(k) performs element-wise division of the index_sorted_experts tensor by k (the number of top-k experts).
    # This operation effectively maps the flat indices back to their batch index.
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    
    # Obtain the gating values for active experts in sorted order.
    # Use the sorted indices (index_sorted_experts) to extract 
    # and reorder the corresponding gating values.
    batch_gates = top_k_gates[index_sorted_experts]
    
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts
    
class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.bias = None
        self.reset_parameters()

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.weight.size(0), self.weight.size(1), self.weight.size(2))

    def reset_parameters(self) -> None:
        # std = math.sqrt(2.0 / float(self.weight.size(1) + self.weight.size(2)))
        # a = math.sqrt(3.0) * std
        nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.weight, self.bias)
        return results

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,  
                 input_size:int, 
                 head_size:int, 
                 num_experts:int=8, 
                 k:int=2, 
                 limit_k:int=0, 
                 task_num:int=9, 
                 gating_activation=nn.GELU(), 
                 moe_multiple_gate:bool=False,
                 bias=True, 
                 activation=nn.Sequential(nn.GELU()),
                 moe_cfg=None
                 ):
        super().__init__()
        # overwrite arguments with config if provided
        if moe_cfg:
            moe_multiple_gate = moe_cfg.multiple_gate
            num_experts = moe_cfg.num_experts
            k = moe_cfg.k
            num_shared_experts = moe_cfg.num_shared_experts
        
        assert activation!=None, 'Activation function cannot be None'
        assert gating_activation!=None, 'gating_activation function cannot be None'
        
        # Write to arguments to class
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        # TODO Experts重新设计
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.limit_k = max(k, limit_k)
        self.activation = activation
        self.task_num = task_num
        self.moe_multiple_gate = moe_multiple_gate
        self.gating_activation = gating_activation

     

class SDPMOE(MoE):
    def __init__(self, 
                 w_MI:float=0, 
                 w_H:float=0, 
                 w_finetune_MI:float=0, 
                 noisy_gating:bool=False, 
                 **kwargs):
        super().__init__(**kwargs)
                
        self.noisy_gating = noisy_gating
        self.w_MI = w_MI
        self.w_H = w_H
        self.w_finetune_MI = w_finetune_MI if w_finetune_MI >= -100 else 0
        
        hidden_size = max(self.input_size//4, 1) # in case input size < 4
        if w_finetune_MI < -100 and self.moe_multiple_gate: # multi gate, w_finetune_MI < -100
            linearlayer = nn.Linear(self.input_size, 2 * self.num_experts if self.noisy_gating else self.num_experts, bias=False)
            self.f_gate = nn.ModuleList([nn.Sequential(linearlayer) for _ in range(self.task_num)])
        elif self.moe_multiple_gate: # multi gate
            linearlayer1 = nn.Linear(self.input_size, hidden_size)
            linearlayer2 = nn.Linear(hidden_size, 2*self.num_experts if self.noisy_gating else self.num_experts, bias=True)
            self.f_gate = nn.ModuleList([nn.Sequential(linearlayer1, self.gating_activation, linearlayer2) for _ in range(self.task_num)])
        else: # single gate
            linearlayer1 = nn.Linear(self.input_size, hidden_size)
            linearlayer2 = nn.Linear(hidden_size, 2*self.num_experts if self.noisy_gating else self.num_experts, bias=True)
            self.f_gate = nn.ModuleList([nn.Sequential(linearlayer1, self.gating_activation, linearlayer2)])
        
        # 初始化门控权重
        if self.moe_multiple_gate:
            for i in range(self.task_num):
                nn.init.zeros_(self.f_gate[i][-1].weight) 
        else:
            nn.init.zeros_(self.f_gate[0][-1].weight)
        
        # VARIABLES FOR ORIGINAL MI LOSS CALCULATION
        # Buffers are tensors that are part of the model's state 
        # but are not learnable parameters (i.e., they are not updated by backpropagation).
        self.register_buffer('PTE', torch.zeros(self.task_num, self.num_experts)) # 任务特定的专家概率表
        self.register_buffer('PE', torch.zeros(self.num_experts))# 专家的概率表
        self.register_buffer('times',torch.zeros(1))# 记录次数的缓冲区
        self.momentum = 0.0# Initialize momemtum for MI loss
        self.task_gate_freq = [0] * self.task_num
        self.topk_acc_probs = [0] * self.task_num
        self.token_probs = [0] * self.task_num
    
    def extra_repr(self):
        return f'k={self.k}, task_num={self.task_num}, num_expert={self.num_experts}, num_gate={len(self.f_gate)}'
        
    def get_MIloss(self, scores:Tensor, task_bh) ->Tensor:
        """Compute Mutual Information (MI) Loss for the current MoE layer. 
        Mutual Information is defined as:

        Args:
            scores (Tensor): The probability of gates
            task_bh (Tensor): The task ID

        Returns:
            Tensor: a scalar MI loss. 
        """        
        if not self.training:
            return torch.tensor([0.0]).cuda()

        top_k_gates, _ = scores.topk(self.k, dim=1)
        self.token_probs[task_bh] = self.token_probs[task_bh] * 0.95 + top_k_gates.mean(0).detach()*0.05

        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + (self.expert_size).detach()*0.05

        self.topk_acc_probs[task_bh] = self.topk_acc_probs[task_bh]*0.95 + (scores.mean(0)).detach()*0.05
        
        # since we want each task to have equal weight P(T_j) = 1/j, j is the task number
        PT = 1 / self.task_num 

        # PTE = P(T_j, E^n_l)
        self.PTE[task_bh] = self.PTE[task_bh] * self.momentum + (1-self.momentum) * (scores.mean(0).detach() * PT)
        loss = -self.w_H * (scores * torch.log(scores + 0.0001)).sum(1).mean() # maximize the entropy

        if self.times[0] < 100:
            self.times[0] = self.times[0] + 1
            self.momentum = 1 - 1/(self.times[0]) 
            return loss.unsqueeze(0)
        else:
            self.momentum = 0.99

        PE = self.PTE.sum(0).detach()

        # P(E,T) in this batch
        MI_task_gate = torch.zeros(self.task_num, self.num_experts).cuda()
        MI_task_gate[task_bh] = MI_task_gate[task_bh] + scores.mean(0) * PT

        # P(E) in this batch
        P_EI = scores.mean(0) * PT

        # get the MI loss, negative since we want to maximize the mutual information
        MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum() - (P_EI * (1 + torch.log(PE + 0.0001))).sum())

        finetune_MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum())

        # Compute auxiliary loss
        loss = loss + self.w_MI * MI_loss + self.w_finetune_MI * finetune_MI_loss 
        loss = loss.unsqueeze(0)
        print(loss)
        return loss
 
    def top_k_gating(self, x:Tensor, task_bh:int, skip_mask:Tensor=None, sample_topk:int=0, noise_epsilon:float=1e-2)->Tensor:
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        
        Args:
            x (Tensor): Input tensor at Shape: (batch_size*length, emb_size)
            task_bh (int): Task index.
            skip_mask (Tensor, optional): Mask to skip certain inputs.. Defaults to None.
            sample_topk (int, optional): Number of experts to sample. Defaults to 0.
            noise_epsilon (float, optional): The noise level for gating. Defaults to 1e-2.

        Returns:
            Tensor: probability [num_experts]
        """        
        # Pass the input to gate to obtain the probability
        clean_logits = self.f_gate[task_bh](x)
        # Sample from noise if noisy gating is applied
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        # use output from gate directly otherwise
        else:
            logits = clean_logits

        
        scores = torch.softmax(logits, dim=1) + 1e-4

        # Skip certain experts if mask is given
        if skip_mask is not None:
            scores = torch.masked_fill(scores, skip_mask, 0)

        # Obtain top k gates (the probabilities) and the top k indices (the expert IDs)
        if self.training and (sample_topk > 0):
            assert sample_topk <= self.k
            _, top_km1_indices = scores.topk(self.k - sample_topk, dim=1)
            masked_probs = scores + 1e-6
            masked_probs[torch.arange(scores.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(scores, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = scores.topk(self.k, dim=1)

        self.batch_gates, self.batch_index, self.expert_size, gates, self.index_sorted_experts = \
            compute_gating(self.k, scores, top_k_gates, top_k_indices)
        
        return scores

    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        topk_weight = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)
        loss = self.get_MIloss(topk_weight, task_bh)
        
        
        h = self.experts(x[self.batch_index], self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        # index expert output y based on batch index, and shape it to the original batch
        zeros = torch.zeros(
            (bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, 
            device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y, loss
    
       
class DeepSeekMoE(MoE):
    def __init__(self, **kwargs:DeepSeekMoEArgs):
        super().__init__(**kwargs)
        self.window_length = 20
        self.window = Window(self.window_length)
        
        hidden_size = max(self.input_size//4, 1)

        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # self.experts= nn.ModuleList([Mlp(in_features=self.input_size,
        #                                   hidden_features=int(hidden_size * 4.0), 
        #                                   act_layer=approx_gelu, 
        #                                   drop=0.1)
        #                                   for _ in range(self.num_experts)])
        
        self.shared_experts = Mlp(in_features=self.input_size, hidden_features=int(hidden_size * 4.0), act_layer=approx_gelu,
                                drop=0.1)

        linearlayer1 = nn.Linear(self.input_size, hidden_size)
        linearlayer2 = nn.Linear(hidden_size, self.num_experts, bias=True)
        self.f_gate = nn.ModuleList([nn.Sequential(linearlayer1, 
                                                   self.gating_activation, 
                                                   linearlayer2) 
                                     for _ in range(self.task_num if self.moe_multiple_gate else 1)])
    
        # 初始化门控权重
        if self.moe_multiple_gate:
            for i in range(self.task_num):
                nn.init.zeros_(self.f_gate[i][-1].weight) 
        else:
            nn.init.zeros_(self.f_gate[0][-1].weight)

    def extra_repr(self):
        return f'k={self.k}, task_num={self.task_num}, num_expert={self.num_experts}, num_gate={len(self.f_gate)}'        

    def top_k_gating(self, x:Tensor, task_bh:Tensor)->Tensor:
        bsz, seq_len, h = x.shape  
        # Generate a mask corresponding to the task id
        if self.moe_multiple_gate:
            categories, task_masks = self.get_task_mask(x, task_bh)
            # logits = torch.zeros(size=[bsz, seq_len, self.num_experts],device=x.device)
            # for i, task_mask in enumerate(task_masks):
            #     logits += self.f_gate[categories[i]](x * task_mask.view(bsz, 1,1))
            masked_x = x.unsqueeze(0) * task_masks.unsqueeze(-1).unsqueeze(-1)  # masked_x.Shape: [num_categories, bsz, seq_len, num_tasks]
            # 将unique_categories从tensor转换为整数列表
            categories_list = [c.item() for c in categories]
            # 使用列表推导式生成所有门控函数的输出
            all_logits = []
            for i, category in enumerate(categories_list):
                # 获取对应的门控函数
                gate = self.f_gate[category]
                # 计算该类别的logits
                logits = gate(masked_x[i])
                all_logits.append(logits)
            # 将所有logits相加
            logits = torch.stack(all_logits, dim=0).sum(0)
        else:
            logits = self.f_gate[0](x)
        
        # compute gating scores
        scores = logits.softmax(dim=-1)
        
        # select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.k, dim=-1, sorted=False)
        
        # expert-level computation auxiliary loss
        if self.training: #and self.alpha > 0.0:
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            # count how many tokens selected per gate in all batch sizes
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.num_experts)
            # the mean of expert selection among all data 
            ce = mask_ce.float().mean(0) # no need to divide by k since maskce is viewed and meaned
            # scale up 
            fi = ce * self.num_experts 
            
            # calculate current Pi: the mean of scores of each expert among all data
            Pi = scores.view(-1, self.num_experts).mean(0)
            
            # initialize history if this is the first time applied
            self.window.init_history(Pi.size(), Pi.device)
            
            # step the window
            moving_average_pi = self.window.step(Pi.detach())
            
            # Get the buffered auxiliary loss, 
            # noticed that the original loss was in range [1, num_expert], 
            # map the loss to range [0,1], -0.6 leaves some space in case loss becomes negative
            # the weight will be applied outside the MoE layer
            aux_loss = ((moving_average_pi * fi).sum() - 0.6)/(self.num_experts - 1)
            
        else: # do not calculate aux when inferencing
            aux_loss = None
            
        self.batch_gates, self.batch_index, self.expert_size, gates, self.index_sorted_experts = \
            compute_gating(self.k, scores.reshape(-1, scores.shape[-1]), topk_weight.reshape(-1, topk_weight.shape[-1]), topk_idx.reshape(-1, topk_idx.shape[-1]))
        
        return topk_idx, topk_weight, aux_loss
    
      
    def get_task_mask(self, x:Tensor, task_ids:Tensor):
        # Find unique categories
        unique_categories:Tensor = torch.unique(task_ids) # shape: [number_categories] 

        # Generate masks for each category
        masks = task_ids.unsqueeze(0) == unique_categories.unsqueeze(1) # shape: [number_categories, batch_size]

        return unique_categories, masks
    
    def forward(self, x:Tensor, task_bh:Tensor):    
        bsz, length, emb_size = x.size()
        orig_shape = x.shape
        identity = x
        
        topk_idx, topk_weight, loss = self.top_k_gating(x, task_bh)
        x = x.view(-1, emb_size)
        h = self.experts(x[self.batch_index], self.expert_size)
        activated_h = self.activation(h)
        expert_outputs = self.output_experts(activated_h, self.expert_size)
        
        # index expert output y based on batch index, and shape it to the original batch
        zeros = torch.zeros(
            (bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, 
            device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)

        # TODO weight
        y = 0.4 * y + 0.6 * self.shared_experts(identity)
        
        if self.training:
            return y, loss
        else:
            return y
         

   
class DeepSeekMoEBlock(nn.Module):
    def __init__(self,
                 num_moe_layers:int=2,
                 **kwargs:DeepSeekMoEArgs):
        super(DeepSeekMoEBlock, self).__init__()
        self.moe_layers = nn.ModuleList([DeepSeekMoE(**kwargs) for _ in range(num_moe_layers)])
    def forward(self, x):
        aux_losses = []
        for moe_layer in self.moe_layers:
            x, aux_loss = moe_layer(x)
            aux_losses.append(aux_loss)
        return x, sum(aux_losses)
    
class MoEMLP(nn.Module):
    def __init__(self, **kwargs:DeepSeekMoEBlockArgs):
        super(MoEMLP, self).__init__()
        self.layers = nn.Sequential(
            DeepSeekMoEBlock(**kwargs),
            nn.GELU,
            DeepSeekMoEBlock(**kwargs)
        )

    def forward(self, x):
        return self.layers(x)

              
if __name__ == '__main__':
    batch_size = 34
    sequence_length = 1
    input_size = 3
    model = DeepSeekMoE( 
            **vars(DeepSeekMoEArgs(input_size=input_size,
                           head_size=input_size//2,
                           num_experts=8,
                           k=2,
                           bias=True,
                           task_num=5,
                           activation=nn.Sequential(nn.GELU()),
                           )
                   )
            )

    input_data = torch.randn(batch_size, sequence_length, input_size)

    DeepSeekMoE.forward
    output, loss = model(input_data,  int(0))
    print(loss)
    print(output.shape)
