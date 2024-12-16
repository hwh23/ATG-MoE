import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor
from dataclasses import dataclass

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



@dataclass
class Beta:
    max_step_size:int
    max:float= 0.9
    min:float=0.99
    step_size:float=0.1
    
    def calculate(self, current_step:int):
        return min(self.min + self.step_size*(current_step/self.max_step_size), self.max)
            
    

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

    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0, switchloss=0, zloss=0,
                 bias=True, 
                 gating_activation=None, activation=nn.Sequential(nn.GELU()), 
                 noisy_gating=True,
                 acc_aux_loss=False):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.activation = activation
        assert activation!=None, 'Activation function cannot be None'
        assert gating_activation!=None, 'gating_activation function cannot be None'

        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()
        
        if gating_activation is None:
            gating_activation = nn.ReLU()
        self.f_gate = nn.Sequential(
            nn.Linear(input_size,
                        2 * num_experts if noisy_gating else num_experts,
                        bias=False)
        )
        nn.init.zeros_(self.f_gate[-1].weight)
        

    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

    def update_aux_statistics(self, logits, topk_weight, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + topk_weight.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        # cvloss = self.acc_gates.mean() / 10000.0
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)
        self.init_aux_statistics()
        return loss

    def compute_cvloss(self, topk_weight):
        return self.cv_squared(F.normalize(topk_weight.sum(0), p=1, dim=0))

    def compute_switchloss(self, topk_weight, freqs):
        loss = F.normalize(topk_weight.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss


    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y
    
    
    
class TaskMoE(MoE):

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
                 num_experts:int, 
                 k:int, 
                 w_MI:float=0, 
                 w_H:float=0.1, 
                 w_finetune_MI:float=0, 
                 limit_k:int=0, 
                 w_topk_loss:float=0.0, 
                 task_num:int=9, 
                 noisy_gating:bool=True, 
                 gating_activation=nn.GELU(), 
                 moe_multiple_gate:bool=False,
                 **kwargs):
        
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI
        self.w_H = w_H
        self.w_finetune_MI = w_finetune_MI
        self.moe_multiple_gate = moe_multiple_gate
        self.limit_k = max(k, limit_k)
        
        assert gating_activation!= None, 'gating_activation cannot be None'

        super(TaskMoE, self).__init__(input_size, 
                                      head_size, 
                                      num_experts, 
                                      k, 
                                      noisy_gating=noisy_gating, 
                                      gating_activation=gating_activation, 
                                      **kwargs)
    

        if w_finetune_MI < -100 and self.moe_multiple_gate: ## hack
            w_finetune_MI = 0
            self.w_finetune_MI = 0
            self.f_gate = nn.ModuleList([nn.Sequential(
                                                nn.Linear(input_size,
                                                      2 * num_experts if noisy_gating else num_experts,
                                                      bias=False)
                                        ) for i in range(task_num)])
        elif self.moe_multiple_gate:
            self.f_gate = nn.ModuleList([nn.Sequential(
                                            nn.Linear(input_size, input_size//4),
                                            gating_activation,
                                            nn.Linear(input_size//4,
                                                      2 * num_experts if noisy_gating else num_experts,
                                                      bias=True)
                                        ) for i in range(task_num)])
        else:
            self.f_gate = nn.ModuleList([nn.Sequential(
                                            nn.Linear(input_size, input_size//4),
                                            gating_activation,
                                            nn.Linear(input_size//4,
                                                      2 * num_experts if noisy_gating else num_experts,
                                                      bias=True)
                                        ) for i in range(1)])
        
        # 初始化门控权重
        if self.moe_multiple_gate:
            for i in range(task_num):
                nn.init.zeros_(self.f_gate[i][-1].weight) 
        else:
            nn.init.zeros_(self.f_gate[0][-1].weight) 


        # VARIABLES FOR MI LOSS CALCULATION
        # Buffers are tensors that are part of the model's state 
        # but are not learnable parameters (i.e., they are not updated by backpropagation).
        self.register_buffer('PTE', torch.zeros(self.task_num, self.num_experts)) # 任务特定的专家概率表
        self.register_buffer('PE', torch.zeros(self.num_experts))# 专家的概率表
        self.register_buffer('times',torch.zeros(1))# 记录次数的缓冲区
        
        self.max_pi_history_length = 20
        self.pi_history = []
        self.beta = Beta(max_step_size = self.max_pi_history_length)
        
        
        self.momentum = 0.0# Initialize momemtum for MI loss
        self.task_gate_freq = [0] * self.task_num
        self.topk_acc_probs = [0] * self.task_num
        self.token_probs = [0] * self.task_num

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
    
    def top_k_gating_deepseek(self, x:Tensor, task_bh:Tensor)->Tensor:
        bsz, seq_len, h = x.shape  
        # Generate a mask corresponding to the task id
        if self.moe_multiple_gate:
            categories, task_masks = self.get_task_mask(x, task_bh)
            logits = torch.zeros(size=[bsz, seq_len, self.num_experts],device=x.device)
            for i, task_mask in enumerate(task_masks):
                logits += self.f_gate[categories[i]](x * task_mask.view(bsz, 1,1))
        else:
            logits = self.f_gate[0](x)
        # 如果有大量不同的任务类别，这可能会导致内存问题，因此应评估任务类别的数量,在batch层级处理
        # for category in categories:
        #     mask = (task_bh == category).unsqueeze(-1).unsqueeze(-1)  # [bsz, 1, 1]
        #     gate_fn = self.f_gate[category]
        #     logits += torch.where(mask, gate_fn(x), torch.tensor(0., device=x.device))
        
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
            ce = mask_ce.float().mean(0) # no need to divide by k since maskce is viewed and meaned
            fi = ce * self.num_experts 
            
            # calculate current Pi
            Pi = scores.view(-1, self.num_experts).mean(0)
            
            # Update history with the latest Pi
            self.pi_history.append(Pi.detach())  # Add the current tensor of Scores to the history
            if len(self.pi_history) > self.max_pi_history_length:       # Keep only the last 10 values
                self.pi_history.pop(0)
            # Compute the overall average of all recorded Pi values
            running_avg_pi = torch.stack(self.pi_history).mean(0)
                        
            # get the linear weight for running average Pi
            beta = self.beta.calculate(len(self.pi_history))
            
            # Compute a moving pi with beta 
            windowed_Pi = (beta * running_avg_pi + (1 - beta) * Pi).flatten()
            
            # Get the buffered auxiliary loss
            aux_loss = (windowed_Pi * fi).sum()
            
        else: # do not calculate aux when inferencing
            aux_loss = None
            
        self.batch_gates, self.batch_index, self.expert_size, gates, self.index_sorted_experts = \
            compute_gating(self.k, scores.reshape(-1, scores.shape[-1]), topk_weight.reshape(-1, topk_weight.shape[-1]), topk_idx.reshape(-1, topk_idx.shape[-1]))
        
        return topk_idx, topk_weight, aux_loss
    
    
    
    def sdpforward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
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
    
    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        return self.deepseekforward(x, task_bh)
        # return self.sdpforward(x, task_bh, skip_mask, sample_topk, multiply_by_gates)
    
    def get_task_mask(self, x:Tensor, task_ids:Tensor):
        # Find unique categories
        unique_categories:Tensor = torch.unique(task_ids) # shape: [number_categories] 

        # Generate masks for each category
        masks = task_ids.unsqueeze(0) == unique_categories.unsqueeze(1) # shape: [number_categories, batch_size]

        return unique_categories, masks
    
    def deepseekforward(self, x:Tensor, task_bh:Tensor):    
        bsz, length, emb_size = x.size()
        orig_shape = x.shape
        
        topk_idx, topk_weight, loss = self.top_k_gating_deepseek(x, task_bh)
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
        
        if self.training:
            return y, loss
        else:
            return y
        
    
if __name__ == '__main__':
    batch_size = 34
    sequence_length = 1
    input_size = 3
    # model = TaskMoE(input_size=20,head_size=10,num_experts=5,k=2,activation=nn.Sequential(
    #                         nn.GELU(),
    #                     ),noisy_gating=False)
    
    #替换测试
    model = TaskMoE( # 这里加入了moe layer
            input_size, # input_size 
            input_size //2, # head_size
            8, # num_experts
            2, # topk’s k
            bias=True,
            acc_aux_loss=True,
            w_MI=0.0005, #0.0005
            w_finetune_MI=0,
            task_num=5, # 任务数量，暂时定为5个
            activation=nn.Sequential(
                nn.GELU(),
            ),
            noisy_gating=False,
        )# 有三个输出y, loss, topk_weight
    

    #对比线性层，最后一维输出不一致，moe 20维，线性层需要3维,经过task moe维度不变
    # model= nn.Sequential(
    #         nn.Linear(input_size , input_size ),
    #         nn.ReLU(),
    #         nn.Linear(input_size , 3)
    #     )

    input_data = torch.randn(batch_size, sequence_length, input_size)

    # Specify the task or task batch you want to perform inference for.
    task_batch_index = int(0) # Replace with the appropriate task batch index.

    # You can skip certain tokens during inference by providing a skip_mask. 
    # Set to None if you don't want to skip any tokens.
    skip_mask = None

    # Perform inference (forward pass) using the TaskMoE model for the specified task.

    output, loss = model(input_data, task_batch_index, skip_mask=skip_mask)####
    print(loss.shape)
