import math
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer.
    Replaces a standard nn.Linear block to inject trainable rank-decomposition matrices
    while freezing the original pre-trained weights.
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original_linear = original_linear
        # freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA matrices A and B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # init weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Original forward
        orig_output = self.original_linear(x)
        
        # LoRA forward
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        
        return orig_output + (lora_output * self.scaling)

def inject_lora(model: nn.Module, rank: int = 8, alpha: int = 16, target_modules=["wq", "wv"]):
    """
    Recursively replaces target nn.Linear layers in the model with LoRALayer blocks.
    Commonly applied to Attention Query and Value projection layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # Replace
            lora_layer = LoRALayer(module, rank=rank, lora_alpha=alpha)
            setattr(model, name, lora_layer)
        else:
            inject_lora(module, rank, alpha, target_modules)
            
def mark_only_lora_as_trainable(model: nn.Module):
    """Ensure only LoRA parameters (and maybe norm layers / heads) require gradients."""
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
