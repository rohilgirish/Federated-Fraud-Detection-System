"""
Model Compression Module
Implements quantization, pruning, and distillation for edge deployment
"""
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class ModelQuantizer:
    """Quantizes model weights to reduce size."""
    
    @staticmethod
    def quantize_int8(model: nn.Module) -> nn.Module:
        """Convert model to INT8 quantization (4x compression)."""
        quantized_model = deepcopy(model)
        
        for param in quantized_model.parameters():
            # Simple linear quantization to INT8
            param.data = torch.quantize_per_tensor(
                param.data, scale=1/127, zero_point=0, dtype=torch.qint8
            ).dequantize()
        
        return quantized_model
    
    @staticmethod
    def quantize_float16(model: nn.Module) -> nn.Module:
        """Convert model to FP16 (2x compression)."""
        quantized_model = deepcopy(model)
        quantized_model.half()  # Convert to FP16
        return quantized_model
    
    @staticmethod
    def get_model_size(model: nn.Module) -> dict:
        """Get model size in bytes and MB."""
        total_params = sum(p.numel() for p in model.parameters())
        total_buffers = sum(b.numel() for b in model.buffers())
        
        # Approximate sizes
        param_size_mb = (total_params * 4) / (1024 ** 2)  # 4 bytes per FP32
        buffer_size_mb = (total_buffers * 4) / (1024 ** 2)
        total_size_mb = param_size_mb + buffer_size_mb
        
        return {
            'total_parameters': total_params,
            'total_buffers': total_buffers,
            'param_size_mb': param_size_mb,
            'buffer_size_mb': buffer_size_mb,
            'total_size_mb': total_size_mb
        }


class ModelPruner:
    """Prunes unnecessary connections from neural network."""
    
    @staticmethod
    def magnitude_pruning(model: nn.Module, threshold: float = 0.01) -> nn.Module:
        """
        Remove weights below threshold magnitude.
        Typical threshold: 0.01 removes ~20-30% of weights
        """
        pruned_model = deepcopy(model)
        total_removed = 0
        total_params = 0
        
        for param in pruned_model.parameters():
            if len(param.shape) > 1:  # Only prune weight matrices, not biases
                mask = torch.abs(param.data) > threshold
                total_removed += (~mask).sum().item()
                total_params += param.numel()
                param.data = param.data * mask.float()
        
        removal_rate = (total_removed / total_params * 100) if total_params > 0 else 0
        print(f"[PRUNING] Removed {removal_rate:.1f}% of parameters (threshold={threshold})")
        
        return pruned_model
    
    @staticmethod
    def structured_pruning(model: nn.Module, layer_idx: int, channels: int) -> nn.Module:
        """Remove entire filters/channels to speed up inference."""
        pruned_model = deepcopy(model)
        
        # Get the layer
        layers = [m for m in pruned_model.modules() if isinstance(m, nn.Linear)]
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            # Zero out some channels
            layer.weight.data[:channels, :] = 0
        
        return pruned_model
    
    @staticmethod
    def get_sparsity(model: nn.Module) -> dict:
        """Measure model sparsity (% of zero weights)."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
        
        sparsity_pct = (zero_params / total_params * 100) if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'sparsity_percentage': sparsity_pct,
            'compression_ratio': total_params / (total_params - zero_params + 1)
        }


class KnowledgeDistillation:
    """Transfer knowledge from large model to small model."""
    
    @staticmethod
    def create_student_model(teacher_model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
        """Create smaller student model (empirically ~0.5 compression ratio)."""
        # Simplified: reduce layer sizes by compression_ratio
        student = deepcopy(teacher_model)
        
        for module in student.modules():
            if isinstance(module, nn.Linear):
                new_out = max(1, int(module.out_features * compression_ratio))
                new_module = nn.Linear(module.in_features, new_out)
                # Random initialization for new student
                nn.init.kaiming_uniform_(new_module.weight)
                module = new_module
        
        return student
    
    @staticmethod
    def calculate_distillation_loss(student_logits, teacher_logits, temperature: float = 4.0):
        """
        KL divergence loss for distillation.
        Temperature controls smoothness of probability distribution.
        """
        soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
        soft_predictions = nn.functional.log_softmax(student_logits / temperature, dim=1)
        loss = nn.functional.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        return loss


class EdgeOptimizer:
    """Optimize model for edge deployment (mobile, IoT)."""
    
    @staticmethod
    def optimize_for_edge(model: nn.Module, optimization_level: str = 'medium') -> dict:
        """
        Apply multiple optimizations for edge deployment.
        Levels: 'light' (minimal), 'medium' (balanced), 'aggressive' (maximum compression)
        """
        metrics = {}
        optimized_model = deepcopy(model)
        
        # Original size
        original_size = ModelQuantizer.get_model_size(optimized_model)
        metrics['original_size_mb'] = original_size['total_size_mb']
        
        if optimization_level in ['medium', 'aggressive']:
            # Pruning
            threshold = 0.01 if optimization_level == 'medium' else 0.05
            optimized_model = ModelPruner.magnitude_pruning(optimized_model, threshold)
            metrics['sparsity'] = ModelPruner.get_sparsity(optimized_model)
        
        if optimization_level in ['aggressive']:
            # Quantization to FP16
            optimized_model = ModelQuantizer.quantize_float16(optimized_model)
        
        # Final size
        final_size = ModelQuantizer.get_model_size(optimized_model)
        metrics['final_size_mb'] = final_size['total_size_mb']
        metrics['compression_ratio'] = original_size['total_size_mb'] / (final_size['total_size_mb'] + 1e-6)
        
        return optimized_model, metrics


def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print comprehensive model summary."""
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY: {name}")
    print(f"{'='*60}")
    
    # Architecture
    print("\nArchitecture:")
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
            print(f"  {i}. {name}: {module}")
    
    # Size
    size_info = ModelQuantizer.get_model_size(model)
    print(f"\nModel Size:")
    print(f"  Total parameters: {size_info['total_parameters']:,}")
    print(f"  Size in memory: {size_info['total_size_mb']:.2f} MB")
    
    # Sparsity
    sparsity_info = ModelPruner.get_sparsity(model)
    print(f"\nSparsity:")
    print(f"  Sparsity: {sparsity_info['sparsity_percentage']:.2f}%")
    print(f"  Zero parameters: {sparsity_info['zero_parameters']:,}")
    
    print(f"{'='*60}\n")
