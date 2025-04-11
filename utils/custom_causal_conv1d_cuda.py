import torch
import torch.nn.functional as F
from einops import rearrange

class CausalConv1dCuda:
    @staticmethod
    def causal_conv1d_fwd(x, weight, bias, seq_idx=None, cu_seqlens=None, dummy=None, apply_silu=False):
        """
        Custom implementation of causal convolution for CUDA
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            weight: Convolution weights of shape (channels, kernel_size)
            bias: Optional bias tensor of shape (channels,)
            seq_idx: Optional sequence indices for document boundaries
            cu_seqlens: Optional cumulative sequence lengths
            dummy: Dummy parameter for API compatibility
            apply_silu: Whether to apply SiLU activation function
        Returns:
            Output tensor of shape (batch, channels, sequence_length)
        """
        batch, channels, seq_len = x.shape
        kernel_size = weight.shape[-1]
        
        # Reshape weight to match conv1d expectations
        weight_reshaped = rearrange(weight, "c k -> c 1 k")
        
        # Add padding for causal convolution
        padding = kernel_size - 1
        x_padded = F.pad(x, (padding, 0))
        
        # Apply convolution
        output = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=channels)
        
        # Handle sequence boundaries if seq_idx is provided
        if seq_idx is not None:
            # Create a mask to zero out positions where sequence indices change
            # This ensures we don't leak information across document boundaries
            seq_idx = seq_idx.to(x.device)
            seq_idx_expanded = seq_idx.unsqueeze(1).expand(-1, channels, -1)
            
            # Create a mask for each position in the sequence
            mask = torch.ones_like(output, dtype=torch.bool)
            
            # For each position, check if any of the kernel_size previous positions
            # have a different seq_idx (indicating a document boundary)
            for i in range(1, kernel_size):
                if i < seq_len:
                    curr_idx = seq_idx_expanded[:, :, i:]
                    prev_idx = seq_idx_expanded[:, :, :-i]
                    curr_mask = (curr_idx == prev_idx)
                    mask[:, :, i:] = mask[:, :, i:] & curr_mask
            
            # Apply the mask
            output = output * mask
        
        # Apply SiLU/Swish activation if requested
        if apply_silu:
            output = F.silu(output)
        
        return output
    
    @staticmethod
    def causal_conv1d_bwd(x, weight, bias, dy, seq_idx=None, cu_seqlens=None, dummy=None, dx=None, use_atomic=False, apply_silu=False):
        """
        Backward pass for custom causal convolution
        Args:
            x: Input tensor from forward pass
            weight: Convolution weights
            bias: Optional bias tensor
            dy: Gradient of output
            seq_idx: Optional sequence indices
            cu_seqlens: Optional cumulative sequence lengths
            dummy: Dummy parameter for API compatibility
            dx: Optional pre-allocated gradient tensor for x
            use_atomic: Whether to use atomic operations (not used in this implementation)
            apply_silu: Whether SiLU was applied in forward pass
        Returns:
            Gradients for x, weight, and bias
        """
        batch, channels, seq_len = x.shape
        kernel_size = weight.shape[-1]
        
        # Reshape weight for conv1d
        weight_reshaped = rearrange(weight, "c k -> c 1 k")
        
        # Apply SiLU backward if needed
        if apply_silu:
            # SiLU'(x) = SiLU(x) + sigmoid(x) * (1 - SiLU(x))
            x_padded = F.pad(x, (kernel_size - 1, 0))
            conv_output = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=channels)
            silu_output = F.silu(conv_output)
            sigmoid_x = torch.sigmoid(conv_output)
            dy = dy * (silu_output / conv_output + sigmoid_x * (1 - silu_output / conv_output))
        
        # Allocate output gradient tensors if not provided
        if dx is None:
            dx = torch.zeros_like(x)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(bias) if bias is not None else None
        
        # Handle sequence boundaries for dy if seq_idx is provided
        if seq_idx is not None:
            seq_idx = seq_idx.to(x.device)
            seq_idx_expanded = seq_idx.unsqueeze(1).expand(-1, channels, -1)
            
            # Create a mask for each position in the sequence
            mask = torch.ones_like(dy, dtype=torch.bool)
            
            # For each position, check if any of the kernel_size previous positions
            # have a different seq_idx (indicating a document boundary)
            for i in range(1, kernel_size):
                if i < seq_len:
                    curr_idx = seq_idx_expanded[:, :, i:]
                    prev_idx = seq_idx_expanded[:, :, :-i]
                    curr_mask = (curr_idx == prev_idx)
                    mask[:, :, i:] = mask[:, :, i:] & curr_mask
            
            # Apply the mask to dy
            dy = dy * mask
        
        # Add padding to x for gradient computation
        x_padded = F.pad(x, (kernel_size - 1, 0))
        
        # Compute gradient for bias
        if bias is not None and db is not None:
            db = dy.sum(dim=(0, 2))
        
        # Compute gradient for weight
        # For each output position, the weight grad is the sum of x_t-k * dy_t for each t
        for k in range(kernel_size):
            x_shifted = x_padded[:, :, k:k+seq_len]
            dw[:, k] += (x_shifted * dy).sum(dim=(0, 2))
        
        # Compute gradient for input
        # For each input position, the input grad is the sum of w_k * dy_t+k for each k
        # We need to use transposed convolution (conv_transpose1d) for this
        weight_flipped = torch.flip(weight_reshaped, dims=[-1])
        dx = F.conv_transpose1d(dy, weight_flipped, groups=channels)
        
        return dx, dw, db, None, None, None
        
# Create the module instance to be imported
causal_conv1d_cuda = CausalConv1dCuda()