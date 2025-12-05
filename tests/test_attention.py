"""
Tests for DCM-MSR attention module.
"""

import pytest
import torch
import torch.nn as nn

from dcm_msr.attention import DCMMSRAttention, DCMMSRSelfAttention


class TestDCMMSRAttention:
    """Tests for DCMMSRAttention module."""
    
    def test_output_shape(self):
        """Test that output shape matches expected dimensions."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        x = torch.randn(batch, seq_len, embed_dim)
        output, _ = attn(x, x, x)
        
        assert output.shape == (batch, seq_len, embed_dim)
    
    def test_different_qkv_lengths(self):
        """Test with different query and key/value lengths."""
        batch = 2
        seq_q = 16
        seq_kv = 32
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        q = torch.randn(batch, seq_q, embed_dim)
        kv = torch.randn(batch, seq_kv, embed_dim)
        
        output, _ = attn(q, kv, kv)
        
        assert output.shape == (batch, seq_q, embed_dim)
    
    def test_return_attention(self):
        """Test returning attention weights."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        x = torch.randn(batch, seq_len, embed_dim)
        output, attention_info = attn(x, x, x, return_attention=True)
        
        assert output.shape == (batch, seq_len, embed_dim)
        assert attention_info is not None
        
        coarse_scores, coarse_attn = attention_info
        assert coarse_scores.shape[0] == batch
        assert coarse_attn.shape[0] == batch
    
    def test_with_padding_needed(self):
        """Test with sequence length not divisible by window size."""
        batch = 2
        seq_len = 30  # Not divisible by 8
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        x = torch.randn(batch, seq_len, embed_dim)
        output, _ = attn(x, x, x)
        
        assert output.shape == (batch, seq_len, embed_dim)
    
    def test_top_k_larger_than_windows(self):
        """Test when top_k is larger than number of windows."""
        batch = 2
        seq_len = 16
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,  # 2 windows
            top_k=10,  # More than available windows
        )
        
        x = torch.randn(batch, seq_len, embed_dim)
        output, _ = attn(x, x, x)
        
        # Should handle gracefully
        assert output.shape == (batch, seq_len, embed_dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        x = torch.randn(batch, seq_len, embed_dim, requires_grad=True)
        output, _ = attn(x, x, x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_parameter_count(self):
        """Test that module has expected number of parameter groups."""
        embed_dim = 64
        
        attn = DCMMSRAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        # Should have Q, K, V projections + output projection + temperature
        param_groups = list(attn.parameters())
        
        # Count parameters: q_proj (weight, bias), k_proj, v_proj, out_proj, temperature
        assert len(param_groups) >= 9  # 4 projections with weights and biases + temp
    
    def test_extra_repr(self):
        """Test string representation."""
        attn = DCMMSRAttention(
            embed_dim=64,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        repr_str = attn.extra_repr()
        assert "embed_dim=64" in repr_str
        assert "num_heads=4" in repr_str
        assert "window_size=8" in repr_str
        assert "top_k=2" in repr_str


class TestDCMMSRSelfAttention:
    """Tests for DCMMSRSelfAttention module."""
    
    def test_self_attention(self):
        """Test self-attention interface."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        
        attn = DCMMSRSelfAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        x = torch.randn(batch, seq_len, embed_dim)
        output, _ = attn(x)
        
        assert output.shape == (batch, seq_len, embed_dim)
    
    def test_self_attention_with_return(self):
        """Test self-attention with attention return."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        
        attn = DCMMSRSelfAttention(
            embed_dim=embed_dim,
            num_heads=4,
            window_size=8,
            top_k=2,
        )
        
        x = torch.randn(batch, seq_len, embed_dim)
        output, attention_info = attn(x, return_attention=True)
        
        assert output.shape == (batch, seq_len, embed_dim)
        assert attention_info is not None


class TestDCMMSRIntegration:
    """Integration tests for DCM-MSR attention."""
    
    def test_in_transformer_block(self):
        """Test using DCM-MSR in a transformer-like block."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        
        class SimpleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = DCMMSRSelfAttention(
                    embed_dim=embed_dim,
                    num_heads=4,
                    window_size=8,
                    top_k=2,
                )
                self.norm1 = nn.LayerNorm(embed_dim)
                self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                )
                self.norm2 = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                attn_out, _ = self.attn(self.norm1(x))
                x = x + attn_out
                x = x + self.ffn(self.norm2(x))
                return x
        
        block = SimpleBlock()
        x = torch.randn(batch, seq_len, embed_dim)
        output = block(x)
        
        assert output.shape == (batch, seq_len, embed_dim)
        assert not torch.isnan(output).any()
    
    def test_multiple_layers(self):
        """Test stacking multiple DCM-MSR layers."""
        batch = 2
        seq_len = 32
        embed_dim = 64
        num_layers = 3
        
        layers = nn.ModuleList([
            DCMMSRSelfAttention(
                embed_dim=embed_dim,
                num_heads=4,
                window_size=8,
                top_k=2,
            )
            for _ in range(num_layers)
        ])
        
        x = torch.randn(batch, seq_len, embed_dim)
        
        for layer in layers:
            x, _ = layer(x)
        
        assert x.shape == (batch, seq_len, embed_dim)
        assert not torch.isnan(x).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
