"""
Tests for windowing utilities.
"""

import pytest
import torch

from dcm_msr.windowing import (
    create_windows,
    merge_windows,
    select_windows,
    create_window_position_encoding,
)


class TestCreateWindows:
    """Tests for create_windows."""
    
    def test_exact_divisible(self):
        """Test when sequence length is exactly divisible by window size."""
        batch = 2
        seq_len = 16
        dim = 8
        window_size = 4
        
        tokens = torch.randn(batch, seq_len, dim)
        windowed, mask, orig_len = create_windows(tokens, window_size)
        
        assert windowed.shape == (batch, 4, window_size, dim)
        assert mask.shape == (batch, 4, window_size)
        assert orig_len == seq_len
        assert torch.all(mask)  # All tokens valid
    
    def test_with_padding(self):
        """Test when sequence length requires padding."""
        batch = 2
        seq_len = 15
        dim = 8
        window_size = 4
        
        tokens = torch.randn(batch, seq_len, dim)
        windowed, mask, orig_len = create_windows(tokens, window_size)
        
        # Should pad to 16 (4 windows of size 4)
        assert windowed.shape == (batch, 4, window_size, dim)
        assert orig_len == seq_len
        
        # Last window should have one padded token
        last_window_mask = mask[:, -1, :]
        assert torch.all(last_window_mask[:, :3])  # First 3 valid
        assert not torch.any(last_window_mask[:, 3:])  # Last 1 padded
    
    def test_preserves_values(self):
        """Test that original values are preserved."""
        tokens = torch.arange(12).float().view(1, 12, 1)
        windowed, _, _ = create_windows(tokens, window_size=3)
        
        # Should have 4 windows of size 3
        assert windowed.shape == (1, 4, 3, 1)
        
        # Check values
        expected = torch.arange(12).float().view(1, 4, 3, 1)
        assert torch.allclose(windowed, expected)


class TestMergeWindows:
    """Tests for merge_windows."""
    
    def test_roundtrip_exact(self):
        """Test create -> merge roundtrip for exact division."""
        batch = 2
        seq_len = 16
        dim = 8
        window_size = 4
        
        original = torch.randn(batch, seq_len, dim)
        windowed, _, orig_len = create_windows(original, window_size)
        merged = merge_windows(windowed, orig_len)
        
        assert torch.allclose(merged, original)
    
    def test_roundtrip_with_padding(self):
        """Test create -> merge roundtrip with padding."""
        batch = 2
        seq_len = 13
        dim = 8
        window_size = 4
        
        original = torch.randn(batch, seq_len, dim)
        windowed, _, orig_len = create_windows(original, window_size)
        merged = merge_windows(windowed, orig_len)
        
        assert merged.shape == (batch, seq_len, dim)
        assert torch.allclose(merged, original)


class TestSelectWindows:
    """Tests for select_windows."""
    
    def test_select_subset(self):
        """Test selecting a subset of windows."""
        batch = 2
        num_windows = 8
        window_size = 4
        dim = 16
        k = 3
        
        windowed = torch.randn(batch, num_windows, window_size, dim)
        indices = torch.randint(0, num_windows, (batch, k))
        
        selected = select_windows(windowed, indices)
        
        assert selected.shape == (batch, k, window_size, dim)
        
        # Verify selection is correct
        for b in range(batch):
            for i in range(k):
                idx = indices[b, i].item()
                assert torch.allclose(selected[b, i], windowed[b, idx])
    
    def test_select_with_heads(self):
        """Test selecting windows with head dimension."""
        batch = 2
        heads = 4
        num_windows = 8
        window_size = 4
        dim = 16
        k = 3
        
        windowed = torch.randn(batch, num_windows, window_size, dim)
        indices = torch.randint(0, num_windows, (batch, heads, k))
        
        selected = select_windows(windowed, indices)
        
        assert selected.shape == (batch, heads, k, window_size, dim)


class TestPositionEncoding:
    """Tests for create_window_position_encoding."""
    
    def test_output_shape(self):
        """Test output shape is correct."""
        num_windows = 4
        window_size = 8
        dim = 64
        
        pe = create_window_position_encoding(num_windows, window_size, dim)
        
        assert pe.shape == (num_windows * window_size, dim)
    
    def test_different_positions(self):
        """Test that different positions have different encodings."""
        pe = create_window_position_encoding(4, 4, 32)
        
        # Check that adjacent positions are different
        for i in range(pe.shape[0] - 1):
            assert not torch.allclose(pe[i], pe[i + 1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
