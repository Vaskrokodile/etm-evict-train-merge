#!/usr/bin/env python
"""Patch etm_v5.py to pre-stack cache into contiguous tensors and use efficient batching."""
import re

with open('/root/etm/src/etm_v5.py') as f:
    code = f.read()

# Replace the cache storage in cache_features: store as pre-stacked tensors
# Instead of list of (tok, d) tensors, stack into (n_samples, tok, d) at the end

# 1. Add a post-processing step after caching to stack lists into single tensors
old_cache_done = """    print(f"\\n  [Cache] DONE: {n_cached} samples cached in {t_cache:.1f}s")"""
new_cache_done = """    # Pre-stack cache into contiguous tensors for fast indexing
    print(f"  [Cache] pre-stacking cache into contiguous tensors...")
    for name in cache:
        if cache[name]['x']:
            cache[name]['x'] = torch.stack(cache[name]['x'])  # (n, tok, d)
            cache[name]['grad_h'] = torch.stack(cache[name]['grad_h'])  # (n, tok, d)
    print(f"  [Cache] stacking done")

    print(f"\\n  [Cache] DONE: {n_cached} samples cached in {t_cache:.1f}s")"""

code = code.replace(old_cache_done, new_cache_done)

# 2. Replace the batch loading in train_mse to use tensor indexing instead of list comprehension
old_load = """            # Gather batch from CPU cache, move to GPU
            x = torch.stack([c["x"][i] for i in batch_idx]).cuda().float()  # (b, tok, d_in)
            g = torch.stack([c["grad_h"][i] for i in batch_idx]).cuda().float()  # (b, tok, d_out)"""
new_load = """            # Gather batch from pre-stacked CPU cache, move to GPU
            idx_tensor = torch.tensor(batch_idx)
            x = c['x'][idx_tensor].cuda().float()  # (b, tok, d_in)
            g = c['grad_h'][idx_tensor].cuda().float()  # (b, tok, d_out)"""

code = code.replace(old_load, new_load)

# 3. Fix the norm display (was showing per-element norm, should be per-tensor)
old_norm = """            total_u_norm += u.detach().norm().item() / u.numel()
            total_g_norm += g.detach().norm().item() / g.numel()"""
new_norm = """            total_u_norm += u.detach().norm().item() / math.sqrt(u.numel())
            total_g_norm += g.detach().norm().item() / math.sqrt(g.numel())"""
code = code.replace(old_norm, new_norm)

with open('/root/etm/src/etm_v5.py', 'w') as f:
    f.write(code)
print('Patch applied successfully')
