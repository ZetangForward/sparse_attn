import torch
from typing import List, Tuple, Optional, Dict
from transformers.cache_utils import Cache


def locret_kv_eviction_for_cache(
    key_cache: List[torch.Tensor],
    value_cache: List[torch.Tensor],
    scores: List[torch.Tensor],
    attentions: List[torch.Tensor],
    input_len: int,
    local_len: int,
    budget_size: int,
    stabilizers: int,
    start_index: int,
    end_index: int,
    n_layers: int,
    key_devices: Optional[Dict[int, torch.device]] = None,
    value_devices: Optional[Dict[int, torch.device]] = None,
):
    """
    Locret KV eviction function adapted for separate key/value cache lists.
    This version works directly with the cache structure instead of tuples.
    
    Args:
        key_devices: Dictionary mapping layer_idx to target device for keys.
        value_devices: Dictionary mapping layer_idx to target device for values.
    
    Returns:
        scores: Updated attention scores
        key_cache: Modified key cache (same objects, modified in-place)  
        value_cache: Modified value cache (same objects, modified in-place)
    """
    # Initialize scores if empty
    if len(scores) == 0:
        scores = [None] * n_layers
    
    # Get shape information from the first key tensor
    kv_shape = key_cache[0].shape
    
    for layer in range(n_layers):
        # Determine target devices for this layer
        target_key_device = None
        target_value_device = None
        if key_devices and layer in key_devices:
            target_key_device = key_devices[layer]
        elif len(key_cache) > layer:
            target_key_device = key_cache[layer].device
            
        if value_devices and layer in value_devices:
            target_value_device = value_devices[layer]
        elif len(value_cache) > layer:
            target_value_device = value_cache[layer].device
        
        # Ensure all tensors for this layer are on the correct devices
        if target_key_device is not None:
            if key_cache[layer].device != target_key_device:
                key_cache[layer] = key_cache[layer].to(target_key_device)
            
        if target_value_device is not None:
            if value_cache[layer].device != target_value_device:
                value_cache[layer] = value_cache[layer].to(target_value_device)
        
        # Get attention tensor for this layer and move to correct device if needed
        attention_tensor = attentions[layer]
        if target_key_device is not None and attention_tensor.device != target_key_device:
            attention_tensor = attention_tensor.to(target_key_device)
        
        # Update or initialize scores for this layer (use key device for scores)
        if scores[layer] is not None:
            # Ensure scores are on the correct device (use key device)
            if target_key_device is not None and scores[layer].device != target_key_device:
                scores[layer] = scores[layer].to(target_key_device)
            
            scores[layer] = torch.cat(
                (scores[layer], attention_tensor[:end_index-start_index]),
                dim=-2
            )
        else:
            scores[layer] = attention_tensor[:end_index-start_index]
        
        n_selected = min(budget_size, scores[layer].shape[-2])
        
        sc = scores[layer].clone()
        if end_index < input_len - local_len:
            sc[:, -stabilizers:, :] = torch.finfo(sc.dtype).max  # always keep `stabilizers` last kvs if this is not the last chunk
        indices = torch.topk(sc[0, :, :], k=n_selected, dim=-2).indices

        indices = indices.transpose(0, 1).sort().values  # Sort the indices in ascending order
        scores[layer] = torch.gather(
            scores[layer],
            1,
            indices.transpose(0, 1).unsqueeze(0)  # back to the original shape after sorting
        )
        
        # Create indices tensors for key and value separately (in case they have different shapes)
        key_indices = indices.unsqueeze(0).unsqueeze(-1).repeat(kv_shape[0], 1, 1, kv_shape[3])
        value_indices = indices.unsqueeze(0).unsqueeze(-1).repeat(kv_shape[0], 1, 1, kv_shape[3])
        
        # Move indices to the correct devices
        if target_key_device is not None and key_indices.device != target_key_device:
            key_indices = key_indices.to(target_key_device)
        if target_value_device is not None and value_indices.device != target_value_device:
            value_indices = value_indices.to(target_value_device)
        
        # Apply eviction directly to the cache tensors
        key_cache[layer] = torch.gather(key_cache[layer], 2, key_indices)
        value_cache[layer] = torch.gather(value_cache[layer], 2, value_indices)
    
    return scores, key_cache, value_cache


class LocretCache(Cache):
    """
    Locret KV cache that maintains accurate position tracking even after KV eviction.
    
    This cache addresses the critical issue where KV cache pruning breaks position ID calculation
    during generation. It tracks the true sequence length independently of the cached KV size.
    """
    
    def __init__(
        self, 
        budget_size: int = 6000,
        local_len: int = 1000,
        stabilizers: int = 10,
        num_layers: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        # Track REAL sequence position - this is the key to fixing the position issue
        self._seen_tokens = 0
        
        # Standard cache storage
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        # Locret-specific parameters
        self.budget_size = budget_size
        self.local_len = local_len
        self.stabilizers = stabilizers
        self.num_layers = num_layers  # Can be None, will be inferred during usage
        
        # Track devices for each layer separately - keys and values can be on different devices
        self.key_devices: Dict[int, torch.device] = {}
        self.value_devices: Dict[int, torch.device] = {}
        
        # Track attention scores for locret eviction - only populated during eviction
        self.scores: List[torch.Tensor] = []
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key-value states.
        
        This method focuses purely on KV storage and position tracking.
        Attention scores are handled separately during eviction.
        """
        if cache_kwargs is None:
            cache_kwargs = {}
            
        # Store the original input devices to return tensors on the same devices
        input_key_device = key_states.device
        input_value_device = value_states.device
            
        # Track real sequence length (only update once per forward pass)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # Initialize cache for this layer if needed
        if len(self.key_cache) == layer_idx:
            # Log and store the devices for this layer the first time
            self.key_devices[layer_idx] = input_key_device
            self.value_devices[layer_idx] = input_value_device
            
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Get the expected devices for this layer
            expected_key_device = self.key_devices[layer_idx]
            expected_value_device = self.value_devices[layer_idx]
            
            # Cast incoming tensors to the layer's devices if needed
            if key_states.device != expected_key_device:
                key_states = key_states.to(expected_key_device)
            if value_states.device != expected_value_device:
                value_states = value_states.to(expected_value_device)
            
            # Ensure existing cache is on the correct devices (defensive)
            if self.key_cache[layer_idx].device != expected_key_device:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(expected_key_device)
            if self.value_cache[layer_idx].device != expected_value_device:
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(expected_value_device)
            
            # Concatenate new KVs with existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            
        # Return cached states on the same devices as the input (what caller expects)
        returned_keys = self.key_cache[layer_idx]
        returned_values = self.value_cache[layer_idx]
        
        # Cast back to input devices if different from layer devices
        if returned_keys.device != input_key_device:
            returned_keys = returned_keys.to(input_key_device)
        if returned_values.device != input_value_device:
            returned_values = returned_values.to(input_value_device)
            
        return returned_keys, returned_values
        
    def apply_locret_eviction(
        self, 
        attentions: List[torch.Tensor], 
        input_len: int,
        start_index: int,
        end_index: int,
        budget_size: Optional[int] = None
    ):
        """
        Apply locret eviction algorithm to compress the cache.
        
        Args:
            attentions: List of attention tensors from the model
            input_len: Total input length being processed
            start_index: Start index of current chunk
            end_index: End index of current chunk
            budget_size: Override the default budget size for this eviction (optional)
        
        This is called externally when attention information is available,
        typically after processing a chunk during prefill.
        """
        # Use provided budget_size or fall back to default
        if budget_size is None:
            budget_size = self.budget_size
            
        # Infer num_layers if not provided during initialization
        if self.num_layers is None:
            self.num_layers = len(self.key_cache)
            
        self.scores, self.key_cache, self.value_cache = locret_kv_eviction_for_cache(
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scores=self.scores,
            attentions=attentions,
            input_len=input_len,
            local_len=self.local_len,
            budget_size=budget_size,
            stabilizers=self.stabilizers,
            start_index=start_index,
            end_index=end_index,
            n_layers=self.num_layers,
            key_devices=self.key_devices,
            value_devices=self.value_devices,
        )
        
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Return the REAL sequence length, not the cached KV size.
        
        This is the critical fix - we return _seen_tokens instead of
        the actual cache size, ensuring position IDs are calculated correctly.
        """
        return self._seen_tokens
        
    def to_legacy_cache(self) -> Tuple:
        """Convert to legacy cache format"""
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache
        
    @classmethod
    def from_legacy_cache(
        cls, 
        past_key_values: Tuple, 
        budget_size: int = 6000,
        local_len: int = 1000,
        stabilizers: int = 10,
        **kwargs
    ) -> "LocretCache":
        """Create LocretCache from legacy cache format"""
        cache = cls(
            budget_size=budget_size,
            local_len=local_len,
            stabilizers=stabilizers,
            num_layers=len(past_key_values),
            **kwargs
        )
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Record the devices for this layer (keys and values can be on different devices)
            cache.key_devices[layer_idx] = k.device
            cache.value_devices[layer_idx] = v.device
            
            cache.key_cache.append(k)
            cache.value_cache.append(v)
            
        # Estimate seen tokens from cache size (not perfect but best we can do)
        if len(cache.key_cache) > 0:
            cache._seen_tokens = cache.key_cache[0].shape[-2]
            
        return cache
        
    def clear(self):
        """Clear the cache"""
        self.key_cache.clear()
        self.value_cache.clear()
        self.scores.clear()
        self.key_devices.clear()
        self.value_devices.clear()
        self._seen_tokens = 0
        
    def crop(self, maximum_cache_length: int):
        """Crop cache to maximum length (used by some generation strategies)"""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, -maximum_cache_length:, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, -maximum_cache_length:, :]
                
        # Don't modify _seen_tokens here - we want to maintain the real position
        # This might cause issues if called during generation, but cropping should
        # ideally not be used with locret
