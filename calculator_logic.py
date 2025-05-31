# calculator_logic.py
from transformers import AutoConfig

def get_bytes_per_element_kv(precision_str):
    """Converts KV cache precision string to bytes per element."""
    precision_str_upper = precision_str.upper()
    if precision_str_upper in ("FP16", "BF16"):
        return 2
    elif "FP8" in precision_str_upper:  # Covers FP8_E4M3, FP8_E5M2
        return 1
    elif precision_str_upper == "AUTO":
        # Defaulting 'auto' to 2 bytes (FP16/BF16) as a common case for unquantized models.
        return 2
    else:
        # Should be caught by Streamlit selectbox, but as a safeguard:
        raise ValueError(f"Unsupported KV Cache Precision: {precision_str}")

def calculate_max_context_window(system_params, model_arch_params):
    """
    Calculates the maximum estimated context window.

    Args:
        system_params (dict): Dictionary of system parameters.
            Expected keys: 'vram_per_gpu_gb', 'num_gpus', 'model_vram_load_gb',
                           'kv_cache_precision', 'gpu_memory_utilization_factor',
                           'vllm_fixed_internal_overhead_gb', 'dynamic_overhead_percentage_decimal'.
        model_arch_params (dict): Dictionary of model architectural parameters.
            Expected keys: 'num_hidden_layers', 'num_kv_heads', 'head_dim'.

    Returns:
        tuple: (max_context_window_tokens, intermediate_results_dict)
               Returns (0, intermediate_results) if calculation is not possible.
    """
    bytes_per_element_kv = get_bytes_per_element_kv(system_params['kv_cache_precision'])

    # KV Cache Memory per Token
    # = 2 (for K and V) * num_hidden_layers * num_kv_heads * head_dim * bytes_per_element_kv
    if not all([model_arch_params['num_hidden_layers'], model_arch_params['num_kv_heads'], model_arch_params['head_dim']]):
        total_kv_mem_per_token_bytes = 0 # Avoid error if params are zero/None
    else:
        total_kv_mem_per_token_bytes = (2 *
                                    model_arch_params['num_hidden_layers'] *
                                    model_arch_params['num_kv_heads'] *
                                    model_arch_params['head_dim'] *
                                    bytes_per_element_kv)

    # Available VRAM for KV Cache
    total_system_vram_gb = system_params['vram_per_gpu_gb'] * system_params['num_gpus']
    total_system_vram_bytes = total_system_vram_gb * (1024**3)
    model_weights_vram_bytes = system_params['model_vram_load_gb'] * (1024**3)

    # VRAM managed by framework (e.g., vLLM)
    effective_framework_vram_bytes = total_system_vram_bytes * system_params['gpu_memory_utilization_factor']

    # VRAM remaining after model load AND fixed vLLM overhead, within the framework's managed pool
    vllm_fixed_internal_overhead_gb = system_params.get('vllm_fixed_internal_overhead_gb', 0.0)
    vllm_fixed_internal_overhead_bytes = vllm_fixed_internal_overhead_gb * (1024**3)

    vram_after_weights_and_fixed_overhead_bytes = effective_framework_vram_bytes - model_weights_vram_bytes - vllm_fixed_internal_overhead_bytes
    
    if vram_after_weights_and_fixed_overhead_bytes < 0:
        vram_after_weights_and_fixed_overhead_bytes = 0 # No space left

    # VRAM available specifically for KV cache data, after accounting for DYNAMIC overheads
    dynamic_overhead_percentage_decimal = system_params['dynamic_overhead_percentage_decimal']
    available_vram_for_kv_cache_bytes = (vram_after_weights_and_fixed_overhead_bytes *
                                         (1 - dynamic_overhead_percentage_decimal))
    if available_vram_for_kv_cache_bytes < 0:
        available_vram_for_kv_cache_bytes = 0


    intermediate_results = {
        "bytes_per_element_kv": bytes_per_element_kv,
        "total_kv_mem_per_token_bytes": total_kv_mem_per_token_bytes,
        "total_system_vram_gb": total_system_vram_gb,
        "effective_framework_vram_gb": effective_framework_vram_bytes / (1024**3),
        "vllm_fixed_internal_overhead_gb": vllm_fixed_internal_overhead_bytes / (1024**3),
        "vram_after_weights_and_fixed_overhead_gb": vram_after_weights_and_fixed_overhead_bytes / (1024**3),
        "available_vram_for_kv_cache_gb": available_vram_for_kv_cache_bytes / (1024**3)
    }

    if total_kv_mem_per_token_bytes == 0:
        max_context_window_tokens = 0 # Cannot calculate if per-token KV size is zero
    else:
        max_context_window_tokens = int(available_vram_for_kv_cache_bytes / total_kv_mem_per_token_bytes)

    if max_context_window_tokens < 0: # Should be caught by available_vram_for_kv_cache_bytes check
        max_context_window_tokens = 0

    return max_context_window_tokens, intermediate_results
# Custom Exceptions for Hugging Face model fetching
class ModelNotFoundError(Exception):
    """Custom exception for when a model is not found on Hugging Face Hub."""
    pass

class ConfigAttributeMissingError(Exception):
    """Custom exception for when a required attribute is missing from model config."""
    pass

def fetch_hf_model_params(model_name_or_path: str) -> dict:
    """
    Fetches model configuration from Hugging Face Hub and extracts key parameters.

    Args:
        model_name_or_path (str): The name or path of the model on Hugging Face Hub.

    Returns:
        dict: A dictionary containing the extracted model parameters:
              num_hidden_layers, hidden_size, num_attention_heads,
              num_key_value_heads, head_dim.

    Raises:
        ModelNotFoundError: If the model cannot be found or accessed.
        ConfigAttributeMissingError: If essential parameters are missing from the config.
        Exception: For other unexpected errors during fetching or parsing.
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
    except OSError as e:
        raise ModelNotFoundError(
            f"Could not fetch model '{model_name_or_path}'. "
            f"Ensure it's a valid Hugging Face model name and you have internet access. Original error: {e}"
        )
    except Exception as e: # Catch other potential errors from from_pretrained
        raise ModelNotFoundError(
            f"An unexpected error occurred while trying to fetch model '{model_name_or_path}'. Original error: {e}"
        )

    params = {}
    try:
        # Directly attempt to access mandatory fields first
        params['num_hidden_layers'] = config.num_hidden_layers
        params['hidden_size'] = config.hidden_size
        params['num_attention_heads'] = config.num_attention_heads
        
        # num_key_value_heads: often defaults to num_attention_heads if not specified (MHA models)
        params['num_key_value_heads'] = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        
        # head_dim: Try to get it directly, then calculate as a fallback.
        params['head_dim'] = getattr(config, 'head_dim', None)
        if params['head_dim'] is None:
            if params['hidden_size'] is not None and params['num_attention_heads'] is not None and params['num_attention_heads'] > 0:
                params['head_dim'] = params['hidden_size'] // params['num_attention_heads']
            else:
                # This case means head_dim is not in config and cannot be calculated from other essential params.
                raise ConfigAttributeMissingError(
                    f"'head_dim' is missing and cannot be calculated for model '{model_name_or_path}'. "
                    f"Ensure 'hidden_size' and 'num_attention_heads' (non-zero) are present."
                )
        
        # Final check for any None values in critical parameters that should have been resolved.
        # num_hidden_layers, hidden_size, num_attention_heads are expected to be direct attributes.
        # num_key_value_heads and head_dim have fallback logic.
        for key, value in params.items():
            if value is None:
                # This check is more for attributes that didn't have a fallback or whose fallback also resulted in None.
                # num_hidden_layers, hidden_size, num_attention_heads should have raised AttributeError if missing.
                # This primarily catches if head_dim calculation failed silently (though logic above tries to prevent it).
                 raise ConfigAttributeMissingError(
                    f"Essential parameter '{key}' could not be determined for model '{model_name_or_path}'."
                )

    except AttributeError as e:
        # This will catch if num_hidden_layers, hidden_size, or num_attention_heads are missing.
        original_attr_name = str(e).split("'")[-2] # Attempt to get the attribute name from the error
        raise ConfigAttributeMissingError(
            f"Configuration for model '{model_name_or_path}' is missing an expected attribute: '{original_attr_name}'. Original error: {e}"
        )
    except Exception as e: # Catch other unexpected errors during parsing
        # This could be DivisionByZero if num_attention_heads is 0 and head_dim calculation is attempted.
        raise Exception(
            f"An unexpected error occurred while parsing config for model '{model_name_or_path}'. Original error: {e}"
        )
            
    return params