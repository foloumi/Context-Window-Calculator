"""
Streamlit web application for estimating the maximum context window for LLMs.

This application allows users to input system hardware parameters (VRAM, number of GPUs),
model architectural details (number of layers, heads, head dimension), and vLLM
specific configurations to calculate an estimated maximum context window in tokens.
It can fetch model parameters directly from Hugging Face model names.
"""
# app.py
import streamlit as st
from calculator_logic import calculate_max_context_window, fetch_hf_model_params, ModelNotFoundError, ConfigAttributeMissingError

# --- Page Configuration ---
st.set_page_config(page_title="LLM Context Window Calculator", layout="wide")
st.title("LLM Context Window Calculator")
st.markdown("""
This tool estimates the maximum context window (in tokens) an LLM can support
based on your hardware configuration, model architecture, and inference settings.
Fill in the parameters below and click "Calculate".
""")

# --- Session State Initialization ---
if 'fetched_params' not in st.session_state:
    st.session_state.fetched_params = None
if 'params_are_read_only' not in st.session_state:
    st.session_state.params_are_read_only = False
if 'hf_model_name_input' not in st.session_state: # To preserve text input across reruns
    st.session_state.hf_model_name_input = ""

# --- Input Sections ---
col1, col2 = st.columns(2)

with col1:
    st.header("System Parameters")
    vram_per_gpu_gb = st.number_input("VRAM per GPU (GB)", min_value=1.0, value=24.0, step=1.0)
    num_gpus = st.number_input("Number of GPUs", min_value=1, value=1, step=1)
    model_vram_load_gb = st.number_input("Model VRAM Load (GB)", min_value=0.1, value=18.0, step=0.1,
                                         help="Memory consumed by the loaded model weights.")
    vllm_fixed_internal_overhead_gb = st.number_input("vLLM Fixed Internal Overhead (GB)", min_value=0.0, value=0.5, step=0.1, help="Estimated fixed VRAM (GB) for vLLM internals (scheduler, PagedAttention metadata, etc.), independent of sequence length or batch size. Subtracted before dynamic overhead.")
    kv_cache_precision_options = ['auto', 'FP16', 'BF16', 'FP8']
    kv_cache_precision = st.selectbox("KV Cache Precision", options=kv_cache_precision_options, index=0,
                                      help="'auto' typically defaults to model's unquantized type (e.g., FP16/BF16, 2 bytes). FP8 uses 1 byte.")
    gpu_memory_utilization_factor = st.slider("GPU Memory Utilization Factor", min_value=0.1, max_value=1.0, value=0.90, step=0.01,
                                             help="Proportion of total GPU VRAM that vLLM aims to manage (matches vLLM's 'gpu_memory_utilization' parameter, typically 0.90). This pool covers model weights, KV cache, activations, and vLLM internal data.")
    dynamic_overhead_percentage = st.slider("Dynamic Overhead Percentage (%)", min_value=0, max_value=50, value=15, step=1,
                                    help="Percentage of VRAM *remaining after model load and fixed vLLM overhead* reserved for sequence-length dependent activations and other dynamic buffers.")

with col2:
    st.header("Model Architectural Parameters")
    
    # Use session state to preserve input value across reruns
    st.session_state.hf_model_name_input = st.text_input(
        "Hugging Face Model Name (e.g., meta-llama/Llama-2-7b-hf)",
        value=st.session_state.hf_model_name_input,
        key="hf_model_name_text_field" 
    )
    
    col_fetch, col_clear = st.columns(2)
    with col_fetch:
        fetch_button = st.button("Fetch Model Parameters", use_container_width=True, key="fetch_params_button")
    with col_clear:
        clear_button = st.button("Clear/Edit Manually", use_container_width=True, key="clear_params_button")

    if fetch_button and st.session_state.hf_model_name_input:
        with st.spinner(f"Fetching parameters for {st.session_state.hf_model_name_input}..."):
            try:
                params = fetch_hf_model_params(st.session_state.hf_model_name_input)
                st.session_state.fetched_params = params
                st.session_state.params_are_read_only = True
                st.success("Parameters fetched successfully!")
            except (ModelNotFoundError, ConfigAttributeMissingError) as e:
                st.error(str(e))
                st.session_state.fetched_params = None 
                st.session_state.params_are_read_only = False
            except Exception as e: 
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.fetched_params = None 
                st.session_state.params_are_read_only = False

    if clear_button:
        st.session_state.fetched_params = None
        st.session_state.params_are_read_only = False
        # Optionally clear the text input: st.session_state.hf_model_name_input = ""
        st.experimental_rerun()

    default_layers_val = 32 
    default_kv_heads_val = 8
    default_head_dim_val = 128

    current_num_hidden_layers = default_layers_val
    current_num_kv_heads = default_kv_heads_val
    current_head_dim = default_head_dim_val

    if st.session_state.fetched_params:
        current_num_hidden_layers = st.session_state.fetched_params.get('num_hidden_layers', default_layers_val)
        current_num_kv_heads = st.session_state.fetched_params.get('num_key_value_heads', default_kv_heads_val)
        current_head_dim = st.session_state.fetched_params.get('head_dim', default_head_dim_val)
        
        fetched_hidden_size = st.session_state.fetched_params.get('hidden_size')
        if fetched_hidden_size:
            st.caption(f"Fetched hidden_size: {fetched_hidden_size}")
        fetched_num_attn_heads = st.session_state.fetched_params.get('num_attention_heads')
        if fetched_num_attn_heads:
            st.caption(f"Fetched num_attention_heads (Q heads): {fetched_num_attn_heads}")

    num_hidden_layers = st.number_input(
        "Number of Layers (num_hidden_layers)", 
        min_value=1, 
        value=current_num_hidden_layers, 
        step=1,
        disabled=st.session_state.params_are_read_only,
        key="num_hidden_layers_input"
    )
    num_kv_heads = st.number_input( 
        "Number of KV Heads (num_key_value_heads)", 
        min_value=1, 
        value=current_num_kv_heads, 
        step=1,
        help="For GQA/MQA, this is the number of unique K/V head groups. Fetched as 'num_key_value_heads'.",
        disabled=st.session_state.params_are_read_only,
        key="num_kv_heads_input"
    )
    head_dim = st.number_input(
        "Head Dimension (head_dim)", 
        min_value=1, 
        value=current_head_dim, 
        step=1,
        help="Dimension of each K/V vector in an attention head. Fetched as 'head_dim'.",
        disabled=st.session_state.params_are_read_only,
        key="head_dim_input"
    )
    st.subheader("vLLM Specifics (Informational)")
    vllm_max_batched_tokens = st.number_input("vLLM Max Batched Tokens (max_num_batched_tokens)", min_value=128, value=2048, step=128, help="Corresponds to vLLM's 'max_num_batched_tokens'. Affects throughput and how vLLM processes long sequences.")
    vllm_kv_cache_block_size = st.number_input("vLLM KV Cache Block Size (tokens)", min_value=1, value=16, step=1, help="The block size (in tokens) used by vLLM's PagedAttention. Typically 16.")

# --- Calculation and Output ---
st.divider()
if st.button("Calculate Maximum Context Window", type="primary", use_container_width=True, key="calculate_button"):
    if not (num_hidden_layers > 0 and num_kv_heads > 0 and head_dim > 0):
        st.error("Model architectural parameters (Layers, KV Heads, Head Dimension) must be greater than zero.")
    else:
        system_params = {
            'vram_per_gpu_gb': vram_per_gpu_gb,
            'num_gpus': num_gpus,
            'model_vram_load_gb': model_vram_load_gb,
            'kv_cache_precision': kv_cache_precision,
            'gpu_memory_utilization_factor': gpu_memory_utilization_factor,
            'vllm_fixed_internal_overhead_gb': vllm_fixed_internal_overhead_gb,
            'dynamic_overhead_percentage_decimal': dynamic_overhead_percentage / 100.0
        }
        model_arch_params = {
            'num_hidden_layers': num_hidden_layers,
            'num_kv_heads': num_kv_heads, 
            'head_dim': head_dim
        }

        max_tokens, intermediates = calculate_max_context_window(system_params, model_arch_params)

        st.subheader("Results")
        if max_tokens > 0:
            st.metric(label="Maximum Estimated Context Window", value=f"{max_tokens:,} tokens")
        else:
            st.error("Could not estimate context window. This might be due to insufficient VRAM after model load and overheads, or invalid model parameters.")

        with st.expander("Show Intermediate Calculation Details"):
            st.write(f"- Bytes per element for KV Cache: `{intermediates['bytes_per_element_kv']}` bytes")
            st.write(f"- Total KV Cache per Token: `{intermediates['total_kv_mem_per_token_bytes']:,}` bytes "
                     f"({intermediates['total_kv_mem_per_token_bytes']/(1024**2):.4f} MB)")
            st.write(f"- Total System VRAM: `{intermediates['total_system_vram_gb']:.2f}` GB")
            st.write(f"- Effective VRAM managed by Framework (after utilization factor): `{intermediates['effective_framework_vram_gb']:.2f}` GB")
            st.write(f"- vLLM Fixed Internal Overhead: `{intermediates['vllm_fixed_internal_overhead_gb']:.2f}` GB")
            st.write(f"- VRAM after Model Load & Fixed Overhead (within utilization): `{intermediates['vram_after_weights_and_fixed_overhead_gb']:.2f}` GB")
            st.write(f"- Available VRAM specifically for KV Cache Data (after dynamic overheads %): "
                     f"`{intermediates['available_vram_for_kv_cache_gb']:.2f}` GB")
            if intermediates['available_vram_for_kv_cache_gb'] <=0:
                 st.warning("Available VRAM for KV cache is zero or negative. Check your VRAM, model load, and overhead settings.")

st.divider()
st.caption("Disclaimer: This calculator provides an estimation. Actual performance may vary based on specific model implementations, framework versions, and other system factors. Always validate with empirical testing.")