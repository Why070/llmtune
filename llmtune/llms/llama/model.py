import torch
import torch.nn as nn
import subprocess

from llmtune.utils import find_layers
from llmtune.engine.quant.converter import make_quant

def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    return result.stdout


def load_llama(llm_config, checkpoint):
    import transformers, accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    
    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(llm_config.hf_config_name)
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(model, layers, llm_config.bits)
    model = accelerate.load_checkpoint_and_dispatch(
        model=model, checkpoint=checkpoint, device_map='auto'
    )
    model.seqlen = 2048
    print("\033[1;31mMemory occupied before 加载 tokenizer:\033[0m:")
    print(get_gpu_memory_usage())
    
    tokenizer = LlamaTokenizer.from_pretrained(llm_config.hf_tokenizer_config)
    tokenizer.truncation_side = 'left'

    print("\033[1;31mMemory occupied after 加载 tokenizer:\033[0m:")
    print(get_gpu_memory_usage())
   
   
   
    return model, tokenizer
