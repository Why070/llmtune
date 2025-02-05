import torch
import torch.nn as nn
import subprocess

from llmtune.utils import find_layers
from llmtune.engine.quant.converter import make_quant



def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '-i', '0', '-q', '-d', 'MEMORY'], capture_output=True, text=True)
    return result.stdout

def get_memory():
    return str(torch.cuda.memory_summary()) 





def load_llama(llm_config, checkpoint):
    import transformers, accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    
    print("\033[1;31mMemory occupied before 加载权重:\033[0m:")
    print(get_memory())
    print("\033[1;31mMemory occupied before 加载权重:\033[0m:")
    print(get_gpu_memory_usage())

    temp=torch.tensor([1.0]).cuda()



    
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
        
        
        
        state_dict = model.state_dict() 
        for name, param in state_dict.items():
            print(f"Parameter: {name}, Size: {param.size()}, Type: {param.dtype}")
        
        make_quant(model, layers, llm_config.bits)
        
        state_dict = model.state_dict() 
       
    
    
    model = accelerate.load_checkpoint_and_dispatch(
        model=model, checkpoint=checkpoint, device_map='auto'
    )
    
   
    
    model.seqlen = 2048

    
    
    print("\033[1;31mMemory occupied after 加载权重:\033[0m:")
    print(get_memory())
    print("\033[1;31mMemory occupied after 加载权重:\033[0m:")
    print(get_gpu_memory_usage())

    print("\033[1;31mMemory occupied before tokenizer:\033[0m:")
    print(get_memory())
    tokenizer = LlamaTokenizer.from_pretrained(llm_config.hf_tokenizer_config)
    tokenizer.truncation_side = 'left'
    print("\033[1;31mMemory occupied after 加载tokenizer:\033[0m:")
    print(get_memory())

    
   
   
   
    return model, tokenizer
