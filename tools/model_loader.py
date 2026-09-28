import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model

def write_to_file(tensor, f):
    tensor.cpu().detach().numpy().astype(np.float32).tofile(f)

def main(): 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(root_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "weights.bin")

    print("Loading model for GPT2...")
    model = GPT2Model.from_pretrained('gpt2')
    state = model.state_dict()
    config = model.config

    with open(output_path, 'wb') as f:
        # header to store model information
        print("Exporting config header onto bin file...")
        header = np.zeros(256, dtype=np.int32)
        header[0] = config.vocab_size
        header[1] = config.n_positions
        header[2] = config.n_layer
        header[3] = config.n_head
        header[4] = config.n_embd
        header.tofile(f)
         
        print("Exporting weights onto bin file...")
        # word token embedding weights and word position embedding weights
        write_to_file(state['wte.weight'], f)
        write_to_file(state['wpe.weight'], f)

        for layer in range(model.config.n_layer):
            prefix = f'h.{layer}'

            # layer normalization weights and bias
            write_to_file(state[f'{prefix}.ln_1.weight'], f)
            write_to_file(state[f'{prefix}.ln_1.bias'], f)

            # attention weights and bias
            weights_Q, weights_K, weights_V = state[f'{prefix}.attn.c_attn.weight'].split(model.config.n_embd, dim=1)
            bias_Q, bias_K, bias_V = state[f'{prefix}.attn.c_attn.bias'].split(model.config.n_embd, dim=0)

            for tensor in [weights_Q, weights_K, weights_V, bias_Q, bias_K, bias_V]:
                write_to_file(tensor, f)

            write_to_file(state[f'{prefix}.attn.c_proj.weight'], f)
            write_to_file(state[f'{prefix}.attn.c_proj.bias'], f)

            # layer normalization weights and bias
            write_to_file(state[f'{prefix}.ln_2.weight'], f)
            write_to_file(state[f'{prefix}.ln_2.bias'], f)

            # multilayer perceptron weights and bias
            write_to_file(state[f'{prefix}.mlp.c_fc.weight'], f)
            write_to_file(state[f'{prefix}.mlp.c_fc.bias'], f)
            write_to_file(state[f'{prefix}.mlp.c_proj.weight'], f)
            write_to_file(state[f'{prefix}.mlp.c_proj.bias'], f)

        # final layer normalization weights and bias
        write_to_file(state['ln_f.weight'], f)
        write_to_file(state['ln_f.bias'], f)

    print("Loading tokenizer for GPT2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if __name__ == "__main__":
    main()