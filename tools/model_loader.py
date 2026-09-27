import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model

def main(): 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(root_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "weights.bin")

    print("Loading model for GPT2...")
    model = GPT2Model.from_pretrained('gpt2')
    state = model.state_dict()

    print("Exporting weights onto bin file...")
    with open(output_path, 'wb') as f:
        # word token embedding weights and word position embedding weights
        state['wte.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
        state['wpe.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)

        for layer in range(model.config.n_layer):
            prefix = f'h.{layer}'

            # layer normalization weights and bias
            state[f'{prefix}.ln_1.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
            state[f'{prefix}.ln_1.bias'].cpu().detach().numpy().astype(np.float32).tofile(f)

            # attention weights and bias
            weights_Q, weights_K, weights_V = state[f'{prefix}.attn.c_attn.weight'].split(model.config.n_embd, dim=1)
            bias_Q, bias_K, bias_V = state[f'{prefix}.attn.c_attn.bias'].split(model.config.n_embd, dim=0)

            for tensor in [weights_Q, weights_K, weights_V, bias_Q, bias_K, bias_V]:
                tensor.cpu().detach().numpy().astype(np.float32).tofile(f)

            state[f'{prefix}.attn.c_proj.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
            state[f'{prefix}.attn.c_proj.bias'].cpu().detach().numpy().astype(np.float32).tofile(f)

            # layer normalization weights and bias
            state[f'{prefix}.ln_2.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
            state[f'{prefix}.ln_2.bias'].cpu().detach().numpy().astype(np.float32).tofile(f)

            # multilayer perceptron weights and bias
            state[f'{prefix}.mlp.c_fc.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
            state[f'{prefix}.mlp.c_fc.bias'].cpu().detach().numpy().astype(np.float32).tofile(f)
            state[f'{prefix}.mlp.c_proj.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
            state[f'{prefix}.mlp.c_proj.bias'].cpu().detach().numpy().astype(np.float32).tofile(f)

        # final layer normalization weights and bias
        state['ln_f.weight'].cpu().detach().numpy().astype(np.float32).tofile(f)
        state['ln_f.bias'].cpu().detach().numpy().astype(np.float32).tofile(f)

    print("Loading tokenizer for GPT2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if __name__ == "__main__":
    main()