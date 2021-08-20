# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from tqdm import tqdm
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

# offset target by 1 if labels start from 1
def target_offset(examples):
    examples["label"] = list(map(lambda x: x - 1, examples["label"]))
    return examples

def get_output_file(args, model, start, end):
    suffix = ''
    if args.finetune:
        suffix += '_finetune'
    if args.dataset == 'mnli':
        dataset_str = f"{args.dataset}_{args.mnli_option}_{args.attack_target}"
    else:
        dataset_str = args.dataset
    attack_str = args.adv_loss
    if args.adv_loss == 'cw':
        attack_str += f'_kappa={args.kappa}'

    model_name = model.replace('/', '-')
    output_file = f"{model_name}_{dataset_str}{suffix}_{start}-{end}"
    output_file += f"_iters={args.num_iters}_{attack_str}_lambda_sim={args.lam_sim}_lambda_perp={args.lam_perp}_emblayer={args.embed_layer}_{args.constraint}.pth"

    return output_file

def load_checkpoints(args):
    if args.dataset == 'mnli':
        adv_log_coeffs = {'premise': [], 'hypothesis': []}
        clean_texts = {'premise': [], 'hypothesis': []}
        adv_texts = {'premise': [], 'hypothesis': []}
    else:
        adv_log_coeffs, clean_texts, adv_texts = [], [], []
    clean_logits, adv_logits, times, labels = [], [], [], []
    
    for i in tqdm(range(args.start_index, args.end_index, args.num_samples)):
        output_file = get_output_file(args, args.surrogate_model, i, i + args.num_samples)
        output_file = os.path.join(args.adv_samples_folder, output_file)
        if os.path.exists(output_file):
            checkpoint = torch.load(output_file)
            clean_logits.append(checkpoint['clean_logits'])
            adv_logits.append(checkpoint['adv_logits'])
            labels += checkpoint['labels']
            times += checkpoint['times']
            if args.dataset == 'mnli':
                adv_log_coeffs['premise'] += checkpoint['adv_log_coeffs']['premise']
                adv_log_coeffs['hypothesis'] += checkpoint['adv_log_coeffs']['hypothesis']
                clean_texts['premise'] += checkpoint['clean_texts']['premise']
                clean_texts['hypothesis'] += checkpoint['clean_texts']['hypothesis']
                adv_texts['premise'] += checkpoint['adv_texts']['premise']
                adv_texts['hypothesis'] += checkpoint['adv_texts']['hypothesis']
            else:
                adv_log_coeffs += checkpoint['adv_log_coeffs']
                clean_texts += checkpoint['clean_texts']
                adv_texts += checkpoint['adv_texts']
        else:
            print('Skipping %s' % output_file)
    clean_logits = torch.cat(clean_logits, 0)
    adv_logits = torch.cat(adv_logits, 0)
    return clean_texts, adv_texts, clean_logits, adv_logits, adv_log_coeffs, labels, times

def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")
        
def embedding_from_weights(w):
    layer = torch.nn.Embedding(w.size(0), w.size(1))
    layer.weight.data = w

    return layer

def load_gpt2_from_dict(dict_path, output_hidden_states=False):
    state_dict = torch.load(dict_path)['model']

    config = GPT2Config(
        vocab_size=30522,
        n_embd=1024,
        n_head=8,
        activation_function='relu',
        n_layer=24,
        output_hidden_states=output_hidden_states
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    # The input embedding is not loaded automatically
    model.set_input_embeddings(embedding_from_weights(state_dict['transformer.wte.weight'].cpu()))

    return model