from collections import OrderedDict, defaultdict
from functools import partial
import itertools
import json
import os
import random
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from peft.utils.save_and_load import set_peft_model_state_dict

from lorahub.algorithm import default_get_loss, load_base_model_and_lora_modules, load_dataset
from lorahub.constant import LORA_MODULE_NAMES


def is_same_param(param1: torch.Tensor, param2: torch.Tensor) -> bool:
    # ! Do not use
    return (param1 == param2).all().item()


def convex_combination(param1: torch.Tensor, param2: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * param1 + (1 - alpha) * param2


def compute_barrier_height(alpha: float, interp_err_list: Sequence[float]) -> float:
    # Linear Mode Connectivity and the Lottery Ticket Hypothesis (ICML 2020)
    sup_interp_err = max(interp_err_list)
    net1_err = interp_err_list[-1]
    net2_err = interp_err_list[0]
    return sup_interp_err - (alpha * net1_err + (1 - alpha) * net2_err)


def accuracy_score(outputs: Sequence[str], ground_truths: Sequence[str]) -> float:
    correct = 0
    total = 0
    for output, truth in zip(outputs, ground_truths):
        if output.strip().lower().replace('.', '') == truth.strip().lower().replace('.', ''):
            correct += 1
        total += 1
    return correct / total


def linear_interp_params(
    alpha: float,
    net1_params: dict[str, torch.Tensor], 
    net2_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    interp_params = OrderedDict()
    with torch.no_grad():
        for name in net1_params.keys():
            # if not is_same_param(net1_params[name].data, net2_params[name].data):
            interp_params[name] = convex_combination(
                net1_params[name].data,
                net2_params[name].data,
                alpha=alpha,
            )
    return interp_params


def get_loss_from_interp_net(
    lora_module_list: Sequence[str],
    example_inputs: Sequence[str], 
    example_outputs: Sequence[str],
    model_name_or_path: Optional[str | os.PathLike] = None,
    batch_size: Optional[str] = None,
    get_loss: Callable = default_get_loss,
    n_alpha: int = 20,
) -> list[float] | None:
    if len(lora_module_list) == 0:
        print('> No LoRA modules are provided. Please provide at least one LoRA module.')
        return
    
    interp_model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path)
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 

    net1_state_dict = cache[lora_module_list[0]]
    net2_state_dict = cache[lora_module_list[1]]
    fast_linear_interp_params = partial(
        linear_interp_params,
        net1_params=net1_state_dict, 
        net2_params=net2_state_dict,
    )
    
    alpha_list = torch.linspace(0, 1, steps=n_alpha)
    loss_list = []
    for alpha in tqdm(alpha_list):
        interp_params = fast_linear_interp_params(alpha)
        set_peft_model_state_dict(interp_model, interp_params)

        loss = get_loss(dataset, interp_model, batch_size)
        loss_list.append(loss)

    loss_list_for_print = ' '.join([str(l)[:7] for l in loss_list])
    print(f'loss_list: [{loss_list_for_print}]\n')
    return loss_list


def get_acc_from_interp_net(
    lora_module_list: Sequence[str],
    example_inputs: Sequence[str], 
    example_outputs: Sequence[str],
    model_name_or_path: Optional[str | os.PathLike] = None,
    batch_size: Optional[str] = None,
    n_alpha: int = 20,
) -> list[float] | None:
    # What is being transferred in transfer learning? (NeurIPS 2020)

    if len(lora_module_list) == 0:
        print('> No LoRA modules are provided. Please provide at least one LoRA module.')
        return
    
    interp_model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path)
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 

    net1_state_dict = cache[lora_module_list[0]]
    net2_state_dict = cache[lora_module_list[1]]
    fast_linear_interp_params = partial(
        linear_interp_params,
        net1_params=net1_state_dict, 
        net2_params=net2_state_dict,
    )
    
    alpha_list = torch.linspace(0, 1, steps=n_alpha)
    acc_list = []
    for alpha in tqdm(alpha_list):
        interp_params = fast_linear_interp_params(alpha)
        set_peft_model_state_dict(interp_model, interp_params)

        interp_model = interp_model.to('cuda')
                
        example_predictions = []
        for i in range(0, len(dataset['input']), batch_size):
            inputs = tokenizer(
                dataset['input'][i : i + batch_size],
                max_length=2048,
                return_tensors='pt',
                padding=True,
            ).to('cuda')
            outputs = interp_model.generate(
                input_ids=inputs['input_ids'], max_new_tokens=256
            )
            outputs = tokenizer.batch_decode(
                outputs.to('cpu'), skip_special_tokens=True
            )
            example_predictions.extend(outputs)
    
        if example_outputs is not None:
            acc = accuracy_score(example_predictions, example_outputs)
        else:
            acc = -1
            
        acc_list.append(acc)

    acc_list_for_print = ' '.join([str(l)[:7] for l in acc_list])
    print(f'acc_list: [{acc_list_for_print}]\n')
    return acc_list


def get_loras_lmc(
    folder: os.PathLike,
    batch_size: int,
    save_dir: os.PathLike,
    err_type: str,
    dataset_indices: Optional[Sequence[int]] = None,
    n_seed: int = 5,
    n_modules: int = 20,
) -> None:
    assert err_type in ['loss', 'acc'], 'parameter `err_type` must be either `loss` or `acc`'

    sub_dirs = sorted(os.listdir(folder))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        # example_inputs = []
        # examples_outputs = []
        # example_file_path = os.path.join(folder, sub_dir, 'example.jsonl')
        # with open(example_file_path, 'r', encoding='utf-8') as lines:
        #     for line in lines:
        #         example = json.loads(line)
        #         example_inputs.append(example['context'])
        #         examples_outputs.append(example['completion'])

        # random.seed(42)
        # random_indices = random.sample(range(len(example_inputs)), k=5)
        # example_inputs = np.array(example_inputs)[random_indices].tolist()
        # examples_outputs = np.array(examples_outputs)[random_indices].tolist()

        task_inputs = []
        task_outputs = []
        test_file_path = os.path.join(folder, sub_dir, 'zero_shot.jsonl')
        with open(test_file_path, 'r', encoding='utf-8') as lines:
            for line in lines:
                example = json.loads(line)
                task_inputs.append(example['context'])
                task_outputs.append(example['completion'])

        errs_per_seed = {}
        for seed in range(1, 1+n_seed):           
            random.seed(seed)

            modules = random.sample(LORA_MODULE_NAMES, n_modules)
            module_id_comb = list(itertools.combinations(range(len(modules)), r=2))
            errs_per_comb = defaultdict(list)
            for selected_ids in module_id_comb:
                net1_name = modules[selected_ids[0]]
                net2_name = modules[selected_ids[1]]
                
                print(f'dataset: {sub_dir}')
                print(f'seed: {seed}')
                print(f'combination: \n - {net1_name} (id: {selected_ids[0]})\n - {net2_name} (id: {selected_ids[1]})')

                err_func = get_loss_from_interp_net if err_type == 'loss' else get_acc_from_interp_net
                err_list = err_func(
                    lora_module_list=[net1_name, net2_name],
                    example_inputs=task_inputs,
                    example_outputs=task_outputs,
                    batch_size=batch_size,
                )
                errs_per_comb[f'{net1_name}+{net2_name}'].extend(err_list)
            
            errs_per_seed[f'seed:{seed}'] = errs_per_comb

        real_save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, f'{err_type}_list_seed{",".join(map(str, range(1, 1+n_seed)))}.json')
        # breakpoint()
        with open(save_path, 'w') as f:
            json.dump(errs_per_seed, f)

        print(f'Saved a json file at {save_path}')


def visualize_lmc(
    json_path: os.PathLike,
    save_dir: os.PathLike,
    dataset_name: str,
    reverse: bool = False,
    plot_len: int = 4
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    err_type = os.path.basename(json_path).split('_')[0]
    alpha_list = np.linspace(0, 1, num=20)
    for seed in json_data.keys():
        seed_num = int(seed.split(':')[-1])
        data_per_seed = json_data[seed]
        comb_list = list(data_per_seed.keys())

        barrier_height_list = []
        for i, comb in enumerate(comb_list):
            if err_type == 'acc':
                data_per_seed[comb] = [1 - e for e in data_per_seed[comb]]
            # alpha_max = alpha_list[np.array(data_per_seed[comb]).argmax()]
            
            barrier_height = compute_barrier_height(0.5, data_per_seed[comb])
            # if error_type == 'acc':
            #     barrier_height *= -1
            barrier_height_list.append((i, barrier_height))
        
        n_plots = plot_len ** 2
        barrier_height_list = sorted(
            barrier_height_list, reverse=reverse, key=lambda x: x[-1]
        )[:n_plots]

        fig, ax = plt.subplots(plot_len, plot_len, figsize=(10, 8))
        for i in range(n_plots):
            n_col = i // plot_len
            n_row = i % plot_len
            comb_id, comb_barrier_height = barrier_height_list[i]
            comb_name = comb_list[comb_id]
            comb_name1, comb_name2 = comb_name.split('+')
            comb_id1 = LORA_MODULE_NAMES.index(comb_name1)
            comb_id2 = LORA_MODULE_NAMES.index(comb_name2)
            # breakpoint()
            ax[n_col, n_row].plot(alpha_list, data_per_seed[comb_name], marker='o')
            ax[n_col, n_row].set_title(
                f'lora_id: {comb_id1} + {comb_id2} ({comb_barrier_height:5f})',
                fontsize=8,
            )
            ax[n_col, n_row].set_ylim([0, 1])
            ax[n_col, n_row].set_xlabel('alpha', fontsize=8)
            # ax[n_col, n_row].set_ylabel(error_type, fontsize=8)
            y_label = f'1 - {err_type}' if err_type == 'acc' else err_type
            ax[n_col, n_row].set_ylabel(y_label, fontsize=8)

        fig.suptitle(f'lorahub LMC (dataset: {dataset_name}, seed: {seed_num}, reverse: {reverse})', fontweight='bold')

        os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
        save_path = os.path.join(save_dir, dataset_name, f'{err_type}-seed{seed_num}{"-reverse" if reverse else ""}.png')
        fig.tight_layout()
        plt.savefig(save_path)
        plt.clf()


def visualize_conf_matrix(
    json_path: os.PathLike,
    save_dir: os.PathLike,
    dataset_name: str,
) -> None:
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    err_type = os.path.basename(json_path).split('_')[0]
    for seed in json_data.keys():
        seed_num = int(seed.split(':')[-1])
        data_per_seed = json_data[seed]
        comb_list = list(data_per_seed.keys())

        barrier_height_list = []
        for comb in comb_list:
            if err_type == 'acc':
                data_per_seed[comb] = [1 - e for e in data_per_seed[comb]]
            # alpha_max = alpha_list[np.array(data_per_seed[comb]).argmax()]
            
            barrier_height = compute_barrier_height(0.5, data_per_seed[comb])
            # if error_type == 'acc':
            #     barrier_height *= -1
            barrier_height_list.append(barrier_height)
        
        height_matrix = np.ones((20, 20)) * -1
        module_id_comb = list(itertools.combinations(range(20), r=2))
        for i in range(len(barrier_height_list)):
            scaled_height = barrier_height_list[i]
            col = module_id_comb[i][0]
            row = module_id_comb[i][1]
            height_matrix[col, row] = scaled_height
            height_matrix[row, col] = scaled_height

        random.seed(seed_num)
        modules = random.sample(LORA_MODULE_NAMES, 20)
        comb_id_list = [LORA_MODULE_NAMES.index(m) for m in modules]

        height_matrix = pd.DataFrame(height_matrix, index=comb_id_list, columns=comb_id_list)
        fig, ax = plt.subplots(figsize=(16, 16))
        ax = sns.heatmap(height_matrix, annot=True, fmt='.3f', cmap='Blues', vmin=0)
        ax.set_title(
            'lorahub error barrier heights (dataset: {}, seed: {}, err_type: {})'.format(
                dataset_name, seed_num, f'1 - {err_type}' if err_type == 'acc' else err_type
            ),
            fontweight='bold'
        )
        fig.tight_layout()
        os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
        save_path = os.path.join(save_dir, dataset_name, f'{err_type}-seed{seed_num}-height_matrix.png')
        plt.savefig(save_path)
        plt.clf()
