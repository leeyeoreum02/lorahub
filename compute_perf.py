import json
import os
import random
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from peft.utils.save_and_load import set_peft_model_state_dict

from lorahub.algorithm import load_base_model_and_lora_modules, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES


def get_perf_every_module(
    folder: os.PathLike,
    batch_size: int,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    model_name_or_path: Optional[str] = None,
) -> None:
    sub_dirs = sorted(os.listdir(folder))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    model, tokenizer, cache = load_base_model_and_lora_modules(LORA_MODULE_NAMES, model_name_or_path)

    for sub_dir in sub_dirs:
        task_inputs = []
        task_outputs = []
        test_file_path = os.path.join(folder, sub_dir, 'zero_shot.jsonl')
        with open(test_file_path, 'r', encoding='utf-8') as lines:
            for line in lines:
                example = json.loads(line)
                task_inputs.append(example['context'])
                task_outputs.append(example['completion'])
        
        lora_module_perf = {}
        for cnt, peft_model_id in enumerate(LORA_MODULE_NAMES):
            print(f'dataset: {sub_dir}, lora module name: {peft_model_id} ({cnt+1}/{len(LORA_MODULE_NAMES)})')

            set_peft_model_state_dict(model, cache[peft_model_id])
            _, task_acc = lorahub_inference(
                example_inputs=task_inputs,
                model_or_name_path=model,
                tokenizer_or_tokenizer_path=tokenizer,
                batch_size=batch_size,
                example_outputs=task_outputs
            )
            task_acc = task_acc / 100 if task_acc is not None else -1
            lora_module_perf[peft_model_id] = task_acc
            print(f'accuracy: {task_acc}\n')
        
        real_save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'acc_every_modules.json')
        # breakpoint()
        with open(save_path, 'w') as f:
            json.dump(lora_module_perf, f)

        print(f'Saved a json file at {save_path}\n')


def get_perf_per_module(
    folder: os.PathLike,
    batch_size: int,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    model_name_or_path: Optional[str] = None,
    n_seed: int = 5,
    n_modules: int = 20,
) -> None:
    sub_dirs = sorted(os.listdir(folder))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        task_inputs = []
        task_outputs = []
        test_file_path = os.path.join(folder, sub_dir, 'zero_shot.jsonl')
        with open(test_file_path, 'r', encoding='utf-8') as lines:
            for line in lines:
                example = json.loads(line)
                task_inputs.append(example['context'])
                task_outputs.append(example['completion'])

        perf_per_seed = {}
        for seed in range(1, 1+n_seed):
            print(f'dataset: {sub_dir}')
            print(f'seed: {seed}')

            random.seed(seed)
            modules = random.sample(LORA_MODULE_NAMES, n_modules)

            model, tokenizer, cache = load_base_model_and_lora_modules(modules, model_name_or_path)
            lora_module_perf = {}

            for peft_model_id in modules:
                print(f'lora module name: {peft_model_id}')

                set_peft_model_state_dict(model, cache[peft_model_id])
                _, task_acc = lorahub_inference(
                    example_inputs=task_inputs,
                    model_or_name_path=model,
                    tokenizer_or_tokenizer_path=tokenizer,
                    batch_size=batch_size,
                    example_outputs=task_outputs
                )
                task_acc = task_acc / 100 if task_acc is not None else -1
                lora_module_perf[peft_model_id] = task_acc
                print(f'accuracy: {task_acc}\n')
            
            perf_per_seed[f'seed:{seed}'] = lora_module_perf
            
            real_save_dir = os.path.join(save_dir, sub_dir)
            os.makedirs(real_save_dir, exist_ok=True)
            save_path = os.path.join(real_save_dir, f'acc_per_modules_seed{seed}.json')
            # breakpoint()
            with open(save_path, 'w') as f:
                json.dump(perf_per_seed, f)

            print(f'Saved a json file at {save_path}\n')


def visualize_perf_per_module(
    json_path: os.PathLike,
    save_dir: os.PathLike,
    dataset_name: str,
) -> None:
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for seed in json_data.keys():
        seed_num = int(seed.split(':')[-1])
        data_per_seed = json_data[seed]
        perf_list = [
            (LORA_MODULE_NAMES.index(module_name), perf) for module_name, perf in data_per_seed.items()
        ]
        perf_list = sorted(perf_list, reverse=True, key=lambda x: x[-1])
        
        perf_dict = {module_id: round(perf * 100, 2) for module_id, perf in perf_list}
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = sns.barplot(data=perf_dict)
        ax.bar_label(ax.containers[0], fontsize=8)
        ax.set_title(
            f'lora modules performance in lorahub (dataset: {dataset_name}, seed: {seed_num})',
            fontweight='bold'
        )
        fig.tight_layout()
        os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
        save_path = os.path.join(save_dir, dataset_name, f'seed{seed_num}-lora_module_perf.png')
        plt.savefig(save_path)
        plt.clf()


def show_analysis(dataset_dir: os.PathLike = 'data_bbh', acc_json_dir: os.PathLike = 'acc_every_module') -> None:
    sub_dirs = os.listdir(dataset_dir)
    
    for sub_dir in sorted(sub_dirs):
        acc_json_path = os.path.join(acc_json_dir, sub_dir, 'acc_every_modules.json')
    
        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)
        module_list = [
            (LORA_MODULE_NAMES.index(module_name), perf) for module_name, perf in acc_data.items()
        ]
        module_list = sorted(module_list, reverse=True, key=lambda x: x[-1])
        top_20 = module_list[:20]
        top_acc = np.array([acc * 100 for _, acc in top_20])

        print('{}: {} ({}) - acc: {:.1f}, top20_mean: {:.1f}, top20_std: {:.1f}').format(
            sub_dir,
            LORA_MODULE_NAMES[module_list[0][0]],
            module_list[0][0],
            top_acc[0],
            top_acc.mean(),
            top_acc.std()
        )
        print(f'module_lists: {[i for i, _ in module_list]}\n')
