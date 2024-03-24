from collections import OrderedDict
import copy
from functools import partial
import itertools
import json
import os
import random
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import nevergrad as ng
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict

from compute_lmc import compute_barrier_height
from lorahub.algorithm import (
    default_get_loss,
    get_final_weights,
    load_base_model_and_lora_modules,
    load_dataset, 
    lorahub_inference, 
    lorahub_learning, 
    default_l1_regularization
)
from lorahub.constant import LORA_MODULE_NAMES


def load_base_model(lora_module_list: list[str], model_name_or_path: Optional[str] = None):
    # use gpu if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        # breakpoint()
        
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
        
    peft_model = peft_model.to(device)
    peft_model.eval()
               
    return peft_model, tokenizer


def load_lora_modules(lora_module_list: list[str], model_name_or_path: Optional[str] = None):
    default_peft_model_id = lora_module_list[0]

    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path

    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    print('> Begin to load lora modules')
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print('> Loading {} ...'.format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
        
    return cache


def default_l2_regularization(weights):
    sum_of_squares = sum([x ** 2 for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def faster_get_score(weights, model, lora_module_list, cache, example_dataset, batch_size, get_loss, get_regular):
    # the composed lora state dict
    final_state_dict = {}
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term (weights are the `lorahub weights`)
    metric_val = loss + get_regular(weights)
    
    return metric_val


def faster_lorahub_learning(
    cache,
    lora_module_list: list[str], 
    example_inputs: list[str], 
    example_outputs: list[str], 
    max_inference_step: int,
    model_name_or_path=None,
    batch_size=None,
    get_loss=default_get_loss, 
    get_regular=default_l1_regularization,
    seed=42,
    is_limit: bool = True,
):
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    model, tokenizer = load_base_model(lora_module_list, model_name_or_path)
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 
    get_score_partial = partial(
        faster_get_score, 
        model=model, 
        lora_module_list=lora_module_list,
        cache=cache,
        example_dataset=dataset,
        batch_size=batch_size,
        get_loss=get_loss, 
        get_regular=get_regular
    )
    # set up the limit of the weights
    if is_limit:
        instrum = ng.p.Array(
            init=[0] * number_of_loras,
            upper=[1.5] * number_of_loras,
            lower=[-1.5] * number_of_loras,
        )
    else:
        instrum = ng.p.Array(init=[0] * number_of_loras)

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    print('> Begin to perform gradient-free optimization ...')
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return recommendation.value, model, tokenizer


def lorahub_with_best_lmc(
    dataset_dir: os.PathLike,
    lmc_json_dir: os.PathLike,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    sampling_seed: int = 42,
    get_regular: Callable = default_l1_regularization,
) -> None:
    sub_dirs = sorted(os.listdir(dataset_dir))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        perf_dict = {}

        real_save_dir = os.path.join(save_dir, sub_dir, f'seed:{sampling_seed}')
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'acc_lorahub_with_best_lmc.json')

        if os.path.exists(save_path):
            print(f'accuracy of {sub_dir} already exists\n')
            continue

        print(f'dataset: {sub_dir}')

        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(dataset_dir, sub_dir, 'example.jsonl')
        for line in open(example_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            # breakpoint()
            example_inputs.append(example['context'])
            examples_outputs.append(example['completion'])

        test_file_path = os.path.join(dataset_dir, sub_dir, 'zero_shot.jsonl')
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            task_inputs.append(example['context'])
            task_outputs.append(example['completion'])

        lmc_json_path = os.path.join(lmc_json_dir, sub_dir, 'acc_list_best20.json')

        with open(lmc_json_path, 'r') as f:
            lmc_data = json.load(f)

        comb_list = list(lmc_data.keys())
        module_list = [comb_list[0].split('+')[0]]
        for i in range(19):
            module_list.append(comb_list[i].split('+')[-1])
        
        barrier_height_list = []
        for i, comb in enumerate(comb_list):
            lmc_data[comb] = [1 - e for e in lmc_data[comb]]
            barrier_height = compute_barrier_height(0.5, lmc_data[comb])
            barrier_height_list.append(barrier_height)

        height_matrix = np.zeros((20, 20))
        module_id_comb = list(itertools.combinations(range(20), r=2))
        for i in range(len(barrier_height_list)):
            scaled_height = barrier_height_list[i]
            col = module_id_comb[i][0]
            row = module_id_comb[i][1]
            # breakpoint()
            height_matrix[col, row] = scaled_height
            height_matrix[row, col] = scaled_height

        mean_height_list = height_matrix.sum(axis=0) / 19
        print(f'heights: {mean_height_list}')
        pos_modules = np.array(module_list)[mean_height_list < 0.02].tolist()

        print(f'sampling seed: {sampling_seed}')
        # random select 5 examples for each task
        random.seed(sampling_seed)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        module_weights, model, tokenizer = lorahub_learning(
            lora_module_list=pos_modules,
            example_inputs=example_inputs,
            example_outputs=examples_outputs,
            max_inference_step=40,
            batch_size=5,
            get_regular=get_regular,
        )
        print(f'module_weights: {module_weights}')

        _, task_acc = lorahub_inference(
            example_inputs=task_inputs,
            model_or_name_path=model,
            tokenizer_or_tokenizer_path=tokenizer,
            batch_size=32,
            # can set as None if you do not have the ground truth
            example_outputs=task_outputs
        )
        
        print(f'accuracy: {task_acc}\n')

        perf_dict[f'sampling_seed:{sampling_seed}'] = {
            'accuracy': task_acc,
            'lorahub_weights': module_weights.tolist(),
        }

        with open(save_path, 'w') as f:
            json.dump(perf_dict, f)

        print(f'Saved a json file at {save_path}\n')


def lorahub_with_just_best(
    dataset_dir: os.PathLike,
    acc_json_dir: os.PathLike,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    sampling_seed: int = 42,
    n_modules: int = 20,
    get_regular: Callable = default_l1_regularization,
    weight_range_limit: bool = True,
) -> None:
    sub_dirs = sorted(os.listdir(dataset_dir))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        perf_dict = {}

        real_save_dir = os.path.join(save_dir, sub_dir, f'seed:{sampling_seed}')
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'acc_lorahub_with_just_best.json')

        if os.path.exists(save_path):
            print(f'accuracy of {sub_dir} is already evaluated\n')
            continue

        print(f'dataset: {sub_dir}')

        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(dataset_dir, sub_dir, 'example.jsonl')
        for line in open(example_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            # breakpoint()
            example_inputs.append(example['context'])
            examples_outputs.append(example['completion'])

        test_file_path = os.path.join(dataset_dir, sub_dir, 'zero_shot.jsonl')
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            task_inputs.append(example['context'])
            task_outputs.append(example['completion'])

        acc_json_path = os.path.join(acc_json_dir, sub_dir, 'acc_every_modules.json')

        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)

        module_list = [
            (LORA_MODULE_NAMES.index(module_name), perf) for module_name, perf in acc_data.items()
        ]
        module_list = sorted(module_list, reverse=True, key=lambda x: x[-1])
        module_list = [LORA_MODULE_NAMES[module_id] for module_id, _ in module_list[:n_modules]]
            
        print(f'sampling seed: {sampling_seed}')
        # random select 5 examples for each task
        random.seed(sampling_seed)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]
        # print(example_inputs)

        module_weights, model, tokenizer = lorahub_learning(
            lora_module_list=module_list,
            example_inputs=example_inputs,
            example_outputs=examples_outputs,
            max_inference_step=40,
            batch_size=5,
            get_regular=get_regular,
            is_limit=weight_range_limit,
        )
        print(f'module_weights: {module_weights}')

        _, task_acc = lorahub_inference(
            example_inputs=task_inputs,
            model_or_name_path=model,
            tokenizer_or_tokenizer_path=tokenizer,
            batch_size=32,
            # can set as None if you do not have the ground truth
            example_outputs=task_outputs
        )
        
        print(f'accuracy: {task_acc}')

        perf_dict[f'sampling_seed:{sampling_seed}'] = {
            'accuracy': task_acc,
            'lorahub_weights': module_weights.tolist(),
        }

        with open(save_path, 'w') as f:
            json.dump(perf_dict, f)

        print(f'Saved a json file at {save_path}\n')


def average_params(params_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    averaged_params = OrderedDict()
    module_list = list(params_dict.keys())
    
    with torch.no_grad():
        temp_params = params_dict[module_list[0]]
        for key, value in temp_params.items():
            averaged_params[key] = torch.zeros_like(value)

        for module_name in params_dict.keys():
            for key, value in params_dict[module_name].items():
                averaged_params[key] += value

        for key in averaged_params.keys():
            averaged_params[key] /= len(module_list)
    
    return averaged_params


def model_soups_with_just_best(
    dataset_dir: os.PathLike,
    acc_json_dir: os.PathLike,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    model_name_or_path: Optional[str] = None,
    n_modules: int = 20,
) -> None:
    sub_dirs = sorted(os.listdir(dataset_dir))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        perf_dict = {}

        real_save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'model_soups_with_just_best.json')

        if os.path.exists(save_path):
            print(f'accuracy of {sub_dir} is already evaluated\n')
            continue

        print(f'dataset: {sub_dir}')

        test_file_path = os.path.join(dataset_dir, sub_dir, 'zero_shot.jsonl')
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            task_inputs.append(example['context'])
            task_outputs.append(example['completion'])

        acc_json_path = os.path.join(acc_json_dir, sub_dir, 'acc_every_modules.json')

        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)

        module_list = [
            (LORA_MODULE_NAMES.index(module_name), perf) for module_name, perf in acc_data.items()
        ]
        module_list = sorted(module_list, reverse=True, key=lambda x: x[-1])
        module_list = [LORA_MODULE_NAMES[module_id] for module_id, _ in module_list[:n_modules]]

        model, tokenizer, cache = load_base_model_and_lora_modules(module_list, model_name_or_path)
        averaged_params = average_params(cache)
        set_peft_model_state_dict(model, averaged_params)
        model = model.merge_and_unload()

        _, task_acc = lorahub_inference(
            example_inputs=task_inputs,
            model_or_name_path=model,
            tokenizer_or_tokenizer_path=tokenizer,
            batch_size=32,
            # can set as None if you do not have the ground truth
            example_outputs=task_outputs
        )
        
        print(f'accuracy: {task_acc}')
        perf_dict[sub_dir] = {'accuracy': task_acc}

        # breakpoint()
        with open(save_path, 'w') as f:
            json.dump(perf_dict, f)

        print(f'Saved a json file at {save_path}\n')


def mean_lora_weights_with_just_best(
    dataset_dir: os.PathLike,
    acc_json_dir: os.PathLike,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    model_name_or_path: Optional[str] = None,
    n_modules: int = 20,
) -> None:
    sub_dirs = sorted(os.listdir(dataset_dir))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        perf_dict = {}

        real_save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'acc_mean_lora_weights_with_just_best.json')

        if os.path.exists(save_path):
            print(f'accuracy of {sub_dir} is already evaluated\n')
            continue

        print(f'dataset: {sub_dir}')

        test_file_path = os.path.join(dataset_dir, sub_dir, 'zero_shot.jsonl')
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            task_inputs.append(example['context'])
            task_outputs.append(example['completion'])

        acc_json_path = os.path.join(acc_json_dir, sub_dir, 'acc_every_modules.json')

        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)

        module_list = [
            (LORA_MODULE_NAMES.index(module_name), perf) for module_name, perf in acc_data.items()
        ]
        module_list = sorted(module_list, reverse=True, key=lambda x: x[-1])
        module_list = [LORA_MODULE_NAMES[module_id] for module_id, _ in module_list[:n_modules]]

        model, tokenizer, cache = load_base_model_and_lora_modules(module_list, model_name_or_path)
        averaged_weights = np.ones((n_modules,)) / n_modules
        averaged_lora = get_final_weights(averaged_weights, module_list, cache)
        set_peft_model_state_dict(model, averaged_lora)
        model = model.merge_and_unload()
        print(f'module_weights: {averaged_weights}')

        _, task_acc = lorahub_inference(
            example_inputs=task_inputs,
            model_or_name_path=model,
            tokenizer_or_tokenizer_path=tokenizer,
            batch_size=32,
            # can set as None if you do not have the ground truth
            example_outputs=task_outputs
        )
        
        print(f'accuracy: {task_acc}\n')
        perf_dict[sub_dir] = {
            'accuracy': task_acc,
            'lorahub_weights': averaged_weights
        }

        # breakpoint()
        with open(save_path, 'w') as f:
            json.dump(perf_dict, f)

        print(f'Saved a json file at {save_path}\n')


def mean_lora_weights_with_best_lmc(
    dataset_dir: os.PathLike,
    lmc_json_dir: os.PathLike,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    model_name_or_path: Optional[str] = None,
) -> None:
    sub_dirs = sorted(os.listdir(dataset_dir))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    for sub_dir in sub_dirs:
        perf_dict = {}

        real_save_dir = os.path.join(save_dir, sub_dir)
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'acc_mean_lora_weights_with_best_lmc.json')

        if os.path.exists(save_path):
            print(f'accuracy of {sub_dir} is already evaluated\n')
            continue

        print(f'dataset: {sub_dir}')

        test_file_path = os.path.join(dataset_dir, sub_dir, 'zero_shot.jsonl')
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            task_inputs.append(example['context'])
            task_outputs.append(example['completion'])

        lmc_json_path = os.path.join(lmc_json_dir, sub_dir, 'acc_list_best20.json')

        with open(lmc_json_path, 'r') as f:
            lmc_data = json.load(f)

        comb_list = list(lmc_data.keys())
        module_list = [comb_list[0].split('+')[0]]
        for i in range(19):
            module_list.append(comb_list[i].split('+')[-1])
        
        barrier_height_list = []
        for i, comb in enumerate(comb_list):
            lmc_data[comb] = [1 - e for e in lmc_data[comb]]
            barrier_height = compute_barrier_height(0.5, lmc_data[comb])
            barrier_height_list.append(barrier_height)

        height_matrix = np.zeros((20, 20))
        module_id_comb = list(itertools.combinations(range(20), r=2))
        for i in range(len(barrier_height_list)):
            scaled_height = barrier_height_list[i]
            col = module_id_comb[i][0]
            row = module_id_comb[i][1]
            # breakpoint()
            height_matrix[col, row] = scaled_height
            height_matrix[row, col] = scaled_height

        mean_height_list = height_matrix.sum(axis=0) / 19
        print(f'heights: {mean_height_list}')
        pos_modules = np.array(module_list)[mean_height_list < 0.02].tolist()

        model, tokenizer, cache = load_base_model_and_lora_modules(pos_modules, model_name_or_path)
        averaged_weights = np.ones((len(pos_modules),)) / len(pos_modules)
        averaged_lora = get_final_weights(averaged_weights, pos_modules, cache)
        set_peft_model_state_dict(model, averaged_lora)
        model = model.merge_and_unload()
        print(f'module_weights: {averaged_weights}')

        _, task_acc = lorahub_inference(
            example_inputs=task_inputs,
            model_or_name_path=model,
            tokenizer_or_tokenizer_path=tokenizer,
            batch_size=32,
            # can set as None if you do not have the ground truth
            example_outputs=task_outputs
        )
        
        print(f'accuracy: {task_acc}')
        perf_dict[sub_dir] = {
            'accuracy': task_acc,
            'lorahub_weights': averaged_weights
        }

        # breakpoint()
        with open(save_path, 'w') as f:
            json.dump(perf_dict, f)

        print(f'Saved a json file at {save_path}\n')


def lorahub_greedy_soups(
    dataset_dir: os.PathLike,
    acc_json_dir: os.PathLike,
    save_dir: os.PathLike,
    dataset_indices: Optional[Sequence[int]] = None,
    sampling_seed: int = 42,
    get_regular: Callable = default_l1_regularization,
    weight_range_limit: bool = True,
    n_patience: int = 30,
) -> None:
    sub_dirs = sorted(os.listdir(dataset_dir))

    if dataset_indices is not None:
        assert isinstance(dataset_indices, Sequence), 'parameter `dataet_indices` must be `list` or `tuple`'
        sub_dirs = np.array(sub_dirs)[dataset_indices].tolist()

    cache = load_lora_modules(LORA_MODULE_NAMES)

    for sub_dir in sub_dirs:
        perf_dict = {}

        real_save_dir = os.path.join(save_dir, sub_dir, f'seed:{sampling_seed}')
        os.makedirs(real_save_dir, exist_ok=True)
        save_path = os.path.join(real_save_dir, 'acc_lorahub_greedy_soups.json')

        if os.path.exists(save_path):
            print(f'accuracy of {sub_dir} is already evaluated\n')
            continue

        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(dataset_dir, sub_dir, 'example.jsonl')
        for line in open(example_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            # breakpoint()
            example_inputs.append(example['context'])
            examples_outputs.append(example['completion'])

        test_file_path = os.path.join(dataset_dir, sub_dir, 'zero_shot.jsonl')
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, 'r', encoding='utf-8'):
            example = json.loads(line)
            task_inputs.append(example['context'])
            task_outputs.append(example['completion'])

        acc_json_path = os.path.join(acc_json_dir, sub_dir, 'acc_every_modules.json')

        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)

        module_list = [
            (LORA_MODULE_NAMES.index(module_name), perf) for module_name, perf in acc_data.items()
        ]
        module_list = sorted(module_list, reverse=True, key=lambda x: x[-1])
        module_list = [LORA_MODULE_NAMES[module_id] for module_id, _ in module_list]

        random.seed(sampling_seed)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        random.seed(42)
        np.random.seed(42)

        ingredients = []
        best_weights = None
        max_acc = 0
        patience = 0
        for module_idx in range(len(module_list)):
            print(f'dataset: {sub_dir}')
            print(f'sampling seed: {sampling_seed}')
            print(f'module_name: {LORA_MODULE_NAMES[module_idx]} ({module_idx+1}/{len(LORA_MODULE_NAMES)})')

            ingredients.append(module_list[module_idx])

            module_weights, model, tokenizer = faster_lorahub_learning(
                cache=cache,
                lora_module_list=ingredients,
                example_inputs=example_inputs,
                example_outputs=examples_outputs,
                max_inference_step=40,
                batch_size=5,
                get_regular=get_regular,
                is_limit=weight_range_limit,
            )
            print(f'module_weights: {module_weights}')

            _, task_acc = lorahub_inference(
                example_inputs=task_inputs,
                model_or_name_path=model,
                tokenizer_or_tokenizer_path=tokenizer,
                batch_size=32,
                # can set as None if you do not have the ground truth
                example_outputs=task_outputs
            )
        
            print(f'accuracy: {task_acc}')

            if max_acc <= task_acc:
                max_acc = task_acc
                best_weights = module_weights
                patience = 0
                print(f'module idx {module_idx} is appended to ingredients')
                print(f'ingredients (idx): {[LORA_MODULE_NAMES.index(ing) for ing in ingredients]}')
            else:
                ingredients.pop()
                patience += 1

            print(f'max accuracy: {max_acc}')

            if patience >= n_patience:
                break

        perf_dict[f'sampling_seed:{sampling_seed}'] = {
            'accuracy': max_acc,
            'lorahub_weights': best_weights.tolist(),
        }

        with open(save_path, 'w') as f:
            json.dump(perf_dict, f)

        print(f'Saved a json file at {save_path}\n')

        

        