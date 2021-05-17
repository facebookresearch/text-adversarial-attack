# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from os.path import join

import argparse
from bert_score import BERTScorer
import csv
import datasets
import json
import numpy as np
import time
import torch
from transformers import AutoTokenizer, pipeline
import transformers
import textattack
from textattack.attack_recipes import BERTAttackLi2020, BAEGarg2019, CLARE2020
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from textattack.commands.attack.attack_args import HUGGINGFACE_DATASET_BY_MODEL
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapMaskedLM
from textattack.attack_recipes.attack_recipe import AttackRecipe
from textattack.goal_functions.classification.targeted_classification import TargetedClassification
from textattack.shared import AttackedText
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import BeamSearch
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapGradientBased

from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult

from src.utils import bool_flag
from src.attacks import build_baegarg2019, build_attack, USE
from src.dataset import load_data


def get_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument("--dataset", choices=['dbpedia14', 'sst2', 'ag_news', 'yelp', 'imdb'])
    parser.add_argument("--data-folder", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dump_path", type=str)
    parser.add_argument('--model', type=str, default='bert-base-uncased')

    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)

    # Attack parameters
    parser.add_argument("--attack", choices=["bae", "bert-attack", "custom"])
    parser.add_argument("--bae-threshold", type=float, default=0.8)
    parser.add_argument("--query-budget", type=int, default=None)
    parser.add_argument("--radioactive", type=bool_flag)
    parser.add_argument("--targeted", type=bool_flag, default=True)
    parser.add_argument("--ckpt", type=str)

    return parser



def main(params):
    # Loading data
    dataset, num_labels = load_data(params)
    dataset = dataset["train"]
    text_key = 'text'
    if params.dataset == "dbpedia14":
        text_key = 'content'
    print(f"Loaded dataset {params.dataset}, that has {len(dataset)} rows")

    # Load model and tokenizer from HuggingFace
    model_class = transformers.AutoModelForSequenceClassification
    model = model_class.from_pretrained(params.model, num_labels=num_labels).cuda()

    if params.ckpt != None:
        state_dict = torch.load(params.ckpt)
        model.load_state_dict(state_dict)
    tokenizer = textattack.models.tokenizers.AutoTokenizer(params.model)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer, batch_size=params.batch_size)

    # Create radioactive directions and modify classification layer to use those
    if params.radioactive:
        torch.manual_seed(0)
        radioactive_directions = torch.randn(num_labels, 768)
        radioactive_directions /= torch.norm(radioactive_directions, dim=1, keepdim=True)
        print(radioactive_directions)
        model.classifier.weight.data = radioactive_directions.cuda()
        model.classifier.bias.data = torch.zeros(num_labels).cuda()

    start_index = params.chunk_id * params.chunk_size
    end_index  = start_index + params.chunk_size

    if params.target_dir is not None:
        target_file = join(params.target_dir, f"{params.chunk_id}.csv")
        f = open(target_file, "w")
        f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

    # Creating attack
    print(f"Building {params.attack} attack")
    if params.attack == "custom":
        current_label = -1
        if params.targeted:
            current_label = dataset[start_index]['label']
            assert all([dataset[i]['label'] == current_label for i in range(start_index, end_index)])
        attack = build_attack(model_wrapper, current_label)
    elif params.attack == "bae":
        print(f"Building BAE method with threshold={params.bae_threshold:.2f}")
        attack = build_baegarg2019(model_wrapper, threshold_cosine=params.bae_threshold, query_budget=params.query_budget)
    elif params.attack == "bert-attack":
        assert params.query_budget is None
        attack = BERTAttackLi2020.build(model_wrapper)
    elif params.attack == "clare":
        assert params.query_budget is None
        attack = CLARE2020.build(model_wrapper)

    # Launching attack
    begin_time = time.time()
    samples = [(dataset[i][text_key], attack.goal_function.get_output(AttackedText(dataset[i][text_key]))) for i in range(start_index, end_index)]
    results = list(attack.attack_dataset(samples))

    # Storing attacked text
    bert_scorer = BERTScorer(model_type="bert-base-uncased", idf=False)

    n_success = 0
    similarities = []
    queries = []
    use = USE()

    for i_result, result in enumerate(results):
        print("")
        print(50 * "*")
        print("")
        text = dataset[start_index + i_result][text_key]
        ptext = result.perturbed_text()
        i_data = start_index + i_result
        if params.target_dir is not None:
            if params.dataset == 'dbpedia14':
                f.writerow([dataset[i_data]['label'] + 1, dataset[i_data]['title'], ptext])
            else:
                f.writerow([dataset[i_data]['label'] + 1, ptext])

        print("True label ", dataset[i_data]['label'])
        print(f"CLEAN TEXT\n {text}")
        print(f"ADV TEXT\n {ptext}")

        if type(result) not in [SuccessfulAttackResult, FailedAttackResult]:
            print("WARNING: Attack neither succeeded nor failed...")
        print(result.goal_function_result_str())
        precision, recall, f1 = [r.item() for r in bert_scorer.score([ptext], [text])]
        print(f"Bert scores: precision {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")
        initial_logits = model_wrapper([text])
        final_logits = model_wrapper([ptext])
        print("Initial logits", initial_logits)
        print("Final logits", final_logits)
        print("Logits difference", final_logits - initial_logits)

        # Statistics
        n_success += 1 if type(result) is SuccessfulAttackResult else 0
        queries.append(result.num_queries)
        similarities.append(use.compute_sim([text], [ptext]))

    print("Processing all samples took %.2f" % (time.time() - begin_time))
    print(f"Total success: {n_success}/{len(results)}")
    logs = {
        "success_rate": n_success / len(results),
        "avg_queries": sum(queries) / len(queries),
        "queries": queries,
        "avg_similarity": sum(similarities) / len(similarities),
        "similarities": similarities,
    }
    print("__logs:" + json.dumps(logs))
    if params.target_dir is not None:
        f.close()


if __name__ == "__main__":
    print("Using text attack from ", textattack.__file__)
    # Parse arguments
    parser = get_parser()
    params = parser.parse_args()
    # if not params.radioactive:
    #     assert params.ckpt is not None, "Should specify --ckpt if not radioactive."
    assert not (params.radioactive and not params.targeted), "Radioactive means targeted"

    # Run main code
    begin_time = time.time()
    main(params)
    print("Running program took %.2f" % (time.time() - begin_time))
