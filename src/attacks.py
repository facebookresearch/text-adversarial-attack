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
import numpy as np
import time
import torch
from transformers import AutoTokenizer, pipeline
import transformers
import tensorflow as tf
import tensorflow_hub as hub
import textattack
from textattack.attack_recipes import BERTAttackLi2020, BAEGarg2019
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

from src.utils import bool_flag


class USE:
    def __init__(self):
        self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_sim(self, clean_texts, adv_texts):
        clean_embeddings = self.encoder(clean_texts)
        adv_embeddings = self.encoder(adv_texts)
        cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

        return float(cosine_sim.numpy())


def build_baegarg2019(model_wrapper, threshold_cosine=0.936338023, query_budget=None, max_candidates=50):
    """
    Modified from https://github.com/QData/TextAttack/blob/04b7c6f79bdb5301b360555bd5458c15aa2b8695/textattack/attack_recipes/bae_garg_2019.py
    """
    transformation = WordSwapMaskedLM(
        method="bae", max_candidates=max_candidates, min_confidence=0.0
    )
    constraints = [RepeatModification(), StopwordModification()]

    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

    use_constraint = UniversalSentenceEncoder(
        threshold=threshold_cosine,
        metric="cosine",
        compare_against_original=True,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    constraints.append(use_constraint)
    goal_function = UntargetedClassification(model_wrapper)
    if query_budget is not None:
        goal_function.query_budget = query_budget
    search_method = GreedyWordSwapWIR(wir_method="delete")

    return Attack(goal_function, constraints, transformation, search_method)


def build_attack(model_wrapper, target_class=-1):
    """
    Same as bert-attack except:
    - it is TargetedClassification instead of Untargeted when target_class != -1
    - using "bae" instead of "bert-attack" because of bert-attack's problem for subtokens
    Modified from https://github.com/QData/TextAttack/blob/36dfce6bdab933bdeed3a2093ae411e93018ebbf/textattack/attack_recipes/bert_attack_li_2020.py
    """

    # transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
    transformation = WordSwapMaskedLM(method="bae", max_candidates=100)
    constraints = [RepeatModification(), StopwordModification()]
    constraints.append(MaxWordsPerturbed(max_percent=0.4))

    use_constraint = UniversalSentenceEncoder(
        threshold=0.2,
        metric="cosine",
        compare_against_original=True,
        window_size=None,
    )
    constraints.append(use_constraint)
    if target_class == -1:
        goal_function = UntargetedClassification(model_wrapper)
    else:
        # We modify the goal
        goal_function = TargetedClassification(model_wrapper, target_class=target_class)
    search_method = GreedyWordSwapWIR(wir_method="unk")

    return Attack(goal_function, constraints, transformation, search_method)


# def build_attack_2(model_wrapper, target_class):
#     """
#     Same as HotFlipEbrahimi2017 attack except:
#     - it is TargetedClassification instead of Untargeted
#     """
#     transformation = WordSwapGradientBased(model_wrapper, top_n=1)
#     constraints = [RepeatModification(), StopwordModification()]
#     constraints.append(MaxWordsPerturbed(max_num_words=2))
#     constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
#     constraints.append(PartOfSpeech())
#     goal_function = TargetedClassification(model_wrapper)
    
#     search_method = BeamSearch(beam_width=10)

#     return Attack(goal_function, constraints, transformation, search_method)