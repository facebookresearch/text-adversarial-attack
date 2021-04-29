# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from datasets import list_datasets, load_dataset, list_metrics, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
import os

from src.dataset import load_data
from src.utils import bool_flag


# function for computing accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if type(predictions) == tuple:
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    acc = np.mean(predictions == labels)
    return {
        'accuracy': acc
    }

def main(args):
    
    dataset, num_labels = load_data(args)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    if args.model == 'gpt2':
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    if args.dataset == "mnli":
        # only evaluate on matched validation set
        testset_key = "validation_matched"
        preprocess_function = lambda examples: tokenizer(
            examples["premise"], examples["hypothesis"], max_length=256, truncation=True)
    else:
        text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'sentence'
        testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'validation'
        preprocess_function = lambda examples: tokenizer(examples[text_key], max_length=256, truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    train_args = TrainingArguments(
        args.checkpoint_folder,
        disable_tqdm=not args.tqdm,
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[testset_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    if not args.finetune:
        # freeze parameters of transformer
        transformer = list(model.children())[0]
        for param in transformer.parameters():
            param.requires_grad = False

    trainer.train()
    trainer.evaluate()
    suffix = ''
    if args.finetune:
        suffix += '_finetune'
    torch.save(model.state_dict(),
               os.path.join(args.result_folder, "%s_%s%s.pth" % (args.model.replace('/', '-'), args.dataset, suffix)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text classification model training.")

    # Bookkeeping
    parser.add_argument("--checkpoint_folder", default="checkpoint/", type=str,
        help="folder in which to store temporary model checkpoints")
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder in which to store trained models")
    parser.add_argument("--tqdm", default=True, type=bool_flag,
        help="Use tqdm in output")

    # Data 
    parser.add_argument("--data_folder", required=True, type=str,
        help="folder in which to store data")
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli"],
        help="classification dataset to use")

    # Model
    parser.add_argument("--model", default="gpt2", type=str,
        help="type of model")

    # Optimization
    parser.add_argument("--batch_size", default=16, type=int,
        help="batch size for training and evaluation")
    parser.add_argument("--epochs", default=5, type=int,
        help="number of epochs to train for")
    parser.add_argument("--lr", default=2e-5, type=float,
        help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="weight decay")
    parser.add_argument("--finetune", default=False, type=bool_flag,
        help="finetune the transformer; if False, only train linear layer")

    args = parser.parse_args()

    if args.result_folder == 'none':
        args.result_folder = args.checkpoint_folder

    main(args)