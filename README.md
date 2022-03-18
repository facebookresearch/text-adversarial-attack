# Gradient-based Adversarial Attacks against Text Transformers

## 0. Install Dependencies and Data Download:
1. Install HuggingFace dependences
```
conda install -c huggingface transformers
pip install datasets
```
2. (Optional) For attacks against DBPedia14, download from [Kaggle](https://www.kaggle.com/danofer/dbpedia-classes) and setup data directory to contain:
```
<data_dir>/dbpedia_csv/
       train.csv
       test.csv
```

## 1. Training finetuned models
Use the following training script to finetune a pre-trained transformer model from HuggingFace:
```
python text_classification.py --data_folder <data_dir> --dataset <dataset_name> --model <model_name> --finetune True
```

## 2. Attacking a finetuned model
To attack a finetuned model after running ```text_classification.py``` or from the TextAttack library:
```
python whitebox_attack.py --data_folder <data_dir> --dataset <dataset_name> --model <model_name> --finetune True --start_index 0 --num_samples 100 --gumbel_samples 100
```
This runs the GBDA on the first 100 samples from the test set.

### 2.1. Downloading GPT-2 trained on BERT tokenizer (optional)
To attack a BERT model, GBDA requires a casual language model trained on the BERT tokenizer. We provide a pretrained GPT-2 model for this purpose. Before the attack, please run the following script to download the model from the Amazon S3 bucket:
```
curl https://dl.fbaipublicfiles.com/text-adversarial-attack/transformer_wikitext-103.pth -o transformer_wikitext-103.pth
```

## 3. Evaluating transfer attack
After attacking a model, run the following script to query a target model from the optimized adversarial distribution:
```
python evaluate_adv_samples.py --data_folder <data_dir> --dataset <dataset_name> --surrogate_model <surrogate_model_name> --target_model <target_model_name> --finetune True --start_index 0 --num_samples 100 --end_index 100 --gumbel_samples 1000
```

## Citation

Please cite [[1]](https://arxiv.org/abs/2104.13733) if you found the resources in this repository useful.


[1] C. Guo *, A. Sablayrolles *, Herve Jegou, Douwe Kiela.  [*Gradient-based Adversarial Attacks against Text Transformers*](https://arxiv.org/abs/2104.13733). EMNLP 2021.


```
@article{guo2021gradientbased,
  title={Gradient-based Adversarial Attacks against Text Transformers},
  author={Guo, Chuan and Sablayrolles, Alexandre and Jégou, Hervé and Kiela, Douwe},
  journal={arXiv preprint arXiv:2104.13733},
  year={2021}
}
```


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
