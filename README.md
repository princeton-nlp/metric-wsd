# MetricWSD
Implementation of the paper: [Non-Parametric Few-Shot Learning for Word Sense Disambiguation](https://arxiv.org/abs/2104.12677).

We propose a learning approach to mitigate the heavily skewed word and sense distribution issue for word sense disambiguation with the following core ideas:

1. Few-shot learning by constructing episodes tailored to word and sense frequencies
2. Non-parametric evaluation by using training examples as supports

This approach results in better sense representations compared with common supervised approaches, especially for rare words and senses as shown in the image below.

<img src="images/tsne.png" width="800"/>

## Dependencies
We use the following training framework and versions. Note that the newer versions of the PyTorch Lightning training framework have compatibility breaking changes. Please make sure to use the version below.

```
torch==1.6.0
pytorch-lightning==0.9.0
transformers==3.3.1
wandb==0.10.4
```

You can directly install through

```
pip install -r requirements.txt
```

## Dataset
We use the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) to train and evaluate our models. Note that the `Scorer.java` in the WSD Framework needs to be compiled, with the `Scorer.class` file in the original directory (`Evaluation_Datasets`) of the Scorer file.

## Specifying Paths
Edit the `metric_wsd/config.py` file to specify project level configurations such as the root directory path of the repo (`ROOT_DIR`) and the data directories. We recommend you place the downloaded `WSD_Evaluation_Framework` folder under `metric_wsd/data/`.

## Run Training
To train, go to the root of the repo and run:

```
python -m metric_wsd.run \
    --run-name <your_run_name> \
    --mode train \
    --gpus 1 \
    --batch_size 5 \
    --max_epochs 200 \
    --lr 1e-5 \
    --model-type cbert-proto \
    --episodic \
    --dist dot \
    --ks_support_kq_query 5 50 5 \
    --max_inference_supports 30
```
Arguments:

* `--model-type`: we provide both the proposed MetricWSD model `cbert-proto` and the baseline model `cbert-linear`.
* `--episodic`: whether to use episodic training. Should be turned on when using `cbert-proto`.
* `--dist`: dot product (`dot`) or L2 distance (`l2`) when calculating the scoring function between the query and the support.
* `--ks_support_kq_query`: a list of 3 numbers specifying 1) the max number of supports to select per sense, 2) the max number of queries for each sense, and 3) the max number of senses per word. E.g., `5 50 5` means for each word, the eposide will contain at most 5 supports, at most 50 queries per sense, and at most 5 senses per word. This is a slightly differnet implementation from the sampling strategy described in the paper but achieves the same result. See argument `--episodic_k` and `--support_query_ratio` for more details of the paper implementation.
* `--max_inference_supports`: Maximum number of training examples to use as support during evaluation.

We provide two ways to control how the examples of a word to be split into the support set and the query set. The above is *Max Query* and the other option is *Ratio Split*. We present the *Ratio Split* sampling strategy in the paper, but *Max Query* achieves the same performance and is more stable (recommended).

Other available arguments (not shown in the above example):

* `--episodic_k`: the maximum number of examples to sample when use the *Ratio Split* strategy.
* `--support_query_ratio`: percentage of examples to be split into the support set. 
* `--sampling`: `balance` or `uniform`.
* `--mix-strategy`: whether or not to switch between *Max Query* and *Split Ratio*.
* `--sample-strategy-threshold`: Threshold on word frequency. Lower than threshold uses *Ratio Split*. Higher than threshold uses *Max Query* (only if `mix-strategy` is specified).

## Run Evaluation
To run evaluation on `SE07` or `ALL`, go the the root of the repo and run:

```
python -m metric_wsd.run_eval \
    --dir $MODEL_DIR_NAME \
    --name $MODEL_CKPT_NAME \
    --max_inference_supports 30 \
    --evalsplit ALL
```
Arguments:

* `--dir`: the folder name of your run. The model directory `MODEL_DIR_NAME` should be placed under `metric_wsd/experiments/` (e.g., `MetricWSD`).
* `--name`: the name of the checkpoint (e.g., `model-epoch=017-f1=71.2.pt`)
* `--max_inference_supports`: Maximum number of training examples to use as support during evaluation. Recommended to use the same value as the one used during training.
* `--evalsplit`: `SE07` or `ALL`

## Checkpoint
We provide the checkpoint [here](https://drive.google.com/file/d/1TF6cSCq8moSvFAPhNiwioFXO7GnyOVJd/view?usp=sharing).
Set `MODEL_DIR_NAME=MetricWSD` and `MODEL_CKPT_NAME=model-epoch=017-f1=71.2.pt` in the evaluation script to rerun evaluation.

The expected performance is 71.2 F1 on `SE07` and 75.3 F1 on the aggregated dataset `ALL`.

## Contact
Please email Howard Chen (howardchen@cs.princeton.edu) or Mengzhou Xia (mengzhou@cs.princeton.edu) for questions or feedback.

## Citation
```
@inproceedings{chen2021metricwsd,
   title={Non-Parametric Few-Shot Learning for Word Sense Disambiguation},
   author={Chen, Howard and Xia, Mengzhou and Chen, Danqi},
   booktitle={North American Association for Computational Linguistics (NAACL)},
   year={2021}
}
```
