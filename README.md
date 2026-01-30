<div align="center">
  <h2><b>PrismGTS: A Lightweight Foundation Model of Time Series Forecasting via Disentangled Representation Learning </b></h2>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

PrismGTS, a lightweight general time series forecaster with disentangled seasonal-trend representation learning. 

## Introduction

> This work proposes PrismGTS, a lightweight foundation model that explic-itly decomposes time series into transferable trend and seasonal representations. PrismGTS comprises two key modules: (1) Decomposed Tokenization, which extracts trend and seasonal features via physics-aware multi-scale convolutions and learnable frequency filtering; and (2) PrismFormer, a dual-stream Transformer with Semantic Mixture-of-Experts routing for domain-adaptive representation learning. Extensive experiments on seven benchmarks demonstrate that PrismGTS with only 5.5M parameters achieves state-of-the-art performance in both zero-shot and few-shot forecasting, outperforming models 50–100× larger. 

## Pretraining Data

All pretraining datasets, except PEMS04 and PEMS07, use the [numpy format UTSD](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/)., then place the downloaded contents under `./dataset`. 

## Installation

Install Python 3.10, and then install the dependencies:

```setup
pip install -r requirements.txt
```

## Pretraining

To pretrain the model(s) in the paper, run the follow command. For convenience, we provide a single-GPU pretrained model in the folder `./checkpoints/0-shot/PrismGTS.sh`.

```pre-training
bash ./scripts/time_series_forecatsing/pretrain/prismgts_3_3_128_Pre_100M_pretrain.sh
```

## Evaluation

To evaluate the model(s) in the paper, use the zero-shot examples in the folder `./scripts/time_series_forecasting/zero_shot` and `./scripts/time_series_forecasting/few_shot`. 

```0-shot eval
bash ./scripts/time_series_forecatsing/zero_shot/Prism_3_3_128_zero_shot_run_all.sh
```
To fine-tune the model(s) in the paper, use the few-shot examples in the folder `./scripts/time_series_forecasting/few_shot`. 

```fine-tuning
bash ./scripts/time_series_forecatsing/few_shot/prismgts_few_shot_10percent.sh
```

## License

This project is licensed under the Apache-2.0 License.

