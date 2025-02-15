# <div align="center"> Multi-Scale Hypergraph Meets LLMs: Aligning Large Language Models for Time Series Analysis

✨ This repository provides the official implementation of MSH-LLM that aligns large language models for time series analysis.
# 1 The framework of MSH-LLM
MSH-LLM focuses on reprogramming an embedding-visible large language model, e.g., LLaMA and GPT-2, for general time series analysis, while accounting for the multi-scale structures of natural language and time series. MSH-LLM consists four main parts: **Multi-Scale Extraction (ME) Module**, **Hyperedging Mechanism**, **Cross-Modality Alignment (CMA) Module**, and **Mixture of Prompts (MoP) Mechanism**. The framework of MSH-LLM is shown as follows: 
![framework](https://github.com/shangzongjiang/MSH-LLM/blob/main/figures/framework.png)
# 2 Prerequisites

* Python 3.8.5
* PyTorch 1.13.1
* math, sklearn, numpy, torch_geometric
  
To install all dependencies:
```
pip install -r requirements.txt
```
# 3 Datasets && Description
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1t7jOkctNJ0rt3VMwZaqmxSuA75TFEo96/view?usp=sharing)
[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/0a758154e0d44de890e3/), then put the downloaded datasets under the folder `./datasets`.

# 4 Running
## 4.1 Install all dependencies listed in prerequisites

## 4.2 Download the dataset

## 4.3 Training
# 5 Main results

## 5.4 Visualization
