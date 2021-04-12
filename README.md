# Graph Neural Networks
This repository contains implementations of several graph neural networks using Tensorflow. Some of the implementations are adapted from the existed ones using other deep learning frameworks. I would like to give credit to all the authors who dedicated to create the implementations.

## Implementations
| Name | Description | Thesis |
|:-----|:------------|:-------|
|[Graph Attention Network](./graph_neural_networks/layers/GraphAttentionLayer.py)| 1. Slightly modified from the keras implementations from [danielegrattarola/keras-gat](https://github.com/danielegrattarola/keras-gat).<br>2. to adapt to Tensorflow Dataset and GradientTape.<br>3. Add additional functionality to compute concat attention without using mask (memory efficient on the cost of computational time)  | [ICLR 2018](https://arxiv.org/abs/1710.10903) |