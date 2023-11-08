# Official implementation of HyperGPA [[arxiv](https://arxiv.org/abs/2211.12034)]

## 1. Software Environment

1. python: 3.7.11

2. torch: 1.8.0 / dgl: 0.7.1

3. other package:
   ```
   pip install -r requirements.txt
   ```

## 2. Reproduce HyperGPA's result

You can experiment with four datasets, (Flu, Stock-US, Stock-China, USHCN)

```
cd hypergpa
main.py [-h] [--data DATA] [--l_model L_MODEL] [--l_h L_H] [--l_l L_L]
             [--emb_size EMB_SIZE] [--attn_dim ATTN_DIM] [--num_class NUM_CLASS]
             [--task2_ratio TASK2_RATIO] [--len LEN] [--graph1 GRAPH1] [--graph2 GRAPH2] [--gpu GPU] [--r R] [-not_default]
DATA: dataset, {flu, usa30, china30, ushcn}
L_MODEL: target model, {lstm, gru, seq2seq, geq2geq, odernn, ncde}
-not_default: the same flag as the previous one
GPU: gpu number to use
R: random seed

L_H: hidden size in target model
L_L: the number of layer in target model
EMB_SIZE: dim(h')
ATTN_DIM: dim(z)
NUM_CLASS: C
TASK2_RATIO: \lambda
LEN: K (window size)
GRAPH1: graph neural network in ncde {empty, avwgcn} * (empty means no graph)
GRAPH2: other graph neural network {empty, gat, gcn, avwgcn} 
```
 The experiment results are saved in a 'result' folder named with "{flu_ILINet, stock_usa30, stock_china30, ushcn_ushcn}/{DATA}\_{L\_MODEL}\_{L_H}\_{L_L}\_{EMB\_SIZE}\_{ATTN\_DIM}\_{GRAPH1}\_{GRAPH2}\_{LEN}\_{NUM\_CLASS}\_{TASK2\_RATIO}^{R}".
