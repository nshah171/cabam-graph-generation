# Scale-Free, Attributed and Class-Assortative Graph Generation

This repository contains code for the paper "Scale-Free, Attributed and Class-Assortative Graph Generation to Facilitate Introspection of Graph Neural Networks," published at KDD MLG 2020.  For clarity:

- `CABAM Simulation Examples.ipynb`: Several examples of generated graphs with various desiderata, their assortativity properties, illustrations of theoretical and empirical quantities as discussed in the paper, and example class-conditional feature generation given a graph and assigned node labels.
- `Dataset Summaries.ipynb`: Driver to generate the dataset statistics for existing GNN benchmarks provided in the paper.
- `cabam_utils.py`: Main graph generation code and helpers.
- `graph_preprocessing_utils.py`: Misc. helpers to load and process existing graph datasets from data/.
- `graph_summary_utils.py`: Misc. helpers for driver to generate benchmark dataset statistics and properties.


If you use the model, or graphs generated with the model for evaluation in your own work, please cite

```
 @inproceedings{cabam2020shah,
     author = {Shah, Neil},
     title = {Scale-Free, Attributed and Class-Assortative Graph Generation to Facilitate Introspection of Graph Neural Networks},
     booktitle = {KDD Mining and Learning with Graphs},
     year = {2020}
   }
```