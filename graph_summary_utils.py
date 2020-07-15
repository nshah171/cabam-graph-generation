import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import powerlaw


def test_degree_distribution(degrees, xmin=None):
    if xmin:
        results = powerlaw.Fit(degrees, xmin=xmin)
    else:
        results = powerlaw.Fit(degrees)
    return results.power_law.alpha, results.power_law.xmin


def plot_degree_label_assortativity(G, y, dataset=None, bins=10):
    degs = []
    assortativities = []
    for node in G.nodes():
        deg = G.degree(node)
        neighbor_labels = [int(y[i] == y[node]) for i in list(G.neighbors(node))]
        assortativity = np.sum(neighbor_labels) / len(neighbor_labels)
        degs.append(deg)
        assortativities.append(assortativity)
    degs = np.array(degs)
    assortativities = np.array(assortativities)
    
    # visualize
    boxplot_arrs = []
    boxplot_labels = []
    deg_bins = np.logspace(np.log10(min(degs)), np.log10(max(degs)+1), bins)
    for deg_st, deg_end in zip(deg_bins, deg_bins[1:]):
        boxplot_arrs.append(assortativities[(degs >= deg_st) & (degs < deg_end)])
        boxplot_labels.append('{}'.format(int(deg_end)))
    
    plt.figure()
    plt.boxplot(boxplot_arrs, labels=boxplot_labels)
    if dataset:
        plt.title(dataset[0].upper() + dataset[1:], fontsize=16)
    plt.xlabel('Degree', fontsize=16)
    plt.ylabel('Class assortativity', fontsize=16)
    plt.savefig('figures/{}_class-assortativity.png'.format(dataset), bbox_inches='tight', dpi=150)
    return degs, assortativities
    
    
def overall_label_assortativity(G, y):
    assortativities = []
    intra_class = 0.0
    inter_class = 0.0
    for src, dst in G.edges():
        if y[src] == y[dst]:
            intra_class +=1
        else:
            inter_class +=1
    
    assortativity = intra_class / (intra_class + inter_class)
    print('Overall assortativity:', assortativity)
    return assortativity


def plot_node_level_assortativity(G, y, dataset=None):
    assortativities = []
    
    for node in G.nodes():
        same_class = 0
        cross_class = 0
        neighbors = np.array(list(G.neighbors(node)))
        
        for neighbor in neighbors:
            if y[node] == y[neighbor]:
                same_class += 1
            else:
                cross_class += 1
        assortativity = float(same_class) / (same_class + cross_class)
        assortativities.append(assortativity)
    
    plt.figure()
    plt.hist(assortativities, rwidth=0.8)
    plt.xlabel('Assortativity')
    plt.ylabel('Count')
    plt.title(dataset.upper())
    plt.savefig('figures/{}.png'.format(dataset))


def emb_degree_prediction(A, H):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    
    d = A.sum(axis=1)
    d = (d > np.mean(d)).astype(int)
    scores = []
    for train_idx, test_idx in StratifiedKFold(n_splits=5).split(H, d):
        clf = RandomForestClassifier().fit(H[train_idx,:], d[train_idx])
        dhat = clf.predict_proba(H[test_idx,:])[:,1]
        scores.append(roc_auc_score(d[test_idx], dhat))
        
    auc_val = np.mean(scores)
    print('Above mean degree prediction: {}'.format(auc_val))
    return auc_val


def simple_mmd(H, y):
    from itertools import combinations
    normalizer = np.linalg.norm(H, ord=2, axis=1).reshape(-1,1)
    normalizer[np.isnan(normalizer)] = 1
    normalizer[normalizer == 0] = 1
    H = H / normalizer
    mmd=0
    class_means = dict()
    classes = np.unique(y)
    for (c1, c2) in combinations(classes, 2):
        if c1 not in class_means:
            class_means[c1] = np.mean(H[y==c1], axis=0)
        if c2 not in class_means:
            class_means[c2] = np.mean(H[y==c2], axis=0)
        mmd = max(mmd, np.max(np.abs(class_means[c1] - class_means[c2])))
    return mmd


def dataset_summary(A, H):
    nodes = A.shape[0]
    edges = A.sum()
    features = H.shape[1]
    print('Nodes: ', nodes)
    print('Edges: ', edges)
    print('Features: ', features)
    return nodes, edges, features


def plot_degree_distribution(A, dataset):
    degs = A.sum(axis=1)
    c = Counter(degs)
    degs = sorted(c.items())
    plt.figure()
    plt.loglog([deg for (deg, ct) in degs], [ct for (deg, ct) in degs])
    plt.title(dataset[0].upper() + dataset[1:], fontsize=16)
    plt.xlabel('Degree', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.savefig('figures/{}_degree.png'.format(dataset), bbox_inches='tight', dpi=150)
