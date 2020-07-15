import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import networkx as nx
import pickle


def make_undirected_and_self_looped(A):
    '''takes scipy sparse matrix'''
    A_hat = A.maximum(A.T)  # make symmetric
    A_hat.setdiag(1)  # populate diagonal with self-loops
    return A_hat


def normalize_adjacency(A, style='left'):
    '''takes scipy sparse matrix'''
    d = np.asarray(A.sum(axis=1)).flatten().astype(float)  # degree vector
    if style == 'left':
        return (np.diag(d**-1.0)) @ A  # D^-1 * A
    return np.diag(d**-0.5) @ A @ np.diag(d**-0.5)  # D^0.5 * A * D^0.5


def normalize_features(H, p=1, style='row'):
    '''takes scipy sparse matrix where rows are node features'''
    if style == 'row':
        return H / (norm(H, ord=p, axis=1).reshape(H.shape[0], 1))
    else:
        return H / (norm(H, ord=p, axis=0).reshape(1, H.shape[1]))


def tensorize(*args, tensor_type):
    '''cast input arrays to torch tensors'''
    return [tensor_type(x) for x in args]


def get_train_val_test_masks(y, train_ratio, val_ratio, test_ratio):
    '''get indices for train, val and test sets given ratios'''
    n = len(y)
    ratio_arr = np.array([0, train_ratio, val_ratio, test_ratio]) / (train_ratio + val_ratio + test_ratio)
    split_idx = np.cumsum(ratio_arr * n, dtype=int)
    mask = np.ones(n)
    for mask_id, (st, end) in enumerate(zip(split_idx, split_idx[1:])):
        mask[st:end] = mask_id
    np.random.shuffle(mask)
    idx_train, idx_val, idx_test = [np.where(mask == mask_id)[0] for mask_id in [0, 1, 2]]
    return idx_train, idx_val, idx_test


def encode_onehot(y):
    '''produce onehot encoding of label vector y given a flattened class vector y'''
    n_samples = len(y)
    n_classes = len(y.unique())
    y_onehot = torch.LongTensor(n_samples, n_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


def load_simulated_random_data():
    '''simulate some random data with random binary labels'''
    A = np.random.binomial(n=1, p=0.2, size=(50,50))
    H = np.random.rand(50, 128)
    y = np.random.binomial(n=1, p=0.2, size=(50,))
    
    # preprocess adjacency and features 
    A = make_undirected_and_self_looped(A)
    A = normalize_adjacency(A)
    H = normalize_features(H)
    
    idx_train, idx_val, idx_test = get_train_val_test_masks(y, 0.7, 0.1, 0.2)
    
    A, H = tensorize(A, H, tensor_type=torch.FloatTensor)
    y, idx_train, idx_val, idx_test = tensorize(y, idx_train, idx_val, idx_test, tensor_type=torch.LongTensor)
    
    return A, H, y, idx_train, idx_val, idx_test 


def load_simulated_clustered_data(n_clusters, n_samples_per_cluster, n_dims):
    '''simulate clustered multivariate gaussian data with adjacency stochastically sampled based on distance in R^d'''
    H = []
    y = []
    for i in range(n_clusters):
        H.append(np.random.multivariate_normal(mean=np.random.randint(10, size=n_dims), 
                                                      cov = np.eye(n_dims), 
                                                      size=n_samples_per_cluster))
        y.extend([i] * n_samples_per_cluster)
    H = np.vstack(H)
    y = np.array(y)
    
    # preprocess adjacency and features 
    A = make_undirected_and_self_looped(A)
    A = normalize_adjacency(A)
    H = normalize_features(H)
    
    A = cdist(H, H, metric='euclidean')
    A = 1.0 - (A - A.min(axis=1)) / (A.max(axis=1) - A.min(axis=1))
    A = np.random.binomial(n=1, p=A)
    idx_train, idx_val, idx_test = get_train_val_test_masks(y, 0.7, 0.1, 0.2)
    
    A, H = tensorize(A, H, tensor_type=torch.FloatTensor)
    y, idx_train, idx_val, idx_test = tensorize(y, idx_train, idx_val, idx_test, tensor_type=torch.LongTensor)
    
    return A, H, y, idx_train, idx_val, idx_test


def accuracy(output, labels):
    '''get accuracy given output tensor and labels '''
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_dataset(path, train_ratio, test_ratio, val_ratio):
    '''Load a graph from a Numpy binary file.
    Adapted from https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py
    '''
    
    if not path.endswith('.npz'):
        path += '.npz'
    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)
        A = csr_matrix((loader['adj_data'], loader['adj_indices'],
                        loader['adj_indptr']), shape=loader['adj_shape'])

        H = csr_matrix((loader['attr_data'], loader['attr_indices'],
                        loader['attr_indptr']), shape=loader['attr_shape'])

        y = loader.get('labels')
        
        # preprocess adjacency and features 
        A = make_undirected_and_self_looped(A)
        #A = normalize_adjacency(A)
        #H = normalize_features(H)
        return A, H, y
        
#         idx_train, idx_val, idx_test = get_train_val_test_masks(y, train_ratio, val_ratio, test_ratio)
        
#         A, H = tensorize(A, H, tensor_type=torch.FloatTensor)
#         y, idx_train, idx_val, idx_test = tensorize(y, idx_train, idx_val, idx_test, tensor_type=torch.LongTensor) 
        
#         return A, H, y, idx_train, idx_val, idx_test
#         graph = {'A': A, 'H': H, 'y': y}

#         idx_to_node = loader.get('idx_to_node')
#         if idx_to_node:
#             idx_to_node = idx_to_node.tolist()
#             graph['idx_to_node'] = idx_to_node

#         idx_to_attr = loader.get('idx_to_attr')
#         if idx_to_attr:
#             idx_to_attr = idx_to_attr.tolist()
#             graph['idx_to_attr'] = idx_to_attr

#         idx_to_class = loader.get('idx_to_class')
#         if idx_to_class:
#             idx_to_class = idx_to_class.tolist()
#             graph['idx_to_class'] = idx_to_class

#         return graph


def produce_processed_data(dataset):
    print('Loading dataset {}'.format(dataset))
    if dataset in ['airport', 'flickr', 'blogcatalog']:
        with open('data/alternative/{}_features.pkl'.format(dataset), 'rb') as f_obj, open('data/alternative/{}_adj.pkl'.format(dataset), 'rb') as a_obj, open('data/alternative/{}_labels.pkl'.format(dataset), 'rb') as l_obj:
                features = pickle.load(f_obj)
                adj = pickle.load(a_obj)
                labels = pickle.load(l_obj)

                if sp.issparse(features):
                    H = features.toarray()
                else:
                    H = features.numpy()

                if sp.issparse(adj):
                    A = adj.toarray()
                else:
                    A = adj

                if type(labels) != np.ndarray:
                    y = labels.numpy().ravel()
                else:
                    y = labels.ravel()
    else:
        A, H, y = load_dataset('data/{}/raw/{}.npz'.format(dataset, dataset), 
                               train_ratio=0.1, val_ratio=0.1, test_ratio=0.8)
        A = A.toarray()
        H = H.toarray()
    
    H[np.isnan(H)] = 0
    G = nx.from_numpy_matrix(A)
    
    return A, H, y, G