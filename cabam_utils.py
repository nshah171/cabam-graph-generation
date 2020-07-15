import numpy as np
import networkx as nx
from tqdm.notebook import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def theoretical_intra_class_estimates(c_probs, native_probs, m, approx_limit=100000):
    '''
    Estimate the theoretical (in-the-limit) intra-class edge probabilities. Has some variance due to the stochasticity in empirical power-law generation and theoretical probabilities from the underlying BA model.
    
    Note: Supports 3 variants of c_probs:
    -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
    -Precomputed (degree-dependent) dictionary.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
    -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}
    
    `approx_limit` is used to approximate the infinite sums in Eqs. 5-6 from the paper.
    '''
    
    g = np.sum(np.power(native_probs, 2))  # probability of an intra-class edge to arise given certain class propensities
    if callable(c_probs):
        # Dynamically computed, degree dependent
        within = np.sum([(g * c_probs(k)[1] * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
        cross = np.sum([((1-g) * (1-c_probs(k)[1]) * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
    else:
        if len(c_probs) == 2:
            # Fixed
            within = np.sum([(g * c_probs[1] * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
            cross = np.sum([((1-g) * (1-c_probs[1]) * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
        else:
            # Precomputed, degree-dependent
            within = np.sum([(g * c_probs[k][1] * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
            cross = np.sum([((1-g) * (1-c_probs[k][1]) * (m+1)) / ((k+1) * (k+2)) for k in range(m, approx_limit)])
    
    intra_class_ratio = within / (within + cross)
    return intra_class_ratio


def draw_network_with_labels(G, node_labels):
    '''
    Given a networkx graph and labels, draw nodes with relevant colors.
    
    G: networkx graph on N nodes
    node_labels: label vector of length N (assuming K labels, this would be labeled 0...K-1)
    
    Note: color_mapping currently fixed to maximum 5 colors, but can trivially be extended.
    '''
    
    color_mapping = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}
    node_colors = [color_mapping[i] for i in node_labels]
    nx.draw_kamada_kawai(G, with_labels=False, node_size=50, node_color=node_colors)


def cabam_graph_generation(n, m, c=2, native_probs=[0.5, 0.5], c_probs={1: 0.5, 0: 0.5}, logger=None):
    '''
    Main function for CABAM graph generation.
    
    n: maximum number of nodes
    m: number of edges to add at each timestep (also the minimum degree)
    c: number of classes
    native_probs: c-length vector of native class probabilities (must sum to 1)
    c_probs: p_c from the paper.  Entry for 1 (0) is the intra-class (inter-class) link strength.  Entries must sum to 1.
    
    Supports 3 variants of c_probs:
    -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
    -Precomputed (degree-dependent) dictionary.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
    -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}
    '''
    
    if m < 1 or n < m:
        raise nx.NetworkXError(
                "NetworkXError must have m>1 and m<n, m=%d,n=%d" % (m, n))

    logger = logger if logger else logging
    
    # graph initialization
    G = nx.empty_graph(m)
    intra_class_edges = 0
    inter_class_edges = 0
    total_intra_class = 0
    total_inter_class = 0
    
    intra_class_ratio_tracker = []
    alpha_tracker = []

    class_tbl = list(range(c))
    node_labels = np.array([np.random.choice(class_tbl, p=native_probs) for x in range(G.number_of_nodes())])
    node_degrees = np.array([1] * m) # technically degree 0, but using 1 here to make the math work out.

    # start adding nodes
    source = m
    source_label = np.random.choice(class_tbl, p=native_probs)
    pbar = tqdm(total=n)
    pbar.update(m)
    empirical_edge_fraction_to_degree_k = np.zeros(10)
    n_added = 0
    
    while source < n:
        logger.debug('Adding node {} with label {}.'.format(source, source_label))
        if type(c_probs) == dict:
            if len(c_probs) == 2:
                # no funny business, just constants
                node_class_probs = np.array([c_probs[abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
            else:
                # pre-generated custom probabilities
                node_class_probs = np.array([c_probs[node_degrees[i]][abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
        else:
            # callable (function) probs
            node_class_probs = np.array([c_probs(node_degrees[i])[abs(node_labels[i] == source_label)] for i in range(len(node_labels))])
            
        
        # determine m target nodes to connect to
        targets = []
        while len(targets) != m: 
            node_class_degree_probs = node_class_probs * node_degrees
            candidate_targets = np.where(node_class_degree_probs > 0)[0]

            if len(candidate_targets) >= m:
                logger.debug('Have enough targets...sampling from AWPA.\n')
                # if we have enough qualifying nodes, sample from assortativity-weighted PA probs
                candidate_node_class_degree_probs = node_class_degree_probs[candidate_targets]
                candidate_node_class_degree_probs = candidate_node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
                targets = np.random.choice(candidate_targets, 
                                           size=m, 
                                           p=candidate_node_class_degree_probs, 
                                           replace=False)
            else:
                logger.debug('Not enough targets...sampling from PA.\n')
                # else, use as many qualifying nodes as possible, and just sample from the PA probs for the rest.
                n_remaining_targets = m - len(candidate_targets)
                other_choices = np.where(node_class_degree_probs == 0)[0]
                other_node_degree_probs = node_degrees[other_choices]
                other_node_degree_probs = other_node_degree_probs / np.linalg.norm(other_node_degree_probs, ord=1)
                other_targets = np.random.choice(other_choices, 
                                                 size=n_remaining_targets, 
                                                 p=other_node_degree_probs,
                                                 replace=False)
                #print(candidate_targets, candidate_targets.shape, other_targets, other_targets.shape)
                targets = np.concatenate((candidate_targets, other_targets))
            assert len(targets) == m

        G.add_edges_from([(source, target) for target in targets])
        edge_types = np.array([source_label == node_labels[target] for target in targets])
        intra_class_edges += np.count_nonzero(edge_types) # intra-class edges
        inter_class_edges += np.count_nonzero(edge_types == 0) # inter-class edges
        
        total_intra_class += np.count_nonzero(edge_types)
        total_inter_class +=  np.count_nonzero(edge_types == 0)
        total_intra_frac = total_intra_class / (total_intra_class + total_inter_class)
        
        #intra_class_ratio_tracker.append(total_intra_frac)
        #alpha_tracker.append(test_degree_distribution(node_degrees)[0])
        
        ncdp = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
        empirical_edge_fraction_to_degree_k += np.array([m*np.sum(ncdp[node_degrees == k]) for k in range(m, m+10)])
        
        if source % 500 == 0:
            theoretical_edge_fraction_to_degree_k = [((m*(m+1))/((k+1)*(k+2))) for k in range(m, m+10)]
            ncdp = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
            avgd_empirical_edge_fraction_to_degree_k = empirical_edge_fraction_to_degree_k / n_added
            
            logger.info('Theor. edge prob to deg k: {}'.format(np.round(theoretical_edge_fraction_to_degree_k, 3)))
            logger.info('Empir. edge prob to deg k: {}'.format(np.round(avgd_empirical_edge_fraction_to_degree_k, 3)))
            snapshot_intra_frac = intra_class_edges / (intra_class_edges + inter_class_edges)
            logger.info('Snapshot: ({}/{})={:.3f}\t Overall: {:.3f}'.format(intra_class_edges, intra_class_edges+inter_class_edges,
                                                                             snapshot_intra_frac, total_intra_frac))
            intra_class_edges = 0
            inter_class_edges = 0
            logger.info('Max node degree: {}'.format(max(node_degrees)))

        # book-keeping
        node_degrees[targets] += 1
        node_labels = np.append(node_labels, source_label)
        node_degrees = np.append(node_degrees, m)
        pbar.update(1)

        # move onto next node!
        n_added += 1
        source += 1
        source_label = np.random.choice(class_tbl, p=native_probs)
    
    pbar.close()
    return G, node_degrees, node_labels, total_intra_class, total_inter_class, intra_class_ratio_tracker, alpha_tracker