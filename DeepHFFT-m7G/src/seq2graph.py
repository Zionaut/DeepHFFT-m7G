import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import torch

alphabet="ACGT"


def to_categorical(y):
    return np.eye(len(alphabet), dtype='uint8')[y]

def one_hot_encoding(seq):
    mp = dict(zip(alphabet, range(len(alphabet))))
    seq_2_number = [mp[nucleotide] for nucleotide in seq]
    return to_categorical(seq_2_number).flatten()


def get_kmer_count_from_sequence(sequence, k, cyclic=True):


    kmers = {}


    for i in range(0, len(sequence)):
        kmer = sequence[i:i + k]

        length = len(kmer)
        if cyclic:
            if len(kmer) != k:
                kmer += sequence[:(k - length)]


        else:
            if len(kmer) != k:
                continue

        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1

    return kmers

def get_debruijn_edges_from_kmers(kmers):


    edges = set()


    for k1 in kmers:
        for k2 in kmers:
            if k1 != k2:

                if k1[1:] == k2[:-1]:
                    edges.add((k1[:-1], k2[:-1]))
                if k1[:-1] == k2[1:]:
                    edges.add((k2[:-1], k1[:-1]))

    return edges

def create_graph(sequence, k):
    features = []
    kmers = get_kmer_count_from_sequence(sequence, k, cyclic=False)
    e = get_debruijn_edges_from_kmers(kmers)

    g = nx.from_edgelist(e)
    for n in g.nodes:
        c = one_hot_encoding(n)
        features.append(c)

    features = np.stack(features).astype(np.float32)
    G=from_networkx(g)
    G["x"]=torch.tensor(features)
    
    return G