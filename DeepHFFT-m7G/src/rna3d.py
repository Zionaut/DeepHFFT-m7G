import sys
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
min_count = 1   
dim = 25        
window = 5      
def get_k_trids(k):
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = base ** k
    for i in range(end):
        n = i
        kmer = []
        for _ in range(k):
            kmer.insert(0, chars[n % base])  
            n //= base
        nucle_com.append("".join(kmer))
    return nucle_com
def get_k_nucleotide_composition(kmers_dict, seq):
    tri_feature = []
    k = len(next(iter(kmers_dict)))  
    seq = seq.upper().replace('T', 'U')  
    for x in range(len(seq) + 1 - k):
        kmer = seq[x:x + k]
        if kmer in kmers_dict:
            tri_feature.append(kmer) 
    return tri_feature
def read_csv_file(csv_file):
    df = pd.read_csv(csv_file, header=None, names=['sequence', 'label'])
    return df['sequence'].tolist()
def train_rnas_for_k(seq_file, k):
    print(f"\n {k}-mer, dim: {dim}, window: {window}")
    sequences = read_csv_file(seq_file)
    kmers = get_k_trids(k)
    kmers_dict = {kmer: idx for idx, kmer in enumerate(kmers)}
    sentences = []
    for seq in sequences:
        trvec = get_k_nucleotide_composition(kmers_dict, seq)
        sentences.append(trvec)
    print(f"{len(sentences)}")
    model = Word2Vec(
        sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,           
        epochs=10,
        workers=4,
    )
    vocab = list(model.wv.index_to_key)
    print(f"{k}-mer: {len(vocab)}")
    dict_file = f'rna_dict_{k}mer.txt'
    with open(dict_file, 'w') as fw:
        for val in vocab:
            fw.write(val + '\n')
    embeddingWeights = np.empty([len(vocab), dim])
    for i, word in enumerate(vocab):
        embeddingWeights[i, :] = model.wv[word]
    pickle_file = f'rnaEmbedding_{k}mer.pickle'
    with open(pickle_file, 'wb') as f:
        pickle.dump(embeddingWeights, f)
    print(f"{k}-mer embedding  {pickle_file}")
if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else '../datasets/rna7m/data.txt'
    for kmer_len in [4, 5, 6]:
        train_rnas_for_k(input_file, kmer_len)
