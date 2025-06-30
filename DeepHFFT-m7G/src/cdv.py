import os
from multiprocessing import Process
import numpy as np
import pickle
from sklearn.preprocessing import normalize  
NULL_vec = np.zeros((25,)) 
def load_vocab_dict(dict_file):
    with open(dict_file, 'r') as f:
        kmers = [line.strip() for line in f]
    return {kmer: idx for idx, kmer in enumerate(kmers)}
def load_embedding_matrix(pickle_file):
    with open(pickle_file, 'rb') as f:
        embedding_matrix = pickle.load(f)
    return embedding_matrix
def get_kmer(seq, K):
    seq = seq.upper()
    l = len(seq)
    return [seq[i:i + K] for i in range(l - K + 1)]
def process_txt_file(file_path, output_folder, dict_file, embedding_file, K):
    print(f"The process {K} is running")
    kmer2idx = load_vocab_dict(dict_file)
    embedding_matrix = load_embedding_matrix(embedding_file)
    output_file = os.path.join(output_folder, f"{K}mer_{os.path.basename(file_path).split('.')[0]}_features.npy")
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  
    features_list = []
    for line in lines:
        data, label = line.strip().split(",")
        label = int(label)
        kmers = get_kmer(data, K)
        code = []
        for kmer in kmers:
            if 'N' not in kmer and 'n' not in kmer:
                idx = kmer2idx.get(kmer)
                if idx is not None:
                    vec = embedding_matrix[idx]
                    code.append(vec)
                else:
                    code.append(NULL_vec)
            else:
                code.append(NULL_vec)
        if len(code) == 0:
            ave = NULL_vec
        else:
            array = np.array(code)
            ave = array.mean(axis=0) 
            ave = normalize(ave.reshape(1, -1), norm='l2')[0]  
        features_list.append(ave)
        print(f"Processed one sequence in process {K}")
    np.save(output_file, np.array(features_list))
    print(f"Saved {output_file}")
    print(f"The process {K} is done")
if __name__ == "__main__":
    train_file = "../datasets/rna7m/rna7train.txt"
    test_file = "../datasets/rna7m/rna7test.txt"
    output_folder = "../experiments/501bp/dna2vec/"
    os.makedirs(output_folder, exist_ok=True)
    embedding_files = {
        4: ("rna_dict_4mer.txt", "rnaEmbedding_4mer.pickle"),
        5: ("rna_dict_5mer.txt", "rnaEmbedding_5mer.pickle"),
        6: ("rna_dict_6mer.txt", "rnaEmbedding_6mer.pickle"),
    }
    ps = []
    for K in [4, 5, 6]:
        dict_file, embedding_file = embedding_files[K]
        p1 = Process(target=process_txt_file, args=(train_file, output_folder, dict_file, embedding_file, K))
        p2 = Process(target=process_txt_file, args=(test_file, output_folder, dict_file, embedding_file, K))
        ps.append(p1)
        ps.append(p2)
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("The main process is done")
