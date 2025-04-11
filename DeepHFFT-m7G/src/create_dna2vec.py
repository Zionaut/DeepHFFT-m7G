





import os
import sys
from multiprocessing import Process
import numpy as np
from multi_k_model import MultiKModel


NULL_vec = np.zeros((100))


def get_kmer(dnaSeq, K):

    dnaSeq = dnaSeq.upper()
    l = len(dnaSeq)
    return [dnaSeq[i:i+K] for i in range(0, l-K+1, K)]

def process_txt_file(file_path, output_folder, mk_model, K):
   
    print(f"The process {K} is running")
    

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
                try:
                    code.append(mk_model.vector(kmer))
                except KeyError:
                    code.append(NULL_vec) 
            else:
                code.append(NULL_vec)
        

        array = np.array(code)
        ave = array.sum(axis=0)
        

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
    

    filepath = './dna2vec/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
    mk_model = MultiKModel(filepath)
    
    print("The main process is running...")
    

    ps = []
    for K in range(4, 7):  
        p1 = Process(target=process_txt_file, args=(train_file, output_folder, mk_model, K))
        p2 = Process(target=process_txt_file, args=(test_file, output_folder, mk_model, K))
        ps.append(p1)
        ps.append(p2)
    
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    
    print("The main process is done")
