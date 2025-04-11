


import os
import numpy as np
import pandas as pd
from collections import Counter
import itertools


def get_feature(df, max_length=501):
    def binary(s):
        Encode = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], 'T': [0, 0, 0, 1],
                  'N': [0, 0, 0, 0]}
        return np.array([Encode[x] for x in s])

    def ENAC(sequence, window=5):
        AA = 'ACGT'
        code = []
        sequence = "NN" + sequence + "NN" 
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                count = Counter(sequence[j:j + window])
                for key in count:
                    count[key] = count[key] / len(sequence[j:j + window])
                for aa in AA:
                    code.append(count[aa])
        return code

    def EIIP(s):
        dic = {'A': [0.1260], 'C': [0.1340], 'G': [0.0806], 'T': [0.1335], 'U': [0.1335], 'N': [0]}
        return np.array([dic[x] for x in s])

    def calculate_chem(s):
        bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1], 'U': [0, 0, 1], 'N': [0, 0, 0]}
        chem_features = []
        for base in s:
            chem_features.append(bases.get(base, [0, 0, 0]))  # 默认为零向量
        return np.array(chem_features)

    def Kmer(sequence, k=3):
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmers.append(sequence[i:i + k])
        code = []
        header = []
        count = Counter()
        count.update(kmers)
        for key in count:
            count[key] = count[key] / len(kmers)
        NA = 'ACGT'
        for kmer in itertools.product(NA, repeat=3):
            header.append(''.join(kmer))
        for j in range(len(header)):
            if header[j] in count:
                code.append(count[header[j]])
            else:
                code.append(0)
        return code

    def CKSNAP(sequence, gap=3):
        AA = 'ACGT'
        code = []
        aaPairs = []
        for aa1 in AA:
            for aa2 in AA:
                aaPairs.append(aa1 + aa2)

        header = []
        for g in range(gap + 1):
            for aa in aaPairs:
                header.append(aa + '.gap' + str(g))

        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        return code
        
    one_hot = []
    chem = []
    eiip = []
    enac = []
    kmer = []
    cksnap = []
    for i in df['data'].values:
        one_hot.append(binary(i))
        chem.append(calculate_chem(i))
        eiip.append(EIIP(i))
        enac.append(np.array(ENAC(i, window=5)).reshape(4, -1).T)
        kmer.append(Kmer(i, k=3))
        cksnap.append(CKSNAP(i, gap=3))
    one_hot = np.array(one_hot)
    chem = np.array(chem)
    eiip = np.array(eiip)
    enac = np.array(enac)
    kmer = np.array(kmer)
    cksnap = np.array(cksnap)
    return one_hot, chem, eiip, enac, kmer, cksnap



def process_and_save_features(train_path, test_path, output_dir):

    feature_names = ["one_hot", "chem", "eiip", "enac", "kmer", "cksnap"]
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)
    print("Extracting features for training data...")
    train_features = get_feature(train_df)
    print("Extracting features for testing data...")
    test_features = get_feature(test_df)

    for train_feature, test_feature, name in zip(train_features, test_features, feature_names):

        feature_dir = os.path.join(output_dir, name)
        os.makedirs(feature_dir, exist_ok=True)
        train_feature_path = os.path.join(feature_dir, f"train_{name}.npy")
        np.save(train_feature_path, train_feature)
        print(f"Saved {name} features for training data to {train_feature_path}")
        test_feature_path = os.path.join(feature_dir, f"test_{name}.npy")
        np.save(test_feature_path, test_feature)
        print(f"Saved {name} features for testing data to {test_feature_path}")

if __name__ == "__main__":
    train_file = "../datasets/rna7m/rna7train.txt"
    test_file = "../datasets/rna7m/rna7test.txt"
    output_directory = "../experiments/501bp/"

    process_and_save_features(train_file, test_file, output_directory)



