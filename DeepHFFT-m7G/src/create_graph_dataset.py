import argparse
import os
from time import perf_counter as pc
from datetime import timedelta
from seq2graph import create_graph
import torch
from tqdm import tqdm
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--ksize", type=int, required=True, help="K-size")
    args = parser.parse_args()
    k = args.ksize
    return k

def load_data(file_path):
    sequences = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  
            sequence, label = line.strip().split(',')
            sequences.append(sequence)
            labels.append(int(label))

    print("Dataset:", file_path)
    print("Number of sequences in the dataset:", len(sequences))
    print("Number of classes:", len(set(labels)))
    return sequences, labels



def create_graphs(sequences, labels, k, dataset_type):
    graphs = []
    print(f"Building graphs for {dataset_type} dataset...")
    start = pc()
    for s, l in tqdm(zip(sequences, labels), total=len(sequences)):
        g = create_graph(s, k)
        g.y = torch.tensor(l)
        graphs.append(g)
    end = pc()
    t = end - start
    total_time = timedelta(seconds=t)
    print(f"Ended in: {str(total_time)}")

    directory = f"../experiments/501bp/K{k}/"
    os.makedirs(directory, exist_ok=True)

    path = f"{directory}{dataset_type}_RNA7N_{k}"

    with open(path, "wb") as file:
        pickle.dump(graphs, file)

    path = f"{directory}{dataset_type}_RNA7N_{k}.txt"

    with open(path, 'w') as file:
        file.write(f"{dataset_type} Dataset\n")
        file.write(f"Number of sequences: {len(graphs)} \n")
        file.write(f"Total time: {str(total_time)}")

def main():
    k = parse_arguments()
    #RNA
    train_file = "../datasets/rna7m/rna7train.txt"  # Hardcoded train dataset file path
    test_file = "../datasets/rna7m/rna7test.txt"  # Hardcoded test dataset file path

    train_sequences, train_labels = load_data(train_file)
    test_sequences, test_labels = load_data(test_file)
    create_graphs(train_sequences, train_labels, k, "train")
    create_graphs(test_sequences, test_labels, k, "test")
    exit(0)


if __name__ == "__main__":
    main()