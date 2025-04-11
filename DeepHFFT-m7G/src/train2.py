import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle
import numpy as np
import os
import platform
import argparse
import random
import cpuinfo
import yaml
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from model import *


seed = 41
nfolds = 5
mini_batch_size = 32
hidden_channels = 128
#lr = 0.00007
lr = 0.00007

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--ksize", type=int, required=True, help="K-size")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs")
    args = parser.parse_args()
    k = args.ksize
    e = args.epochs
    return k, e


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def test_randomness():
    print("Random:", random.random())
    print("Numpy Random:", np.random.rand())
    print("Torch Random:", torch.rand(1))
    
    if torch.cuda.is_available():
        print("Torch CUDA Random:", torch.cuda.FloatTensor(1).uniform_())
    else:
        print("No CUDA")


def check_model_weights(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.view(-1)[:5]}")  # 打印每个参数前5个值
        


def identify_device():
    so = platform.system()
    if so == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = torch.cuda.get_device_name() if device.type == 'cuda' else cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name


def load_data(k):

    base_path = "../experiments/501bp/"
    one_hot_path = os.path.join(base_path, "one_hot/train_one_hot.npy")
    chem_path = os.path.join(base_path, "chem/train_chem.npy")
    eiip_path = os.path.join(base_path, "eiip/train_eiip.npy")
    enac_path = os.path.join(base_path, "enac/train_enac.npy")

    dna2vec_4mer_path = os.path.join(base_path, "dna2vec/4mer_rna7train_features.npy")
    dna2vec_5mer_path = os.path.join(base_path, "dna2vec/5mer_rna7train_features.npy")
    dna2vec_6mer_path = os.path.join(base_path, "dna2vec/6mer_rna7train_features.npy")


    one_hot_features = np.load(one_hot_path)
    chem_features = np.load(chem_path)
    eiip_features = np.load(eiip_path)
    enac_features = np.load(enac_path)
    dna2vec_4mer_features = np.load(dna2vec_4mer_path)
    dna2vec_5mer_features = np.load(dna2vec_5mer_path)
    dna2vec_6mer_features = np.load(dna2vec_6mer_path)


    path_graph = f"../experiments/501bp/K{k}/train_RNA7N_{k}"
    with open(path_graph, "rb") as file:
        sequences = pickle.load(file)


    dict_data = [{
        'sequence': s,
        'one_hot': one_hot_features[idx],
        'chem': chem_features[idx],
        'eiip': eiip_features[idx],
        'enac': enac_features[idx],
        'dna2vec_4mer': dna2vec_4mer_features[idx],
        'dna2vec_5mer': dna2vec_5mer_features[idx],
        'dna2vec_6mer': dna2vec_6mer_features[idx],
        'label': int(s.y)
    } for idx, s in enumerate(sequences)]

    return dict_data


def load_test_data(k):

    base_path = "../experiments/501bp/"
    one_hot_path = os.path.join(base_path, "one_hot/test_one_hot.npy")
    chem_path = os.path.join(base_path, "chem/test_chem.npy")
    eiip_path = os.path.join(base_path, "eiip/test_eiip.npy")
    enac_path = os.path.join(base_path, "enac/test_enac.npy")

    dna2vec_4mer_path = os.path.join(base_path, "dna2vec/4mer_rna7test_features.npy")
    dna2vec_5mer_path = os.path.join(base_path, "dna2vec/5mer_rna7test_features.npy")
    dna2vec_6mer_path = os.path.join(base_path, "dna2vec/6mer_rna7test_features.npy")


    one_hot_features = np.load(one_hot_path)
    chem_features = np.load(chem_path)
    eiip_features = np.load(eiip_path)
    enac_features = np.load(enac_path)
    dna2vec_4mer_features = np.load(dna2vec_4mer_path)
    dna2vec_5mer_features = np.load(dna2vec_5mer_path)
    dna2vec_6mer_features = np.load(dna2vec_6mer_path)


    path_graph = f"../experiments/501bp/K{k}/test_RNA7N_{k}"
    with open(path_graph, "rb") as file:
        sequences = pickle.load(file)


    dict_data = [{
        'sequence': s,
        'one_hot': one_hot_features[idx],
        'chem': chem_features[idx],
        'eiip': eiip_features[idx],
        'enac': enac_features[idx],
        'dna2vec_4mer': dna2vec_4mer_features[idx],
        'dna2vec_5mer': dna2vec_5mer_features[idx],
        'dna2vec_6mer': dna2vec_6mer_features[idx],
        'label': int(s.y)
    } for idx, s in enumerate(sequences)]

    return dict_data



class CustomDataset(Dataset):
    def __init__(self, dict_data):
        self.dict_data = dict_data

    def __len__(self):
        return len(self.dict_data)

    def __getitem__(self, idx):
        data = self.dict_data[idx]


        sequence = data['sequence']
        one_hot = torch.tensor(data['one_hot']).float()
        chem = torch.tensor(data['chem']).float()
        eiip = torch.tensor(data['eiip']).float()
        enac = torch.tensor(data['enac']).float()
        dna2vec_4mer = torch.tensor(data['dna2vec_4mer']).float()
        dna2vec_5mer = torch.tensor(data['dna2vec_5mer']).float()
        dna2vec_6mer = torch.tensor(data['dna2vec_6mer']).float()

        label = data['label']


        return one_hot, chem, eiip, enac, dna2vec_4mer, dna2vec_5mer,dna2vec_6mer,label


def create_data_loader(dict_data, indices, shuffle, batch_size=32):
    subset_data = [dict_data[i] for i in indices]
    dataset = CustomDataset(subset_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,  
)

import pandas as pd
import yaml
import numpy as np

def compute_metrics(y_true, y_pred, y_prob, f):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')
    mcc_coeff = matthews_corrcoef(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()  
    specificity = tn / (tn + fp)

    metrics = {
        'AUC': auc_score,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1score,
        'Specificity': specificity,
        'MCC': mcc_coeff
    }


    print(f"Fold {f}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


    return [f] + list(metrics.values()), y_true, y_prob


def save_metrics(metrics, k):
    columns = ["Fold", "AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "MCC"]  
    data = pd.DataFrame(metrics, columns=columns)
    path = f"../experiments/501bp/result/metrics_{k}.csv"
    data.to_csv(path, index=False)


    yaml_data = {col: {"Mean": float(data[col].mean()), "Standard Deviation": float(data[col].std())} for col in columns[1:]}
    path_yaml = f"../experiments/501bp/result/results_{k}.yaml"
    with open(path_yaml, "w") as file:
        yaml.dump(yaml_data, file)


    for col in columns[1:]:
        print(f"AVG. {col} = {yaml_data[col]['Mean']} SD = {yaml_data[col]['Standard Deviation']}")

def save_report(times, k, epochs, devname):
    path = "../experiments/501bp/result/times_%d.csv" % (k)
    columns = ["Fold", "Training time", "Testing time"]

    df = pd.DataFrame(times, columns=columns)
    df.to_csv(path, index=False)

    train_time_mu = np.mean(df["Training time"].values)
    train_time_sigma = np.std(df["Training time"].values)

    test_time_mu = np.mean(df["Testing time"].values)
    test_time_sigma = np.std(df["Testing time"].values)

    yaml_data = {}
    yaml_data["Device"] = devname
    yaml_data["Training Time"] = {
        "Mean": float(train_time_mu),
        "SD": float(train_time_sigma)
    }

    yaml_data["Testing Time"] = {
        "Mean": float(test_time_mu),
        "SD": float(test_time_sigma)
    }

    yaml_data["Epochs"] = int(epochs)

    path = "../experiments/501bp/result/times_avg_%d.yaml" % (k)

    with open(path, "w") as file:
        yaml.dump(yaml_data, file)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    print(f"超参数设置：")
    print(f"seed = {seed}")
    print(f"nfolds = {nfolds}")
    print(f"mini_batch_size = {mini_batch_size}")
    print(f"hidden_channels = {hidden_channels}")
    print(f"learning_rate = {lr}")

    set_seed(seed)  
    test_randomness()  
    device, devname = identify_device()
    print("Using %s - %s" % (device, devname))
    times = []
    metrics = []

    k, epochs = parse_arguments()
    train_data = load_data(k)
    test_data = load_test_data(k)

    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    print(f"M7G - k={k} + train ({epochs} epochs)")

    all_y_true = []
    all_y_prob = []

    plt.figure(figsize=(4, 4))

    for f, (train_idx, val_idx) in enumerate(skf.split(train_data, [d['label'] for d in train_data])):
        print("================================================")
        print("FOLD", f + 1)

        trainloader = create_data_loader(train_data, train_idx, shuffle=True)
        valloader = create_data_loader(train_data, val_idx, shuffle=False)
        testloader = create_data_loader(test_data, range(len(test_data)), shuffle=False)

        print(f"Training set size: {len(train_idx)}")
        print(f"Validation set size: {len(val_idx)}")
        print(f"Test set size: {len(test_data)}")

        net = MainModel()

        net, train_time = train_net(device, net, trainloader, epochs, lr)

        model_dir = './models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'model_fold{f+1}.pth')
        torch.save(net.state_dict(), model_path)
        print(f"Saved model for Fold {f + 1} to {model_path}")

        y_true, y_pred, y_prob, test_time = predict(device, net, testloader)
        m, y_true_fold, y_prob_fold = compute_metrics(y_true, y_pred, y_prob, f + 1)

        metrics.append(m)
        times.append((f + 1, train_time, test_time))

        fpr, tpr, _ = roc_curve(y_true_fold, y_prob_fold)
        auc = roc_auc_score(y_true_fold, y_prob_fold)
        plt.plot(fpr, tpr, label=f'Fold {f + 1} (AUC = {auc:.4f})')

        print("================================================")

    save_metrics(metrics, k)
    save_report(times, k, epochs, devname)
if __name__ == "__main__":
    main()