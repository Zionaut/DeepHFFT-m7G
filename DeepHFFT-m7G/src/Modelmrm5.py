import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool
from torch.nn import Linear
import torch.nn as nn
from time import perf_counter as pc
from datetime import timedelta
from tqdm import tqdm

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertConfig
from time import perf_counter as pc
from tqdm import tqdm
from datetime import timedelta


from transformers import AutoModel, AutoTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import *  




class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1_1 = nn.Conv1d(4, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(64, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.max_pool1 = nn.MaxPool1d(2, stride=2)
        self.conv2_1 = nn.Conv1d(3, 64, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(64, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(2, stride=2)
        self.conv3_1 = nn.Conv1d(1, 64, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv1d(64, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.max_pool3 = nn.MaxPool1d(2, stride=2)
        self.conv4_1 = nn.Conv1d(4, 64, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv1d(64, 32, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.max_pool4 = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv5 = nn.Conv1d(112, 32, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        self.max_pool5 = nn.MaxPool1d(2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4000, 64)  
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
    def forward(self, x1, x2, x3, x4):
        x1 = x1.permute(0, 2, 1)  
        x2 = x2.permute(0, 2, 1)  
        x3 = x3.permute(0, 2, 1)  
        x4 = x4.permute(0, 2, 1)  
        x1 = F.relu(self.conv1_1(x1))
        x1 = F.relu(self.conv1_2(x1))
        x1 = self.bn1(x1)
        x1 = self.max_pool1(x1)
        x2 = F.relu(self.conv2_1(x2))
        x2 = F.relu(self.conv2_2(x2))
        x2 = self.bn2(x2)
        x2 = self.max_pool2(x2)
        x3 = F.relu(self.conv3_1(x3))
        x3 = F.relu(self.conv3_2(x3))
        x3 = self.bn3(x3)
        x3 = self.max_pool3(x3)
        x4 = F.relu(self.conv4_1(x4))
        x4 = F.relu(self.conv4_2(x4))
        x4 = self.bn4(x4)
        x4 = self.max_pool4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.dropout1(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = self.max_pool5(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        return x


class DNA2VecProcessor(nn.Module):
    def __init__(self, num_labels):
        super(DNA2VecProcessor, self).__init__()
        self.CNNmodel = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding='same'), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, kernel_size=3, padding='same'),  
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.drop = nn.Dropout(0.5)
    def forward(self, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer):
        Vec = torch.cat((dna2vec_4mer, dna2vec_5mer, dna2vec_6mer), dim=-1)
        Vec = Vec.unsqueeze(1)  
        Vec = self.CNNmodel(Vec)
        Vec = self.drop(Vec)
        Vec = Vec[:, -1, :]
        features = Vec
        return features
    


import torch
import torch.nn as nn
import torchvision.models as models

class HilbertProcessor(nn.Module):
    def __init__(self, input_channels=4, embed_dim=128, num_heads=4, num_transformer_layers=3):
        super(HilbertProcessor, self).__init__()
        resnet = models.resnet18(pretrained=True)  
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=1),  
            *list(resnet.children())[:-2]  
        )
        self.conv1x1 = nn.Conv2d(512, embed_dim, kernel_size=1) 
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )
        
       
        self.global_pool = nn.AdaptiveAvgPool1d(1)  

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        resnet_out = self.feature_extractor(x)  

        embed_out = self.conv1x1(resnet_out)  

        batch_size, embed_dim, h, w = embed_out.shape
        embed_out = embed_out.permute(0, 2, 3, 1).reshape(batch_size, -1, embed_dim)  
        transformer_out = self.transformer(embed_out)  
        pooled_out = self.global_pool(transformer_out.permute(0, 2, 1)).squeeze(-1)  
        return pooled_out







class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(128, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=16, nhead=2),  
            num_layers=3  
        )
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)

        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)

        x = x.unsqueeze(1)  
        x = self.transformer_encoder(x)
        x = x.squeeze(1)   
        x = F.relu(self.fc3(x))
        x = self.bn3(x)

        x = self.fc4(x)  
        return x



    

class MainModel(nn.Module):
    def __init__(self, num_labels=2):
        super(MainModel, self).__init__()
        
        self.custom_model = CustomModel()  # 提取 CustomModel 特征

        self.hilbert_processor = HilbertProcessor(input_channels=4, embed_dim=128)
  
        self.net = Net()

    def forward(self, x1, x2, x3, x4, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer, hilbert):

        custom = self.custom_model(x1, x2, x3, x4)


        hilbert_processed = self.hilbert_processor(hilbert)

        combined_features = torch.cat((custom,hilbert_processed), dim=-1)

        logits = self.net(hilbert_processed)

        return logits



from tqdm import tqdm
import time

from tqdm import tqdm
import time

def train_net(device, net, trainloader, epochs, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    net.to(device)

    start_time = time.time()  
    print("Training....")


    with tqdm(total=epochs, desc="Training", position=0, leave=True) as progress:
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0

            for i, (one_hot, chem, eiip, enac, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer, hilbert, label) in enumerate(trainloader):
               
                one_hot, chem, eiip, enac, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer, hilbert, label = [
                    feat.to(device) for feat in (one_hot, chem, eiip, enac, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer, hilbert, label)
                ]


                label = label.float()


                logits = net(one_hot, chem, eiip, enac, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer, hilbert).squeeze()
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            progress.set_postfix(loss=epoch_loss / len(trainloader))
            progress.update(1)

    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    return net, train_time  



def predict(device, model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    start_time = time.time()  

    with torch.no_grad():
        for data in dataloader:
            one_hot, chem, eiip, enac, dna2vec_4mer, dna2vec_5mer, dna2vec_6mer, hilbert, labels = data
            inputs = (one_hot.to(device), chem.to(device), eiip.to(device), enac.to(device),
                      dna2vec_4mer.to(device), dna2vec_5mer.to(device), dna2vec_6mer.to(device), hilbert.to(device))
            labels = labels.to(device)

            logits = model(*inputs).squeeze()
            probabilities = torch.sigmoid(logits)  
            predictions = (probabilities > 0.5).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    end_time = time.time()  

    return y_true, y_pred, y_prob, test_time

    
