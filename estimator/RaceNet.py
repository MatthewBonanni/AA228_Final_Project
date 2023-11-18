import os
import pdb
from datetime import datetime

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

from tqdm import tqdm


Timedelta = pd._libs.tslibs.timedeltas.Timedelta
args = {
      'num_layers': 6,
      'hidden_dim': 512,
      'out_dim': 512,
      'emb_dim': 10,
      'dropout': 0.5,
  }

cols = [
    'Year',
    'TrackID',
    'DriverID',
    'TeamID',
    'LapNumber',
    'TyreLife',
    'CompoundID',
    'Stint',
    'Position',
    'PitStop',
    'YellowFlag',
    'RedFlag',
    'SC',
    'VSC',
    ]

ids = [
    'TrackID',
    'TrackStatusID',
    'DriverID',
    'TeamID',
]
weather_cols = [
    'AirTemp',
    'Humidity',
    'Pressure',
    'Rainfall',
    'TrackTemp',
    'WindDirection',
    'WindSpeed']

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class RaceNet(torch.nn.Module):
    def __init__(self, args, num_drivers, num_tracks, num_teams,  activation=F.relu):
        super().__init__()
        
        self.in_dim = len(cols) - len(ids) + len(weather_cols) + args["emb_dim"]*len(ids)
        self.num_layers = args["num_layers"]

        self.in_dim = len(cols) - len(ids) + len(weather_cols) + 10*len(ids)
        self.activation = activation
        # Initialize Activation Fn
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=args["hidden_dim"])\
                                                 for i in range(args["num_layers"])])

        ## Initialize Linear Layers
        self.linears = \
            torch.nn.ModuleList(
                [torch.nn.Linear(in_features=self.in_dim, out_features=args["hidden_dim"])])
        self.linears.extend([torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["hidden_dim"])\
                              for i in range(args["num_layers"]-1)])
        self.linears.append(torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["out_dim"]))

        # Initialize Embeddings For Categorical Data
        track_emb = torch.nn.Embedding(num_embeddings = num_tracks, embedding_dim = args["emb_dim"])
        track_stat_emb = torch.nn.Embedding(num_embeddings = 7, embedding_dim = args["emb_dim"])
        driver_emb = torch.nn.Embedding(num_embeddings = num_drivers, embedding_dim = args["emb_dim"])
        team_emb = torch.nn.Embedding(num_embeddings = num_teams, embedding_dim = args["emb_dim"])
        self.embs = [track_emb, track_stat_emb, driver_emb, team_emb]

        self.dropout = args["dropout"]
        
    def forward(self, batched_vals, batched_ids):
        beds = []
        batched_ids[:,ids.index('TrackStatusID')] = batched_ids[:,ids.index('TrackStatusID')] -1
        for i, embed in enumerate(self.embs):
            beds.append(embed(batched_ids[:,i]))         
        input = batched_vals
        for bed in beds:
            input = torch.cat((input, bed),dim=1)
        input = input.to(torch.float32)

        for i in range(self.num_layers-1):
            input = self.linears[i](input)      
            input = self.batch_norms[i](input)
            input = F.dropout(input, p=self.dropout, training = self.training)
        
        output = self.linears[-1](input)
        output = torch.sum(output,dim=1)

        return output

class F1Dataset(Dataset):
    def __init__(self, data):
        self.lap_times = data["LapTime"]
        self.inputs = data[cols + weather_cols]
        
    def __len__(self):
        return len(self.lap_times)
    
    def __getitem__(self, idx):
        data = self.inputs.iloc[idx]
        
        val_cols = [i for i in cols if i not in ids]
        input_vals = data.loc[val_cols + weather_cols]
        for key, value in input_vals.items():
            if type(value) == Timedelta:
                input_vals.loc[key] = value.total_seconds()*1000
        input_vals = input_vals.to_numpy(dtype=float)

        input_ids = data.loc[ids].to_numpy(dtype=int)
        label = self.lap_times.iloc[idx].total_seconds()*1000
        return input_vals, input_ids, label

def train(dataloader, model, optimizer, loss_fn=F.mse_loss):

    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        optimizer.zero_grad()

        out = model(batch[0],batch[1]).squeeze()
        label = batch[2].squeeze().to(torch.float32)
        loss = loss_fn(out, label,reduction="mean")

        loss.backward()
        optimizer.step()

    return loss.item()

def eval(dataloader, model, loss_fn=F.mse_loss):

    model.eval()
    loss = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

        out = model(batch[0],batch[1]).squeeze()
        label = batch[2].squeeze().to(torch.float32)

        loss += loss_fn(out, label)/torch.mean(label)/len(dataloader)

    return loss.item()

if __name__=="__main__":
    print(torch.cuda.is_available())
    now = datetime.now()
    filename = 'outputs/racenet_' + now.strftime('%m_%d_%H_%M_%S') + str(args["num_layers"]) + "l_" + str(args["hidden_dim"]) + '.pt'

    writer = SummaryWriter()
    data = pd.read_hdf("data/f1_dataset.h5")
    train_data = F1Dataset(data)
    generator = torch.Generator().manual_seed(228)
    splits = random_split(train_data, [0.8, 0.1, 0.1], generator)

    train_data = splits[0]

    train_dataloader = DataLoader(train_data, batch_size = 1000, shuffle=True)
    val_dataloader = DataLoader(splits[1],batch_size=100, shuffle=False)
    test_dataloader = DataLoader(splits[2],batch_size=100, shuffle=False)

    model = RaceNet(args, num_drivers=26, num_tracks=27, num_teams=11)

    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,)

    stopper = EarlyStopper(20, 0.01)
    min_val_loss = np.inf
    for i in range(epochs):
        print("Epoch:", i)
        train_loss = train(train_dataloader, model, optimizer)
        writer.add_scalar('Loss/train', train_loss, i)
        val_loss = eval(val_dataloader,model,loss_fn=F.l1_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if val_loss <= 0.05:
                torch.save(model.state_dict(), filename)
        writer.add_scalar('Accuracy/eval', val_loss, i)
        
        print("Loss:", train_loss)
        if stopper.early_stop(val_loss):
                break

    test_loss = eval(test_dataloader, model, loss_fn =F.l1_loss)
    print(test_loss)

    now = datetime.now()
    filename = 'outputs/racenet_' + now.strftime('%H_%M_%S') + '.pt'
    torch.save(model.state_dict(), filename)