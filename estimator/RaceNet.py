import os
import pdb

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import pandas as pd
import numpy as np

from tqdm import tqdm


Timedelta = pd._libs.tslibs.timedeltas.Timedelta
args = {
      'num_layers': 6,
      'in_dim': 40,
      'hidden_dim': 256,
      'out_dim': 1,
      'emb_dim': 10,
      'dropout': 0.5,
      'lr': 0.01,
      'epochs': 100,
  }

cols = [
    'Year',
    'TrackID',
    'DriverID',
    'TeamID',
    'LapNumber',
    'TrackStatusID',
    'TyreLife',
    'CompoundID',
    'Stint',
    'Sector1Time',
    'Sector2Time',
    'Sector3Time',
    'Position']
ids = [
    'TrackID',
    'DriverID',
    'TeamID'
]
weather_cols = [
    'AirTemp',
    'Humidity',
    'Pressure',
    'Rainfall',
    'TrackTemp',
    'WindDirection',
    'WindSpeed']

class RaceNet(torch.nn.Module):
    def __init__(self, args, num_drivers, num_tracks, num_teams,  activation=F.relu):
        super().__init__()

        self.num_layers = args["num_layers"]
        self.activation = activation
        # Initialize Activation Fn
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=args["hidden_dim"])\
                                                 for i in range(args["num_layers"]-1)])

        ## Initialize Linear Layers
        self.linears = \
            torch.nn.ModuleList(
                [torch.nn.Linear(in_features=args["in_dim"], out_features=args["hidden_dim"])])
        self.linears.extend([torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["hidden_dim"])\
                              for i in range(args["num_layers"])])
        self.linears.append(torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["out_dim"]))

        # Initialize Embeddings For Categorical Data
        track_emb = torch.nn.Embedding(num_embeddings = num_tracks, embedding_dim = args["emb_dim"])
        driver_emb = torch.nn.Embedding(num_embeddings = num_drivers, embedding_dim = args["emb_dim"])
        team_emb = torch.nn.Embedding(num_embeddings = num_teams, embedding_dim = args["emb_dim"])
        self.embs = [track_emb, driver_emb, team_emb]

        self.dropout = args["dropout"]
        
    def forward(self, batched_vals, batched_ids):
        beds = []
        for i, embed in enumerate(self.embs):
            beds.append(embed(batched_ids[:,i]))         
        input = batched_vals
        for bed in beds:
            input = torch.cat((input, bed),dim=1)
        input = input.to(torch.float32)

        for i in range(self.num_layers-1):
            input = self.linears[i](input)      
            input = self.batch_norms[i](input)
            input = self.activation(input)
            input = F.dropout(input, p=self.dropout, training = self.training)
        
        output = self.linears[-1](input)

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
        input_vals = data.loc[val_cols]
        for key, value in input_vals.items():
            if type(value) == Timedelta:
                input_vals.loc[key] = value.total_seconds()
        input_vals = input_vals.to_numpy(dtype=float)

        input_ids = data.loc[ids].to_numpy(dtype=int)
        label = self.lap_times.iloc[idx].total_seconds()
        return input_vals, input_ids, label

def train(dataloader, model, optimizer, loss_fn=F.mse_loss):

    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
        optimizer.zero_grad()

        out = model(batch[0],batch[1]).squeeze()
        label = batch[2].squeeze().to(torch.float32)
        loss = loss_fn(out, label)

        loss.backward()
        optimizer.step()

    return loss.item()


if __name__=="__main__":
    data = pd.read_hdf("data/f1_dataset.h5")
    train_data = F1Dataset(data)
    generator = torch.Generator().manual_seed(228)
    splits = random_split(train_data, [0.7, 0.2, 0.1], generator)

    train_data = splits[0]    
    train_dataloader = DataLoader(train_data, batch_size = 100, shuffle=True)

    model = RaceNet(args, num_drivers=26, num_tracks=27, num_teams=11)
    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"],)
    for i in range(epochs):
        print("Epoch:", i)
        loss = train(train_dataloader, model, optimizer)
        breakpoint()
        print("Loss:", loss)




