import os
import pdb
from datetime import datetime

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss 

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

from tqdm import tqdm


Timedelta = pd._libs.tslibs.timedeltas.Timedelta
args = {
      'num_layers': 3,
      'hidden_dim': 640,
      'out_dim': 1,
      'emb_dim': 10,
      'dropout': 0.5,
      'num_monte_carlo': 100,
  }

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
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
        
        self.in_dim = len(cols) - len(ids) + len(weather_cols) + 10*len(ids)
        self.num_layers = args["num_layers"]
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

        for i in range(self.num_layers):
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
        input_vals = data.loc[val_cols + weather_cols]
        for key, value in input_vals.items():
            if isinstance(value, Timedelta):
                input_vals.loc[key] = value.total_seconds()
        input_vals = input_vals.to_numpy(dtype=float)

        input_ids = data.loc[ids].to_numpy(dtype=int)
        label = self.lap_times.iloc[idx].total_seconds()
        return input_vals, input_ids, label

def train(dataloader, model, optimizer, loss_fn=F.mse_loss):

    model.train()
    loss = 0

    for batch in tqdm(dataloader, desc="Iteration"):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            out = model(batch[0],batch[1]).squeeze()
            label = batch[2].squeeze().to(torch.float32)
            kl = get_kl_loss(model)
            loss = loss_fn(out, label) + kl

        loss.backward()
        optimizer.step()

    return loss.item()

def eval(dataloader, model, loss_fn=F.mse_loss):

    model.eval()
    test_acc = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Iteration"):
            output_mc = []
            for mc_run in range(args["num_monte_carlo"]):
                outs = model(batch[0],batch[1]).squeeze()
                output_mc.append(outs)
            output = torch.stack(output_mc)  
            y_pred = output.mean(dim=0)
            test_acc += (torch.abs(y_pred - batch[2])/batch[2]).mean()/len(dataloader)

    return test_acc

if __name__=="__main__":
    print(torch.cuda.is_available())
    writer = SummaryWriter()
    data = pd.read_hdf("data/f1_dataset.h5")
    train_data = F1Dataset(data)
    generator = torch.Generator().manual_seed(228)
    splits = random_split(train_data, [0.7, 0.2, 0.1], generator)

    train_data = splits[0]

    train_dataloader = DataLoader(train_data, batch_size = 1000, shuffle=True)
    val_dataloader = DataLoader(splits[1],batch_size=1000, shuffle=False)
    test_dataloader = DataLoader(splits[2],batch_size=100, shuffle=False)

    model = RaceNet(args, num_drivers=26, num_tracks=27, num_teams=11)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    epochs = 60
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,)

    stopper = EarlyStopper(5, 0.01)

    for i in range(epochs):
        print("Epoch:", i)
        train_loss = train(train_dataloader, model, optimizer)
        writer.add_scalar('Loss/train', train_loss, i)
        val_loss = eval(val_dataloader,model,loss_fn=F.l1_loss)
        writer.add_scalar('Accuracy/eval', val_loss, i)
        print("Loss:", train_loss)
        if stopper.early_stop(val_loss):
            break

    test_loss = eval(test_dataloader, model, loss_fn =F.l1_loss)
    print("Final Test Loss:",test_loss)

    now = datetime.now()
    filename = 'outputs/racenet_' + now.strftime('%H_%M_%S') + '.pt'
    torch.save(model.state_dict(), filename)