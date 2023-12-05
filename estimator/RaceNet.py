import os
import pdb
import copy
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
      'out_dim': 128,
      'emb_dim': 10,
      'dropout': 0.4,
      'num_monte_carlo': 100,
      'num_heads': 4,
  }

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.4,
      'num_heads': 4,
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
    'PitStop',
    'YellowFlag',
    'RedFlag',
    'SC',
    'VSC',
    "Rainfall",
    ]

lap_cols = [
    "LapNumber",
    "TyreLife",
    "CompoundID",
    "Stint",
    "PitStop",
    "YellowFlag",
    "RedFlag",
    "SC",
    "VSC",
    'Rainfall',
]

time_cols = [
    ]

ids = [
    'TrackID',
    'DriverID',
    'TeamID',
]

weather_cols = [
    'AirTemp',
    'Humidity',
    'Pressure',
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

class RaceNetBranched(torch.nn.Module):
    def __init__(self, args, num_drivers, num_tracks, num_teams,  activation=F.elu):
        super().__init__()
        
        self.num_layers = args["num_layers"]
        in_dim_cons = len(cols) - len(ids) - len(lap_cols) + len(weather_cols) + args["emb_dim"]*len(ids) + 1
        in_dim_laps = len(lap_cols)

        self.cons_dim = args["hidden_dim"]//2
        self.lap_dim = args["hidden_dim"]

        in_laps = len(lap_cols) + 1
        in_cons = len(cols) - len(ids) - len(lap_cols) + args["emb_dim"]*len(ids)
        in_weath = len(weather_cols)

        # Initialize Linear Layer Activation Fn
        self.activation = activation

        self.batch_norms_cons = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.cons_dim)\
                                                 for i in range(args["num_layers"])])
        self.batch_norms_weath = copy.deepcopy(self.batch_norms_cons)
        self.batch_norms_lap = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features=self.lap_dim)\
                                                 for i in range(args["num_layers"])])

        ## Initialize Linear Layers for constant vals
        self.cons_linears = \
            torch.nn.ModuleList(
                [torch.nn.LazyLinear(out_features=args["hidden_dim"])])
        self.cons_linears.extend([torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["hidden_dim"])\
                              for i in range(args["num_layers"]-1)])
        self.cons_linears.append(torch.nn.Linear(in_features=self.cons_dim, out_features=self.cons_dim))
        self.cons_linears.append(torch.nn.Linear(in_features=self.cons_dim,out_features=args["out_dim"]))

        self.weath_linears = copy.deepcopy(self.cons_linears)
        self.weath_linears[0] = torch.nn.Linear(in_features=in_weath, out_features=self.cons_dim)

        ## Initialize Linear Layers for lap vals
        self.laps_linears = \
            torch.nn.ModuleList(
                [torch.nn.LazyLinear(out_features=args["hidden_dim"])])
        self.laps_linears.extend([torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["hidden_dim"])\
                              for i in range(args["num_layers"]-1)])

        self.final_lin = torch.nn.Linear(in_features=args["hidden_dim"]*2, out_features=args["out_dim"])

        # Initialize Embeddings For Categorical Data
        track_emb = torch.nn.Embedding(num_embeddings = num_tracks, embedding_dim = args["emb_dim"])
        driver_emb = torch.nn.Embedding(num_embeddings = num_drivers, embedding_dim = args["emb_dim"])
        team_emb = torch.nn.Embedding(num_embeddings = num_teams, embedding_dim = args["emb_dim"])
        self.embs = [track_emb, driver_emb, team_emb]

        self.dropout = args["dropout"]
        
    def forward(self, batched_lap, batched_cons, batched_weath, batched_ids):
        beds = []
        for i, embed in enumerate(self.embs):
            beds.append(embed(batched_ids[:,i]))         
        input_cons = batched_cons
        for bed in beds:
            input_cons = torch.cat((input_cons, bed),dim=1)
        input_cons = input_cons.to(torch.float32)
        input_weath = batched_weath.to(torch.float32)
        input_lap = batched_lap.to(torch.float32)

        for i in range(self.num_layers):
            input_cons = self.cons_linears[i](input_cons)
            input_cons = self.activation(input_cons)      
            input_cons = self.batch_norms_cons[i](input_cons)
            input_cons = F.dropout(input_cons, p=self.dropout, training = self.training)

            input_weath = self.weath_linears[i](input_weath)
            input_weath = self.activation(input_weath)      
            input_weath = self.batch_norms_weath[i](input_weath)
            input_weath = F.dropout(input_weath, p=self.dropout, training = self.training)

            input_lap = self.laps_linears[i](input_lap)
            input_lap = self.activation(input_lap)      
            input_lap = self.batch_norms_lap[i](input_lap)
            input_lap = F.dropout(input_lap, p=self.dropout, training = self.training)
        
        output = self.final_lin(torch.cat((input_lap,input_cons),dim=1))
        output = torch.sum(output,dim=1)

        return output 

class RaceNet(torch.nn.Module):
    def __init__(self, args, num_drivers, num_tracks, num_teams,  activation=F.elu):
        super().__init__()
        
        self.in_dim = len(cols) - len(ids) + len(weather_cols) + args["emb_dim"]*len(ids) + 1
        #self.in_dim = len(lap_cols) + 1
        self.num_layers = args["num_layers"]

        # Initialize Activation Fn
        self.activation = activation

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
        
    def forward(self, batched_lap, batched_cons, batched_weath, batched_ids):
        beds = []
        for i, embed in enumerate(self.embs):
            beds.append(embed(batched_ids[:,i]))         
        input = torch.cat((batched_lap,batched_cons,batched_weath), dim=1)
        for bed in beds:
            input = torch.cat((input, bed),dim=1)

        input = input.to(torch.float32)

        for i in range(self.num_layers):
            input = self.linears[i](input)
            input = self.activation(input)      
            input = self.batch_norms[i](input)
            input = F.dropout(input, p=self.dropout, training = self.training)
        
        output = self.linears[-1](input).sum(dim=-1)

        return output

class F1Dataset(Dataset):
    def __init__(self, data):
        data = data.copy()
        for key in cols + weather_cols + time_cols:
            data = data[pd.isnull(data[key])==False]
        self.inputs = data[cols + weather_cols + time_cols]

        self.lap_times = data["LapTime"]
        self.deltas = data["LapTime"] - data["PrevLapTime"]

    def __len__(self):
        return len(self.lap_times)
    
    def __getitem__(self, idx):
        data = self.inputs.iloc[idx]
        
        con_cols = [i for i in cols if i not in ids]
        con_cols = [i for i in con_cols if i not in lap_cols ]
        input_cons = data.loc[con_cols]
        input_cons = input_cons.to_numpy(dtype=float)

        time_vals = data.loc[time_cols]
        two_sector_time = 0
        for time in time_vals:
            two_sector_time += time.total_seconds()*1000
        input_cons = np.concatenate([input_cons,[two_sector_time]])

        input_lap = data.loc[lap_cols]
        input_lap = input_lap.to_numpy(dtype=float)

        input_weath = data.loc[weather_cols]
        input_weath = input_weath.to_numpy(dtype=float)

        time_vals = data.loc[time_cols]
        time_dat = 0
        for time in time_vals:
            time_dat += time.total_seconds()
        input_lap = np.concatenate([input_lap,[time_dat]])

        input_ids = data.loc[ids].to_numpy(dtype=int)
        label = self.lap_times.iloc[idx].total_seconds()*1000
        return input_lap, input_cons, input_ids, label

def train(dataloader, model, optimizer, loss_fn=F.mse_loss):

    model.train()
    loss = 0

    for batch in tqdm(dataloader, desc="Iteration"):
        optimizer.zero_grad()

        out = model(batch[0],batch[1],batch[2]).squeeze()
        label = batch[-1].squeeze().to(torch.float32)
        loss = loss_fn(out, label, reduction="mean")

        loss.backward()
        optimizer.step()
    scheduler.step()

    return loss.item(), kl.item()

def eval(dataloader, model, loss_fn=F.mse_loss):

    model.eval()
    loss = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

        out = model(batch[0],batch[1],batch[2]).squeeze()
        label = batch[-1].squeeze().to(torch.float32)

        loss += loss_fn(out, label)/torch.mean(label)/len(dataloader)

    return test_acc

if __name__=="__main__":
    print(torch.cuda.is_available())
    now = datetime.now()
    filename = 'outputs/racenet_branch_' + now.strftime('%m_%d_%H_%M_%S') +"_"+ str(args["num_layers"]) + "l_" + str(args["hidden_dim"]) + '.pt'

    writer = SummaryWriter()
    data = pd.read_hdf("data/f1_dataset.h5")
    train_data = F1Dataset(data)
    generator = torch.Generator().manual_seed(228328)
    splits = random_split(train_data, [0.8, 0.1, 0.1], generator)

    train_data = splits[0]

    train_dataloader = DataLoader(train_data, batch_size = 1280, shuffle=True)
    val_dataloader = DataLoader(splits[1],batch_size=1280, shuffle=False)
    test_dataloader = DataLoader(splits[2],batch_size=1000, shuffle=False)

    model = RaceNet(args, num_drivers=26, num_tracks=27, num_teams=11)
    #model_state_dict = torch.load("outputs/racenet_branch_11_28_21_32_30_3l_640.pt")
    #model.load_state_dict(model_state_dict)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    epochs = 100
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,)

    stopper = EarlyStopper(10, 0.005)
    min_val_loss = np.inf
    for i in range(epochs):
        print("Epoch:", i)
        train_loss = train(train_dataloader, model, optimizer)
        writer.add_scalar('Loss/train', train_loss, i)
        val_loss = eval(val_dataloader,model,loss_fn=F.l1_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if val_loss <= 0.1:
                torch.save(model.state_dict(), filename)
        writer.add_scalar('Accuracy/eval', val_loss, i)
        print("Loss:", train_loss)
        if stopper.early_stop(val_loss):
                break
    model_state_dict = torch.load(filename)
    model.load_state_dict(model_state_dict)
    test_loss = eval(test_dataloader, model, loss_fn =F.l1_loss)
    print("Best Model Test Loss:", test_loss)