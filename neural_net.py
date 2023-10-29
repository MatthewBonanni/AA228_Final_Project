import pytorch as torch
from pytorch.nn import Functional as F

import pandas as pd
import numpy as np

args = {
      'num_layers': 6,
      'in_dim': 100,
      'hidden_dim': 256,
      'out_dim': 1,
      'emb_dim': 10,
      'dropout': 0.5,
      'lr': 0.01,
      'epochs': 100,
  }

class RaceNet(torch.nn.Module):
    def __init__(self, args, activation, num_drivers, num_tracks, num_teams):
        self.num_layers = args["num_layers"]

        # Initialize Activation Fn
        self.activations = torch.nn.ModuleList([activation() for i in range(args["num_layers"]-1)])

        ## Initialize Linear Layers
        self.linears = \
            torch.nn.ModuleList(
                [torch.nn.Linear(in_features=args["in_dim"], out_features=args["hidden_dim"])])
        self.linears.extend([torch.nn.Linear(in_features=self.args["hidden_dim"], out_features=args["hidden_dim"])\
                              for i in range(args["num_layers"])])
        self.linears.append(torch.nn.Linear(in_features=args["hidden_dim"], out_features=args["out_dim"]))

        # Initialize Embeddings For Categorical Data
        self.track_emb = torch.nn.Embedding(num_embeddings = num_tracks, embedding_dim = args["emb_dim"])
        self.driver_emb = torch.nn.Embedding(num_embeddings = num_tracks, embedding_dim = args["emb_dim"])
        self.team_emb = torch.nn.Embedding(num_embeddings = num_teams, embedding_dim = args["emb_dim"])


        self.dropout = args["dropout"]
        
    def forward(self, x):
        input = np.empty((0,))
        for key, item in x.items():
            # embed the categorical data
            if key == "TeamID":
                vec_team = self.team_embed(item)
                input.extend(vec_team)
            elif key == "TrackID":
                vec_track = self.track_embed(item)
                input.extend(vec_track)
            elif key == "DriverNumber":
                vec_driver = self.driver_embed(item)
                input.extend(vec_driver)
            else:
                input.extend(item)

        




