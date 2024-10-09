import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DataClass:
    
    def __init__(self, 
                 interactions: pd.DataFrame = None, 
                 users: pd.DataFrame = None, 
                 items: pd.DataFrame = None, 
                 user_id_col='user_id', 
                 item_id_col='item_id', 
                 feedback_col='feedback'
                 ) -> None:
        
        if interactions is not None:
            assert user_id_col in interactions.columns
            assert item_id_col in interactions.columns
            assert feedback_col in interactions.columns
        
        if users is not None:
            assert user_id_col in users.columns
        
        if items is not None:
            assert item_id_col in items.columns
        
        self.interactions = interactions
        self.users = users
        self.items = items

        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.feedback_col = feedback_col


class TwoTowerDataset(Dataset):
    def __init__(self, context, action_emb, target=None):
        self.context = torch.tensor(context).float()
        self.action_emb = torch.tensor(action_emb).float()

        if target is not None:
            self.target = torch.tensor(target).float()
        else:
            self.target = torch.zeros(len(context))

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        return self.context[index], self.action_emb[index], self.target[index]
    

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return torch.tensor(sample)
    