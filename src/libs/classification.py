from src.libs.dataclass import TwoTowerDataset
from src.libs.utils import prepare_cls_data

import numpy as np
from torch.utils.data import random_split
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torch.functional as F
import torch

"""
Input
    context: (n_sample, dim_context)
    action_emb_log: (n_sample, dim_act_emb)
    action_emb_tar: (n_sample, dim_act_emb)
"""

# class Trainer:

#     def __init__(self, model) -> None:
        
#         self.model = model


#     def train(self, context, action_emb_log, action_emb_tar):

#         pass


class Classifier:

    def __init__(self, 
                 bandit_data,
                 action_tar, 
                 model, 
                 num_epochs,
                 verbose=False
                 ) -> None:
        
        self.bandit_data = bandit_data
        self.model = model
        self.optimizer = Adam(self.model.parameters())
        self.criterion = BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.verbose = verbose

        self.context, self.action_emb, target = prepare_cls_data(bandit_data['context_obs'],
                                                                bandit_data['action_context_obs'],
                                                                bandit_data['action'],
                                                                action_tar)

        context_train, context_test, action_train, action_test, y_train, y_test = train_test_split(
            self.context, self.action_emb, target, test_size=0.1, random_state=42, stratify=target)
        
        context_train, context_val, action_train, action_val, y_train, y_val = train_test_split(
            context_train, action_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

        train_dataset = TwoTowerDataset(context_train, action_train, y_train)
        val_dataset = TwoTowerDataset(context_val, action_val, y_val)
        test_dataset = TwoTowerDataset(context_test, action_test, y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

        self.best_model = None

    def train(self):

        best_score = 0
        for epoch in range(self.num_epochs):

            self.model.train()

            for context, action, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model((context, action))
                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

            # Validation
            val_score = self.test(self.val_loader)
            if self.verbose:
                print('validation score:', val_score)
            if val_score > best_score:
                self.best_model = self.model

    def test(self, loader):

        with torch.no_grad():
            self.model.eval()
            correct = 0
            total = 0
            for context, action, labels in loader:
                logits = self.model((context, action))
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                total += labels.size(0)
                correct += (preds == labels.view(-1, 1)).sum().item()

        return correct / total
    
    def predict(self):

        context = torch.tensor(self.bandit_data['context_obs']).float()
        action_emb = torch.tensor(self.bandit_data['action_context_obs'][self.bandit_data['action']]).float()

        logits = self.best_model((context, action_emb))
        probs = torch.sigmoid(logits).detach().numpy().flatten()

        return probs
