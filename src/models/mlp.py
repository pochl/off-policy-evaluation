import torch
import torch.nn as nn
import torch.nn.functional as F


# class MLPClassifier(nn.Module):
#   """
#   simple MLP classifier (e.g. for classifying in z-space)
#   slightly deeper than MLPClassifier
#   """
#   def __init__(self, in_dim, h_dim):
#       super(MLPClassifier, self).__init__()
#       self.h_dim = h_dim
#       # self.n_classes = config.model.n_classes
#       self.in_dim = in_dim

#       self.fc1 = nn.Linear(self.in_dim, self.h_dim)
#       self.fc2 = nn.Linear(self.h_dim, self.h_dim)
#       self.fc3 = nn.Linear(self.h_dim, self.h_dim)
#       self.fc4 = nn.Linear(self.h_dim, 1)

#   def forward(self, x):
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = F.relu(self.fc3(x))
#     logits = self.fc4(x)
#     probas = torch.sigmoid(logits)

#     return logits, probas
  

class TwoTowerClassifier(nn.Module):
  """
  simple MLP classifier (e.g. for classifying in z-space)
  slightly deeper than MLPClassifier
  """
  def __init__(self, dim_context, dim_action, h_dim):
      super(TwoTowerClassifier, self).__init__()
      self.h_dim = h_dim
      # self.n_classes = config.model.n_classes
      self.dim_context = dim_context
      self.dim_action = dim_action

      self.context_tower = nn.Sequential(
            nn.Linear(self.dim_context, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )
      
      self.action_tower = nn.Sequential(
            nn.Linear(self.dim_action, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

      self.classifier = nn.Sequential(
            nn.Linear(self.h_dim * 2, 1)
        )

  def forward(self, X):
    out1 = self.context_tower(X[0])
    out2 = self.action_tower(X[1])
    combined = torch.cat((out1, out2), dim=1)
    logits = self.classifier(combined)
    return logits  # return logits instead of probs so to use BCEWithLogitsLoss() which is more numerically stable
  

class MLPClassifier(nn.Module):
  """
  simple MLP classifier (e.g. for classifying in z-space)
  slightly deeper than MLPClassifier
  """
  def __init__(self, dim_context, dim_action, h_dim):
      super(MLPClassifier, self).__init__()
      self.h_dim = h_dim
      # self.n_classes = config.model.n_classes
      self.dim_context = dim_context
      self.dim_action = dim_action

      self.layers = nn.Sequential(
            nn.Linear(self.dim_context + self.dim_action, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, 1)
        )

  def forward(self, X):
    logits = self.layers(torch.cat((X[0], X[1]), dim=1))

    return logits  # return logits instead of probs so to use BCEWithLogitsLoss() which is more numerically stable