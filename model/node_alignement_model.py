import torch
import torch.nn as nn


class NodeAlignmentModel(nn.Module):
    def __init__(self, embed_dim):
        super(NodeAlignmentModel, self).__init__()
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(2 * 4 * embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(
        self,
        e1,
        e2,
    ):
        z = torch.cat((e1, e2), dim=-1)

        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.dropout(z, p=0.2, training=self.training)
        z = nn.functional.relu(self.fc2(z))
        z = self.fc3(z)
        return z
