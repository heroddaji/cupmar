import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DMBase.base import DMBase


class NewsEncoder(DMBase):
    def __init__(self, config):
        super(NewsEncoder, self).__init__(config)

        self.category_embedding = nn.Embedding(config.num_categories, config.category_embedding_dim)
        self.subcategory_embedding = nn.Embedding(config.num_categories, config.category_embedding_dim)
        self.linear1 = nn.Linear(config.category_embedding_dim * 2, int(config.category_embedding_dim * 1.5))
        self.linear2 = nn.Linear(int(config.category_embedding_dim * 1.5), config.category_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                }
        """
        category_tensor = news['category'].to(self.device)  # batch_size
        subcategory_tensor = news['subcategory'].to(self.device)  # batch_size

        category_embedded = self.category_embedding(category_tensor)  # batch_size, category_embedding_dim
        subcategory_embedded = self.subcategory_embedding(subcategory_tensor)  # batch_size, category_embedding_dim
        out = torch.cat((category_embedded, subcategory_embedded), dim=1)  # batch_size, category_embedding_dim * 2
        out = F.relu(self.linear1(out))  # batch_size, category_embedding * 1.5
        out = F.relu(self.linear2(out))  # batch_size, category_embedding
        return out
