import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DMBase.base import DMBase


class UserEncoder(DMBase):
    def __init__(self, config, news_encoder):
        super(UserEncoder, self).__init__(config)
        self.news_encoder = news_encoder
        self.user_embedding = nn.Embedding(config.num_users, config.user_embedding_dim)
        self.gru = nn.GRU(config.user_embedding_dim, config.user_embedding_dim)
        self.linear = nn.Linear(config.user_embedding_dim * 2, config.user_embedding_dim)
        self.out_of_vocab_users = {}
        self.empty_clicked_users = {}

    def forward(self, batch_users, batch_clicked_news, empty_clicked_news_batch_size=1):
        """
        Args:
            batch_users: Tensor(batch_size)
            batch_clicked_news:
                [news:
                    {
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                    "title": Tensor(batch_size) * num_words_title,
                    "abstract": Tensor(batch_size) * num_words_abstract,
                    "title_entities": Tensor(batch_size) * num_words_title,
                    "abstract_entities": Tensor(batch_size) * num_words_abstract,
                    }
                ]
        """
        if not self.training:
            # todo: how to handle out of vocab user_id?
            '''
            if not training, the input is only one user, so check the user_id, if it is bigger than
            user_embedding total rows, and make a random user_embedding for now
            '''
            user_id = batch_users[0]
            if user_id >= self.config.num_users:
                if self.out_of_vocab_users.get(user_id, None) is None:
                    user_embedding_tensor = torch.rand(empty_clicked_news_batch_size,
                                                       self.config.user_embedding_dim).to(self.device)
                    self.out_of_vocab_users[user_id] = user_embedding_tensor

                user_embedding_tensor = self.out_of_vocab_users[user_id]
            else:
                batch_users_tensor = batch_users.to(self.device)  # batch_size
                user_embedding_tensor = self.user_embedding(batch_users_tensor)  # batch_size, user_embedding_dim
        else:
            batch_users_tensor = batch_users.to(self.device)  # batch_size
            user_embedding_tensor = self.user_embedding(batch_users_tensor)  # batch_size, user_embedding_dim

        if len(batch_clicked_news) == 0:
            batch_clicked_news_tensor = torch.rand(self.config.num_words_abstract,
                                                   1,
                                                   self.config.user_embedding_dim).to(self.device)
        else:
            # num_clicked_news, batch_size, user_embedding_dim
            batch_clicked_news_tensor = torch.stack([self.news_encoder(news) for news in batch_clicked_news]).to(
                self.device)

        lpe = self.get_lpe(user_embedding_tensor, batch_clicked_news)

        output, hidden = self.gru(batch_clicked_news_tensor)

        final_user_vector = torch.cat((output.mean(dim=0), hidden.squeeze(0)), dim=1)
        final_user_vector = F.relu(self.linear(final_user_vector))
        return final_user_vector

    def get_lpe(self, user_tensor, batch_clicked_news):
        '''
        get the long-term preferences of the user
        '''
        # todo: why there is 0-category?
        all_categories = torch.stack([news['category'] for news in batch_clicked_news]).view(-1)
        unique_categories, count_cate = torch.unique(all_categories, sorted=True, return_counts=True)
        all_subcategories = torch.stack([news['subcategory'] for news in batch_clicked_news]).view(-1)
        unique_subcategories, count_subcate = torch.unique(all_subcategories, sorted=True, return_counts=True)

        # find lpe_top_cate_num of categories with the most count
        topk = self.config.lpe_top_cate_num
        if topk > len(unique_categories) or \
                topk > len(unique_subcategories):
            topk = min(len(unique_categories), len(unique_subcategories))

        _, cate_lpe_indices = count_cate.topk(topk)
        _, subcate_lpe_indices = count_subcate.topk(topk)

        # remove zero index
        cate_lpe_indices = cate_lpe_indices[cate_lpe_indices.nonzero()]
        subcate_lpe_indices = subcate_lpe_indices[subcate_lpe_indices.nonzero()]

        # get the categories embedding
        cate_tensor = unique_categories[cate_lpe_indices]
        subcate_tensor = unique_subcategories[subcate_lpe_indices]

        cate_tensor = self.news_encoder.category_embedding(cate_tensor.to(self.device))
        subcate_tensor = self.news_encoder.category_embedding(subcate_tensor.to(self.device))

        #sum them up and batch it
        lpe_tensor = torch.add(subcate_tensor, cate_tensor).squeeze().sum(dim=0).unsqueeze(0)
        lpe_tensor = lpe_tensor.repeat(user_tensor.shape[0],1)

        return lpe_tensor
