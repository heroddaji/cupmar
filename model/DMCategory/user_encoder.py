import torch

from model.DMBase.base import DMBase

class UserEncoder(DMBase):
    def __init__(self, config, news_encoder):
        super(UserEncoder, self).__init__(config)
        self.news_encoder = news_encoder

    def forward(self, clicked_news, empty_clicked_news_batch_size=1):
        """
        Args:
            clicked_news:
                [news:
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                    }
                ]
        """

        user_vector = torch.rand(empty_clicked_news_batch_size,
                                 self.config.category_embedding_dim)
        if len(clicked_news) > 0:
            # batch_size, num_clicked_news_a_user, cate_embed_dim
            clicked_news_vector = torch.stack([self.news_encoder(news) for news in clicked_news], dim=1)
            if self.config.user_encoding_strategy == 'sum_cate':
                # batch_size, cate_embed_dim
                user_vector = torch.mean(clicked_news_vector, dim=1)

        return user_vector
