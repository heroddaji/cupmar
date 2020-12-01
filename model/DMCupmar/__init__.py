import torch
import torch.nn as nn

from model.DMBase.base import DMBase
from model.DMCupmar.news_encoder import NewsEncoder
from model.DMCupmar.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor


class DMCupmar(DMBase):
    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        super(DMCupmar, self).__init__(config)
        self.news_encoder = NewsEncoder(self.config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(self.config, self.news_encoder)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, batch_impressions):
        candidate_news = batch_impressions['candidate_news']
        clicked_news = batch_impressions['clicked_news']
        users = batch_impressions['user']
        """
        Args:
            candidate_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title,
                        "abstract": Tensor(batch_size) * num_words_abstract
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": Tensor(batch_size),
                        "subcategory": Tensor(batch_size),
                        "title": Tensor(batch_size) * num_words_title,
                        "abstract": Tensor(batch_size) * num_words_abstract
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """

        # 1+K, batch_size, cate_embed_dim
        candidate_news_vector = torch.stack([self.news_encoder(news) for news in candidate_news])

        # batch_size, cate_embed_dim
        user_vector = self.user_encoder(users, clicked_news)

        # batch_size, 1+K
        click_probability = torch.stack(
            [self.click_predictor(news_vector, user_vector) for news_vector in candidate_news_vector],
            dim=1)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                    "title": Tensor(batch_size) * num_words_title,
                    "abstract": Tensor(batch_size) * num_words_abstract
                }
        Returns:
            (shape) batch_size, cate_embed_dim
        """
        # batch_size, cate_embed_dim
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news, empty_clicked_news_batch_size=1):
        """
        Args:
            user: batch_size
            clicked_news: list of news
            [news:
                {
                    "category":
                    "subcategory":
                    "title":
                    "abstract":
                    "title_entities":
                    "abstract_entities":
                }
            ]
        Returns:
            (shape) batch_size, user_embed_dim
        """
        # batch_size, user_embed_dim
        return self.user_encoder(user, clicked_news, empty_clicked_news_batch_size)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: cate_embed_dim
            user_vector: cate_embed_dim
        Returns:
            click_probability: 0-dim tensor
        """

        # 0-dim tensor
        click_probability = torch.dot(news_vector, user_vector)
        return click_probability
