import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DMBase.base import DMBase
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


class NewsEncoder(DMBase):
    def __init__(self, config, pretrained_word_embedding=None):
        super(NewsEncoder, self).__init__(config, pretrained_word_embedding)

        # todo: get pre-trained word embedding

        self.category_embedding = nn.Embedding(config.num_categories, config.category_embedding_dim)
        self.entity_embedding = nn.Embedding(config.num_entities, config.entity_embedding_dim, padding_idx=0)

        if config.use_pretrain_word_embedding:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding(config.num_words, config.word_embedding_dim,
                                               padding_idx=0)
        self.linear = nn.Linear(config.word_embedding_dim, config.category_embedding_dim)
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.category_embedding_dim, config.num_attention_heads)

        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.category_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": Tensor(batch_size),
                    "subcategory": Tensor(batch_size),
                    "title": Tensor(batch_size) * num_words_title,
                    "abstract": Tensor(batch_size) * num_words_abstract,
                    "title_entities": Tensor(batch_size) * num_words_title,
                    "abstract_entities": Tensor(batch_size) * num_words_abstract,
                }
        """

        # batch_size
        batch_cate_tensor = news['category'].to(self.device)
        # batch_size, category_embedding_dim
        batch_cate_tensor = self.category_embedding(batch_cate_tensor)

        # batch_size
        batch_subcate_tensor = news['subcategory'].to(self.device)
        # batch_size, category_embedding_dim
        batch_subcate_tensor = self.category_embedding(batch_subcate_tensor)

        # batch_size, num_entities
        batch_entities_title_tensor = torch.stack(news["title_entities"], dim=1).to(self.device)
        # batch_size, num_entities, entity_embedding_dim
        batch_entities_title_tensor = self.entity_embedding(batch_entities_title_tensor)

        # batch_size, num_entities
        batch_entities_abstract_tensor = torch.stack(news["abstract_entities"], dim=1).to(self.device)
        # batch_size, num_entities, entity_embedding_dim
        batch_entities_abstract_tensor = self.entity_embedding(batch_entities_abstract_tensor)

        # batch_size, num_words_title
        batch_title_tensor = torch.stack(news["title"], dim=1).to(self.device)

        # batch_size, num_words_abstract
        batch_abstract_tensor = torch.stack(news["abstract"], dim=1).to(self.device)

        # batch_size, num_words_title + num_word_abstract,
        batch_title_abstract_tensor = torch.cat((batch_title_tensor, batch_abstract_tensor), dim=1)

        # batch_size, num_words_title + num_word_abstract , word_embedding_dim
        batch_title_abstract_tensor = F.dropout(self.word_embedding(batch_title_abstract_tensor),
                                                p=self.config.dropout_probability,
                                                training=self.training)

        # batch_size, category_embedding_dim
        batch_title_abstract_tensor = F.relu(self.linear(batch_title_abstract_tensor))

        # batch_size,  category_embedding_dim
        batch_title_abstract_tensor = self.multihead_self_attention(batch_title_abstract_tensor)

        final_news_tensor = torch.cat((batch_cate_tensor.unsqueeze(1),
                                      batch_subcate_tensor.unsqueeze(1),
                                      batch_entities_title_tensor,
                                      batch_entities_abstract_tensor,
                                      batch_title_abstract_tensor),
                                      dim=1)
        final_news_tensor = self.additive_attention(final_news_tensor)
        return final_news_tensor
