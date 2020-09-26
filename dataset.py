from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import model_name
import importlib

try:
    Config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, attributes):
        super(BaseDataset, self).__init__()
        self.attributes = attributes
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            converters={
                attribute: literal_eval
                for attribute in set(attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.padding = {}
        if 'category' in attributes['news']:
            self.padding['category'] = 0
        if 'subcategory' in attributes['news']:
            self.padding['subcategory'] = 0
        if 'title' in attributes['news']:
            self.padding['title'] = [0] * Config.num_words_title
        if 'abstract' in attributes['news']:
            self.padding['abstract'] = [0] * Config.num_words_abstract
        if 'title_entities' in attributes['news']:
            self.padding['title_entities'] = [0] * Config.num_words_title
        if 'abstract_entities' in attributes['news']:
            self.padding['abstract_entities'] = [0] * Config.num_words_abstract

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        def news2dict(news, df):
            return {key: df.loc[news][key]
                    for key in self.attributes['news']
                    } if news in df.index else self.padding

        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in self.attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [news2dict(x, self.news_parsed)
                                  for x in row.candidate_news.split()]
        item["clicked_news"] = [
            news2dict(x, self.news_parsed)
            for x in row.clicked_news.split()[:Config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in self.attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = Config.num_clicked_news_a_user - \
                         len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend([self.padding] * repeated_times)

        return item


class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """

    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(news_path,
                                         converters={
                                             'title': literal_eval,
                                             'abstract': literal_eval,
                                             'title_entities': literal_eval,
                                             'abstract_entities': literal_eval
                                         })

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        row = self.news_parsed.iloc[idx]
        item = {
            "id": row.id,
            "category": row.category,
            "subcategory": row.subcategory,
            "title": row.title,
            "abstract": row.abstract,
            "title_entities": row.title_entities,
            "abstract_entities": row.abstract_entities
        }
        return item



class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """

    def __init__(self, config, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.config = config
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0

        if config.model_name == 'LSTUR':
            print(f'User miss rate: {user_missed / user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
                row.user,
            "clicked_news_string":
                row.clicked_news,
            "clicked_news":
                row.clicked_news.split()[:self.config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = self.config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"].extend(['PADDED_NEWS'] * repeated_times)

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """

    def __init__(self, config, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.config = config
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions,
            "clicked_news":
                row.clicked_news.split()[:self.config.num_clicked_news_a_user]

        }
        item['clicked_news_length'] = len(item["clicked_news"])
        return item
