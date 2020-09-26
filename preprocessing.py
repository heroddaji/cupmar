import sys
import json
import random
from os import path
from pathlib import Path
from shutil import copyfile
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize

import config as cf
import argparse


def parse_behaviors(config, source, target, val_target, user2int_path):
    """
    Parse behaviors file in training set.
    Args:
        config: config
        source: source behaviors file
        target: target behaviors file
        val_target: target behaviors file used for validation split from training set
        user2int_path: path for saving user2int file
    """
    print(f"Parse {source}")
    with open(source, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(val_target, 'w') as f:
        f.writelines(lines[:int(len(lines) * config.validation_proportion)])

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = {}
    for row in behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1

    pd.DataFrame(user2int.items(), columns=['user',
                                            'int']).to_csv(user2int_path,
                                                           sep='\t',
                                                           index=False)
    print(
        f'Please modify `num_users` in `src/config.py` into 1 + {len(user2int)}'
    )

    # Drop rows in val_behaviors
    val_behaviors = pd.read_table(val_target,
                                  header=None,
                                  usecols=[0],
                                  names=['impression_id'])
    behaviors.drop(behaviors[behaviors['impression_id'].isin(
        val_behaviors['impression_id'])].index,
                   inplace=True)
    print(
        f'Drop {len(val_behaviors)} sessions from training set to be used in validation.'
    )

    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int[row.user]

    with tqdm(total=len(behaviors), desc="Balancing data") as pbar:
        for row in behaviors.itertuples():
            positive = iter([x for x in row.impressions if x.endswith('1')])
            negative = [x for x in row.impressions if x.endswith('0')]
            random.shuffle(negative)
            negative = iter(negative)
            pairs = []
            try:
                while True:
                    pair = [next(positive)]
                    for _ in range(config.negative_sampling_ratio):
                        pair.append(next(negative))
                    pairs.append(pair)
            except StopIteration:
                pass
            behaviors.at[row.Index, 'impressions'] = pairs
            pbar.update(1)
    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]).reset_index(drop=True)
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_behaviors_dev(config, source, target, user2int_path):
    """
        Parse behaviors file in dev set.
        Args:
            config: config
            source: source behaviors file
            target: target behaviors file
            user2int_path: path of the user2int file, usually in the train_dir
        """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = pd.read_table(user2int_path)
    user2int_dict = user2int.to_dict('l')
    user2int = dict(zip(user2int_dict['user'], user2int_dict['int']))

    for row in behaviors.itertuples():
        exist_user_id = user2int.get(row.user, None)
        if exist_user_id is None:
            user2int[row.user] = len(user2int) + 1
        else:
            behaviors.at[row.Index, 'user'] = user2int[row.user]

    def convert_to_clicked(row, ):
        return ' '.join([label.split('-')[1] for label in row])

    def convert_to_candidate_news(row):
        return ' '.join([label.split('-')[0] for label in row])

    behaviors['clicked'] = behaviors['impressions'].apply(lambda row: convert_to_clicked(row))
    behaviors['candidate_news'] = behaviors['impressions'].apply(lambda row: convert_to_candidate_news(row))
    behaviors.drop(columns=['impressions'])

    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_news(config,
               source, target, category2int_path, word2int_path,
               entity2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, word2int_path, entity2int_path: Path to load from
    """
    print(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4, 6, 7],
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities'
                         ])
    news.title_entities.fillna('[]', inplace=True)
    news.abstract_entities.fillna('[]', inplace=True)
    news.fillna(' ', inplace=True)

    def parse_row(row):
        new_row = [
            row.id,
            category2int[row.category] if row.category in category2int else 0,
            category2int[row.subcategory]
            if row.subcategory in category2int else 0,
            [0] * config.num_words_title, [0] * config.num_words_abstract,
            [0] * config.num_words_title, [0] * config.num_words_abstract
        ]

        # Calculate local entity map (map lower single word to entity)
        local_entity_map = {}
        for e in json.loads(row.title_entities):
            if e['Confidence'] > config.entity_confidence_threshold and e[
                'WikidataId'] in entity2int:
                for x in ' '.join(e['SurfaceForms']).lower().split():
                    local_entity_map[x] = entity2int[e['WikidataId']]
        for e in json.loads(row.abstract_entities):
            if e['Confidence'] > config.entity_confidence_threshold and e[
                'WikidataId'] in entity2int:
                for x in ' '.join(e['SurfaceForms']).lower().split():
                    local_entity_map[x] = entity2int[e['WikidataId']]

        try:
            for i, w in enumerate(word_tokenize(row.title.lower())):
                if w in word2int:
                    new_row[3][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[5][i] = local_entity_map[w]
        except IndexError:
            pass

        try:
            for i, w in enumerate(word_tokenize(row.abstract.lower())):
                if w in word2int:
                    new_row[4][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[6][i] = local_entity_map[w]
        except IndexError:
            pass

        return pd.Series(new_row,
                         index=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities'
                         ])

    if mode == 'train':
        category2int = {}
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        for row in news.itertuples(index=False):
            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

            for w in word_tokenize(row.title.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.abstract.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            for e in json.loads(row.title_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

            for e in json.loads(row.abstract_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

        for k, v in word2freq.items():
            if v >= config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        parsed_news = news.apply(parse_row, axis=1)

        parsed_news.to_csv(target, sep='\t', index=False)

        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )

        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
        print(
            f'Please modify `num_entities` in `src/config.py` into 1 + {len(entity2int)}'
        )

    elif mode == 'test':
        category2int = dict(pd.read_table(category2int_path).values.tolist())
        # na_filter=False is needed since nan is also a valid word
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        parsed_news = news.apply(parse_row, axis=1)

        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


def generate_word_embedding(config, source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    word2int = pd.read_table(word2int_path, na_filter=False, index_col='word')
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(config.word_embedding_dim))
    # word, vector
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1),
                                merged.index.values)
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), config.word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    np.save(target, final_embedding.values)

    print(
        f'Rate of word missed in pretrained embedding: {(len(missed_index) - 1) / len(word2int):.4f}'
    )


def transform_entity_embedding(config, source, target, entity2int_path):
    """
    Args:
        source: path of embedding file
        target: path of transformed embedding file in numpy format
        entity2int_path
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector'
                                         ]].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(entity_embedding, entity2int,
                         on='entity').sort_values('int')
    entity_embedding_transformed = np.zeros(
        (len(entity2int) + 1, config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)


def transform2txt(config, source, target):
    """
    Transform behaviors file in tsv to txt
    """
    behaviors = pd.read_table(source,
                              header=None,
                              usecols=[0, 4],
                              names=['impression_id', 'impression'])
    f = open(target, "w")
    with tqdm(total=len(behaviors), desc="Transforming tsv to txt") as pbar:
        for row in behaviors.itertuples(index=False):
            f.write(
                f"{row.impression_id} {str([int(x.split('-')[1]) for x in row.impression.split()]).replace(' ', '')}\n"
            )

            pbar.update(1)

    f.close()


if __name__ == '__main__':
    preconf = cf.BaseConfig()
    parser = argparse.ArgumentParser(description='Preprocessing params')
    parser.add_argument('-ds',
                        '--datasize',
                        default='small',
                        choices=['large', 'small'],
                        help='Mind dataset size')

    args = parser.parse_args()
    datasize = args.datasize
    if datasize == 'large':
        train_dir = preconf.large_train_dir
        val_dir = preconf.large_val_dir
        dev_dir = preconf.large_dev_dir
        test_dir = preconf.large_test_dir
    elif datasize == 'small':
        train_dir = preconf.small_train_dir
        val_dir = preconf.small_val_dir
        dev_dir = preconf.small_dev_dir
        test_dir = preconf.small_dev_dir

    Path(train_dir).mkdir(exist_ok=True)
    Path(val_dir).mkdir(exist_ok=True)
    Path(dev_dir).mkdir(exist_ok=True)
    Path(test_dir).mkdir(exist_ok=True)

    print('Process data for training')

    # print('Parse train behaviors')
    # parse_behaviors(preconf,
    #                 path.join(train_dir, 'behaviors.tsv'),
    #                 path.join(train_dir, 'behaviors_parsed.tsv'),
    #                 path.join(val_dir, 'behaviors.tsv'),
    #                 path.join(train_dir, 'user2int.tsv'))
    #
    # print('Parse news')
    # parse_news(preconf,
    #            path.join(train_dir, 'news.tsv'),
    #            path.join(train_dir, 'news_parsed.tsv'),
    #            path.join(train_dir, 'category2int.tsv'),
    #            path.join(train_dir, 'word2int.tsv'),
    #            path.join(train_dir, 'entity2int.tsv'),
    #            mode='train')
    #
    # # For validation in training
    # copyfile(path.join(train_dir, 'news_parsed.tsv'),
    #          path.join(val_dir, 'news_parsed.tsv'))

    print('Parse val behaviors')
    parse_behaviors_dev(preconf,
                        path.join(val_dir, 'behaviors.tsv'),
                        path.join(val_dir, 'behaviors_parsed.tsv'),
                        path.join(train_dir, 'user2int.tsv'))

    # print('Generate word embedding')
    # generate_word_embedding(
    #     preconf,
    #     preconf.word_embedding_file,
    #     path.join(train_dir, 'pretrained_word_embedding.npy'),
    #     path.join(train_dir, 'word2int.tsv'))
    #
    # print('Transform entity embeddings')
    # transform_entity_embedding(
    #     preconf,
    #     path.join(train_dir, 'entity_embedding.vec'),
    #     path.join(train_dir, 'pretrained_entity_embedding.npy'),
    #     path.join(train_dir, 'entity2int.tsv'))

    # todo: why do we need to do this transformation
    # print('Transform test data')
    # transform2txt(
    #     preconf,
    #     path.join(test_dir, 'behaviors.tsv'),
    #     path.join(test_dir, 'truth.txt'))

    # print('Parse dev behaviors')
    # parse_behaviors_dev(preconf,
    #                     path.join(dev_dir, 'behaviors.tsv'),
    #                     path.join(dev_dir, 'behaviors_parsed.tsv'),
    #                     path.join(train_dir, 'user2int.tsv'))

    # print('Parse dev news')
    # parse_news(preconf,
    #            path.join(test_dir, 'news.tsv'),
    #            path.join(test_dir, 'news_parsed.tsv'),
    #            path.join(train_dir, 'category2int.tsv'),
    #            path.join(train_dir, 'word2int.tsv'),
    #            path.join(train_dir, 'entity2int.tsv'),
    #            mode='test')
