import os
import importlib
import argparse
from sys import platform
from pathlib import Path
from ast import literal_eval
import threading

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader

from dataset import NewsDataset, UserDataset, BehaviorsDataset
from utils import mrr_score, ndcg_score, value2rank

if platform != 'win32':
    import resource

    # torch.multiprocessing.set_sharing_strategy('file_system')
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

news_dataset = None
user2int = None


@torch.no_grad()
def evaluate_dm(config,
                model,
                eval_dir,
                train_dir,
                generate_txt=False,
                txt_path=None,
                has_labels=True,
                ):
    """
        Evaluate model on target directory.
        Args:
            config: the configuration
            model: model to be evaluated
            eval_dir: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
            train_dir: the train directory
            generate_txt: whether to generate txt file from inference result
            txt_path: file path
            has_labels: if False, there is no click label, just write predictions to file
        Returns:
            AUC
            nMRR
            nDCG@5
            nDCG@10
        """
    print(f'Evaluating model {config.model_name}')
    model.eval()

    global user2int
    if user2int is None:
        user2int = pd.read_table(os.path.join(train_dir, 'user2int.tsv'))
        user2int_dict = user2int.to_dict('l')
        user2int = dict(zip(user2int_dict['user'], user2int_dict['int']))

    ##### News vectors
    print('Loading News Vector.....')
    global news_dataset
    if news_dataset is None:
        news_dataset = NewsDataset(os.path.join(eval_dir, 'news_parsed.tsv'))

    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False,
                                 pin_memory=config.pin_memory)

    news2vector = {}
    with tqdm(total=len(news_dataloader),
              desc="Calculating vectors for news") as pbar:
        for minibatch in news_dataloader:
            batch_news_ids = minibatch["id"]
            category = minibatch['category']
            subcategory = minibatch['subcategory']
            title = minibatch['title']
            abstract = minibatch['abstract']
            title_entities = minibatch['title_entities']
            abstract_entities = minibatch['abstract_entities']

            if any(id not in news2vector for id in batch_news_ids):
                news_vector = model.get_news_vector(minibatch)
                for idx, (id, vector) in enumerate(zip(batch_news_ids, news_vector)):
                    batch_title = torch.stack([t for t in title], dim=1).to('cpu')
                    batch_abstract = torch.stack([t for t in abstract], dim=1).to('cpu')
                    batch_title_entities = torch.stack([t for t in title_entities], dim=1).to('cpu')
                    batch_abstract_entities = torch.stack([t for t in abstract_entities], dim=1).to('cpu')
                    if id not in news2vector:
                        values = {}
                        values['vector'] = vector.to('cpu')
                        values['category'] = category[idx].to('cpu')
                        values['subcategory'] = subcategory[idx].to('cpu')
                        values['title'] = batch_title[idx]
                        values['abstract'] = batch_abstract[idx]
                        values['title_entities'] = batch_title_entities[idx]
                        values['abstract_entities'] = batch_abstract_entities[idx]
                        news2vector[id] = values

            pbar.update(1)

    behaviors_dataset = BehaviorsDataset(config,
                                         os.path.join(eval_dir, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers,
                                      pin_memory=config.pin_memory)
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    if generate_txt:
        answer_file = open(txt_path, 'w')

    with tqdm(total=len(behaviors_dataloader),
              desc="Calculating probabilities") as pbar:
        for minibatch in behaviors_dataloader:
            clicked_news_ids = minibatch['clicked_news_string'][0].split()
            impression_news = minibatch['impressions']
            cnews = []
            for news_id in clicked_news_ids:
                clicked_news_cate = {
                    'category': news2vector[news_id]['category'].unsqueeze(0),
                    'subcategory': news2vector[news_id]['subcategory'].unsqueeze(0),
                    'title': [x for x in news2vector[news_id]['title'].view(-1, 1)],
                    'abstract': [x for x in news2vector[news_id]['abstract'].view(-1, 1)],
                    'title_entities': [x for x in news2vector[news_id]['title_entities'].view(-1, 1)],
                    'abstract_entities': [x for x in news2vector[news_id]['abstract_entities'].view(-1, 1)]
                }
                cnews.append(clicked_news_cate)
            if config.model_name == 'DMUserGru' or config.model_name == 'DMCateContext' or config.model_name == 'DMCupsan':
                user_original_id = minibatch['user'][0]
                if user2int.get(user_original_id, None) is None:
                    user2int[user_original_id] = len(user2int) + 1
                user_id = user2int[user_original_id]
                user = torch.tensor(user_id).unsqueeze(0)
                user_vector = model.get_user_vector(user, cnews, empty_clicked_news_batch_size=1).squeeze(0).to(
                    config.device_str)
            else:
                user_vector = model.get_user_vector(cnews, empty_clicked_news_batch_size=1).squeeze(0).to(
                    config.device_str)
            y_pred_list = []
            impression = {}
            for news in impression_news:
                news_id = news[0].split('-')[0]
                news_vector = news2vector[news_id]['vector'].to(config.device_str)
                prediction = model.get_prediction(news_vector, user_vector)
                y_pred_list.append(prediction.item())
                impression[news_id] = prediction.item()

            y_list = [int(news[0].split('-')[1]) for news in minibatch['impressions']]

            auc = roc_auc_score(y_list, y_pred_list)
            mrr = mrr_score(y_list, y_pred_list)
            ndcg5 = ndcg_score(y_list, y_pred_list, 5)
            ndcg10 = ndcg_score(y_list, y_pred_list, 10)

            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)

            if generate_txt:
                answer_file.write(
                    f"{minibatch['impression_id'][0]} {str(list(value2rank(impression).values())).replace(' ', '')}\n"
                )

            pbar.update(1)

    if generate_txt:
        answer_file.close()

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)


@torch.no_grad()
def evaluate(config,
             model,
             eval_dir,
             train_dir,
             generate_txt=False,
             txt_path=None,
             has_labels=True,
             ):
    """
    Evaluate model on target directory.
    Args:
        config: the configuration
        model: model to be evaluated
        eval_dir: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        train_dir: the train directory
        generate_txt: whether to generate txt file from inference result
        txt_path: file path
        has_labels: if False, there is no click label, just write predictions to file
    Returns:
        AUC
        nMRR
        nDCG@5
        nDCG@10
    """
    print(f'Evaluating model {model}')
    model.eval()
    device = torch.device(config.device_str if torch.cuda.is_available() else "cpu")

    ##### News vectors
    news_dataset = NewsDataset(os.path.join(eval_dir, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=config.pin_memory)

    news2vector = {}
    with tqdm(total=len(news_dataloader),
              desc="Calculating vectors for news") as pbar:
        for minibatch in news_dataloader:
            news_ids = minibatch["id"]
            if any(id not in news2vector for id in news_ids):
                news_vector = model.get_news_vector(minibatch)
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector.to('cpu')
            pbar.update(1)

    news2vector['PADDED_NEWS'] = torch.zeros(list(news2vector.values())[0].size())

    ###### User vectors
    user_dataset = UserDataset(config,
                               os.path.join(eval_dir, 'behaviors.tsv'),
                               train_dir + '/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=config.pin_memory)

    user2vector = {}
    with tqdm(total=len(user_dataloader),
              desc="Calculating vectors for users") as pbar:
        for minibatch in user_dataloader:
            user_strings = minibatch["clicked_news_string"]
            if any(user_string not in user2vector for user_string in user_strings):
                clicked_news_vector = torch.stack([
                    torch.stack([news2vector[x].to(device) for x in news_list], dim=0) for news_list in
                    minibatch["clicked_news"]
                ], dim=0).transpose(0, 1)

                if config.model_name == 'LSTUR':
                    user_vector = model.get_user_vector(minibatch['user'],
                                                        minibatch['clicked_news_length'],
                                                        clicked_news_vector)
                else:
                    user_vector = model.get_user_vector(clicked_news_vector)

                for user, vector in zip(user_strings, user_vector):
                    if user not in user2vector:
                        user2vector[user] = vector.to('cpu')

            pbar.update(1)

    ##### Evaluation
    behaviors_dataset = BehaviorsDataset(config,
                                         os.path.join(eval_dir, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers,
                                      pin_memory=config.pin_memory)
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    if generate_txt:
        answer_file = open(txt_path, 'w')

    with tqdm(total=len(behaviors_dataloader),
              desc="Calculating probabilities") as pbar:
        for minibatch in behaviors_dataloader:
            impression = {
                news[0].split('-')[0]: model.get_prediction(
                    news2vector[news[0].split('-')[0]].to(config.device_str),
                    user2vector[minibatch['clicked_news_string'][0]].to(config.device_str)).item()
                for news in minibatch['impressions']
            }

            y_pred_list = list(impression.values())
            y_list = [int(news[0].split('-')[1]) for news in minibatch['impressions']]

            auc = roc_auc_score(y_list, y_pred_list)
            mrr = mrr_score(y_list, y_pred_list)
            ndcg5 = ndcg_score(y_list, y_pred_list, 5)
            ndcg10 = ndcg_score(y_list, y_pred_list, 10)

            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)

            if generate_txt:
                answer_file.write(
                    f"{minibatch['impression_id'][0]} {str(list(value2rank(impression).values())).replace(' ', '')}\n"
                )
            pbar.update(1)

    if generate_txt:
        answer_file.close()

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)


if __name__ == '__main__':
    # avoid circular import
    from train import parse_arguments, get_model, restore_checkpoint

    parser = argparse.ArgumentParser(description='Eval params')
    config = parse_arguments(parser)

    model = get_model(config)
    model, is_sucessfull = restore_checkpoint(config, model, is_train=False)

    if not is_sucessfull:
        print('No checkpoint file found!')
        exit()

    prediction_folder = f'{config.val_dir}/{config.model_name}'
    Path(prediction_folder).mkdir(parents=True, exist_ok=True)
    if config.model_name.startswith('DM'):
        auc, mrr, ndcg5, ndcg10 = evaluate_dm(config,
                                              model,
                                              config.dev_dir,
                                              config.train_dir,
                                              generate_txt=True,
                                              txt_path=prediction_folder + '/prediction.txt',
                                              has_labels=True)
    else:
        auc, mrr, ndcg5, ndcg10 = evaluate(config,
                                           model,
                                           config.dev_dir,
                                           config.train_dir,
                                           generate_txt=True,
                                           txt_path=prediction_folder + '/prediction.txt',
                                           has_labels=True)
    result = f'AUC:{auc:.4f} MRR:{mrr:.4f} nDCG@5:{ndcg5:.4f} nDCG@10:{ndcg10:.4f}'
    print(result)
    remark = config.remark if config.remark == '' else f',{config.remark}'
    with open(config.score_path, 'a+') as f:
        f.write(f'{config.model_name}, {config.datasize}{remark}: {result}\n')
