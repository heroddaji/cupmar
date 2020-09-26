import os
import argparse
import importlib
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import latest_checkpoint
from evaluate import NewsDataset, UserDataset, BehaviorsDataset
from evaluate import ndcg_score, mrr_score, value2rank
import utils


def cal_news_vector(models_configs, test_dir):
    # add news2vector map as 2nd-index element to models_configs
    [model_list.append({}) for _, model_list in models_configs.items()]

    general_config = models_configs[list(models_configs)[0]][1]
    news_dataset = NewsDataset(os.path.join(test_dir, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=general_config.batch_size,
                                 shuffle=False,
                                 num_workers=general_config.num_workers,
                                 drop_last=False,
                                 pin_memory=general_config.pin_memory)
    with torch.no_grad():
        with tqdm(total=len(news_dataloader), desc="Calculating vectors for news") as pbar:
            for minibatch in news_dataloader:
                news_ids = minibatch["id"]
                for model_name, components in models_configs.items():
                    model = components[0]
                    news2vector = components[2]
                    if any(id not in news2vector for id in news_ids):
                        news_vector = model.get_news_vector(minibatch)
                        for id, vector in zip(news_ids, news_vector):
                            if id not in news2vector:
                                news2vector[id] = vector
                pbar.update(1)

    for model_name, components in models_configs.items():
        news2vector = components[2]
        news2vector['PADDED_NEWS'] = torch.zeros(
            list(news2vector.values())[0].size())


# def cal_user_vector(models_configs, test_dir, train_dir):
#     # add user2vector map as 3rd-index element to models_configs
#     [model_list.append({}) for _, model_list in models_configs.items()]
#
#     general_config = models_configs[list(models_configs)[0]][1]
#     user_dataset = UserDataset(general_config,
#                                os.path.join(test_dir, 'behaviors.tsv'),
#                                train_dir + '/user2int.tsv')
#
#     user_dataloader = DataLoader(user_dataset,
#                                  batch_size=general_config.batch_size,
#                                  shuffle=False,
#                                  num_workers=general_config.num_workers,
#                                  drop_last=False,
#                                  pin_memory=general_config.pin_memory)
#
#     with tqdm(total=len(user_dataloader),
#               desc="Calculating vectors for users") as pbar:
#         for minibatch in user_dataloader:
#             for model_name, components in models_configs.items():
#                 model = components[0]
#                 config = components[1]
#                 device = torch.device(config.device_str if torch.cuda.is_available() else "cpu")
#                 news2vector = components[2]
#                 user2vector = components[3]
#                 user_strings = minibatch["clicked_news_string"]
#                 if any(user_string not in user2vector
#                        for user_string in user_strings):
#                     clicked_news_vector = torch.stack([
#                         torch.stack([news2vector[x].to(device) for x in news_list],
#                                     dim=0)
#                         for news_list in minibatch["clicked_news"]
#                     ],
#                         dim=0).transpose(0, 1)
#                     if config.model_name == 'LSTUR':
#                         user_vector = model.get_user_vector(minibatch['user'],
#                                                             minibatch['clicked_news_length'],
#                                                             clicked_news_vector)
#                     else:
#                         user_vector = model.get_user_vector(clicked_news_vector)
#                     for user, vector in zip(user_strings, user_vector):
#                         if user not in user2vector:
#                             user2vector[user] = vector.to('cpu')
#             pbar.update(1)


def evaluate(models_configs,
             test_dir,
             train_dir,
             generate_txt=False,
             txt_path=None,
             has_labels=True):
    cal_news_vector(models_configs, test_dir)
    # cal_user_vector(models_configs, test_dir, train_dir)

    general_config = models_configs[list(models_configs)[0]][1]
    print('Loading Behavior dataset')
    behaviors_dataset = BehaviorsDataset(
        general_config,
        os.path.join(test_dir, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=general_config.num_workers,
                                      pin_memory=general_config.pin_memory)
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    if generate_txt:
        answer_file = open(txt_path, 'w')

    with torch.no_grad():
        with tqdm(total=len(behaviors_dataloader),
                  desc="Calculating probabilities") as pbar:
            for minibatch in behaviors_dataloader:
                impression = {
                    news[0]: [] for news in minibatch['impressions']
                }
                if has_labels:
                    impression = {
                        news[0].split('-')[0]: [] for news in minibatch['impressions']
                    }

                for model_name, components in models_configs.items():
                    model = components[0]
                    config = components[1]
                    news2vector = components[2]

                    # get user vector
                    device = torch.device(config.device_str if torch.cuda.is_available() else "cpu")
                    try:
                        if len(minibatch["clicked_news"]) == 0:
                            minibatch["clicked_news"] = [['N12246']]
                        clicked_news_vector = torch.stack([
                            torch.stack([news2vector[x].to(device) for x in news_list], dim=0)
                            for news_list in minibatch["clicked_news"]], dim=0).transpose(0, 1)
                        if config.model_name == 'LSTUR':
                            user_vector = model.get_user_vector(minibatch['user'],
                                                                minibatch['clicked_news_length'],
                                                                clicked_news_vector)
                        else:
                            user_vector = model.get_user_vector(clicked_news_vector)
                    except Exception as e:
                        print(e)
                    for news in minibatch['impressions']:
                        news_id = news[0]
                        if has_labels:
                            news_id = news[0].split('-')[0]
                        if len(user_vector.shape) == 2:
                            prediction = model.get_prediction(
                                news2vector[news_id],
                                user_vector.squeeze(0)).item()
                        elif len(user_vector.shape) == 1:
                            prediction = model.get_prediction(
                                news2vector[news_id],
                                user_vector).item()
                        else:
                            prediction = model.get_prediction(
                                news2vector[news_id],
                                user_vector.squeeze(0).mean(dim=0)).item()
                        impression[news_id].append(prediction)

                final_impression_scores = {}
                for imp_id, impression_values in impression.items():
                    tensor = torch.tensor([impression_values])
                    # softmax = F.softmax(tensor)
                    # scaled_tensor = tensor * (1 - softmax)
                    final_prediction = torch.sum(tensor).detach().item()
                    final_impression_scores[imp_id] = final_prediction

                y_pred_list = list(final_impression_scores.values())

                if has_labels:
                    y_list = [
                        int(news[0].split('-')[1]) for news in minibatch['impressions']
                    ]

                    auc = roc_auc_score(y_list, y_pred_list)
                    mrr = mrr_score(y_list, y_pred_list)
                    ndcg5 = ndcg_score(y_list, y_pred_list, 5)
                    ndcg10 = ndcg_score(y_list, y_pred_list, 10)

                    aucs.append(auc)
                    mrrs.append(mrr)
                    ndcg5s.append(ndcg5)
                    ndcg10s.append(ndcg10)

                    if pbar.n % 1000 == 0 and pbar.n > 0:
                        st = f'AUC: {np.mean(aucs):.4f}\nMRR: {np.mean(mrrs):.4f}\nnDCG@5: {np.mean(ndcg5s):.4f}\nnDCG@10: {np.mean(ndcg10s):.4f}'
                        print(st)

                if generate_txt:
                    answer_file.write(
                        f"{minibatch['impression_id'][0]} {str(list(value2rank(final_impression_scores).values())).replace(' ', '')}\n"
                    )
                pbar.update(1)

    if generate_txt:
        answer_file.close()

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)


def parse_arguments(parser):
    parser.add_argument('-ms',
                        '--models',
                        default=['NAML'],
                        help='Models to be used in the ensemble')
    parser.add_argument('-dv',
                        '--device',
                        default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help='Device to train the network on')
    parser.add_argument('-ds',
                        '--datasize',
                        default='large',
                        choices=['large', 'small'],
                        help='Mind dataset size')

    args = parser.parse_args()
    models_configs = get_models_and_configs(args)
    return models_configs


def get_models_and_configs(args):
    ensemble_model_names = args.models
    assert type(ensemble_model_names) == type([])
    models_configs = {model_name: [] for model_name in ensemble_model_names}

    for idx, (model_name, model_list) in enumerate(models_configs.items()):
        Config_cls = getattr(importlib.import_module('config'), f"{model_name}Config")
        config = Config_cls()
        config.datasize = args.datasize
        config.model_name = model_name
        config.configure_datasize()
        config.device_str = args.device

        checkpoint_path = latest_checkpoint(os.path.join(config.checkpoint_dir,
                                                         config.datasize,
                                                         config.model_name))
        if checkpoint_path is None:
            print('No checkpoint file found!')
            exit()
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        Model_cls = getattr(importlib.import_module(f"model.{model_name}"), model_name)
        model = Model_cls(config).to(config.device_str)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        model_list.append(model)
        model_list.append(config)

    return models_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble params')
    models_configs = parse_arguments(parser)

    first_model_config = models_configs[list(models_configs)[0]][1]
    prediction_folder = f'{first_model_config.test_dir}/NAML'
    Path(prediction_folder).mkdir(parents=True, exist_ok=True)
    auc, mrr, ndcg5, ndcg10 = evaluate(models_configs,
                                       first_model_config.test_dir,
                                       first_model_config.train_dir,
                                       generate_txt=True,
                                       txt_path=prediction_folder + '/prediction.txt',
                                       has_labels=False)
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )
