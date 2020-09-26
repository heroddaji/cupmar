import os
import sys
import time
import importlib
import datetime
import argparse
from sys import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset import BaseDataset
from evaluate import evaluate, evaluate_dm
from utils import latest_checkpoint, EarlyStopping

if platform != 'win32':
    # need this code to fix a pytorch bug
    # torch.multiprocessing.set_sharing_strategy('file_system')
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def get_model(config):
    if config.use_pretrain_word_embedding:
        try:
            pretrained_word_embedding = torch.from_numpy(np.load(config.train_dir + config.pretrained_word)).float()
        except FileNotFoundError:
            pretrained_word_embedding = None

    Model_cls = getattr(importlib.import_module(f"model.{config.model_name}"), config.model_name)
    if config.model_name == 'DKN':
        try:
            pretrained_entity_embedding = torch.from_numpy(np.load(config.train_dir + config.pretrained_entity)).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None

        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(config.train_dir + config.pretrained_context)).float()
        except FileNotFoundError:
            pretrained_context_embedding = None

        model = Model_cls(config, pretrained_word_embedding,
                          pretrained_entity_embedding,
                          pretrained_context_embedding).to(config.device_str)
    else:
        model = Model_cls(config, pretrained_word_embedding).to(config.device_str)
    print(model)
    print(f'total trainable parameters:{sum([param.nelement() for param in model.parameters()])}')
    return model


def restore_checkpoint(config, model, optimizer=None, early_stopping=None, is_train=True):
    checkpoint_dir = os.path.join(config.checkpoint_dir, config.datasize, config.model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        config.step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if early_stopping:
            early_stopping(checkpoint['early_stop_value'])
        if is_train:
            model.train()
        else:
            model.eval()

    return model, checkpoint_path is not None


def get_train_dataset(config):
    dataset = BaseDataset(config.train_dir + '/behaviors_parsed.tsv',
                          config.train_dir + '/news_parsed.tsv',
                          config.dataset_attributes)

    print(f"Load training dataset with size {len(dataset)}.")
    return dataset


def train(config):
    loss_full = []
    exhaustion_count = 0
    step = 0
    config.step = step
    writer = None
    start_time = time.time()

    if config.log_tensorboard: 
        writer = SummaryWriter(
            log_dir=
            f"{config.train_dir}/runs/{config.model_name}/{config.turn}-{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
        )

    model = get_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    early_stopping = EarlyStopping()
    early_stopping_loss = EarlyStopping(patience=200)
    model, _ = restore_checkpoint(config, model, optimizer, early_stopping)
    step = config.step
    dataset = get_train_dataset(config)

    for epoch in range(config.epochs):
        print(f'epoch {epoch}/{config.epochs}')
        dataloader = iter(
            DataLoader(dataset,
                       batch_size=config.batch_size,
                       shuffle=True,
                       num_workers=config.num_workers,
                       drop_last=True,
                       pin_memory=config.pin_memory))

        total_batches = int(len(dataset) / config.batch_size)
        with tqdm(total=total_batches, desc=f'Training epoch {epoch}/{config.epochs}') as pbar:
            for i in range(1, total_batches + 1):
                model.train()
                try:
                    minibatch = next(dataloader)
                except StopIteration:
                    exhaustion_count += 1
                    tqdm.write(
                        f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                    )
                    dataloader = iter(
                        DataLoader(dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   pin_memory=config.pin_memory))
                    minibatch = next(dataloader)

                step += 1
                if config.model_name == 'LSTUR':
                    y_pred = model(minibatch["user"],
                                   minibatch["clicked_news_length"],
                                   minibatch["candidate_news"],
                                   minibatch["clicked_news"])
                elif config.model_name == 'HiFiArk':
                    y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                                     minibatch["clicked_news"])
                elif config.model_name == 'TANR':
                    y_pred, topic_classification_loss = model(
                        minibatch["candidate_news"], minibatch["clicked_news"])
                elif config.model_name.startswith('DM'):
                    y_pred = model(minibatch)
                else:
                    y_pred = model(minibatch["candidate_news"],
                                   minibatch["clicked_news"])

                loss = torch.stack([x[0] for x in -F.log_softmax(y_pred, dim=1)]).mean()
                if config.model_name == 'HiFiArk':
                    if i % config.num_iters_show_loss == 0:
                        if config.log_tensorboard:
                            writer.add_scalar('Train/BaseLoss', loss.item(), step)
                            writer.add_scalar('Train/RegularizerLoss',
                                              regularizer_loss.item(), step)
                            writer.add_scalar('Train/RegularizerBaseRatio',
                                              regularizer_loss.item() / loss.item(),
                                              step)
                    loss += config.regularizer_loss_weight * regularizer_loss
                elif config.model_name == 'TANR':
                    if i % config.num_iters_show_loss == 0:
                        if config.log_tensorboard:
                            writer.add_scalar('Train/BaseLoss', loss.item(), step)
                            writer.add_scalar('Train/TopicClassificationLoss',
                                              topic_classification_loss.item(), step)
                            writer.add_scalar(
                                'Train/TopicBaseRatio',
                                topic_classification_loss.item() / loss.item(), step)
                    loss += config.topic_classification_loss_weight * topic_classification_loss
                loss_full.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_f = np.mean(loss_full)

                if i % config.num_iters_show_loss == 0:
                    if config.log_tensorboard:
                        writer.add_scalar('Train/Loss', loss.item(), step)

                if i % config.num_batches_show_loss == 0:
                    tqdm.write(
                        f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {loss_f:.4f}"
                    )
                    stopping_loss, _ = early_stopping_loss(loss_f)
                    if stopping_loss:
                        tqdm.write('Early stop due to no improvement on loss.')
                        eval_and_save_checkpoint(config,
                                                 model,
                                                 optimizer,
                                                 early_stopping,
                                                 writer,
                                                 loss_f,
                                                 step,
                                                 start_time,
                                                 i)
                        break

                if i % config.num_batches_validate == 0 or i == total_batches:
                    should_break = eval_and_save_checkpoint(config,
                                                            model,
                                                            optimizer,
                                                            early_stopping,
                                                            writer,
                                                            loss_f,
                                                            step,
                                                            start_time,
                                                            i)
                    if should_break:
                        break

                pbar.update(1)


def eval_and_save_checkpoint(config,
                             model,
                             optimizer,
                             early_stopping,
                             writer,
                             loss_f,
                             step,
                             start_time,
                             i):
    should_break = False

    if config.model_name.startswith('DM'):
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate_dm(
            config,
            model,
            config.val_dir,
            config.train_dir,
            generate_txt=False,
            txt_path=config.val_dir + f'/{config.model_name}-prediction.txt')
    else:
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
            config,
            model,
            config.val_dir,
            config.train_dir,
            generate_txt=False,
            txt_path=config.val_dir + f'/{config.model_name}-prediction.txt')

    if config.log_tensorboard:
        writer.add_scalar('Validation/AUC', val_auc, step)
        writer.add_scalar('Validation/MRR', val_mrr, step)
        writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
        writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
    tqdm.write(
        f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
    )

    early_stop, get_better = early_stopping(-val_auc)
    if early_stop:
        tqdm.write('Early stop.')
        should_break = True
    elif get_better:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'early_stop_value': -val_auc,
            },
            f"{config.checkpoint_dir}/{config.datasize}/{config.model_name}/ckpt-{step}_loss-{loss_f:.4f}_auc-{val_auc:.4f}.pth")
    return should_break


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def parse_arguments(parser):
    parser.add_argument('-m',
                        '--model',
                        default='DMUserGru',
                        choices=['NRMS', 'NAML', 'DKN', 'TANR', 'HiFiArk', 'FIM', 'LSTUR',
                                 'DMCategory', 'DMUserGru','DMCateContext','DMCupsan'],
                        help='Model to train')

    parser.add_argument('-ds',
                        '--datasize',
                        default='small',
                        choices=['large', 'small'],
                        help='Mind dataset size')

    parser.add_argument('-dv',
                        '--device',
                        default='cuda:0',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help='Device to train the network on')
    parser.add_argument('-t',
                        '--turn',
                        default=5,
                        help='turn to log to tensorboard')

    parser.add_argument('-b',
                        '--batches',
                        default=32,
                        help='batch size ')

    parser.add_argument('-bv',
                        '--batchvalid',
                        default=1000,
                        help='batch size ')

    parser.add_argument('-r',
                        '--remark',
                        default='',
                        help='Remark for this model')

    args = parser.parse_args()
    config = get_config(args)
    return config


def get_config(args):
    model_name = args.model
    Config_cls = getattr(importlib.import_module('config'), f"{model_name}Config")
    config = Config_cls()
    config.model_name = model_name
    config.device_str = args.device
    config.turn = args.turn
    config.batch_size = int(args.batches)
    config.num_batches_validate = int(args.batchvalid)
    config.datasize = args.datasize
    config.remark = args.remark
    config.configure_datasize()

    print('Using device:', config.device_str)
    print(f'Using model {config.model_name}')

    return config


# def train_multiprocesses(config, model, dataloader):
#     loss_full = []
#     step = 0
#     writer = None
#     start_time = time.time()
#
#     if config.log_tensorboard:
#         writer = SummaryWriter(
#             log_dir=
#             f"{config.train_dir}/runs/{config.model_name}/{config.turn}-{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
#         )
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
#     early_stopping = EarlyStopping()
#
#     for epoch in range(config.epochs):
#         print(f'epoch {epoch}/{config.epochs}')
#
#         for minibatch in dataloader:
#             step += 1
#             print(f'training {step}')
#             if config.model_name == 'LSTUR':
#                 y_pred = model(minibatch["user"],
#                                minibatch["clicked_news_length"],
#                                minibatch["candidate_news"],
#                                minibatch["clicked_news"])
#             elif config.model_name == 'HiFiArk':
#                 y_pred, regularizer_loss = model(minibatch["candidate_news"],
#                                                  minibatch["clicked_news"])
#             elif config.model_name == 'TANR':
#                 y_pred, topic_classification_loss = model(
#                     minibatch["candidate_news"], minibatch["clicked_news"])
#             elif config.model_name.startswith('DM'):
#                 y_pred = model(minibatch)
#             else:
#                 y_pred = model(minibatch["candidate_news"],
#                                minibatch["clicked_news"])
#
#             loss = torch.stack([x[0] for x in -F.log_softmax(y_pred, dim=1)]).mean()
#             if config.model_name == 'HiFiArk':
#                 if step % config.num_iters_show_loss == 0:
#                     if config.log_tensorboard:
#                         writer.add_scalar('Train/BaseLoss', loss.item(), step)
#                         writer.add_scalar('Train/RegularizerLoss',
#                                           regularizer_loss.item(), step)
#                         writer.add_scalar('Train/RegularizerBaseRatio',
#                                           regularizer_loss.item() / loss.item(),
#                                           step)
#                 loss += config.regularizer_loss_weight * regularizer_loss
#             elif config.model_name == 'TANR':
#                 if step % config.num_iters_show_loss == 0:
#                     if config.log_tensorboard:
#                         writer.add_scalar('Train/BaseLoss', loss.item(), step)
#                         writer.add_scalar('Train/TopicClassificationLoss',
#                                           topic_classification_loss.item(), step)
#                         writer.add_scalar(
#                             'Train/TopicBaseRatio',
#                             topic_classification_loss.item() / loss.item(), step)
#                 loss += config.topic_classification_loss_weight * topic_classification_loss
#             loss_full.append(loss.item())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if step % config.num_iters_show_loss == 0:
#                 if config.log_tensorboard:
#                     writer.add_scalar('Train/Loss', loss.item(), step)
#
#             if step % config.num_batches_show_loss == 0:
#                 tqdm.write(
#                     f"Time {time_since(start_time)}, batches {step}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}"
#                 )
#             if step % config.num_batches_validate == 0:
#                 ###validate model
#                 if config.model_name.startswith('DM'):
#                     val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate_dm(
#                         config,
#                         model,
#                         config.val_dir,
#                         config.train_dir,
#                         generate_txt=False,
#                         txt_path=config.val_dir + f'/{config.model_name}-prediction.txt')
#                 else:
#                     val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
#                         config,
#                         model,
#                         config.val_dir,
#                         config.train_dir,
#                         generate_txt=False,
#                         txt_path=config.val_dir + f'/{config.model_name}-prediction.txt')
#
#                 if config.log_tensorboard:
#                     writer.add_scalar('Validation/AUC', val_auc, step)
#                     writer.add_scalar('Validation/MRR', val_mrr, step)
#                     writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
#                     writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
#                 tqdm.write(
#                     f"Time {time_since(start_time)}, batches {step}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
#                 )
#
#                 early_stop, get_better = early_stopping(-val_auc)
#                 if early_stop:
#                     tqdm.write('Early stop.')
#                     break
#                 elif get_better:
#                     torch.save(
#                         {
#                             'model_state_dict': model.state_dict(),
#                             'optimizer_state_dict': optimizer.state_dict(),
#                             'step': step,
#                             'early_stop_value': -val_auc
#                         }, f"{config.checkpoint_dir}/{config.datasize}/{config.model_name}/ckpt-{step}.pth")
#
#
# def run_multiprocesses(config):
#     model = get_model(config)
#     model.share_memory()
#     processes = []
#     dataset = get_dataset(config)
#
#     num_processes = config.num_processes
#     for rank in range(num_processes):
#         dataloader = DataLoader(
#             dataset=dataset,
#             sampler=DistributedSampler(
#                 dataset=dataset,
#                 shuffle=True,
#                 num_replicas=num_processes,
#                 rank=rank
#             ),
#             batch_size=config.batch_size
#         )
#         p = mp.Process(target=train_multiprocesses, args=(config, model, dataloader))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train params')
    config = parse_arguments(parser)

    train(config)
    # run_multiprocesses(config)
