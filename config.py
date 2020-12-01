import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'NRMS'
# Currently included model
assert model_name in ['NRMS', 'NAML', 'LSTUR', 'DKN', 'TANR', 'FIM', 'HiFiArk']


class BaseConfig():
    """
        General configurations appiled to all models
    """
    epochs = 2
    # num_batches = 3000  # 300  # Number of batches to train
    num_batches_show_loss = 10  # Number of batchs to show loss
    num_iters_show_loss = 10  # write to tensorboard
    # Number of batchs to check metrics on validation dataset, aslo check point
    num_batches_validate = 200
    batch_size = 256
    learning_rate = 0.001
    validation_proportion = 0.1
    num_workers = 0  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 3
    entity_freq_threshold = 3
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 44774
    num_categories = 1 + 295
    num_entities = 1 + 14697
    num_users = 1 + 711222
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200

    use_pretrain_word_embedding = True
    log_tensorboard = False
    # paths
    local = f'e:/dev/datasets'
    remote = f'/disks/sheng/scratch-ssd/dai/datasets/mind'
    localize_path = local
    word_embedding_file = f'{localize_path}/glove.840B.300d.txt'
    checkpoint_dir = './checkpoint1'
    pretrained_word = '/pretrained_word_embedding.npy'
    pretrained_entity = '/pretrained_entity_embedding.npy'
    pretrained_context = '/pretrained_context_embedding.npy'

    large_train_dir = f'{localize_path}/MINDlarge_train'
    large_val_dir = f'{localize_path}/MINDlarge_val'
    large_dev_dir = f'{localize_path}/MINDlarge_dev'
    large_test_dir = f'{localize_path}/MINDlarge_test'

    small_train_dir = f'{localize_path}/MINDsmall_train'
    small_val_dir = f'{localize_path}/MINDsmall_val'
    small_dev_dir = f'{localize_path}/MINDsmall_dev'

    score_path = f'{localize_path}/scores.txt'

    device_str = 'cuda:0'
    turn = 1
    datasize = 'small'
    model_name = 'NAML'
    train_dir = ''
    val_dir = ''
    test_dir = ''
    dev_dir = ''
    remark = ''
    pin_memory = True

    def configure_datasize(self):
        if self.datasize == 'large':
            self.__large()
        elif self.datasize == 'small':
            self.__small()

    def __large(self):
        self.num_words = 1 + 44774
        self.num_categories = 1 + 295
        self.num_entities = 1 + 14697
        self.num_users = 1 + 711222
        self.train_dir = self.large_train_dir
        self.val_dir = self.large_val_dir
        self.dev_dir = self.large_dev_dir
        self.test_dir = self.large_test_dir

    def __small(self):
        self.num_words = 1 + 31312
        self.num_categories = 1 + 274
        self.num_entities = 1 + 8312
        self.num_users = 1 + 50000
        self.train_dir = self.small_train_dir
        self.val_dir = self.small_val_dir
        self.dev_dir = self.small_dev_dir
        self.test_dir = self.small_dev_dir


class DMCupmarConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities'],
        "record": ['user']
    }
    user_embedding_dim = 100
    num_attention_heads = 10
    query_vector_dim = 100
    lpe_top_cate_num= 6 #how many top categories to keep as long term preferences


class DMUserGruConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities'],
        "record": ['user']
    }
    user_embedding_dim = 100
    num_attention_heads = 10
    query_vector_dim = 100


class DMCateContextConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title_entities', 'abstract_entities'],
        "record": ['user']
    }
    user_embedding_dim = 100
    num_attention_heads = 10
    query_vector_dim = 100


class DMCategoryConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory'],
        "record": []
    }
    category_embedding_dim = 100
    user_encoding_strategy = 'sum_cate'
    assert user_encoding_strategy in ['sum_cate', 'cate_frequency']


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = 15


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1


class FIMConfig(BaseConfig):
    dataset_attributes = {
        # Currently only title is used
        "news": ['category', 'subcategory', 'title'],
        "record": []
    }
    news_rep = {"num_filters": 300, "window_size": 3, "dilations": [1, 2, 3]}
    cross_matching = {
        "layers": [{
            "num_filters": 32,
            "window_size": (3, 3, 3),
            "stride": (1, 1, 1)
        }, {
            "num_filters": 16,
            "window_size": (3, 3, 3),
            "stride": (1, 1, 1)
        }],
        "max_pooling": {
            "window_size": (3, 3, 3),
            "stride": (3, 3, 3)
        }
    }
