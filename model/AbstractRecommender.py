import os
import time
from abc import ABC
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from NeuRec.data.dataset import Dataset
from NeuRec.evaluator import ProxyEvaluator
from NeuRec.util import Logger, Configurator
from NeuRec.util.tool import csr_to_user_dict_bytime

# note: keep below caches in memory only
cache__loggers: Dict[str, Logger] = {}
cache__evaluators: Dict[str, ProxyEvaluator] = {}
cache__sessions: Dict[str, tf.Session] = {}


class AbstractRecommender(object):
    def __init__(self, conf: Configurator):
        self.conf: Configurator = conf

        # generate cache key
        param_str = "%s_%s" % (self.dataset_name, self.conf.params_str())
        model_name = self.conf["recommender"]
        timestamp = time.time()
        run_id = "%s_%.8f" % (param_str[:150], timestamp)
        self.cache_key = f"{param_str}_{model_name}"

        # generate logger name
        log_dir = os.path.join("log", self.dataset_name, model_name)
        self.logger_name: str = os.path.join(log_dir, run_id + ".log")

        self.logger.info(self.dataset)
        self.logger.info(conf)

    @property
    def dataset_name(self) -> str:
        return self.conf["data.input.dataset"]

    @property
    def dataset(self) -> Dataset:
        return Dataset.cache[self.dataset_name]

    @property
    def evaluator(self) -> ProxyEvaluator:
        # To save storage, inference the Evaluator from our meta automatically
        if self.cache_key not in cache__evaluators:
            dataset = self.dataset
            cache__evaluators[self.cache_key] = ProxyEvaluator(
                dataset.get_user_train_dict(),
                dataset.get_user_test_dict(),
                dataset.get_user_test_neg_dict(),
                metric=self.conf["metric"],
                group_view=self.conf["group_view"],
                top_k=self.conf["topk"],
                batch_size=self.conf["test_batch_size"],
                num_thread=self.conf["num_thread"])
        return cache__evaluators[self.cache_key]

    @property
    def logger(self) -> Logger:
        if self.logger_name not in cache__loggers:
            cache__loggers[self.logger_name] = Logger(self.logger_name)

        return cache__loggers[self.logger_name]

    def build_graph(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def predict(self, user_ids, items):
        raise NotImplementedError

    @property
    def sess(self) -> tf.Session:
        assert self.cache_key in cache__sessions
        return cache__sessions[self.cache_key]

    def plug_tf_session(self, sess: tf.Session):
        """ We don't want to serialize connection data like session,
        put them in memory instead. """
        cache__sessions[self.cache_key] = sess


class SeqAbstractRecommender(AbstractRecommender, ABC):
    def __init__(self, conf):
        if self.dataset.time_matrix is None:
            raise ValueError("Dataset does not contant time infomation!")
        super(SeqAbstractRecommender, self).__init__(conf)

    @property
    def train_dict(self) -> Dict[int, List[int]]:
        return csr_to_user_dict_bytime(self.dataset.time_matrix,
                                       self.dataset.train_matrix)

    @property
    def user_pos_train(self):
        return self.train_dict  # just an alias

    @property
    def num_users(self) -> int:
        return self.dataset.num_users

    @property
    def num_items(self) -> int:
        return self.dataset.num_items


class SocialAbstractRecommender(AbstractRecommender, ABC):
    def __init__(self, conf):
        super(SocialAbstractRecommender, self).__init__(conf)
        social_users = pd.read_csv(conf["social_file"],
                                   sep=conf["data.convert.separator"],
                                   header=None, names=["user", "friend"])
        users_key = np.array(list(self.dataset.userids.keys()))
        index = np.in1d(social_users["user"], users_key)
        social_users = social_users[index]

        index = np.in1d(social_users["friend"], users_key)
        social_users = social_users[index]

        user = social_users["user"]
        user_id = [self.dataset.userids[u] for u in user]
        friend = social_users["friend"]
        friend_id = [self.dataset.userids[u] for u in friend]
        num_users, num_items = self.dataset.train_matrix.shape
        self.social_matrix = sp.csr_matrix(
            ([1] * len(user_id), (user_id, friend_id)),
            shape=(num_users, num_users))


__all__ = [
    "SeqAbstractRecommender",
    "SocialAbstractRecommender",
    "AbstractRecommender",
]
