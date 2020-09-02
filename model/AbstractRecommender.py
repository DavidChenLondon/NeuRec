import os
import shutil
import time
from abc import ABC
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf

from NeuRec.data.dataset import Dataset
from NeuRec.evaluator import ProxyEvaluator
from NeuRec.util import Logger, Configurator
from NeuRec.util.tool import timer
from smk_recsys.utils.const import project_dir  # TODO fix it

# note: keep below caches in memory only
cache__loggers: Dict[str, Logger] = {}
cache__evaluators: Dict[str, ProxyEvaluator] = {}
cache__sessions: Dict[str, tf.Session] = {}


class EvaluationReport(object):
    FIELDS_EPOCH = ["epoch", "loss", "seconds"]

    def __init__(self, metrics: List[str]):
        assert isinstance(metrics, list), metrics
        self.metrics: List[str] = metrics
        self.data: Dict[str, list] = {f: [] for f in
                                      (self.FIELDS_EPOCH + self.metrics)}

    def record(self,
               epoch: int, loss: float, seconds: float,
               metric_values: Union[str, Dict[str, float]] = None):
        # TODO: currently we use existing str API, but we should refactor
        #  Evaluator to support metadata instead of printed string.
        self.data["epoch"].append(epoch)
        self.data["loss"].append(loss)
        self.data["seconds"].append(seconds)

        if metric_values is not None:
            if isinstance(metric_values, str):
                metric_values = dict(zip(
                    self.metrics, map(float, metric_values.split("\t"))))
            for k, v in metric_values.items():
                self.data[k].append(v)

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)


class LossRecorder(object):

    def __init__(self):
        self.loss_list: List[float] = []
        self.training_start_time = time.time()

    def add_loss(self, loss: float):
        self.loss_list.append(loss)

    @property
    def avg_loss(self) -> float:
        avg_loss = None
        if self.loss_list:
            avg_loss = sum(self.loss_list) / len(self.loss_list)
        return avg_loss

    @property
    def seconds(self) -> float:
        return time.time() - self.training_start_time


class AbstractRecommender(object):
    DEFAULT_RANDOM_STATE: int = 42

    def __init__(self, conf: Configurator):
        self.conf: Configurator = conf
        self.report: EvaluationReport = EvaluationReport(conf["metric"])
        self.verbose = conf.alg_arg.get("verbose", 5)

        # generate cache key
        self.param_str = "%s_%s" % (self.dataset_name, self.conf.params_str())
        model_name = self.conf["recommender"]
        timestamp = time.time()
        self.run_id = "%s_%.8f" % (self.param_str[:150], timestamp)

        self.random_state = conf.lib_arg.get("random_state",
                                             self.DEFAULT_RANDOM_STATE)
        self.cache_key = f"{self.param_str}_{model_name}"
        if self.random_state != self.DEFAULT_RANDOM_STATE:
            self.cache_key = f"{self.cache_key}_rs={self.random_state}"

        # generate logger name
        self.cache_dir = os.path.join(f"{project_dir}/cache/NeuRec/",
                                      self.dataset_name, model_name)
        self.log_dir = os.path.join(f"{project_dir}/log/NeuRec/",
                                    self.dataset_name, model_name)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.logger_name: str = os.path.join(self.log_dir, self.run_id + ".log")

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
    def tf_cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"tf_{self.cache_key}")

    @property
    def sess(self):
        if self.cache_key not in cache__sessions:
            # TODO looks multiple-CPUs option not works
            config = tf.ConfigProto(
                device_count={"CPU": 4},
                inter_op_parallelism_threads=2,
                intra_op_parallelism_threads=2,
                # log_device_placement=True,
                allow_soft_placement=True,
            )
            session = tf.Session(config=config)

            meta_path = f"{self.tf_cache_path}.meta"
            if os.path.exists(meta_path):
                saver = tf.train.import_meta_graph(meta_path)
                saver.restore(session, self.tf_cache_path)
                print(f"Restored session from tf_cache_path:"
                      f" {self.tf_cache_path}")
                self.logger.info(f"Restored session from tf_cache_path:"
                                 f" {self.tf_cache_path}")
            session.run(tf.initialize_all_variables())
            session.run(tf.global_variables_initializer())
            cache__sessions[self.cache_key] = session
        return cache__sessions[self.cache_key]

    def save_tf_model(self):
        saver = tf.train.Saver()
        # TODO add checkpoints
        print(f"saving to {self.tf_cache_path}")
        shutil.rmtree(self.tf_cache_path, ignore_errors=True)
        saver.save(self.sess, self.tf_cache_path, save_debug_info=True)
        self.logger.info(f"Saved session to tf_cache_path:"
                         f" {self.tf_cache_path}")

    def log_loss_and_evaluate(self, epoch: int, lr: LossRecorder):
        self.logger.info(
            f"[iter {epoch}: loss : {lr.avg_loss}, time: {lr.seconds}]")

        evaluate_result = None
        if epoch % int(self.verbose) == 0:
            evaluate_result = self.evaluate()
            self.logger.info("epoch %d:\t%s" % (epoch, evaluate_result))
        self.report.record(epoch, lr.avg_loss, lr.seconds, evaluate_result)

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    @property
    def num_users(self) -> int:
        return self.dataset.num_users

    @property
    def num_items(self) -> int:
        return self.dataset.num_items


class SeqAbstractRecommender(AbstractRecommender, ABC):
    def __init__(self, conf):
        super(SeqAbstractRecommender, self).__init__(conf)
        if self.dataset.time_matrix is None:
            raise ValueError("Dataset does not contant time infomation!")

    @property
    def train_dict(self) -> Dict[int, List[int]]:
        return self.dataset.train_dict

    @property
    def user_pos_train(self):
        return self.train_dict  # just an alias


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
