import copy
import logging
import os

import math
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from gym.utils import seeding

import constants
from gym.spaces import Discrete, Box, MultiDiscrete
import ServEnv_base
import definitions as defs
from sys import maxsize
from ServEnv_base import Worker
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
# from keras.optimizers import Adam, RMSprop
from kubernetes import client, config, watch
from datetime import datetime
import time
import json
import threading
from collections import deque
from queue import Queue
import xlwt
from tensorflow.python.keras.layers import LSTM
from xlwt import Workbook
from tensorflow.keras.callbacks import TensorBoard
# import keras.backend.tensorflow_backend as backend
import tensorflow as tf
import subprocess

MODEL_NAME = "Serverless_Scaling"
# PATH = 'D:/WL generation/Third_work/Simulator/training_agent'
PATH = ''
mode = 'normal'
FN_TYPE = ""

action_cpu = [-920, -690, -460, -230, -115, 0, 115, 230, 460, 690, 920]

action_mem = [-375, -300, -225, -150, -75, 0, 75, 150, 225, 300, 375]

action_util = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]


class ActorCriticModel(Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.total_action_size = action_size[0] + action_size[1] + action_size[2]
        self.dense1 = tf.keras.layers.Dense(150, activation='relu')
        self.dense2 = tf.keras.layers.Dense(150, activation='relu')
        self.policy_logits_act = tf.keras.layers.Dense(self.total_action_size)

        self.dense3 = tf.keras.layers.Dense(150, activation='relu')
        self.dense4 = tf.keras.layers.Dense(150, activation='relu')
        self.values = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        # Forward pass
        x1 = self.dense1(inputs)
        x2 = self.dense2(x1)
        logits_act = self.policy_logits_act(x2)

        x3 = self.dense3(inputs)
        v1 = self.dense4(x3)
        values = self.values(v1)
        # return logits_cpu, logits_mem, logits_rep, values

        return logits_act, values


class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        # self.model = model
        self.model = model
        self._log_write_dir = self.log_dir
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        # pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class ThreadLogFilter(logging.Filter):
    """
    This filter only show log entries for specified thread name
    """

    def __init__(self, thread_name, *args, **kwargs):
        logging.Filter.__init__(self, *args, **kwargs)
        self.thread_name = thread_name

    def filter(self, record):
        return record.threadName == self.thread_name


def start_thread_logging(id, level, type):
    thread_name = threading.Thread.getName(threading.current_thread())
    log_file = "log/" + str(id) + "/" + level + "-" + type + "-logfile.log"
    log_handler = logging.FileHandler(log_file)
    log_handler.setLevel(logging.DEBUG)
    log_filter = ThreadLogFilter(thread_name)
    log_handler.addFilter(log_filter)
    logger = logging.getLogger()
    logger.addHandler(log_handler)

    return log_handler


class ServerlessEnv(gym.Env):
    """A serverless environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, level, fn_type, worker, episode):
        self.fn_type = fn_type
        self.level = level
        self.episode_no = episode
        self.worker_id = worker
        logging.basicConfig(
            filename="log/" + str(self.worker_id) + "/" + self.level + "-logfile.log",
            filemode="w",
            level=logging.DEBUG)

        # thread_log_handler = start_thread_logging(self.worker_id, self.level, self.fn_type)
        logging.info(
            "Serverlessenv to be initialized for level: {} worker {} episode: {}".format(level, worker, episode))
        self.lr = 0.0001
        self.num_max_replicas = constants.max_num_replicas
        self.num_max_vms = constants.max_num_vms
        # State includes server metrics, pod metrics, individual function metrics
        self.observation_space = Box(low=np.array(np.zeros(148)),
                                     high=np.array([maxsize] * 148),
                                     dtype=np.float32)
        self.state_size = self.observation_space.shape[0]
        self.action_space = MultiDiscrete([len(action_cpu), len(action_mem), len(action_util)])
        self.action_size = [len(action_cpu), len(action_mem), len(action_util)]
        self.state = np.zeros(148)
        self.opt = tf.optimizers.Adam(self.lr)
        self.done = False
        self.action = 0
        self.act_type = ""
        self.reward = 0
        self.episode_success = False
        self.current_count = 1
        self.info = {}
        self.average_task_duration = []
        self.ave_task_time = 0
        self.episode_cost = 0
        self.simulation_running = True

        if self.level == "master":
            self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
            self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

        else:
            self.worker = ServEnv_base.Worker(self.worker_id, self.episode_no)
            self.local_model = ActorCriticModel(self.state_size, self.action_size)  # local network
            self.local_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
            self.clock = self.worker.clock
            self.save_dir = "model/{}-{}".format(self.level, self.worker_id)

        self.tensorboard = ModifiedTensorBoard(MODEL_NAME, log_dir="logs/{}-{}-{}".format(self.level, self.worker_id,
                                                                                          int(time.time())))

    def action_spec(self):
        return self.action_space

    def observation_spec(self):
        return self.observation_space

    def reset(self):
        self.episode_no += 1
        self.worker = ServEnv_base.Worker(self.worker_id, self.episode_no)
        self.simulation_running = True
        self.done = False
        self.reward = 0

        self.clock = self.worker.clock
        self.episode_success = False
        self.current_count = 1
        self.info = {}

    def graphs(self, write_graph):
        total_req_count = 0
        failed_count = 0
        fn_failure_rate = 0
        req_info = {}
        fn_latency = 0

        for re_type in self.worker.sorted_request_history_per_window:
            for req in self.worker.sorted_request_history_per_window[re_type]:
                total_req_count += 1
                if req.status != "Dropped":
                    if req.type in req_info:
                        # self.logging.info(
                        #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                        #                                                            req.finish_time))
                        req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] += 1
                    else:
                        req_info[req.type] = {}
                        # self.logging.info(
                        #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                        #                                                            req.finish_time))
                        req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] = 1
                else:
                    failed_count += 1

        for req_type, req_data in req_info.items():
            # self.logging.info(
            #     "FN type: {}, Total exec time: {}, Req count: {}, MIPS for type: {}".format(req_type, req_data[
            #         'execution_time'], req_data['req_count'], ServEnv_base.fn_features[str(
            #         req_type) + "_req_MIPS"]))
            fn_latency += (req_data['execution_time'] / req_data['req_count']) / float(
                self.worker.fn_features[str(req_type) + "_req_exec_time"])

        if len(req_info) != 0:
            self.worker.Episodic_latency = fn_latency / len(req_info)
        logging.info("Worker: {} CLOCK: {} Overall latency: {}".format(self.worker_id, self.worker.clock, fn_latency))
        vm_up_time_cost, vm_up_time = self.calc_total_vm_up_time_cost()
        if total_req_count != 0:
            self.worker.Episodic_failure_rate = failed_count / total_req_count
        logging.info(
            "CLOCK: {} Worker: {} Cum vm_up_time_cost at episode end: {}".format(self.worker.clock, self.worker_id,
                                                                                 vm_up_time_cost))
        logging.info(
            "CLOCK: {} Worker: {} fn_failure_rate: {}".format(self.worker.clock, self.worker_id, fn_failure_rate))

        if write_graph:
            self.tensorboard.update_stats(Episodic_reward=self.worker.episodic_reward)
            self.tensorboard.update_stats(Function_Latency_step_based=self.worker.function_latency)
            self.tensorboard.update_stats(Function_failure_rate_total_step_based=self.worker.fn_failures)
            self.tensorboard.update_stats(Total_VM_COST_DIFF=self.worker.total_vm_cost_diff)
            self.tensorboard.update_stats(Vertical_cpu_action_total=self.worker.ver_cpu_action_total)
            self.tensorboard.update_stats(Vertical_mem_action_total=self.worker.ver_mem_action_total)
            self.tensorboard.update_stats(Horizontal_action_total=self.worker.hor_action_total)
            self.tensorboard.update_stats(Episodic_latency=self.worker.Episodic_latency)
            self.tensorboard.update_stats(Episodic_failure_rate=self.worker.Episodic_failure_rate)
            self.tensorboard.update_stats(Episodic_vm_cost=vm_up_time_cost)
            self.tensorboard.update_stats(Episodic_vm_uptime=vm_up_time)
            self.tensorboard.update_stats(loss=self.worker.episodic_loss)

    def graphs_test(self, write_graph):
        total_req_count = 0
        failed_count = 0
        fn_failure_rate = 0
        req_info = {}
        fn_latency = 0

        # with ServEnv_base.history_lock:
        for re_type in self.worker.sorted_request_history_per_window:
            for req in self.worker.sorted_request_history_per_window[re_type]:
                total_req_count += 1
                if req.status != "Dropped":
                    if req.type in req_info:
                        # self.logging.info(
                        #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                        #                                                            req.finish_time))
                        req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] += 1
                    else:
                        req_info[req.type] = {}
                        # self.logging.info(
                        #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                        #                                                            req.finish_time))
                        req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                        req_info[req.type]['req_count'] = 1
                else:
                    failed_count += 1

        for req_type, req_data in req_info.items():
            # self.logging.info(
            #     "FN type: {}, Total exec time: {}, Req count: {}, MIPS for type: {}".format(req_type, req_data[
            #         'execution_time'], req_data['req_count'], ServEnv_base.fn_features[str(
            #         req_type) + "_req_MIPS"]))
            fn_latency += (req_data['execution_time'] / req_data['req_count']) / float(
                self.worker.fn_features[str(req_type) + "_req_exec_time"])

        if len(req_info) != 0:
            self.worker.Episodic_latency = fn_latency / len(req_info)
        logging.info("Worker: {} CLOCK: {} Overall latency: {}".format(self.worker_id, self.worker.clock, fn_latency))
        vm_up_time_cost, vm_up_time = self.calc_total_vm_up_time_cost()
        if total_req_count != 0:
            self.worker.Episodic_failure_rate = failed_count / total_req_count
        logging.info(
            "CLOCK: {} Worker: {} Cum vm_up_time_cost at episode end: {}".format(self.worker.clock, self.worker_id,
                                                                                 vm_up_time_cost))
        logging.info(
            "CLOCK: {} Worker: {} fn_failure_rate: {}".format(self.worker.clock, self.worker_id, fn_failure_rate))

        if write_graph:
            self.tensorboard.update_stats(Episodic_reward=self.worker.episodic_reward)
            self.tensorboard.update_stats(Function_Latency_step_based=self.worker.function_latency)
            self.tensorboard.update_stats(Function_failure_rate_total_step_based=self.worker.fn_failures)
            self.tensorboard.update_stats(Total_VM_COST_DIFF=self.worker.total_vm_cost_diff)
            self.tensorboard.update_stats(Vertical_cpu_action_total=self.worker.ver_cpu_action_total)
            self.tensorboard.update_stats(Vertical_mem_action_total=self.worker.ver_mem_action_total)
            self.tensorboard.update_stats(Horizontal_action_total=self.worker.hor_action_total)
            self.tensorboard.update_stats(Episodic_latency=self.worker.Episodic_latency)
            self.tensorboard.update_stats(Episodic_failure_rate=self.worker.Episodic_failure_rate)
            self.tensorboard.update_stats(Episodic_vm_cost=vm_up_time_cost)
            self.tensorboard.update_stats(Episodic_vm_uptime=vm_up_time)
            self.tensorboard.update_stats(loss=self.worker.episodic_loss)

    def filtered_unavail_action_list(self, fn_type):
        global action_cpu
        global action_mem
        global action_util
        unavail_action_list_cpu = []
        unavail_action_list_mem = []
        # unavail_action_list_rep = []
        unavail_action_list_util = []
        for vm in self.worker.vms:
            cpu_added = 0
            pod_count = 0
            fntype_pod_cpu_req_total = 0
            fntype_pod_mem_req_total = 0
            pod_cpu_util_min = 0
            pod_ram_util_min = 0
            if fn_type in vm.running_pod_list:
                for pod in vm.running_pod_list[fn_type]:
                    pod_count += 1
                    fntype_pod_cpu_req_total += pod.cpu_req
                    fntype_pod_mem_req_total += pod.ram_req
                    if pod.cpu_util > pod_cpu_util_min:
                        pod_cpu_util_min = pod.cpu_util
                    if pod.ram_util > pod_ram_util_min:
                        pod_ram_util_min = pod.ram_util
            if pod_count * self.worker.fn_features[str(fn_type) + "_pod_cpu_req"] > vm.cpu_allocated:
                print("debug at unav action list")
            for a in range(len(action_cpu)):
                if a not in unavail_action_list_cpu:
                    if action_cpu[a] * pod_count + vm.cpu_allocated > vm.cpu or self.worker.fn_features[
                        str(fn_type) + "_pod_cpu_req"] + action_cpu[a] > constants.max_pod_cpu_req or \
                            action_cpu[a] * pod_count + vm.cpu_allocated < 0 or self.worker.fn_features[
                        str(fn_type) + "_pod_cpu_req"] + action_cpu[a] < constants.min_pod_cpu_req or \
                            self.worker.fn_features[
                                str(fn_type) + "_pod_cpu_req"] + action_cpu[a] < pod_cpu_util_min:
                        if a == 0:
                            print(str(self.worker.fn_features[
                                          str(fn_type) + "_pod_cpu_req"] + action_cpu[a]))
                            print("Thus cpu action {} added to list".format(a))
                        unavail_action_list_cpu.append(a)
                        # logging.info("Worker: {} Adding cpu Action: {} to unavail list".format(self.worker_id, a))

            for b in range(len(action_mem)):
                if b not in unavail_action_list_mem:
                    if action_mem[b] * pod_count + vm.mem_allocated > vm.ram or action_mem[
                        b] * pod_count + vm.mem_allocated < 0 or self.worker.fn_features[
                        str(fn_type) + "_pod_ram_req"] + math.floor(
                        action_mem[b]) > constants.max_total_pod_mem or self.worker.fn_features[
                        str(fn_type) + "_pod_ram_req"] + math.floor(
                        action_mem[b]) < constants.min_pod_mem_req or self.worker.fn_features[
                        str(fn_type) + "_pod_ram_req"] + math.floor(
                        action_mem[b]) < pod_ram_util_min:
                        unavail_action_list_mem.append(b)
                        # logging.info("Worker: {} Adding mem Action: {} to unavail list".format(self.worker_id, b))

            for c in range(len(action_util)):
                if c not in unavail_action_list_util:
                    if self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"] + action_util[
                        c] > constants.pod_scale_cpu_util_high or \
                            self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"] + action_util[
                        c] < constants.pod_scale_cpu_util_low:
                        unavail_action_list_util.append(c)

        return unavail_action_list_cpu, unavail_action_list_mem, unavail_action_list_util

    def act(self, step, w_id, fn):
        print("Action selection")
        discarded_action_list_c, discarded_action_list_m, discarded_action_list_r = self.filtered_unavail_action_list(
            fn)

        logits_action, v = self.local_model(
            tf.convert_to_tensor(self.state[None, :], dtype=tf.float32))

        logits_c, logits_m, logits_r = tf.split(logits_action,
                                                [self.action_size[0], self.action_size[1], self.action_size[2]], 2)

        probs_c = tf.nn.softmax(logits_c)
        probs_m = tf.nn.softmax(logits_m)
        probs_r = tf.nn.softmax(logits_r)

        probs_c = probs_c.numpy()[0].flatten()
        probs_m = probs_m.numpy()[0].flatten()
        probs_r = probs_r.numpy()[0].flatten()

        # set the probabilities of all illegal moves to zero and renormalise the output vector before we choose our move.
        for a in discarded_action_list_c:
            probs_c[a] = 0
        for a in discarded_action_list_m:
            probs_m[a] = 0
        for a in discarded_action_list_r:
            probs_r[a] = 0

        # print("prob_c after " + str(probs_c))
        # print("prob_m after " + str(probs_m))
        # print("prob_r after " + str(probs_r))

        probs_c /= np.array(probs_c).sum()
        probs_m /= np.array(probs_m).sum()
        probs_r /= np.array(probs_r).sum()

        # print("Prob"+str(probs_c.numpy()[0]))
        action_c = np.random.choice(self.action_size[0], p=probs_c)
        # print("Discarded actions for thread %s: %s" % (str(step-1), str(discarded_action_list)))

        # print("Prob"+str(probs_c.numpy()[0]))
        action_m = np.random.choice(self.action_size[1], p=probs_m)

        # print("Prob"+str(probs_c.numpy()[0]))
        action_r = np.random.choice(self.action_size[2], p=probs_r)

        action_t = "Network"
        sel_action = [action_c, action_m, action_r]
        print("worker: %s Selected action for step %s: %s" % (str(self.worker_id), step, str(sel_action)))
        self.worker.ver_cpu_action_total += action_cpu[sel_action[0]]
        self.worker.ver_mem_action_total += action_mem[sel_action[1]]
        self.worker.hor_action_total += action_util[sel_action[2]]

        vm_up_time_cost, vm_up_time = self.calc_total_vm_up_time_cost()
        self.worker.vm_up_time_cost_prev = vm_up_time_cost / constants.max_step_vmcost

        self.execute_scaling(sel_action, step, w_id, fn)
        self.worker.sorted_events.append(
            defs.EVENT(self.worker.clock + constants.reward_interval, constants.calc_reward,
                       step))
        self.worker.pod_scaler()

        return sel_action, action_t, self.done

    def act_test(self, model, step, w_id, fn):

        discarded_action_list_c, discarded_action_list_m, discarded_action_list_r = self.filtered_unavail_action_list(
            fn)
        logits_action, v = model(
            tf.convert_to_tensor(self.state[None, :], dtype=tf.float32))
        logits_c, logits_m, logits_r = tf.split(logits_action,
                                                [self.action_size[0], self.action_size[1], self.action_size[2]], 2)

        probs_c = tf.nn.softmax(logits_c).numpy()[0][0]
        probs_m = tf.nn.softmax(logits_m).numpy()[0][0]
        probs_r = tf.nn.softmax(logits_r).numpy()[0][0]

        # print("Action Probabilities: %s and %s and %s " % (str(probs_c), str(probs_m), str(probs_r)))

        for a in discarded_action_list_c:
            probs_c[a] = 0
        for a in discarded_action_list_m:
            probs_m[a] = 0
        for a in discarded_action_list_r:
            probs_r[a] = 0

        if mode == 'normal':
            action_c = np.argmax(probs_c)
            action_m = np.argmax(probs_m)
            action_r = np.argmax(probs_r)
            action_t = "Network"
        else:
            action_c = 5
            action_m = 5
            action_r = 5
            action_t = "comparison"

        sel_action = [action_c, action_m, action_r]
        print("worker: %s Selected action for step %s: %s" % (str(self.worker_id), step, str(sel_action)))
        self.worker.ver_cpu_action_total += action_cpu[sel_action[0]]
        self.worker.ver_mem_action_total += action_mem[sel_action[1]]
        self.worker.hor_action_total += action_util[sel_action[2]]

        vm_up_time_cost, vm_up_time = self.calc_total_vm_up_time_cost()
        self.worker.vm_up_time_cost_prev = vm_up_time_cost / constants.max_step_vmcost

        self.execute_scaling(sel_action, step, w_id, fn)
        self.worker.sorted_events.append(
            defs.EVENT(self.worker.clock + constants.reward_interval, constants.calc_reward,
                       step))
        self.worker.pod_scaler()
        return sel_action, action_t, self.done

    def get_state(self, step_c, w_id, fn):
        self.state = self.worker.gen_serv_env_state(fn)
        self.state = np.reshape(self.state, [1, self.state_size])
        return self.state, self.worker.clock

    def select_action_test(self, step_c, ep):
        state = self.worker.gen_serv_env_state(self.fn_type)
        state = np.reshape(state, [1, self.state_size])
        self.action, self.act_type, done = self.act_test(step_c, state, self.fn_type, ep)
        return state, self.act_type, self.action, self.worker.clock

    def execute_scaling(self, action, step_c, idx, fn_type):
        self.worker.fn_features[str(fn_type) + "_scale_cpu_threshold"] += action_util[action[2]]
        new_cpu = self.worker.fn_features[str(fn_type) + "_pod_cpu_req"] + action_cpu[action[0]]
        self.worker.fn_features[str(fn_type) + "_pod_cpu_req"] = new_cpu
        new_mem = self.worker.fn_features[str(fn_type) + "_pod_ram_req"] + action_mem[action[1]]
        self.worker.fn_features[str(fn_type) + "_pod_ram_req"] = new_mem

        if fn_type in self.worker.pod_object_dict_by_type:
            for pod in self.worker.pod_object_dict_by_type[fn_type]:
                pod.allocated_vm.cpu_allocated += new_cpu - pod.cpu_req
                pod.cpu_req = new_cpu
                pod.allocated_vm.mem_allocated += new_mem - pod.ram_req
                pod.ram_req = new_mem

        # Allocate the current reward to the previous step
        tuple0 = (self.current_count - 1, self.reward)
        self.info["step_reward"] = tuple0

        if step_c == constants.max_steps:
            self.done = True
            self.episode_success = True

        return True

    def create_scaling_events(self):
        scaling_time = constants.scaling_start
        while scaling_time < constants.WL_duration:
            # create events for each scaling step
            self.worker.sorted_events.append(defs.EVENT(scaling_time, constants.invoke_step_scaling, None))
            scaling_time += constants.step_interval

        self.worker.sorted_events = sorted(self.worker.sorted_events, key=self.worker.sorter_events)
        # print(ServEnv_base.sorted_events)

    def create_scaling_events_test(self):
        scaling_time = constants.scaling_start
        while scaling_time < constants.WL_duration:
            # create events for each scaling step
            # self.logging.info("Scaling event created at: {}".format(scaling_time))
            self.worker.sorted_events.append(defs.EVENT(scaling_time, constants.invoke_step_scaling, None))
            scaling_time += constants.step_interval

        self.worker.sorted_events = sorted(self.worker.sorted_events, key=self.worker.sorter_events)

    def execute_events(self):
        while self.worker.sorted_events:
            ev = self.worker.sorted_events.pop(0)
            prev_clock = self.worker.clock
            self.worker.clock = float(ev.received_time)
            ev_name = ev.event_name
            # print("Event is " + ev_name)
            if ev_name == "SCHEDULE_REQ":
                # set the current arrival rate for a fn
                # self.worker.fn_request_rate = ev.entity_object.arrival_rate
                self.worker.fn_request_rate[ev.entity_object.type] = ev.entity_object.arrival_rate
                # second parameter specifies if a request is a new one or a rescheduling request
                self.worker.req_scheduler(ev.entity_object, False)
            elif ev_name == "RE_SCHEDULE_REQ":
                self.worker.req_scheduler(ev.entity_object, True)
            elif ev_name == "REQ_COMPLETE":
                self.worker.req_completion(ev.entity_object)
            elif ev_name == "CREATE_POD":
                self.worker.pod_creator(ev.entity_object.type)
            elif ev_name == "SCALE_POD":
                self.worker.pod_creator(ev.entity_object)
            elif ev_name == "SCHEDULE_POD":
                self.worker.pod_scheduler(ev.entity_object)
            elif ev_name == "STEP_SCALING":
                if self.worker.fn_iterator < (len(self.worker.fn_types) - 1):
                    self.worker.fn_iterator += 1
                else:
                    self.worker.fn_iterator = 0
                return True
            elif ev_name == "REWARD":
                return True

        self.simulation_running = False
        return True

    def calc_total_vm_up_time_cost(self):
        vm_cost = 0
        vm_time = 0
        for vm in self.worker.vm_up_time_dict:
            if self.worker.vm_up_time_dict[vm]['status'] == "ON":
                self.worker.vm_up_time_dict[vm]['total_time'] += self.worker.clock - self.worker.vm_up_time_dict[vm][
                    'time_now']
                self.worker.vm_up_time_dict[vm]['time_now'] = self.worker.clock
            vm_time += self.worker.vm_up_time_dict[vm]['total_time']
            vm_cost += vm.price * self.worker.vm_up_time_dict[vm]['total_time']
        return vm_cost, vm_time

    def calculate_reward(self, step_c):
        req_info = {}
        fn_latency = 0
        fn_failure_rate = 0
        failed_count = 0
        total_req_count_within_window = 0

        logging.info(
            "CLOCK: {} Worker: {} Calculating reward for previous step".format(self.worker.clock, self.worker_id))

        for re_type in self.worker.sorted_request_history_per_window:
            for req in reversed(self.worker.sorted_request_history_per_window[re_type]):
                if req.finish_time > self.worker.clock - constants.reward_window_size:
                    # self.logging.info(
                    #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time, req.finish_time))
                    total_req_count_within_window += 1
                    if req.status != "Dropped":
                        if req.type in req_info:
                            # self.logging.info("Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                            #                                                                     req.finish_time))
                            req_info[req.type]['execution_time'] += req.finish_time - req.arrival_time
                            req_info[req.type]['req_count'] += 1
                        else:
                            req_info[req.type] = {}
                            # self.logging.info("Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                            #                                                                     req.finish_time))
                            req_info[req.type]['execution_time'] = req.finish_time - req.arrival_time
                            req_info[req.type]['req_count'] = 1
                    else:
                        failed_count += 1
                else:
                    break

        for req_type, req_data in req_info.items():
            # self.logging.info("FN type: {}, Total exec time: {}, Req count: {}, MIPS for type: {}".format(req_type, req_data[
            #     'execution_time'], req_data['req_count'], ServEnv_base.fn_features[str(req_type) + "_req_MIPS"]))
            fn_latency += (req_data['execution_time'] / req_data['req_count']) / float(
                self.worker.fn_features[str(req_type) + "_req_exec_time"])

        if len(req_info) != 0:
            fn_latency = fn_latency / len(req_info) / constants.max_step_latency_perfn
        logging.info("CLOCK: {} worker: {}  step latency: {}".format(self.worker.clock, self.worker_id, fn_latency))
        vm_up_time_cost, vm_up_time = self.calc_total_vm_up_time_cost()
        if total_req_count_within_window != 0:
            fn_failure_rate = failed_count / total_req_count_within_window
        logging.info(
            "CLOCK: {} Worker: {} Cum vm_up_time_cost: {}".format(self.worker.clock, self.worker_id, vm_up_time_cost))
        logging.info("CLOCK: {} fn_failure_rate: {}".format(self.worker.clock, self.worker_id, fn_failure_rate))

        self.worker.fn_failures += fn_failure_rate
        self.worker.function_latency += fn_latency

        vm_cost_step = vm_up_time_cost / constants.max_step_vmcost - self.worker.vm_up_time_cost_prev
        self.worker.total_vm_cost_diff += vm_cost_step
        logging.info("CLOCK: {} Worker: {} vm_up_time_cost diff for step: {}".format(self.worker.clock, self.worker_id,
                                                                                     vm_cost_step))

        x = 1
        y = 1 - x
        reward = - (vm_cost_step * x + (fn_latency + fn_failure_rate) * y)
        logging.info("CLOCK: {} Worker: {} Step reward: {}".format(self.worker.clock, self.worker_id, reward))

        self.worker.episodic_reward += reward

        return reward

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        # s = "state: {}  reward: {}  info: {}"
        # print(s.format(self.state, self.reward, self.info))

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
