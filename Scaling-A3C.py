import logging
import sys
import warnings
from queue import Queue
import random

import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
# import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow import keras
from xlwt import Workbook

from env.ServerlessEnv import ServerlessEnv
from env.ServerlessEnv import ActorCriticModel
import constants
# from ServEnv_base
import ServEnv_base
from ServEnv_base import Worker

tf.disable_v2_behavior()
tf.enable_eager_execution()
warnings.filterwarnings('ignore')

MODEL_NAME = "Serverless_Scheduling"


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        if reward != 1000000:
            self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class WorkerThread(threading.Thread):
    global_episode = 0
    best_score = -sys.maxsize
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 opt, train_model,
                 idx):
        super(WorkerThread, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.current_state = np.zeros(148)
        self.action = np.zeros(3)
        self.reward_e = 0
        self.done = False
        self.todo = False
        self.global_model = global_model
        self.opt = opt
        self.fn_type = "fn" + str(idx)
        self.episode_no = 0
        self.worker_idx = idx
        self.episodic_reward = 0
        self.ep_loss = 0.0
        self.gamma = 0.6
        self.update_freq = 30
        self.total_step = 1
        self.checkpoint_freq = 19
        self.train = train_model
        self.env_worker = ServerlessEnv("worker", self.fn_type, self.worker_idx, self.episode_no)
        self.save_dir = self.env_worker.save_dir
        if not self.train:
            # Give the path of the saved model
            self.save_dir = ""
            model_path = os.path.join(self.save_dir)
            print('Loading model from: {}'.format(model_path))
            self.global_model.load_weights(model_path)

    def compute_loss(self,
                     done,
                     cur_state,
                     memory):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = \
                self.env_worker.local_model(tf.convert_to_tensor(cur_state[None, :], dtype=tf.float32))[
                    -1].numpy()[0]

        new_action_c = []
        new_action_m = []
        new_action_r = []

        if len(memory.states) != len(memory.rewards):
            new_actions = memory.actions[:-1]
            new_state = memory.states[:-1]
        else:
            new_actions = memory.actions
            new_state = memory.states

        for act in new_actions:
            new_action_c.append(act[0])
            new_action_m.append(act[1])
            new_action_r.append(act[2])

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits_action, values = self.env_worker.local_model(
            tf.convert_to_tensor(np.vstack(new_state),
                                 dtype=tf.float32))

        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2
        logits_c, logits_m, logits_r = tf.split(logits_action,
                                                [self.action_size[0], self.action_size[1], self.action_size[2]], 1)

        # Calculate our policy loss
        actions_one_hot_c = tf.one_hot(new_action_c, self.action_size[0], dtype=tf.float32)
        actions_one_hot_m = tf.one_hot(new_action_m, self.action_size[1], dtype=tf.float32)
        actions_one_hot_r = tf.one_hot(new_action_r, self.action_size[2], dtype=tf.float32)

        policy_c = tf.nn.softmax(logits_c)
        policy_m = tf.nn.softmax(logits_m)
        policy_r = tf.nn.softmax(logits_r)
        entropy_c = tf.reduce_sum(policy_c * tf.math.log(policy_c + 1e-20), axis=1)
        entropy_m = tf.reduce_sum(policy_m * tf.math.log(policy_m + 1e-20), axis=1)
        entropy_r = tf.reduce_sum(policy_r * tf.math.log(policy_r + 1e-20), axis=1)

        policy_loss_c = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot_c,
                                                                logits=logits_c)
        policy_loss_m = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot_m,
                                                                logits=logits_m)
        policy_loss_r = tf.nn.softmax_cross_entropy_with_logits(labels=actions_one_hot_r,
                                                                logits=logits_r)

        policy_loss_c *= tf.stop_gradient(advantage)
        policy_loss_c -= 0.01 * entropy_c

        policy_loss_m *= tf.stop_gradient(advantage)
        policy_loss_m -= 0.01 * entropy_m

        policy_loss_r *= tf.stop_gradient(advantage)
        policy_loss_r -= 0.01 * entropy_r

        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss_c + policy_loss_m + policy_loss_r))
        return total_loss

    def run_episode_test(self, sheet, wb):
        self.episodic_reward = 0
        self.env_worker.create_scaling_events()
        step_count = 1
        episode_steps_sheet_row_counter = 1

        while self.env_worker.simulation_running:
            if self.env_worker.execute_events():
                if not self.env_worker.simulation_running:
                    continue
                if not self.todo:
                    self.fn_type = "fn" + str(self.env_worker.worker.fn_types[self.env_worker.worker.fn_iterator])
                    self.current_state, clock = self.env_worker.get_state(step_count, self.worker_idx, self.fn_type)
                    self.action, act_t, self.done = self.env_worker.act_test(self.global_model, step_count,
                                                                             self.worker_idx, self.fn_type)
                    logging.info(
                        "CLOCK: {} worker: {} Action selected and executed: {}".format(self.env_worker.worker.clock,
                                                                                       self.worker_idx,
                                                                                       self.action))
                    self.todo = True
                else:
                    self.reward_e = self.env_worker.calculate_reward(step_count)
                    # reward_e, done, info = self.env_worker.take_step(act_t, action, step_count, self.worker_idx)
                    self.episodic_reward += self.reward_e
                    logging.info(
                        "CLOCK: {} worker: {} step reward: {} done: {} total steps: {}".format(
                            self.env_worker.worker.clock, self.worker_idx, self.reward_e, self.done, self.total_step))
                    self.todo = False

                    sheet.write(episode_steps_sheet_row_counter, 0, self.env_worker.worker.clock)
                    sheet.write(episode_steps_sheet_row_counter, 1, self.episode_no)
                    sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                    sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(self.current_state))
                    sheet.write(episode_steps_sheet_row_counter, 4, str(self.action))
                    sheet.write(episode_steps_sheet_row_counter, 5, self.reward_e)
                    sheet.write(episode_steps_sheet_row_counter, 7, self.done)

                    wb.save(
                        "drl_steps/" + str(self.worker_idx) + "/" + "DRL_Steps_Episode" + str(self.episode_no) + ".xls")
                    if self.done:
                        WorkerThread.global_episode += 1
                        write_log = True
                        self.env_worker.graphs_test(write_log)

                if not self.todo:
                    step_count += 1
                    self.total_step += 1
                    episode_steps_sheet_row_counter += 1
        return True

    def run_episode(self, memory, sheet, wb):
        self.episodic_reward = 0
        self.ep_loss = 0
        self.env_worker.create_scaling_events()
        step_count = 1
        episode_steps_sheet_row_counter = 1

        while self.env_worker.simulation_running:
            if self.env_worker.execute_events():
                if not self.env_worker.simulation_running:
                    continue
                if not self.todo:
                    print("CLOCK: {} worker: {} Starting step {}:".format(self.env_worker.worker.clock, self.worker_idx,
                                                                          step_count))
                    self.fn_type = "fn" + str(self.env_worker.worker.fn_types[self.env_worker.worker.fn_iterator])
                    self.current_state, clock = self.env_worker.get_state(step_count, self.worker_idx, self.fn_type)

                    self.action, act_t, self.done = self.env_worker.act(step_count, self.worker_idx, self.fn_type)
                    logging.info(
                        "CLOCK: {} worker: {} Action selected and executed: {}".format(self.env_worker.worker.clock,
                                                                                       self.worker_idx,
                                                                                       self.action))
                    self.todo = True
                else:
                    self.reward_e = self.env_worker.calculate_reward(step_count)
                    self.episodic_reward += self.reward_e
                    logging.info(
                        "CLOCK: {} worker: {} step reward: {} done: {} total steps: {}".format(
                            self.env_worker.worker.clock, self.worker_idx, self.reward_e, self.done, self.total_step))
                    self.todo = False
                    memory.store(self.current_state, self.action, self.reward_e)

                    sheet.write(episode_steps_sheet_row_counter, 0, self.env_worker.worker.clock)
                    sheet.write(episode_steps_sheet_row_counter, 1, self.episode_no)
                    sheet.write(episode_steps_sheet_row_counter, 2, step_count)
                    sheet.write(episode_steps_sheet_row_counter, 3, np.array_str(self.current_state))
                    sheet.write(episode_steps_sheet_row_counter, 4, str(self.action))
                    sheet.write(episode_steps_sheet_row_counter, 5, self.reward_e)
                    sheet.write(episode_steps_sheet_row_counter, 7, self.done)

                    wb.save(
                        "drl_steps/" + str(self.worker_idx) + "/" + "DRL_Steps_Episode" + str(self.episode_no) + ".xls")

                    if ((self.total_step % self.update_freq == 0) or self.done) and len(memory.states) > 1:
                        # Calculate gradient wrt to local model. We do so by tracking the
                        # variables involved in computing the loss by using tf.GradientTape
                        with tf.GradientTape() as tape:
                            total_loss = self.compute_loss(self.done,
                                                           self.current_state,
                                                           memory)
                        self.ep_loss += total_loss
                        # Calculate local gradients
                        grads = tape.gradient(total_loss, self.env_worker.local_model.trainable_weights)
                        # Push local gradients to global model
                        self.opt.apply_gradients(zip(grads,
                                                     self.global_model.trainable_weights))
                        # Update local model with new weights
                        self.env_worker.local_model.set_weights(self.global_model.get_weights())
                        logging.info(
                            "Updated global and local models local episode: {} Worker: {} episode score: {} total step count: {}".format(
                                self.episode_no,
                                self.worker_idx,
                                self.episodic_reward, self.total_step))

                        memory.clear()

                        if self.done:  # done and print information
                            # We must use a lock to save our model and to print to prevent data races.

                            with WorkerThread.save_lock:
                                if self.episodic_reward > WorkerThread.best_score or self.episode_no % self.checkpoint_freq == 0:
                                    logging.info(
                                        "clock: {} Local reward higher than global best or reached checkpoint at end of global episode: {} local Episode: {} Worker: {} best_score: {} episode score: {} step count: {}".format(
                                            self.env_worker.clock, WorkerThread.global_episode, self.episode_no,
                                            self.worker_idx, WorkerThread.best_score,
                                            self.episodic_reward, step_count))
                                    if not os.path.exists(self.save_dir):
                                        os.makedirs(self.save_dir)
                                    self.global_model.save_weights(
                                        self.save_dir + '/model_{}_{}.h5'.format(self.env_worker.clock, str(
                                            random.randint(1, 899999) + 100000)))
                                    # os.path.join(self.save_dir, 'model_{}.h5'.format(self.env_worker.clock)))
                                    if self.episodic_reward > WorkerThread.best_score:
                                        WorkerThread.best_score = self.episodic_reward
                                else:
                                    logging.info(
                                        "clock:{} Finishing Global Episode: {} local episode: {} Worker: {} best_score: {} episode score: {}".format(
                                            self.env_worker.clock, WorkerThread.global_episode, self.episode_no,
                                            self.worker_idx, WorkerThread.best_score,
                                            self.episodic_reward))
                                WorkerThread.global_episode += 1
                            write_log = True
                            self.env_worker.worker.episodic_loss = self.ep_loss
                            self.env_worker.graphs(write_log)

                if not self.todo:
                    step_count += 1
                    self.total_step += 1
                    episode_steps_sheet_row_counter += 1

        return True

    def run(self):
        mem = Memory()
        while WorkerThread.global_episode < constants.num_episodes:
            logging.info("Worker: {} Starting episode: {}".format(self.worker_idx, self.episode_no))
            wb_drl = Workbook()
            drl_steps = wb_drl.add_sheet('Episode_steps')
            drl_steps.write(0, 0, 'Time')
            drl_steps.write(0, 1, 'Episode')
            drl_steps.write(0, 2, 'Step')
            drl_steps.write(0, 3, 'State')
            drl_steps.write(0, 4, 'Action')
            drl_steps.write(0, 5, 'Reward')
            drl_steps.write(0, 6, 'Next State')
            drl_steps.write(0, 7, 'Done')

            ep_data = wb_drl.add_sheet('Episodes')
            ep_data.write(0, 0, 'Time')
            ep_data.write(0, 1, 'Episode')
            ep_data.write(0, 2, 'Ep_reward')

            if self.train:
                result = self.run_episode(mem, drl_steps, wb_drl)
            else:
                result = self.run_episode_test(drl_steps, wb_drl)
            try:
                ep_data.write(1, 0, self.env_worker.worker.clock)
                ep_data.write(1, 1, self.episode_no)
                ep_data.write(1, 2, self.episodic_reward)
                wb_drl.save(
                    "drl_steps/" + str(self.worker_idx) + "/" + "DRL_Steps_Episode" + str(self.episode_no) + ".xls")

            except Exception as inst:
                print(inst.args)

            self.episode_no += 1

            self.env_worker.reset()
        if self.train:
            with WorkerThread.save_lock:
                logging.info(
                    "clock: {} Saving model at end of training  at end of global episode: {} local Episode: {} Worker: {} best_score: {} episode score: {}".format(
                        self.env_worker.clock, WorkerThread.global_episode, self.episode_no,
                        self.worker_idx, WorkerThread.best_score,
                        self.episodic_reward))
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                self.global_model.save_weights(self.save_dir + '/model_{}_{}.h5'.format(self.env_worker.clock, str(
                    random.randint(1, 899999) + 100000)))
                # os.path.join(self.save_dir, 'model_{}.h5'.format(self.env_worker.clock)))
        logging.info("Stopping Worker: {} because global episode is: {} local episode: {}".format(self.worker_idx,
                                                                                                  WorkerThread.global_episode,
                                                                                                  self.episode_no))


def train():
    env_master = ServerlessEnv("master", "fn0", 0, 0)
    env_master.tensorboard.step = 0

    workers = [WorkerThread(env_master.state_size,
                            env_master.action_size,
                            env_master.global_model,
                            env_master.opt, True,
                            i) for i in range(3)]

    for i, worker in enumerate(workers):
        print("Starting worker {}".format(i))
        worker.start()

    [w.join() for w in workers]

    logging.info("All workers joined")


def test():
    env_master = ServerlessEnv("master", "fn0", 0, 0)
    env_master.tensorboard.step = 0

    workers = [WorkerThread(env_master.state_size,
                            env_master.action_size,
                            env_master.global_model,
                            env_master.opt, False,
                            i) for i in range(1)]
    for i, worker in enumerate(workers):
        print("Starting worker {}".format(i))
        worker.start()

    [w.join() for w in workers]


if __name__ == "__main__":
    train()
    # test()
