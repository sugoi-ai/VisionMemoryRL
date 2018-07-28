# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from utils.accum_trainer import AccumTrainer
from scene_loader import THORDiscreteEnvironment as Environment
from network import ActorCriticFFNetwork

from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import VERBOSE


class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               network_scope="network",
               scene_scope="scene",
               training_scene="scene",
               task_scope="task",
               checkpoint_scope="checkpoint"):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.network_scope = network_scope
    self.scene_scope = scene_scope
    self.training_scene = training_scene
    self.task_scope = task_scope
    self.checkpoint_scope = checkpoint_scope
    self.scopes = [network_scope, scene_scope, task_scope]

    self.local_network = ActorCriticFFNetwork(
                           action_size=ACTION_SIZE,
                           device=device,
                           network_scope=network_scope,
                           scene_scopes=[scene_scope])

    self.local_network.prepare_loss(ENTROPY_BETA, self.scopes)

    self.trainer = AccumTrainer(device)
    self.trainer.prepare_minimize(self.local_network.total_loss,
                                  self.local_network.get_vars())

    self.accum_gradients = self.trainer.accumulate_gradients()
    self.reset_gradients = self.trainer.reset_gradients()

    accum_grad_names = [self._local_var_name(x) for x in self.trainer.get_accum_grad_list()]
    global_net_vars = [x for x in global_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names]

    self.apply_gradients = grad_applier.apply_gradients(
      global_net_vars, self.trainer.get_accum_grad_list() )

    self.sync = self.local_network.sync_from(global_network)

    self.env = None

    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0
    self.episode_length = 0
    self.episode_max_q = -np.inf

  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_accum_grad_name(self, var):
    return self._local_var_name(var).replace(':','_') + '_accum_grad:0'

  def _anneal_learning_rate(self, global_time_step):
    time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
    learning_rate = self.initial_learning_rate * time_step_to_go / self.max_global_time_step
    return learning_rate

  def choose_action(self, pi_values):
    values = []
    sum = 0.0
    for rate in pi_values:
      sum = sum + rate
      value = sum
      values.append(value)

    r = random.random() * sum
    for i in range(len(values)):
      if values[i] >= r:
        return i

    # fail safe
    return len(values) - 1

  def _record_score(self, sess, writer, summary_op, placeholders, values, global_t):
    feed_dict = {}
    for k in placeholders:
      feed_dict[placeholders[k]] = values[k]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    if VERBOSE: sys.stdout.write('writing to summary writer at time %d\n' % (global_t))
    writer.add_summary(summary_str, global_t)
    # writer.flush()

  def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):

    if self.env is None:
      # lazy evaluation
      time.sleep(self.thread_index*1.0)
      self.env = Environment({
        'scene_name': self.training_scene,
        'terminal_state_id': self.task_scope,
        'checkpoint_state_id': self.checkpoint_scope
      })
      self.env.reset()

    states = []
    actions = []
    rewards = []
    values = []
    targets = []
    checkpoints = []
    positions = []
    auxilaries = []
    auxilaries_cl = []
    aclists = []
    colists = []
    isCheckpointed = []
    collision = []
    terminal_end = False

    # reset accumulated gradients
    sess.run( self.reset_gradients )

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    # t_max times loop
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.env.s_t, self.env.s_position, self.env.checkpoint, self.env.s_a_t, self.env.s_c_t, self.env.isCheckpoint, self.env.s_aux_cl, self.scopes)
      action = self.choose_action(pi_)

      states.append(self.env.s_t)
      actions.append(action)
      values.append(value_)
      targets.append(self.env.target)
      checkpoints.append(self.env.checkpoint)
      positions.append(self.env.s_position)
      aclists.append(self.env.s_a_t)
      colists.append(self.env.s_c_t)
      collision.append(self.env.s_aux_cl)
      isCheckpointed.append(int(self.env.isCheckpoint))
      if VERBOSE and (self.thread_index == 0) and (self.local_t % 1000) == 0:
        sys.stdout.write("Pi = {0} V = {1}\n".format(pi_, value_))

      # process game
      self.env.step(action)

      # receive game result
      reward = self.env.reward
      terminal = self.env.terminal

      #if self.env.isCheckpoint:
      #  sys.stdout.write("CHECKPOINT \n")
      if self.episode_length > 5e3: terminal = True

      self.episode_reward += reward
      self.episode_length += 1
      self.episode_max_q = max(self.episode_max_q, np.max(value_))

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1

      # s_t1 -> s_t
      self.env.update()
      auxilaries.append(self.env.s_aux)
      auxilaries_cl.append(self.env.s_aux_cl)
      
      if terminal:
        terminal_end = True
        sys.stdout.write("time %d | thread #%d | scene %s | target #%s\n%s %s episode reward = %.3f\n%s %s episode length = %d\n%s %s episode max Q  = %.3f\n" % (global_t, self.thread_index, self.scene_scope, self.task_scope, self.scene_scope, self.task_scope, self.episode_reward, self.scene_scope, self.task_scope, self.episode_length, self.scene_scope, self.task_scope, self.episode_max_q))

        summary_values = {
          "episode_reward_input": self.episode_reward,
          "episode_length_input": float(self.episode_length),
          "episode_max_q_input": self.episode_max_q,
          "learning_rate_input": self._anneal_learning_rate(global_t)
        }

        self._record_score(sess, summary_writer, summary_op, summary_placeholders,
                           summary_values, global_t)
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf
        self.env.reset()

        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.env.s_t, self.env.s_position, self.env.checkpoint, self.env.s_a_t, self.env.s_c_t, self.env.isCheckpoint, self.env.s_aux_cl, self.scopes)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    positions.reverse()
    auxilaries.reverse()
    auxilaries_cl.reverse()
    aclists.reverse()
    colists.reverse()
    isCheckpointed.reverse()
    collision.reverse()
    
    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []
    batch_t = []
    batch_c = []
    batch_p = []
    batch_aux = []
    batch_aux_cl = []
    batch_al = []
    batch_cl = []
    batch_ic = []
    batch_collision = []
    # compute and accmulate gradients
    for(ai, ri, si, Vi, ti, ci, pi, auxi, aux_cl_i, ali, cli, ici, coli) in zip(actions, rewards, states, values, targets, checkpoints, positions, auxilaries, auxilaries_cl, aclists, colists, isCheckpointed, collision):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1
      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)
      batch_t.append(ti)
      batch_c.append(ci)
      batch_p.append(pi)
      batch_aux.append(auxi)
      batch_aux_cl.append(aux_cl_i)
      batch_al.append(ali)
      batch_cl.append(cli)
      batch_ic.append(ici)
      batch_collision.append(coli)

    sess.run( self.accum_gradients,
              feed_dict = {
                self.local_network.s: batch_si,
                self.local_network.a: batch_a,
                self.local_network.t: batch_p,
                self.local_network.c: batch_c,
                self.local_network.td: batch_td,
                self.local_network.aux: batch_aux,
                self.local_network.aux_cl: batch_aux_cl,
                self.local_network.al: batch_al,
                self.local_network.cl: batch_cl,
                self.local_network.ic: batch_ic,
                self.local_network.col: batch_collision,
                self.local_network.r: batch_R} )

    cur_learning_rate = self._anneal_learning_rate(global_t)

    sess.run( self.apply_gradients,
              feed_dict = { self.learning_rate_input: cur_learning_rate } )

    if VERBOSE and (self.thread_index == 0) and (self.local_t % 100) == 0:
      sys.stdout.write("Local timestep %d\n" % self.local_t)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t, terminal

