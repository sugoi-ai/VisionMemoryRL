# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Actor-Critic Network Base Class
# The policy network and value network architecture
# should be implemented in a child class of this one
class ActorCriticNetwork(object):
  def __init__(self,
               action_size,
               device="/GPU:1"):
    self._device = device
    self._action_size = action_size

  #scopes = [network_scope, scene_scope, task_scope]  
  def prepare_loss(self, entropy_beta, scopes):

    # drop task id (last element) as all tasks in
    # the same scene share the same output branch
    scope_key = self._get_key(scopes[:-1])
    print(scope_key)
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))

      # policy entropy
      entropy = -tf.reduce_sum(self.pi[scope_key] * log_pi, axis=1)

      # policy loss (output)
      policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * entropy_beta)

      # R (input for value)
      self.r = tf.placeholder("float", [None])

      # value loss (output)
      # learning rate for critic is half of actor's
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key])
      
      self.aux = tf.placeholder("float", [None, self._action_size])
      self.aux_cl = tf.placeholder("float", [None, self._action_size])

      #loop_value_loss = 0.5 * tf.nn.l2_loss(tf.reduce_sum(tf.multiply(self.aux, self.aux - self.loop_value[scopes[0]])))
      loop_value_loss = 0.5 * tf.nn.l2_loss(tf.reduce_sum(tf.multiply(self.aux, self.aux - self.loop_value[scope_key])))
      #collision_value_loss = 0.5 * tf.nn.l2_loss(tf.reduce_sum(tf.multiply(self.aux_cl, self.aux_cl - self.collision_value[scopes[0]])))
      collision_value_loss = 0.5 * tf.nn.l2_loss(tf.reduce_sum(tf.multiply(self.aux_cl, self.aux_cl - self.collision_value[scope_key])))
      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss + loop_value_loss + collision_value_loss

  def run_policy_and_value(self, sess, s_t, task):
    raise NotImplementedError()

  def run_policy(self, sess, s_t, task):
    raise NotImplementedError()

  def run_value(self, sess, s_t, task):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "ActorCriticNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def _get_key(self, scopes):
    return '/'.join(scopes)

# Actor-Critic Feed-Forward Network
class ActorCriticFFNetwork(ActorCriticNetwork):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    ActorCriticNetwork.__init__(self, action_size, device)

    self.pi = dict()
    self.v = dict()
    self.loop_value = dict()
    self.collision_value = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_fc4 = dict()
    self.b_fc4 = dict()

    self.W_fc5 = dict()
    self.b_fc5 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_value = dict()
    self.b_value = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4])

      # target position
      self.t = tf.placeholder("float", [None, 4, 10])

      # checkpoint (input)
      self.c = tf.placeholder("float", [None, 2048, 4])
      
      self.al = tf.placeholder("float", [None, self._action_size, 10])
      
      self.cl = tf.placeholder("float", [None, 1, 10])

      self.ic = tf.placeholder("float", [None,])

      self.col = tf.placeholder("float", [None, self._action_size])
      
      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.c_flat = tf.reshape(self.c, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 40])
        self.col_flat = tf.reshape(self.col, [-1, 3])
        ic_scalar = tf.reshape(self.ic, [-1, 1])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_c_flat = tf.nn.relu(tf.matmul(self.c_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_c_flat], axis=1)

        #normalize_a = tf.nn.l2_normalize(h_s_flat,0)        
        #normalize_b = tf.nn.l2_normalize(h_c_flat,0)
        #cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))

        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])#*(1-cos_similarity)

        #LSTM
       
        #lstm_in = tf.concat(values=[self.al, self.t], axis=1)
        lstm_in = tf.transpose(self.al)
        lstm_in = tf.reshape(lstm_in,[-1,30])
        lstm_in = tf.split(lstm_in,10,1)
        
        rnn_cell = rnn.BasicLSTMCell(64)

        outputs, states = rnn.static_rnn(rnn_cell, lstm_in, dtype="float32")
        
        

        h_fc2_lstm = tf.concat(values=[h_fc2*(1-ic_scalar), outputs[-1], self.t_flat, self.col_flat], axis=1)
        #h_fc2_lstm = tf.concat(values=[h_fc2, outputs[-1], self.t_flat], axis=1)

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope):
            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([619, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 619)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_lstm, self.W_fc3[key]) + self.b_fc3[key])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy[key]) + self.b_policy[key]
            #pi_ = tf.multiply(1-self.col_flat,tf.nn.softmax(pi_))
            #self.pi[key] = pi_/tf.reduce_sum(pi_)
            self.pi[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(h_fc3, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])


            # weight for value output layer
            self.W_fc4[key] = self._fc_weight_variable([512, action_size])
            self.b_fc4[key] = self._fc_bias_variable([action_size], 512)

            # weight for value output layer
            self.W_fc5[key] = self._fc_weight_variable([512, action_size])
            self.b_fc5[key] = self._fc_bias_variable([action_size], 512)

            self.loop_value[key] = tf.nn.sigmoid(tf.matmul(h_fc3, self.W_fc4[key]) + self.b_fc4[key])
            self.collision_value[key] = tf.nn.sigmoid(tf.matmul(h_fc3, self.W_fc5[key]) + self.b_fc5[key])

  def run_policy_and_value(self, sess, state, target, checkpoint, action_list, collision_list, checkpointed, collision, scopes):
    k = self._get_key(scopes[:2])
    pi_out, v_out = sess.run( [self.pi[k], self.v[k]], feed_dict = {self.s : [state], self.t: [target], self.c: [checkpoint], self.al: [action_list], self.cl: [collision_list], self.ic: [checkpointed], self.col: [collision]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, state, target, checkpoint, action_list, collision_list, checkpointed, collision, scopes):
    k = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[k], feed_dict = {self.s : [state], self.t: [target], self.c: [checkpoint], self.al: [action_list], self.cl: [collision_list], self.ic: [checkpointed], self.col: [collision]} )
    return pi_out[0]

  def run_value(self, sess, state, target, checkpoint, action_list, collision_list, checkpointed, collision, scopes):
    k = self._get_key(scopes[:2])
    v_out = sess.run( self.v[k], feed_dict = {self.s : [state], self.t: [target], self.c: [checkpoint], self.al: [action_list], self.cl: [collision_list], self.ic: [checkpointed], self.col: [collision]} )
    return v_out[0]

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_fc4, self.b_fc4,
      self.W_fc5, self.b_fc5,
      self.W_policy, self.b_policy,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
