# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH

from constants import YOLO
from constants import TARGET
from constants import CHECKPOINT
from constants import DATA 

class THORDiscreteEnvironment(object):

  def __init__(self, config=dict()):

    

    # configurations
    self.scene_name          = config.get('scene_name', 'FloorPlan28')
    self.random_start        = config.get('random_start', True)
    self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling
    self.terminal_state_id   = config.get('terminal_state_id', 0)
    self.checkpoint_state_id   = config.get('checkpoint_state_id', 0)
    self.terminal_state_ids = TARGET[self.terminal_state_id]
    self.checkpoint_state_ids = CHECKPOINT[self.checkpoint_state_id]

    self.h5_file_path = config.get('h5_file_path', DATA+'/%s.h5'%self.scene_name)
    self.h5_file      = h5py.File(self.h5_file_path, 'r')

    self.yolo = YOLO[self.scene_name]

    print(self.scene_name)
    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()]
    self.n_locations = self.locations.shape[0]

    self.terminals = np.zeros(self.n_locations)
    self.checkpoints = np.zeros(self.n_locations)
    self.terminals[self.terminal_state_ids] = 1
    self.terminal_states, = np.where(self.terminals)
    self.checkpoint_states, = np.where(self.checkpoints[0])

    self.transition_graph = self.h5_file['graph'][()]
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

    self.history_length = HISTORY_LENGTH
    self.screen_height  = SCREEN_HEIGHT
    self.screen_width   = SCREEN_WIDTH

    # we use pre-computed fc7 features from ResNet-50
    # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
    self.s_t      = np.zeros([2048, self.history_length])
    self.s_t1     = np.zeros_like(self.s_t)
    #self.s_checkpoint = self._tiled_checkpoint(self.checkpoint_state_ids)
    self.s_checkpoint = self._tiled_state(self.checkpoint_state_ids[0])
    self.s_position = np.zeros([4, self.history_length])
    self.s_position1 = np.zeros([4, self.history_length])
    self.task_key = self.terminal_state_id
    self.reset()
  # public methods

  def reset(self):
    # randomize initial state
    while True:
      k = random.randrange(self.n_locations)
      min_d = np.inf
      # check if target is reachable
      for t_state in self.terminal_states:
        dist = self.shortest_path_distances[k][t_state]
        min_d = min(min_d, dist)
      # min_d = 0  if k is a terminal state
      # min_d = -1 if no terminal state is reachable from k
      if min_d > 0: break

    # reset parameters
    self.current_state_id = k

    self.s_t = self._tiled_state(self.current_state_id)

    for step in range(self.history_length):
      target_pos = self._position(self.current_state_id)
      self.s_position =  np.append(self.s_position[:,1:], np.array([target_pos]).reshape(4,1), axis=1)

    self.checkpoints[self.checkpoint_state_ids[0]] = 1

    self.reward   = 0
    self.collided = False
    self.terminal = False
    self.checkpointed = False

  def step(self, action):
    assert not self.terminal, 'step() called in terminal state'
    k = self.current_state_id
    if self.transition_graph[k][action] != -1:
      self.current_state_id = self.transition_graph[k][action]
      if self.terminals[self.current_state_id]:
        self.terminal = True
        self.collided = False
        self.checkpointed = False
      elif self.checkpoints[self.current_state_id]:
        self.terminal = False
        self.collided = False
        self.checkpointed = True
      else:
        self.terminal = False
        self.collided = False
        self.checkpointed = False
    else:
      self.terminal = False
      self.collided = True
      self.checkpointed = False

    self.reward = self._reward(self.terminal, self.collided, self.checkpointed)
    self.s_t1 = np.append(self.s_t[:,1:], self.state, axis=1)
    self.s_position1  = np.append(self.s_position[:,1:], np.array([self._position(self.current_state_id)]).reshape(4,1), axis=1)
    #print(self.current_state_id)
    #print(self.reward)

  def step_advance(self, id_number):
    self.current_state_id = id_number

  def update(self):
    self.s_t = self.s_t1
    self.s_position = self.s_position1

  # private methodsq

  def _tiled_state(self, state_id):
    k = random.randrange(self.n_feat_per_locaiton)
    f = self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]
    return np.tile(f, (1, self.history_length))

  def _tiled_checkpoint(self, checkpoint_ids):
    k = random.randrange(self.n_feat_per_locaiton)
    cp = self.h5_file['resnet_feature'][checkpoint_ids[0]][k][:,np.newaxis]
    for i in range(self.history_length-1):
      f = self.h5_file['resnet_feature'][checkpoint_ids[i+1]][k][:,np.newaxis]
      cp = np.append(cp,f,axis=1)
    return cp

  def _reward(self, terminal, collided, checkpointed):
    # positive reward upon task completion
    if terminal: return 10.0
    #if checkpointed: 
    #  self.checkpoints[self.checkpoint_state_ids[0]] = 0
    #  return 0.1
    # time penalty or collision penalty
    return -0.01 if collided else -0.01
    #return -0.01 if collided else 0


  def _position(self, state_id):
    target_pos = [0,0,0,0]
    task_object = self.yolo.get(str(state_id), -1)
    if task_object !=-1:
      target_pos = task_object.get(self.task_key, [0,0,0,0])
    #print(target_pos)  
    return target_pos 
  # properties

  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE 

  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
    return action_vocab[:ACTION_SIZE]

  @property
  def observation(self):
    try:
      return self.h5_file['observation'][self.current_state_id]
    except Exception as e:
      print("not exist current_state_id")
      return None

  @property
  def state(self):
    # read from hdf5 cache
    k = random.randrange(self.n_feat_per_locaiton)
    return self.h5_file['resnet_feature'][self.current_state_id][k][:,np.newaxis]

  @property
  def checkpoint(self):
    return self.s_checkpoint

  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def z(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]

if __name__ == "__main__":
  scene_name = 'FloorPlan28'

  env = THORDiscreteEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })
 