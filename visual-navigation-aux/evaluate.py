#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import sys

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread
from scene_loader import THORDiscreteEnvironment as Environment

from utils.ops import sample_action

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE

from constants import TASK_TYPE
from constants import TASK_LIST
from constants import YOLO

from utils.tools import SimpleImageViewer
import time

if __name__ == '__main__':

  device = "/cpu:0" # use CPU for display tool
  network_scope = TASK_TYPE
  list_of_tasks = TASK_LIST
  scene_scopes = list_of_tasks.keys()
  yolo = YOLO
  global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                        device=device,
                                        network_scope=network_scope)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")

  scene_stats = dict()
  print(scene_scopes )
  
  f = open("/home/hmi/Desktop/visual-navigation-aux/eva.py", "w")
  f_op = open("/home/hmi/Desktop/visual-navigation-aux/op.py", "w")
  f_path = open("/home/hmi/Desktop/visual-navigation-aux/path.txt", "w")
  #fo = open("/home/hmi/Desktop/visual-navigation-checkpoint/Dang_result.py", "a")
  NNN = 100
  counter = 0
  viewer = SimpleImageViewer()
  for scene_scope in scene_scopes:
    print(scene_scope) 

    scene_stats[scene_scope] = []
    f_path.write('PATH = {')
    f.write('EVA = {\n')
    f_op.write('OPT = {\n')
    #fo.write(scene_scope +'\n')
    for task_scope in list_of_tasks[scene_scope]:
      #fo.write('%s \n', %task_scope)
      f.write("'"+str(task_scope[0])+"'"+': [')
      f_op.write("'"+str(task_scope[0])+"'"+': [')
      env = Environment({
        'scene_name': scene_scope,
        'terminal_state_id': task_scope[0],
        'checkpoint_state_id': task_scope[1]
      })
      ep_rewards = []
      ep_lengths = []
      ep_collisions = []

      scopes = [network_scope, "scene", task_scope]

      print('evaluation: %s %s' % (scene_scope, task_scope))
      
      
      #NUM_EVAL_EPISODES
      for i_episode in range(NNN):

        optimal = env.reset()
        terminal = False
        ep_reward = 0
        ep_collision = 0
        ep_t = 0

        f_path.write(str(counter*NNN+i_episode)+': [')
        while not terminal:
          
          #viewer.imshow(env.observation)
          #time.sleep(0.3)
          #print(env.isCheckpoint)  
          pi_values = global_network.run_policy(sess, env.s_t, env.s_position, env.checkpoint, env.s_a_t, env.s_c_t, env.isCheckpoint, env.s_aux_cl,scopes)
          #print(pi_values)
          action = sample_action(pi_values)
          env.step(action)

          
              
          env.update()

          terminal = env.terminal
          if ep_t == 500: break
          if env.collided: ep_collision += 1
          ep_reward += env.reward
          ep_t += 1
          #if not terminal:
          #  f_path.write('['+str(int(env.x*2))+', '+str(int(env.z*2))+'], ')
          #else:
          #  f_path.write('['+str(int(env.x*2))+', '+str(int(env.z*2))+']')
            
        f_path.write(']}\n')    
        if i_episode < NNN-1:
          f.write(str(ep_t)+', ')
          f_op.write(str(optimal)+', ')
        else:
          f.write(str(ep_t)+'],\n')
          f_op.write(str(optimal)+'],\n')

        ep_lengths.append(ep_t)
        ep_rewards.append(ep_reward)
        ep_collisions.append(ep_collision)
        if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

       
      counter+=1 
      #print('evaluation: %s %s' % (scene_scope, task_scope))
      print('mean episode reward: %.2f' % np.mean(ep_rewards))
      print('mean episode length: %.2f' % np.mean(ep_lengths))
      print('mean episode collision: %.2f' % np.mean(ep_collisions))
      #fo.write('%.2f , %.2f , %.2f \n' % (np.mean(ep_rewards), np.mean(ep_lengths), np.mean(ep_collisions)))
      scene_stats[scene_scope].extend(ep_lengths)

      #break
    #break
    f_path.write(']}')
    f.write('}')
    f_op.write('}')
#fo.write("average")
print('\nResults (average trajectory length):')
for scene_scope in scene_stats:
  print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
  #fo.write('%s: %.2f steps \n'%(scene_scope, np.mean(scene_stats[scene_scope])))
#fo.close()
