#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python q3_1.py Ant-v1 Humanoid-v1 --render --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from sklearn.utils import shuffle

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    with open('train_data/' + args.envname + '.pkl', 'rb') as f:
        data = pickle.loads(f.read())
    obs_data = np.array(data['observations'])
    act_data = np.array(data['actions'])
    obs_data, act_data = shuffle(obs_data, act_data, random_state=0)
    act_data = act_data.reshape(-1, act_data.shape[2])
    
    f = obs_data.shape[1]
    o = act_data.shape[1]

    y_ = tf.placeholder(tf.float32, [None, o])

    x = tf.placeholder(tf.float32, [None, f])
    W = tf.Variable(tf.truncated_normal([f, 64]))
    b = tf.Variable(tf.truncated_normal([64]))
    h = tf.tanh(tf.matmul(x, W) + b)
    W2 = tf.Variable(tf.truncated_normal([64, 64]))
    b2 = tf.Variable(tf.truncated_normal([64]))
    h2 = tf.tanh(tf.matmul(h, W2) + b2)
    Wend = tf.Variable(tf.truncated_normal([64, o]))
    bend = tf.Variable(tf.truncated_normal([o]))
    y = tf.matmul(h2, Wend) + bend

    loss = tf.reduce_sum(tf.losses.mean_squared_error(y, y_))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver()

    # with tf.Session() as sess:
    #     # tf_util.initialize()
    #     new_saver = tf.train.import_meta_graph("model/model_bc_" + args.envname +'-20000.meta')
    #     new_saver.restore(sess, "model/model_bc_" + args.envname +'-20000')
    #     for i in range(20001):
    #         inds = np.random.choice(len(obs_data), 4096)
    #         obs_batch, act_batch = obs_data[inds], act_data[inds]
    #         sess.run(train_step, feed_dict={x: obs_batch, y_: act_batch})
    #         if i % 1000 == 0:
    #             print(sess.run(loss, feed_dict={x: obs_batch, y_: act_batch}))
    #     saver.save(sess, "model/model_bc_" + args.envname, global_step=i)

    with tf.Session() as sess:
        tf_util.initialize()
        new_saver = tf.train.import_meta_graph("model/model_bc_" + args.envname +'-20000.meta')
        new_saver.restore(sess, "model/model_bc_" + args.envname +'-20000')

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(y, feed_dict={x: np.array([obs])})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            if i % 100 == 0: print("%ith iteration"%(i))

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
if __name__ == '__main__':
    main()
