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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    returns = []

    with tf.Session() as sess:
        tf_util.initialize()
        for i in range(0, args.num_rollouts):
            for i in range(1000):
                inds = np.random.choice(len(obs_data), 4096, replace=False)
                obs_batch, act_batch = obs_data[inds], act_data[inds]
                sess.run(train_step, feed_dict={x: obs_batch, y_: act_batch})
            print(sess.run(loss, feed_dict={x: obs_batch, y_: act_batch}))

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit
            total = []
            for i in range(20):
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = sess.run(y, feed_dict={x: np.array([obs])})
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                total.append(totalr)
            returns.append(sum(total)/len(total))
            print(total, sum(total)/len(total))

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    # train = [1000*x for x in range(0, len(returns))]
    # plt.plot(train, returns)
    # plt.xlabel('training steps')
    # plt.ylabel('returns')
    # pp = PdfPages('q3_2.pdf')
    # plt.savefig(pp, format='pdf')
    # pp.close()
if __name__ == '__main__':
    main()
