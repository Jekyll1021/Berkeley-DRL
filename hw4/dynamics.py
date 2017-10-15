import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        self.env = env
        self.obs_act = tf.placeholder(shape=[None, self.env.observation_space.shape[0] + self.env.action_space.shape[0]], name="input", dtype=tf.float32)
        self.next_states = tf.placeholder(shape=[None, self.env.observation_space.shape[0]], name="next_states", dtype=tf.float32)
        self.pred_next = build_mlp(self.obs_act, self.env.observation_space.shape[0], "pred_next", n_layers, size, activation, output_activation)
        self.loss = tf.reduce_mean((self.pred_next - self.next_states)**2)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.sess = sess
        self.batch_size = batch_size
        self.iterations = iterations
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization
        """ Note: Be careful about normalization """

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        # TODO: do the updating as minibatch GD
        norm_obs = (data['observations'] - self.mean_obs) / (self.std_obs + 1e-8)
        norm_act = (data['actions'] - self.mean_action) / (self.std_action + 1e-8)
        norm_diff = (data['next_observations'] - data['observations'] - self.mean_deltas) / (self.std_deltas+1e-8)
        norm_obs_act = np.append(norm_obs, norm_act, axis=1)
        for _ in range(self.iterations):
          for _ in range(1000):
            ind = np.random.choice(len(norm_obs_act), self.batch_size, replace=False)
            _, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.obs_act:norm_obs_act[ind], self.next_states:norm_diff[ind]})

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        norm_states = (states - self.mean_obs) / (self.std_obs+ 1e-8)
        norm_actions = (actions - self.mean_action) / (self.std_action+1e-8)
        norm_states_actions = np.append(norm_states, norm_actions, axis=1)
        return self.sess.run(self.pred_next, feed_dict={self.obs_act:norm_states_actions}) * (self.std_deltas+1e-8) + self.mean_deltas + states
