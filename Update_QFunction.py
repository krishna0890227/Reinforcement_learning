import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np



class DecisionPolicy:
    def select_action(self, current_state):
        pass

    def update_q(self, state, action, reward, next_state):
        pass



class RandomDecisionPolicy(DecisionPolicy):
    def __init__(self, actions):
        self.actions= actions

    def select_action(self, current_state):
        print('I am here')
        action=np.random.choice(self.actions)
        return action



class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim1):
        input_dim=input_dim1+2
        self.epsilon = 0.95
        self.gamma = 0.3
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 20

        self.x = tf.placeholder(shape=(None, input_dim), dtype=tf.float32)
        self.y = tf.placeholder(shape=output_dim, dtype=tf.float32)

        w1 = tf.Variable(tf.random_normal(shape=[input_dim, h1_dim], stddev=0.1, dtype=tf.float32))
        b1 = tf.Variable(tf.random_normal(shape=[h1_dim], stddev=0.1, dtype=tf.float32))

        h1 = tf.nn.relu(tf.matmul(self.x, w1)+b1)

        w2 = tf.Variable(tf.random_normal(shape=[h1_dim, output_dim], stddev=0.1, dtype=tf.float32))
        b2 = tf.Variable(tf.random_normal(shape=[output_dim], stddev=0.1, dtype=tf.float32))

        self.q = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # setup loss function and optimizer
        loss = tf.square(self.y-self.q)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def select_action(self, current_state, step):
        threshold=min(self.epsilon, step/1000.)
        print('I am ok')
        if np.random.random()<threshold:
            # Exploit best option with probability epsilon
            action_q_vals=self.sess.run(self.q, feed_dict={self.x:current_state})
            action_idx=np.argmax(action_q_vals)
            action=self.actions[action_idx]
        else:
            #Explore random option with probability 1-epsilon

            action=self.actions[np.random.randint(0, len(self.actions)-1)]
        return action



    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x:state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x:next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        current_state_idx = self.actions.index(action)
        action_q_vals[0, current_state_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})

