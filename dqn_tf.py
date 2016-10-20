import numpy as np
import random
from collections import deque
import config
import gym
import tensorflow as tf

class Brain:
	def __init__(self, stateCnt, actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt
		self._createModel()
	
	def weight_variable(self, shape):
		initial = tf.random_uniform(shape, -1.0, 1.0)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.random_uniform(shape, -1.0, 1.0)
		return tf.Variable(initial)
	
	def _createModel(self):
		self.sess = tf.InteractiveSession()
		w1 = self.weight_variable([self.stateCnt, 128])
		b1 = self.bias_variable([128])

		w2 = self.weight_variable([128, 128])
		b2 = self.bias_variable([128])

		w3 = self.weight_variable([128, 128])
		b3 = self.bias_variable([128])

		w4 = self.weight_variable([128, self.actionCnt])
		b4 = self.bias_variable([self.actionCnt])

		self.s = tf.placeholder(tf.float32, [None, self.stateCnt])
		hidden_1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
		hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
		hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
		#hidden_3 = tf.nn.dropout(hidden_3, 0.5)
		self.Q = tf.matmul(hidden_3, w4) + b4

		self.a = tf.placeholder(tf.float32, [None, self.actionCnt])
		self.y = tf.placeholder(tf.float32, [None])
		greedy_action = tf.reduce_sum(tf.mul(self.Q, self.a), reduction_indices=1)

		cost = tf.reduce_mean(tf.square(self.y-greedy_action))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
		self.sess.run(tf.initialize_all_variables())
	
	def train(self, y, a, s):
		self.train_step.run(feed_dict={self.y:y, self.a:a, self.s:s})
	
	def predict(self, s):
		return self.Q.eval(feed_dict={self.s:s})

	def predictOne(self, s):
		s = s.reshape((1, self.stateCnt))
		return self.Q.eval(feed_dict={self.s:s})[0]

class Memory:
	samples = deque()

	def __init__(self, capacity):
		self.capacity = capacity
	
	def add(self, sample):
		self.samples.append(sample)

		if len(self.samples) > self.capacity:
			self.samples.popleft()
	
	def sample(self, n):
		n = min(n, len(self.samples))
		return random.sample(self.samples, n)

class Agent:
	steps = 0
	epsilon = 1.

	def __init__(self, stateCnt, actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt

		self.brain = Brain(stateCnt, actionCnt)
		self.memory = Memory(config.MEMORY_CAPACITY)
	
	def act(self, s):
		if random.random() < self.epsilon:
			return random.randint(0, self.actionCnt-1)
		else:
			return np.argmax(self.brain.predictOne(s))
	
	def observe(self, sample):
		self.memory.add(sample)
		self.steps += 1
		if self.steps % config.EPSILON_DECAY_TIME == 0:
			self.epsilon *= config.EPSILON_DECAY
	
	def replay(self):
		batch = self.memory.sample(config.BATCH_SIZE)
		batchLen = len(batch)

		no_state = np.zeros(self.stateCnt)

		states = np.array([s[0] for s in batch])
		_states = np.array([(no_state if s[3] is None else s[3]) for s in batch])
		action = np.array([s[1] for s in batch])

		Q1 = agent.brain.predict(states)
		Q2 = agent.brain.predict(_states)

		y = []

		for i in range(batchLen):
			sample = batch[i]
			r = sample[2]; _s = sample[3]

			if _s is None:
				y.append(r)
			else:
				#y.append(r + config.GAMMA * np.max(Q2[i]))
				y.append(r + config.GAMMA * Q2[i][np.argmax(Q1[i])])

		self.brain.train(np.array(y), action, states)

class Environment:
	def __init__(self, problem):
		self.env = gym.make(problem)
	
	def run(self, agent):
		self.env.monitor.start('cartpole_v1', force=True)
		for i in range(1000):
			s = self.env.reset()
			R = 0

			while "NP" != "P":
				if config.RENDER:
					self.env.render()
				a = agent.act(s)
				a_t = np.zeros(agent.actionCnt)
				a_t[a] = 1
				_s, r, done, info = self.env.step(a)

				if done:
					_s = None

				agent.observe((s, a_t, r, _s))
				if agent.steps > config.OBSERVE:
					agent.replay()

				s = _s
				R += r

				if done:
					break
			print("episode %d: %f" % (i, R))
		self.env.monitor.close()

if __name__ == "__main__":
	PROBLEM = 'CartPole-v1'
	env = Environment(PROBLEM)

	stateCnt  = env.env.observation_space.shape[0]
	actionCnt = env.env.action_space.n

	agent = Agent(stateCnt, actionCnt)
	env.run(agent)
