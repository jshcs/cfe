import tensorflow as tf
from config import *

class BaseModel():
	def __init__(self):
		self.config=config_params
		self.sess=None
	def global_init_variables(self):
		init=tf.global_variables_initializer()
		self.sess.run(init)
