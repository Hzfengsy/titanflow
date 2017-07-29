from .ops import *
from ._session import *

class GradientDescentOptimizer(object):
	def __init__(self, learning_rate, name = "GradientDescent"):
		self.learning_rate = learning_rate
		self.name = name
	
	def minimize(self, loss, var_list = None, name = None):
		if var_list == None:
			vars = all_var
		elif isinstance(var_list, list):
			vars = var_list
		else:
			vars = [var_list]
		grads = gradients(loss, vars)
		assign_list = []
		for i in range(len(grads)):
			assign_list.append(assign(vars[i], vars[i] - self.learning_rate * grads[i]))
		return runner(assign_list)


class AdamOptimizer(object):
	def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, name = "Adam"):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.name = name
		self.epsilon = epsilon
		

	def minimize(self, loss, var_list = None, name = None):
		if var_list == None:
			vars = all_var
		elif isinstance(var_list, list):
			vars = var_list
		else:
			vars = [var_list]
		grads = gradients(loss, vars)
		self.t = Variable(zeros([1, ]))
		self.m = [Variable(zeros([1, ])) for i in range(len(grads))]
		self.v = [Variable(zeros([1, ])) for i in range(len(grads))]
		assign_m = [0 for i in range(len(grads))]
		assign_v = [0 for i in range(len(grads))]
		delta = [0 for i in range(len(grads))]
		assign_list = []
		assign_t = assign(self.t, self.t + 1)
		It_t = self.learning_rate * sqrt(1 - pow(self.beta2, assign_t)) / (1 - pow(self.beta1, assign_t))
		for i in range(len(grads)):
			assign_m[i] = assign(self.m[i], self.beta1 * self.m[i] + (1. - self.beta1) * grads[i])
			assign_v[i] = assign(self.v[i], self.beta2 * self.v[i] + (1. - self.beta2) * grads[i] * grads[i])
			delta[i] = It_t * assign_m[i] / (sqrt(assign_v[i]) + self.epsilon)
			assign_list.append(assign(vars[i], vars[i] - delta[i]))
		return runner(assign_list)


