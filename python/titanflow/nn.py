from .ops import *
import numpy as np

def softmax_func(y):
	"""Numerically stable softmax."""
	b = y - np.max(y, axis = 1, keepdims = True)
	expb = np.exp(b)
	softmax = expb / np.sum(expb, axis = 1, keepdims = True)
	return softmax

class SoftmaxCrossEntropyOp(Op):
	def __call__(self, node_A, node_B, dim = -1):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "SoftmaxXEntropy(%s, %s)" % (node_A.name, node_B.name)
		new_node.dim = dim
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		y = input_vals[0]
		y_ = input_vals[1]
		Softmax = softmax_func(y)
		cross_entropy = np.mean(-np.sum(y_ * np.log(Softmax), axis = node.dim), keepdims = True)
		return cross_entropy

	def gradient(self, node, output_grad):
		grad_A = (softmax_op(node.inputs[0]) + -1 * node.inputs[1]) * output_grad
		grad_B = zeros_like(node.inputs[1])
		return [grad_A, grad_B]

	def infer_shape(self, node, input_shapes):
		return (1, )


class SoftmaxOp(Op):
	def __call__(self, node_A):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "Softmax(%s)" % (node_A.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return softmax_func(input_vals[0])

	def gradient(self, node, output_grad):
		# Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
		# Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		return input_shapes[0]


class ReluOp(Op):
	def __call__(self, node_A):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "Relu(%s)" % (node_A.name)
		return new_node

	def compute(self, node, input_vals):
		return np.maximum(input_vals[0], 0)

	def gradient(self, node, output_grad):
		return [relu_gradient(node.inputs[0], output_grad)]

	def infer_shape(self, node, input_shapes):
		return input_shapes[0]


class ReluGradientOp(Op):
	def __call__(self, node_A, node_B):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "ReluGradient(%s)" % (node_A.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		return input_shapes[0]



softmax_op = SoftmaxOp()
softmax_cross_entropy = SoftmaxCrossEntropyOp()
relu = ReluOp()
relu_gradient = ReluGradientOp()

def softmax(logits, dim = -1, name = None):
	t = exp(logits)
	k = reduce_sum(t, axis = dim, keep_dims = True)
	return t / k


def softmax_cross_entropy_with_logits(labels, logits, dim = -1):
	return softmax_cross_entropy(logits, labels, dim)

