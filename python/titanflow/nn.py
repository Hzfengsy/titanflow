from .ops import *
import numpy as np


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



relu = ReluOp()
relu_gradient = ReluGradientOp()

def softmax(logits, dim = -1, name = None):
	t = exp(logits)
	k = reduce_sum(t, axis = dim, keep_dims = True)
	return t / k


def softmax_cross_entropy_with_logits(labels, logits, dim = -1):
	y = softmax(logits)
	return reduce_mean(-reduce_sum(labels * log(y), reduction_indices=[1]))

