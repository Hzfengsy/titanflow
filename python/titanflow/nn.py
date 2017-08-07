from .ops import *
import numpy as np
from math import ceil
from ctypes import *
from _type import *


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


class Conv2dOp(Op):
	def __call__(self, input, filter, strides, padding, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, filter]
		new_node.name = "conv2d(%s, %s)" % (input, filter)
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		# print "Conv start"
		node.N, node.in_height, node.in_width, node.in_channel = input_vals[0].shape
		node.filter_height, node.filter_width, C, node.out_channel = input_vals[1].shape
		node.strides = node.strides
		assert node.strides[0] == 1 and node.strides[3] == 1
		assert C == node.in_channel
		if node.padding == "SAME":
			node.out_height = int(ceil(float(node.in_height) / float(node.strides[1])))
			node.out_width = int(ceil(float(node.in_width) / float(node.strides[2])))
			node.pad_height = (node.out_height - 1) * node.strides[1] + node.filter_height - node.in_height
			node.pad_width = (node.out_width - 1) * node.strides[2] + node.filter_width - node.in_width
			node.pad_top = node.pad_height // 2
			node.pad_bottom = node.pad_height - node.pad_top
			node.pad_left = node.pad_width // 2
			node.pad_right = node.pad_width - node.pad_left
			x = np.pad(input_vals[0],
					   ((0, 0), (node.pad_top, node.pad_bottom), (node.pad_left, node.pad_right), (0, 0)),
					   "constant")
			node.N, node.in_height, node.in_width, node.in_channel = x.shape
		if node.padding == "VALID":
			node.out_height = int(ceil(float(node.in_height - node.filter_height + 1) / float(node.strides[1])))
			node.out_width = int(ceil(float(node.in_width - node.filter_width + 1) / float(node.strides[2])))
			x = input_vals[0]

		node.out_shape = (node.N, node.out_height, node.out_width, node.out_channel)
		node.outcols_shape = (node.N * node.out_height * node.out_width, node.out_channel)
		x = x.astype(np.float32)
		x_data = x.ctypes.data_as(POINTER(c_float))

		node.w_cols = input_vals[1].reshape(-1, node.out_channel).astype(np.float32)
		w_data = node.w_cols.ctypes.data_as(POINTER(c_float))

		output = np.zeros((node.N, node.out_height, node.out_width, node.out_channel), dtype = np.float32)
		output_data = output.ctypes.data_as(POINTER(c_float))
		# x_col = N * out_height * out_width, filter_height * filter_width * in_channel
		# output_col = N * out_height * out_width, out_channel
		# node.output = node.output_cols.reshape(node.N, node.out_height, node.out_width, node.out_channel)
		# w_col = filter_height * filter_width * in_channel, out_channel
		lib.conv2d(x_data, w_data,
				   output_data, node.N,
				   node.in_height, node.in_width,
				   node.filter_height, node.filter_width,
				   node.out_height, node.out_width,
				   node.strides[1], node.strides[2],
				   node.in_channel, node.out_channel)

		return output

	def gradient(self, node, output_grad):
		return [conv2d_gradient_dx(node.inputs[0], node.inputs[1], output_grad, node),
				conv2d_gradient_dw(node.inputs[0], node.inputs[1], output_grad, node)]

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class Conv2dGradientXOp(Op):
	def __call__(self, input, filter, output_grad, node):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, filter, output_grad]
		new_node.name = "conv2d_gradient_dx(%s, %s, %s)" % (input, filter, output_grad)
		new_node.node = node
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3
		tnode = node.node
		if tnode.padding == "SAME":
			x = np.pad(input_vals[0],
					   ((0, 0), (tnode.pad_top, tnode.pad_bottom), (tnode.pad_left, tnode.pad_right), (0, 0)),
					   "constant")
		if tnode.padding == "VALID":
			x = input_vals[0]
		w = input_vals[1]
		dout = input_vals[2]

		# dout_cols = N * out_height * out_width, out_channel
		dout_cols = dout.reshape(tnode.N, tnode.out_height * tnode.out_width, tnode.out_channel).astype(np.float32)
		# grad_x_cols = N * out_height * out_width, filter_height * filter_width * in_channel
		# grad_x_cols = np.dot(dout_cols, tnode.w_cols.T)
		dx_padded = np.zeros((tnode.N, tnode.in_height, tnode.in_width, tnode.in_channel), dtype = np.float32)

		w = w.astype(np.float32)
		w_data = w.ctypes.data_as(POINTER(c_float))
		dout_data = dout_cols.ctypes.data_as(POINTER(c_float))
		dx_data = dx_padded.ctypes.data_as(POINTER(c_float))
		lib.conv2d_dx(w_data, dout_data,
					  dx_data, tnode.N,
					  tnode.in_height, tnode.in_width,
					  tnode.filter_height, tnode.filter_width,
					  tnode.out_height, tnode.out_width,
					  tnode.strides[1], tnode.strides[2],
					  tnode.in_channel, tnode.out_channel)

		if (tnode.padding == "SAME"):
			top = tnode.pad_top
			bottom = tnode.in_height - tnode.pad_bottom
			left = tnode.pad_left
			right = tnode.in_width - tnode.pad_right
		else:
			top = 0
			bottom = tnode.in_height
			left = 0
			right = tnode.in_width

		# print "Conv dx finish"
		return dx_padded[:, top: bottom, left: right, :]

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class Conv2dGradientWOp(Op):
	def __call__(self, input, filter, output_grad, node):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, filter, output_grad]
		new_node.name = "conv2d_gradient_dw(%s, %s, %s)" % (input, filter, output_grad)
		new_node.node = node
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3
		tnode = node.node
		if tnode.padding == "SAME":
			x = np.pad(input_vals[0],
					   ((0, 0), (tnode.pad_top, tnode.pad_bottom), (tnode.pad_left, tnode.pad_right), (0, 0)),
					   "constant")
		if tnode.padding == "VALID":
			x = input_vals[0]
		w = input_vals[1]
		dout = input_vals[2]
		# assert dout.shape == tnode.out_shape
		# dout_cols = N * out_height * out_width, out_channel
		dout_cols = dout.reshape((tnode.N, tnode.out_height * tnode.out_width, tnode.out_channel)).astype(np.float32)
		# grad_w = filter_height * filter_width * in_channel, out_channel
		grad_w = np.zeros(
			(tnode.filter_height * tnode.filter_width * tnode.in_channel, tnode.out_channel), dtype = np.float32)

		x_data = x.astype(np.float32).ctypes.data_as(POINTER(c_float))
		dout_data = dout_cols.ctypes.data_as(POINTER(c_float))
		grad_w_data = grad_w.ctypes.data_as(POINTER(c_float))

		lib.conv2d_dw(x_data, dout_data,
					  grad_w_data, tnode.N,
					  tnode.in_height, tnode.in_width,
					  tnode.filter_height, tnode.filter_width,
					  tnode.out_height, tnode.out_width,
					  tnode.strides[1], tnode.strides[2],
					  tnode.in_channel, tnode.out_channel)

		return grad_w.reshape(w.shape)

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class MaxPoolOp(Op):
	def __call__(self, input, ksize, strides, padding, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input]
		new_node.name = "max_pool(%s)" % (input)
		new_node.ksize = ksize
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		# print "Max_pool start"
		assert len(input_vals) == 1
		x = input_vals[0]
		N, in_height, in_width, in_channel = x.shape
		strides = node.strides
		ksize = node.ksize
		assert strides[0] == 1 and strides[3] == 1
		assert ksize[0] == 1 and ksize[3] == 1
		if node.padding == "SAME":
			out_height = int(ceil(float(in_height) / float(strides[1])))
			out_width = int(ceil(float(in_width) / float(strides[2])))
			pad_height = (out_height - 1) * strides[1] + ksize[1] - in_height
			pad_width = (out_width - 1) * strides[2] + ksize[2] - in_width
			pad_top = pad_height / 2
			pad_bottom = pad_height - pad_top
			pad_left = pad_width / 2
			pad_right = pad_width - pad_left
			x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), "constant")
			N, in_height, in_width, in_channel = x.shape
		if node.padding == "VALID":
			out_height = ceil(float(in_height - ksize[1] + 1) / float(strides[1]))
			out_width = ceil(float(in_width - ksize[2] + 1) / float(strides[2]))

		ans = np.zeros((N, out_height, out_width, in_channel), dtype = input_vals[0].dtype)
		xx, yy = -1, -1
		for i in range(0, in_height - ksize[1] + 1, strides[1]):
			yy += 1
			for j in range(0, in_width - ksize[2] + 1, strides[2]):
				xx += 1
				ans[:, yy, xx, :] = np.amax(x[:, i: i + ksize[1], j: j + ksize[2], :], axis = (1, 2))
			xx = -1

		# print "Max_pool finish"
		return ans

	def gradient(self, node, output_grad):
		return [max_pool_gradient(node.inputs[0], output_grad, node.ksize, node.strides, node.padding)]

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class MaxPoolGradientOp(Op):
	def __call__(self, input, output_grad, ksize, strides, padding, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, output_grad]
		new_node.name = "conv2d_gradient(%s, %s)" % (input, output_grad)
		new_node.strides = strides
		new_node.padding = padding
		new_node.ksize = ksize
		return new_node

	def compute(self, node, input_vals):
		# print "max_pool gradient start"
		assert len(input_vals) == 2
		x = input_vals[0]
		dout = input_vals[1]
		N, in_height, in_width, in_channel = x.shape
		_, dout_height, dout_width, _ = dout.shape
		strides = node.strides
		ksize = node.ksize
		assert strides[0] == 1 and strides[3] == 1
		if node.padding == "SAME":
			out_height = int(ceil(float(in_height) / float(strides[1])))
			out_width = int(ceil(float(in_width) / float(strides[2])))
			pad_height = (out_height - 1) * strides[1] + ksize[1] - in_height
			pad_width = (out_width - 1) * strides[2] + ksize[2] - in_width
			pad_top = pad_height / 2
			pad_bottom = pad_height - pad_top
			pad_left = pad_width / 2
			pad_right = pad_width - pad_left
			x_with_pad = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), "constant")
		if node.padding == "VALID":
			out_height = ceil(float(in_height - ksize[1] + 1) / float(strides[1]))
			out_width = ceil(float(in_width - ksize[2] + 1) / float(strides[2]))
			x_with_pad = x

		dx = np.zeros((N, in_height, in_width, in_channel), dtype = input_vals[0].dtype)
		xx, yy = -1, -1
		for j in range(0, in_height - ksize[1] + 1, strides[1]):
			yy += 1
			for k in range(0, in_width - ksize[2] + 1, strides[2]):
				xx += 1
				x_pool = x_with_pad[:, j: j + ksize[1], k: k + ksize[2], :]
				mask = np.equal(x_pool, np.max(x_pool, axis = (1, 2), keepdims = True))
				dx[:, j: j + ksize[1], k: k + ksize[2], :] += dout[:, yy: yy + 1, xx: xx + 1, :] * mask
			xx = -1
		# print "max_pool gradient finish"
		return dx

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class DropOutOP(Op):
	def __call__(self, input, keep_prob, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, keep_prob]
		new_node.name = "dropout(%s)" % (input)
		return new_node

	def compute(self, node, input_vals):
		t = np.random.random(input_vals[0].shape).astype(input_vals[0].dtype)
		node.mask = (t < input_vals[1])
		return input_vals[0] * node.mask

	def gradient(self, node, output_grad):
		return [dropout_gradient(output_grad, node), 0]

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class DropOutGradientOP(Op):
	def __call__(self, input, node, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input]
		new_node.name = "dropout_gradient(%s)" % (input)
		new_node.node = node
		return new_node

	def compute(self, node, input_vals):
		return input_vals[0] * node.node.mask

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


conv2d = Conv2dOp()
conv2d_gradient_dx = Conv2dGradientXOp()
conv2d_gradient_dw = Conv2dGradientWOp()
relu = ReluOp()
relu_gradient = ReluGradientOp()
max_pool = MaxPoolOp()
max_pool_gradient = MaxPoolGradientOp()
dropout = DropOutOP()
dropout_gradient = DropOutGradientOP()


def softmax(logits, dim = -1, name = None):
	t = exp(logits)
	k = reduce_sum(t, axis = dim, keep_dims = True)
	return t / k


def softmax_cross_entropy_with_logits(labels, logits, dim = -1):
	y = softmax(logits)
	return reduce_mean(-reduce_sum(labels * log(y), axis = dim))
