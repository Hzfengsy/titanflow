from .ops import *
import numpy as np
from math import ceil

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
		x = input_vals[0]
		w = input_vals[1]
		N, in_height, in_width, in_channel = x.shape
		filter_height, filter_width, C, out_channel = w.shape
		strides = node.strides
		assert strides[0] == 1 and strides[3] == 1
		assert C == in_channel
		if node.padding == "SAME":
			out_height = int(ceil(float(in_height) / float(strides[1])))
			out_width = int(ceil(float(in_width) / float(strides[2])))
			pad_height = (out_height - 1) * strides[1] + filter_height - in_height
			pad_width = (out_width - 1) * strides[2] + filter_width - in_width
			pad_top = pad_height // 2
			pad_bottom = pad_height - pad_top
			pad_left = pad_width // 2
			pad_right = pad_width - pad_left
			x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), "constant")
			N, in_height, in_width, in_channel = x.shape
		if node.padding == "VALID":
			out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
			out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

		ans = np.zeros((N, out_height, out_width, out_channel))

		for i in range(0, N):
			x_data = x[i]
			xx, yy = -1, -1
			for j in range(0, in_height - filter_height + 1, strides[1]):
				yy += 1
				for k in range(0, in_width - filter_width + 1, strides[2]):
					xx += 1
					x_rf = x_data[j : j + filter_height, k : k + filter_width, :]
					for l in range(0, out_channel):
						conv_value = np.sum(x_rf * w[:, :, :, l])
						ans[i, yy, xx, l] = conv_value
				xx = -1
		return ans

	def gradient(self, node, output_grad):
		return [conv2d_gradient_dx(node.inputs[0], node.inputs[1], output_grad, node.strides, node.padding),
				conv2d_gradient_dw(node.inputs[0], node.inputs[1], output_grad, node.strides, node.padding)]

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class Conv2dGradientXOp(Op):
	def __call__(self, input, filter, output_grad, strides, padding, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, filter, output_grad]
		new_node.name = "conv2d_gradient(%s, %s, %s)" % (input, filter, output_grad)
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3
		x = input_vals[0]
		w = input_vals[1]
		dout = input_vals[2]
		N, in_height, in_width, in_channel = x.shape
		filter_height, filter_width, C, out_channel = w.shape
		_, dout_height, dout_width, _ = dout.shape
		strides = node.strides
		assert strides[0] == 1 and strides[3] == 1
		assert C == in_channel
		if node.padding == "SAME":
			out_height = int(ceil(float(in_height) / float(strides[1])))
			out_width = int(ceil(float(in_width) / float(strides[2])))
			pad_height = (out_height - 1) * strides[1] + filter_height - in_height
			pad_width = (out_width - 1) * strides[2] + filter_width - in_width
			pad_top = pad_height / 2
			pad_bottom = pad_height - pad_top
			pad_left = pad_width / 2
			pad_right = pad_width - pad_left
			x_with_pad = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), "constant")
		if node.padding == "VALID":
			out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
			out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
			x_with_pad = x

		dx = np.zeros((N, in_height, in_width, in_channel))
		
		for nprime in range(N):
			for i in range(in_height):
				for j in range(in_width):
					for f in range(out_channel):
						for k in range(dout_height):
							for l in range(dout_width):
								mask1 = np.zeros_like(w[:, :, :, f])
								mask2 = np.zeros_like(w[:, :, :, f])
								if (i + pad_top - k * strides[1]) < filter_height and (i + pad_top - k * strides[1]) >= 0:
									mask1[i + pad_top - k * strides[1], :, :] = 1.0
								if (j + pad_left - l * strides[2]) < filter_width and (j + pad_left - l * strides[2]) >= 0:
									mask2[:, j + pad_left - l * strides[2], :] = 1.0
								w_masked = np.sum(w[:, :, :, f] * mask1 * mask2, axis=(0, 1))
								dx[nprime, i, j, :] += dout[nprime, k, l, f] * w_masked
		return dx

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		raise NotImplementedError


class Conv2dGradientWOp(Op):
	def __call__(self, input, filter, output_grad, strides, padding, name = None):
		"""node_B is output_grad"""
		new_node = Op.__call__(self)
		new_node.inputs = [input, filter, output_grad]
		new_node.name = "conv2d_gradient(%s, %s, %s)" % (input, filter, output_grad)
		new_node.strides = strides
		new_node.padding = padding
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 3
		x = input_vals[0]
		w = input_vals[1]
		dout = input_vals[2]
		N, in_height, in_width, in_channel = x.shape
		filter_height, filter_width, C, out_channel = w.shape
		_, dout_height, dout_width, _ = dout.shape
		strides = node.strides
		assert strides[0] == 1 and strides[3] == 1
		assert C == in_channel
		if node.padding == "SAME":
			out_height = int(ceil(float(in_height) / float(strides[1])))
			out_width = int(ceil(float(in_width) / float(strides[2])))
			pad_height = (out_height - 1) * strides[1] + filter_height - in_height
			pad_width = (out_width - 1) * strides[2] + filter_width - in_width
			pad_top = pad_height / 2
			pad_bottom = pad_height - pad_top
			pad_left = pad_width / 2
			pad_right = pad_width - pad_left
			x_with_pad = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), "constant")
		if node.padding == "VALID":
			out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
			out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
			x_with_pad = x

		dw = np.zeros((filter_height, filter_width, C, out_channel))

		for l in range(0, out_channel):
			for k in range(0, C):
				for i in range(0, filter_height):
					for j in range(0, filter_width):
						dw[i, j, k, l] = np.sum(dout[:, :, :, l] * x_with_pad[:, i:i + dout_height * strides[0]:strides[0], j:j + dout_width * strides[1]:strides[1], k])
		return dw

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

		ans = np.zeros((N, out_height, out_width, in_channel))

		for i in range(0, N):
			x_data = x[i]
			xx, yy = -1, -1
			for j in range(0, in_height - ksize[1] + 1, strides[1]):
				yy += 1
				for k in range(0, in_width - ksize[2] + 1, strides[2]):
					xx += 1
					x_rf = x_data[j : j + ksize[1], k : k + ksize[2], :]
					for l in range(0, in_channel):
						ans[i, yy, xx, l] = np.max(x_rf[:, :, l])
				xx = -1
		
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

		dx = np.zeros((N, in_height, in_width, in_channel))
		for i in range(0, N):
			x_data = x_with_pad[i]
			xx, yy = -1, -1
			for j in range(0, in_height - ksize[1] + 1, strides[1]):
				yy += 1
				for k in range(0, in_width - ksize[2] + 1, strides[2]):
					xx += 1
					x_rf = x_data[j : j + ksize[1], k : k + ksize[2], :]
					for l in range(0, in_channel):
						x_pool = x_rf[:, :, l]
						mask = x_pool == np.max(x_pool)
						dx[i, j : j + ksize[1], k : k + ksize[2], l] += dout[i, yy, xx, l] * mask
				xx = -1
		return dx

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


def softmax(logits, dim = -1, name = None):
	t = exp(logits)
	k = reduce_sum(t, axis = dim, keep_dims = True)
	return t / k


def softmax_cross_entropy_with_logits(labels, logits, dim = -1):
	y = softmax(logits)
	return reduce_mean(-reduce_sum(labels * log(y), axis = dim))

