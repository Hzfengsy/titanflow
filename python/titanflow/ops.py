""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import numpy as np
from ._type import *

# from . import ndarray, gpu_op

init_assigns = []
all_var = []


class Node(object):
	"""Node in a computation graph."""

	def __init__(self):
		"""Constructor, new node is indirectly created by Op object call method.

			Instance variables
			------------------
			self.inputs: the list of input nodes.
			self.op: the associated op object,
				e.g. add_op if this node is created by adding two other nodes.
			self.const_attr: the add or multiply constant.
				e.g. self.const_attr=5 if this node is created by x+5.
			self.name: node name for debugging.
		"""
		self.inputs = []
		self.op = None
		self.const_attr = None
		self.name = ""

	def __add__(self, other):
		"""Adding two nodes return a new node."""
		new_node = add(self, other)
		return new_node

	def __mul__(self, other):
		"""Multiplying two nodes return a new node."""
		new_node = mul(self, other)
		return new_node

	def __sub__(self, other):
		"""Subing two nodes return a new node."""
		new_node = sub(self, other)
		return new_node

	def __rsub__(self, other):
		"""Subing two nodes return a new node."""
		new_node = sub(other, self)
		return new_node

	def __div__(self, other):
		"""Dividing two nodes return a new node."""
		new_node = div(self, other)
		return new_node

	def __rdiv__(self, other):
		"""Dividing two nodes return a new node."""
		new_node = div(other, self)
		return new_node

	def __neg__(self):
		new_node = neg(self)
		return new_node

	# Allow left-hand-side add and multiply.
	__radd__ = __add__
	__rmul__ = __mul__

	def __str__(self):
		"""Allow print to display node name."""
		return self.name

	def eval(self, feed_dict = {}):
		from ._session import sess_t
		return sess_t.run(self, feed_dict)

	def run(self, feed_dict = {}):
		from ._session import sess_t
		return sess_t.run(self, feed_dict)


class Op(object):
	"""Op represents operations performed on nodes."""

	def __call__(self):
		"""Create a new node and associate the op object with the node.

		Returns
		-------
		The new node object.
		"""
		new_node = Node()
		new_node.op = self
		return new_node

	def compute(self, node, input_vals):
		"""Given values of input nodes, compute the output value.

		Parameters
		----------
		node: node that performs the compute.
		input_vals: values of input nodes.
		output_val: output value of the node, modified in-place.
		use_numpy: bool flag whether to use numpy for compute
		"""
		raise NotImplementedError

	def gradient(self, node, output_grad):
		"""Given output gradient, compute partial gradient to each input node.

		Parameters
		----------
		node: node that performs the gradient.
		output_grad: output gradient summed from children nodes' contributions

		Returns
		-------
		A list of gradient contributions to each input node respectively.
		"""
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		"""Given shapes of input nodes, compute shape of output node.

		Implementation note:
		It's simpler to treat shape of constants as (1,), so that constants can
		be stored as a numpy array too and you would need fewer special case
		handling.

		Parameters
		----------
		node: node whose shape is being inferred.
		input_vals: shapes of input nodes.

		Returns
		-------
		A tuple representing the shape of output node.
		"""
		raise NotImplementedError


class NegOp(Op):
	def __call__(self, input):
		new_node = Op.__call__(self)
		new_node.name = "(-%s)" % (input)
		new_node.inputs = [input]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return -input_vals[0]

	def gradient(self, node, output_grad):
		return [-output_grad]

	def infer_shape(self, node, input_shapes):
		return input_shapes[0]


class AddOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		if not isinstance(node_B, Node):
			node_B = constant(node_B)
		new_node.name = "(%s + %s)" % (node_A.name, node_B.name)
		new_node.inputs = [node_A, node_B]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return input_vals[0] + input_vals[1]

	def gradient(self, node, output_grad):
		return [reduce_sum_op(output_grad, node.inputs[0]), reduce_sum_op(output_grad, node.inputs[1])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 2
		return broadcast_rule(input_shapes[0], input_shapes[1])


class SubOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		if not isinstance(node_B, Node):
			node_B = constant(node_B)
		if not isinstance(node_A, Node):
			node_A = constant(node_A)
		new_node.name = "(%s - %s)" % (node_A.name, node_B.name)
		new_node.inputs = [node_A, node_B]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return input_vals[0] - input_vals[1]

	def gradient(self, node, output_grad):
		return [reduce_sum_op(output_grad, node.inputs[0]), reduce_sum_op(-output_grad, node.inputs[1])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 2
		return broadcast_rule(input_shapes[0], input_shapes[1])


class MulOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		if not isinstance(node_B, Node):
			node_B = constant(node_B)
		new_node.name = "(%s * %s)" % (node_A.name, node_B.name)
		new_node.inputs = [node_A, node_B]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return input_vals[0] * input_vals[1]

	def gradient(self, node, output_grad):
		return [reduce_sum_op(node.inputs[1] * output_grad, node.inputs[0]),
				reduce_sum_op(node.inputs[0] * output_grad, node.inputs[1])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 2
		return broadcast_rule(input_shapes[0], input_shapes[1])


class DivOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		if not isinstance(node_B, Node):
			node_B = constant(node_B)
		if not isinstance(node_A, Node):
			node_A = constant(node_A)
		new_node.name = "(%s / %s)" % (node_A.name, node_B.name)
		new_node.inputs = [node_A, node_B]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return input_vals[0] / input_vals[1]

	def gradient(self, node, output_grad):
		return [reduce_sum_op(output_grad / node.inputs[1], node.inputs[0]),
				reduce_sum_op(-node.inputs[0] * output_grad / node.inputs[1] / node.inputs[1], node.inputs[1])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 2
		return broadcast_rule(input_shapes[0], input_shapes[1])


class MatMulOp(Op):
	def __call__(self, node_A, node_B, trans_A = False, trans_B = False):
		new_node = Op.__call__(self)
		new_node.matmul_attr_trans_A = trans_A
		new_node.matmul_attr_trans_B = trans_B
		new_node.inputs = [node_A, node_B]
		new_node.name = "MatMul(%s, %s, %s, %s)" % (
			node_A.name, node_B.name, str(trans_A), str(trans_B))
		return new_node

	def compute(self, node, input_vals):
		if ((node.matmul_attr_trans_A is False) and (node.matmul_attr_trans_B is False)):
			return np.matmul(input_vals[0], input_vals[1])
		elif ((node.matmul_attr_trans_A is True) and (node.matmul_attr_trans_B is False)):
			return np.matmul(np.transpose(input_vals[0]), input_vals[1])
		elif ((node.matmul_attr_trans_A is False) and (node.matmul_attr_trans_B is True)):
			return np.matmul(input_vals[0], np.transpose(input_vals[1]))
		elif ((node.matmul_attr_trans_A is True) and (node.matmul_attr_trans_B is True)):
			return np.matmul(np.transpose(input_vals[0]), np.transpose(input_vals[1]))

	def gradient(self, node, output_grad):
		if ((node.matmul_attr_trans_A is False) and (node.matmul_attr_trans_B is False)):
			# if Y=AB, then dA=dY B^T, dB=A^T dY
			lhs_grad = matmul(output_grad, node.inputs[1], trans_A = False, trans_B = True)
			rhs_grad = matmul(node.inputs[0], output_grad, trans_A = True, trans_B = False)
		elif ((node.matmul_attr_trans_A is True) and (node.matmul_attr_trans_B is False)):
			# if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
			lhs_grad = matmul(node.inputs[1], output_grad, trans_A = False, trans_B = True)
			rhs_grad = matmul(node.inputs[0], output_grad, trans_A = True, trans_B = False)
		elif ((node.matmul_attr_trans_A is False) and (node.matmul_attr_trans_B is True)):
			# if Y=A B^T, then dA=dY B^T, dB=(A^T dY)^T=dY^T A
			lhs_grad = matmul(output_grad, node.inputs[1], trans_A = False, trans_B = True)
			rhs_grad = matmul(output_grad, node.inputs[0], trans_A = True, trans_B = False)
		elif ((node.matmul_attr_trans_A is True) and (node.matmul_attr_trans_B is True)):
			# if Y=A^T B^T, then dA=(dY B^T)^T=B dY^T, dB=(A^T dY)^T=dY^T A
			lhs_grad = matmul(node.inputs[1], output_grad, trans_A = False, trans_B = True)
			rhs_grad = matmul(output_grad, node.inputs[0], trans_A = True, trans_B = False)
		return [lhs_grad, rhs_grad]

	def infer_shape(self, node, input_shapes):
		x = input_shapes[0][0]
		y = input_shapes[1][1]
		if (node.matmul_attr_trans_A):
			x = input_shapes[0][1]
		if (node.matmul_attr_trans_B):
			y = input_shapes[1][0]
		return (x, y)


class PlaceholderOp(Op):
	def __call__(self, dtype, shape = None, name = None):
		"""Creates a variable node."""
		new_node = Op.__call__(self)
		new_node.dtype = dtype
		new_node.shape = shape
		new_node.name = "placeholder(%s)" % name
		return new_node

	def compute(self, node, input_vals):
		assert False, "placeholder %s values provided by feed_dict" % node.name

	def gradient(self, node, output_grad):
		return None

	def infer_shape(self, node, input_shapes):
		assert False, "placeholder %s shape provided by feed_shape" % node.name


class ZerosLikeOp(Op):
	def __call__(self, node_A):
		"""Creates a node that represents np.zeros(node_A.shape)."""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "Zeroslike(%s)" % node_A.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.zeros(input_vals[0].shape)

	def gradient(self, node, output_grad):
		return [zeroslike(node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		return input_shapes[0]


class ZerosOp(Op):
	def __call__(self, shape, dtype = float32, name = None):
		new_node = Op.__call__(self)
		new_node.name = "zeros %s" % name
		new_node.shape = tuple(shape)
		new_node.dtype = dtype
		return new_node

	def compute(self, node, input_vals):
		return np.zeros(node.shape, node.dtype)

	def gradient(self, node, output_grad):
		return np.zeros(node.shape, node.dtype)

	def infer_shape(self, node, input_shapes):
		if len(node.shape) == 0:
			return (1,)
		return node.shape


class OnesLikeOp(Op):
	def __call__(self, node_A):
		"""Creates a node that represents np.ones(node_A.shape)."""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.name = "Oneslike(%s)" % node_A.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.ones(input_vals[0].shape)

	def gradient(self, node, output_grad):
		return [zeroslike(node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		return input_shapes[0]


class RandomNormalOp(Op):
	def __call__(self, shape, mean = 0.0, stddev = 1.0, dtype = float32, name = None):
		new_node = Op.__call__(self)
		new_node.name = "zeros %s" % name
		new_node.shape = shape
		new_node.dtype = dtype
		new_node.mean = mean
		new_node.stddev = stddev
		return new_node

	def compute(self, node, input_vals):
		return node.dtype().exchange(np.random.normal(node.mean, node.stddev, node.shape))

	def gradient(self, node, output_grad):
		return [zeroslike(node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		if len(node.shape) == 0:
			return (1,)
		return node.shape


class ReduceSum_Op(Op):
	def __call__(self, node_A, node_B, name = None):
		"""Creates a node that represents sum(node_A) to shape node_B.shape"""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "ReduceSum(%s, %s)" % (node_A, node_B)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		ans = input_vals[0]
		while len(ans.shape) > len(input_vals[1].shape):
			ans = np.sum(ans, axis = 0)
		for dim in range(len(ans.shape)):
			if ans.shape[dim] > input_vals[1].shape[dim]:
				assert input_vals[1].shape[dim] == 1
				ans = np.sum(ans, axis = dim, keepdims = True)
		return ans

	def gradient(self, node, output_grad):
		return [broadcastto(output_grad, node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		"""summation reduction axis = 0
		e.g. (3,4,5)->(4,5)
		for vector, simpler to do (3,)->(1,)
		"""
		return input_shapes[1]


class ReduceSumOp(Op):
	def __call__(self, input, axis = None, keep_dims = False, name = None, reduction_indices = None):
		"""Creates a node that represents sum(node_A) to shape node_B.shape"""
		assert axis == None or reduction_indices == None
		new_node = Op.__call__(self)
		if isinstance(axis, int):
			axis = [axis]
		if reduction_indices != None:
			axis = reduction_indices
		new_node.inputs = [input]
		new_node.axis = axis
		new_node.keep_dims = keep_dims
		new_node.name = "ReduceSum(%s, name = %s)" % (input, name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		if isinstance(node.axis, list):
			axis = tuple(node.axis)
		else:
			axis = node.axis
		return np.sum(input_vals[0], axis = axis, keepdims = node.keep_dims)

	def gradient(self, node, output_grad):
		return [broadcastto(output_grad, node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		"""summation reduction axis = 0
		e.g. (3,4,5)->(4,5)
		for vector, simpler to do (3,)->(1,)
		"""
		assert len(input_shapes) == 1
		t = np.ones(input_shapes[0])
		if isinstance(node.axis, list):
			axis = tuple(node.axis)
		else:
			axis = node.axis
		t = np.sum(t, axis = axis, keepdims = node.keep_dims)
		ans = t.shape
		if ans == ():
			return (1,)
		return ans


class ReduceMeanOp(Op):
	def __call__(self, input, axis = None, keep_dims = False, name = None, reduction_indices = None):
		"""Creates a node that represents sum(node_A) to shape node_B.shape"""
		assert axis == None or reduction_indices == None
		new_node = Op.__call__(self)
		if isinstance(axis, int):
			axis = [axis]
		if reduction_indices != None:
			axis = reduction_indices
		new_node.inputs = [input]
		new_node.axis = axis
		new_node.keep_dims = keep_dims
		new_node.name = "ReduceMean(%s, name = %s)" % (input, name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		if isinstance(node.axis, list):
			axis = tuple(node.axis)
		else:
			axis = node.axis
		return np.mean(input_vals[0], axis = axis, keepdims = node.keep_dims)

	def gradient(self, node, output_grad):
		W = reduce_sum(ones_like(node.inputs[0]), axis = node.axis, keep_dims = node.keep_dims)
		return [broadcastto(output_grad, node.inputs[0]) / W]

	def infer_shape(self, node, input_shapes):
		"""summation reduction axis = 0
		e.g. (3,4,5)->(4,5)
		for vector, simpler to do (3,)->(1,)
		"""
		assert len(input_shapes) == 1
		t = np.ones(input_shapes[0])
		if isinstance(node.axis, list):
			axis = tuple(node.axis)
		else:
			axis = node.axis
		t = np.sum(t, axis = axis, keepdims = node.keep_dims)
		ans = t.shape
		if ans == ():
			return (1,)
		return ans


class BroadcastToOp(Op):
	def __call__(self, node_A, node_B):
		"""Creates a node that represents np.broadcast_to(node_A, node_B.shape).
		Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
		"""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "BroadcastTo(%s, %s.shape)" % (node_A.name, node_B.name)
		return new_node

	def compute(self, node, input_vals):
		assert (len(input_vals) == 2)
		tmp = input_vals[0]
		# not complete yet
		if len(tmp.shape) < len(input_vals[1].shape):
			front_align = True
			for dim, in_size in enumerate(tmp.shape):
				if input_vals[1].shape[dim] != in_size:
					front_align = False
					break
			new_shape = tmp.shape
			if front_align:
				while len(new_shape) < len(input_vals[1].shape):
					new_shape = new_shape + (1,)
			tmp.resize(new_shape)
		return np.broadcast_to(tmp, input_vals[1].shape)

	def gradient(self, node, output_grad):
		assert False, "undo"
		grad_A = reduce_sum_op(output_grad, node.input[0])
		grad_B = zeros_like(node.inputs[1])
		return [grad_A, grad_B]

	def infer_shape(self, node, input_shapes):
		return input_shapes[1]


class Initializer(Op):
	def __call__(self):
		new_node = Op.__call__(self)
		new_node.inputs = init_assigns
		new_node.name = "global variables initializer"
		return new_node

	def compute(self, node, input_vals):
		return None

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		return (1,)


class AssignOp(Op):
	def __call__(self, var, input):
		new_node = Op.__call__(self)
		new_node.var = var
		if not isinstance(input, Node):
			node_t = constant(input)
			new_node.inputs = [node_t]
			new_node.name = "assign(%s, %s)" % (var.name, node_t.name)
		else:
			new_node.inputs = [input]
			new_node.name = "assign(%s, %s)" % (var.name, input)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		node.var.value = input_vals[0]
		return node.var.value

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		node.var.shape = input_shapes[0]
		return input_shapes[0]


class VariableOp(Op):
	def __call__(self, initial_value, name = None, dtype = None):
		new_node = Op.__call__(self)
		if dtype == None:
			new_node.dtype = float32
		else:
			new_node.dtype = dtype
		new_node.name = "Variable(%s)" % name
		new_node.value = None
		assign_node = assign(new_node, initial_value)
		init_assigns.append(assign_node)
		all_var.append(new_node)
		return new_node

	def compute(self, node, input_vals):
		return node.dtype().exchange(node.value)

	def gradient(self, node, output_grad):
		return None

	def infer_shape(self, node, input_shapes):
		return node.shape


class ConstantOp(Op):
	def __call__(self, value, dtype = None, shape = None, name = None):
		new_node = Op.__call__(self)
		new_node.name = "Constant %s" % name
		if shape != None:
			new_node.value = np.full(shape, value, dtype)
			new_node.shape = shape
		else:
			new_node.value = np.array(value)
			new_node.shape = new_node.value.shape
		return new_node

	def compute(self, node, input_vals):
		return node.value

	def gradient(self, node, output_grad):
		return [0, ]

	def infer_shape(self, node, input_shapes):
		if len(node.shape) == 0:
			return (1,)
		return node.shape


class ExpOp(Op):
	def __call__(self, input):
		new_node = Op.__call__(self)
		if not isinstance(input, Node):
			input = constant(input)
		new_node.name = "exp(%s)" % (input.name)
		new_node.inputs = [input]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.exp(input_vals[0])

	def gradient(self, node, output_grad):
		return [output_grad * exp(node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 1
		return input_shapes[0]


class SqrtOp(Op):
	def __call__(self, input):
		new_node = Op.__call__(self)
		if not isinstance(input, Node):
			input = constant(input)
		new_node.name = "sqrt(%s)" % (input.name)
		new_node.inputs = [input]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.sqrt(input_vals[0])

	def gradient(self, node, output_grad):
		return [output_grad / sqrt(node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 1
		return input_shapes[0]


class PowOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		if not isinstance(node_B, Node):
			node_B = constant(node_B)
		if not isinstance(node_A, Node):
			node_A = constant(node_A)
		new_node.name = "(%s ^ %s)" % (node_A.name, node_B.name)
		new_node.inputs = [node_A, node_B]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return np.power(input_vals[0], input_vals[1])

	def gradient(self, node, output_grad):
		a = node.inputs[0]
		b = node.inputs[1]
		return [reduce_sum_op(output_grad * b * pow(a, b - 1), a),
				reduce_sum_op(output_grad * pow(a, b) * log(a), node.inputs[1])]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 2
		return broadcast_rule(input_shapes[0], input_shapes[1])


class LogOp(Op):
	def __call__(self, input):
		new_node = Op.__call__(self)
		if not isinstance(input, Node):
			input = constant(input)
		new_node.name = "log(%s)" % (input)
		new_node.inputs = [input]
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.log(input_vals[0])

	def gradient(self, node, output_grad):
		return [output_grad / node.inputs[0]]

	def infer_shape(self, node, input_shapes):
		"""Need to handle input_vals[0].shape != input_vals[1].shape"""
		assert len(input_shapes) == 1
		return input_shapes[0]


class EqualOp(Op):
	def __call__(self, node_A, node_B):
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "Equal(%s)" % (node_A.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return np.equal(input_vals[0], input_vals[1])

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		assert len(input_shapes) == 2
		return broadcast_rule(input_shapes[0], input_shapes[1])


class ArgMaxOp(Op):
	def __call__(self, input, axis = None, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [input]
		new_node.axis = axis
		new_node.name = "ArgMax(%s)" % (input.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.argmax(input_vals[0], axis = node.axis)

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		assert len(input_shapes) == 1
		t = np.ones(input_shapes[0])
		if isinstance(node.axis, list):
			axis = tuple(node.axis)
		else:
			axis = node.axis
		t = np.sum(t, axis = axis)
		ans = t.shape
		if ans == ():
			return (1,)
		return ans


class CastOp(Op):
	def __call__(self, input, dtype = None, name = None):
		new_node = Op.__call__(self)
		new_node.inputs = [input]
		if dtype == "float":
			new_node.dtype = float32
		else:
			new_node.dtype = dtype
		new_node.name = "ArgMax(%s)" % (input.name)
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		if node.dtype != None:
			return node.dtype().exchange(input_vals[0])
		else:
			return input_vals[0]

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		assert len(input_shapes) == 1
		return input_shapes[0]


class RunnerOp(Op):
	def __call__(self, input_list):
		new_node = Op.__call__(self)
		new_node.inputs = input_list
		new_node.name = "initializer"
		return new_node

	def compute(self, node, input_vals):
		return None

	def gradient(self, node, output_grad):
		raise NotImplementedError

	def infer_shape(self, node, input_shapes):
		return (1,)


class ReshapeOp(Op):
	def __call__(self, node_A, shape, name = None):
		"""Creates a node that represents np.zeros(node_A.shape)."""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A]
		new_node.shape = tuple(shape)
		new_node.name = "Reshape(%s)" % node_A.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 1
		return np.reshape(input_vals[0], node.shape)

	def gradient(self, node, output_grad):
		return [reshape_to(output_grad, node.inputs[0])]

	def infer_shape(self, node, input_shapes):
		assert False, "Wrong"
		return node.shape


class ReshapeToOp(Op):
	def __call__(self, node_A, node_B, name = None):
		"""Creates a node that represents np.zeros(node_A.shape)."""
		new_node = Op.__call__(self)
		new_node.inputs = [node_A, node_B]
		new_node.name = "ReshapeTo(%s)" % node_A.name
		return new_node

	def compute(self, node, input_vals):
		assert len(input_vals) == 2
		return np.reshape(input_vals[0], input_vals[1].shape)

	def gradient(self, node, output_grad):
		return [reshape_to(output_grad, node.inputs[0].shape)]

	def infer_shape(self, node, input_shapes):
		assert False, "Wrong"
		return node.shape


# Create global singletons of operators.
add = AddOp()
sub = SubOp()
mul = MulOp()
neg = NegOp()
div = DivOp()
matmul = MatMulOp()
placeholder = PlaceholderOp()
ones_like = OnesLikeOp()
zeros_like = ZerosLikeOp()
zeros = ZerosOp()
random_normal = RandomNormalOp()
reduce_sum_op = ReduceSum_Op()
reduce_sum = ReduceSumOp()
reduce_mean = ReduceMeanOp()
broadcastto = BroadcastToOp()
global_variables_initializer = Initializer()
Variable = VariableOp()
assign = AssignOp()
constant = ConstantOp()
exp = ExpOp()
log = LogOp()
equal = EqualOp()
argmax = ArgMaxOp()
cast = CastOp()
runner = RunnerOp()
sqrt = SqrtOp()
pow = PowOp()
reshape = ReshapeOp()
reshape_to = ReshapeToOp()


def gradients(output_node, node_list):
	"""Take gradient of output node with respect to each node in node_list.

	Parameters
	----------
	output_node: output node that we are taking derivative of.
	node_list: list of nodes that we are taking derivative wrt.

	Returns
	-------
	A list of gradient values, one for each node in node_list respectively.

	"""
	node_to_output_grads_list = {}
	node_to_output_grads_list[output_node] = [ones_like(output_node)]
	node_to_output_grad = {}
	# Traverse forward graph in reverse topological order
	reverse_topo_order = reversed(find_topo_sort([output_node]))
	for node in reverse_topo_order:
		output_grad = sum_node_list(node_to_output_grads_list[node])
		node_to_output_grad[node] = output_grad
		input_grads_list = node.op.gradient(node, output_grad)
		for i in range(len(node.inputs)):
			if node.inputs[i] not in node_to_output_grads_list:
				node_to_output_grads_list[node.inputs[i]] = []
			# Calculate partial adjoint for input nodes.
			node_to_output_grads_list[node.inputs[i]].append(input_grads_list[i])

	grad_node_list = [node_to_output_grad.get(node, 0) for node in node_list]
	return grad_node_list


##################
# Helper Methods #
##################

def find_topo_sort(node_list):
	"""Given a list of nodes, return a topo ordering of nodes ending in them.

	A simple algorithm is to do a post-order DFS traversal on the given nodes,
	going backwards based on input edges. Since a node is added to the ordering
	after all its predecessors are traversed due to post-order DFS, we get a
	topological sort.

	"""
	visited = set()
	topo_order = []
	for node in node_list:
		topo_sort_dfs(node, visited, topo_order)
	return topo_order


def topo_sort_dfs(node, visited, topo_order):
	"""Post-order DFS"""
	if node in visited:
		return
	visited.add(node)
	for n in node.inputs:
		topo_sort_dfs(n, visited, topo_order)
	topo_order.append(node)


def sum_node_list(node_list):
	"""Custom sum func to avoid creating redundant nodes in Python sum func."""
	from operator import add
	from functools import reduce
	return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
	"""Return output shape of broadcast shape_a, shape_b.
	e.g. broadcast_rule((3,2), (4,3,2))
	returns output_shape = (4,3,2)

	Check out explanations and more examples at
	https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
	http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
	"""
	assert (isinstance(shape_a, tuple))
	assert (isinstance(shape_b, tuple))
	if len(shape_a) > len(shape_b):
		longer_shape, shorter_shape = shape_a, shape_b
	else:
		longer_shape, shorter_shape = shape_b, shape_a
	len_diff = len(longer_shape) - len(shorter_shape)
	for i in range(len_diff):
		# pad with leading 1s
		shorter_shape = (1,) + shorter_shape
	assert len(shorter_shape) == len(longer_shape)
	output_shape = list(longer_shape)
	for i in range(len(output_shape)):
		assert (shorter_shape[i] == longer_shape[i]) \
			   or (shorter_shape[i] == 1) \
			   or (longer_shape[i] == 1)
		output_shape[i] = max(shorter_shape[i], longer_shape[i])
	return tuple(output_shape)
