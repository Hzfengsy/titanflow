import numpy as np
# from . import ndarray, gpu_op
from .ops import *

# f = open('log.txt', 'w')
class Session(object):
	def __init__(self, ctx = None):
		"""
		Parameters
		----------
		eval_node_list: list of nodes whose values need to be computed.
		ctx: runtime DLContext, default is None which means np.ndarray on cpu
		topo_order: list of nodes in topological order
		node_to_shape_map: dict from node to shape of the node
		node_to_arr_map: dict from node to ndarray.NDArray allocated for node
		feed_shapes: shapes of feed_dict from last run(...)
		"""
		self.ctx = ctx
		self.last_eval = []
		self.node_to_shape_map = None
		self.node_to_arr_map = None
		self.feed_shapes = None

	def __enter__(self):
		return self

	def __exit__(self, Type, value, traceback):
		pass

	def infer_shape(self, feed_shapes):
		"""Given shapes of feed_dict nodes, infer shape for all nodes in graph.

		Implementation note:
		Iteratively calls node.op.infer_shape to infer shapes.
		Node shapes stored in self.node_to_shape_map.

		Parameters
		----------
		feed_shapes: node->shapes mapping for feed_dict nodes.
		"""
		node_to_shape_map = {}
		for node in self.topo_order:
			if node in feed_shapes:
				# Skip placeholder nodes. Values already provided by feed_dict.
				node_to_shape_map[node] = feed_shapes[node]
				node.ishape = feed_shapes[node]
				continue
			input_shapes = [node_to_shape_map[n] for n in node.inputs]
			node_to_shape_map[node] = node.op.infer_shape(node, input_shapes)
			node.ishape = node_to_shape_map[node]
		self.node_to_shape_map = node_to_shape_map

	def memory_plan(self, feed_shapes):
		"""Allocates ndarray.NDArray for every node except feed_dict nodes.

		Implementation note:
		Option 1: Alloc a ndarray.NDArray per node that persists across run()
		Option 2: Implement a memory pool to reuse memory for nodes of same
				shapes. More details see Lecture 7.

		For both options, self.node_to_arr_map stores node->NDArray mapping to
		allow mapping to persist across multiple executor.run().

		Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

		Parameters
		----------
		feed_shapes: node->shapes mapping for feed_dict nodes.
		"""
		self.node_to_arr_map = {}
		ctx = ndarray.gpu(0)
		for node in self.topo_order:
			if node in feed_shapes:
				# Skip placeholder nodes. Values already provided by feed_dict.
				# self.node_to_arr_map[node] = ndarray.array(feed_shapes[node], ctx = ctx)
				continue
			# self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx = ctx)

	def run(self, eval_node_list, feed_dict = {}, convert_to_numpy_ret_vals = False):
		"""
		Parameters
		----------
		feed_dict: a dictionary of node->np.ndarray supplied by user.
		convert_to_numpy_ret_vals: whether to convert ret vals to np.array

		Returns
		-------
		A list of values for nodes in eval_node_list. NDArray or np.ndarray.
		"""
		if isinstance(eval_node_list, list):
			self.eval_node_list = eval_node_list
		else:
			self.eval_node_list = [eval_node_list]
		self.topo_order = find_topo_sort(self.eval_node_list)

		def are_feed_shapes_equal(sa, sb):
			if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
				return False
			unmatched_item = set(sa.items()) ^ set(sb.items())
			return len(unmatched_item) == 0

		# Assume self.ctx is None implies numpy array and numpy ops.
		use_numpy = self.ctx is None
		node_to_val_map = {}
		for node, value in feed_dict.items():
			if use_numpy:
				# all values passed in feed_dict must be np.ndarray
				assert isinstance(node.op, PlaceholderOp)
				if not isinstance(value, np.ndarray):
					value = np.array(value)
				node_to_val_map[node] = value
				if node.dtype != None:
					node_to_val_map[node] = node.dtype().exchange(node_to_val_map[node])
			else:
				# convert values to ndarray.NDArray if necessary
				if isinstance(value, np.ndarray):
					node_to_val_map[node] = ndarray.array(value, ctx = self.ctx)
				elif isinstance(value, ndarray.NDArray):
					node_to_val_map[node] = value
				else:
					assert False, "feed_dict value type not supported"
		
		# collect shapes for all placeholders
		feed_shapes = {}
		for node in node_to_val_map:
			if len(node_to_val_map[node].shape) == 0:
				feed_shapes[node] = (1, )
			else:
				feed_shapes[node] = node_to_val_map[node].shape
			# if node.shape != None:
			# 	print node.shape
			# 	print feed_shapes[node]
			# 	assert feed_shapes[node] == node.shape
			

		# infer shape if feed_shapes changed since last run
		# e.g. call run() on test data after trainng
		if (not self.last_eval == self.eval_node_list) | (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
			self.infer_shape(feed_shapes)
			self.feed_shapes = feed_shapes
			# plan memory if using GPU
			if (not use_numpy):
				self.memory_plan(feed_shapes)
		# Traverse graph in topo order and compute values for all nodes.
		for node in self.topo_order:
			if node in node_to_val_map:
				# Skip placeholder nodes. Values already provided by feed_dict.
				continue
			input_vals = [node_to_val_map[n] for n in node.inputs]
			if use_numpy:
				node_val = np.empty(shape = self.node_to_shape_map[node])
			else:
				node_val = self.node_to_arr_map[node]
			# node_val is modified in-place whether np.ndarray or NDArray
			# f.write(node.name + '\n')
			node.op.compute(node, input_vals, node_val, use_numpy)
			node_to_val_map[node] = node_val
			# np.savetxt(f, node_val)
			# f.write('\n')

		self.last_eval = self.eval_node_list
		# Collect node values.
		if not use_numpy and convert_to_numpy_ret_vals:
			return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
		if isinstance(eval_node_list, list):
			return [node_to_val_map[n] for n in self.eval_node_list]
		else:
			return node_to_val_map[eval_node_list]


