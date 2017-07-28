from .ops import *

def softmax(logits, dim = -1, name = None):
	t = exp(logits)
	k = reduce_sum(t, axis = dim, keep_dims = True)
	return t / k