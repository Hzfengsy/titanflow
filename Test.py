import multiprocessing as mp
import numpy as np


def f(x):
	ans[x] = np.dot(arr[x], t)


y = 2
shared_arr, shared_t, shared_ans = None, None, None
arr, t, ans = None, None, None

for i in range(100):
	shared_arr, shared_t, shared_ans = None, None, None
	arr, t, ans = None, None, None
	shared_arr = mp.Array("f", 10000000)
	shared_t = mp.Array("f", 10000000)
	shared_ans = mp.Array("f", 10000)
	arr = np.frombuffer(shared_arr.get_obj(), dtype = np.float32).reshape(100, 100000)
	arr[:] = np.random.random((100, 100000))
	t = np.frombuffer(shared_arr.get_obj(), dtype = np.float32).reshape(100000, 100)
	t[:] = arr.T
	ans = np.frombuffer(shared_ans.get_obj(), dtype = np.float32).reshape(100, 100)
	arr = np.frombuffer(shared_arr.get_obj(), dtype = np.float32).reshape(100, 100000)
	arr[:] = np.random.random((100, 100000))
	pool = mp.Pool(processes = 1)
	pool.map(f, range(4))
	pool = None

print ans


