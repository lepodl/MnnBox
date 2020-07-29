import numpy as np


# w = np.ones((5, 5))
# s = np.linspace(1., 10, 5)
# r = np.eye(5)
# inv = 1 / (2 * s)
# ratio = 0.4
#
# s_w = np.einsum('k,ki,i,j,ij->kj', inv, w, s, s, r) * 2 * (1 + ratio ** 2)
#
# temp = np.multiply(np.tensordot(s, s, axes=0), r)
# res = (2 * np.matmul(w, temp)) * (1 / (2 * s[:, None])) * (1 + ratio ** 2)
#
# print(s_w, '\n')
# print(res)

# dim =2
# batch_size = 4
# rou = np.eye(dim) * 0.9 + np.ones(dim) * 0.1
# rou = np.expand_dims(rou, 0).repeat(batch_size, axis=0)
# print(rou)

x = np.arange(16).reshape((2,4,2))
print(x)
y = x.flatten()
print(y)