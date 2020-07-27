from mnnbox import *
import matplotlib.pyplot as plt

# test 1
'''
X, W = Input('X'), Input('W')
t = Input('target')

f = Combine(X, W)
cost = MSE(f, t)

X_ = np.array([[1., 2.], [3., 4.]])
W_ = np.array([[1., 1.], [1, 2]])
t_ = np.array([[1., 2.], [3., 4.]])




feed_dict = {X: X_, W: W_, t: t_}

graph = topological_sort(feed_dict)
forward_and_backward(graph)
gradients = [l.gradients[l] for l in [X, t, W]]

print('\n\n', gradients)
'''

# test 2
'''
y, a = Input('y'), Input('a')
cost = MSE(y, a)

y_ = np.array([[1, 4, 9], [1, 2, 3]])
a_ = np.array([[4.5, 5, 10], [2, 3, 1]])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
# forward pass
# forward_pass_mse(graph)
forward_pass(cost, graph)
'''


# test 3

X = Input('input')
layer = [X]
weight = []
hidd =5
W = [Variable('weight_{}'.format(i)) for i in range(hidd)]
for i in range(hidd):
    combine = Combine(layer[i], W[i])
    activate = Activate(combine)
    layer.append(activate)
target = Input('target')
cost = MSE(layer[-1], target)

u_input = np.ones(100) * 0.1
s_input = np.ones(100) * 0.447
X_ = np.stack([u_input, s_input], axis=1)
W_ = np.ones((100, 100)) * 0.5
target_ = np.stack([u_input, s_input], axis=1)
feed_dict = {X: X_, target: target_}
for i in range(hidd):
    feed_dict[W[i]] = W_
graph = topological_sort(feed_dict)

train_ables = W

total_loss = []
for i in range(20):
    forward_and_backward(graph)
    sgd_update(train_ables)
    loss = graph[-1].value
    total_loss.append(loss)

print('\n\n====================Finish====================================')
plt.figure()
plt.plot(range(len(total_loss)), total_loss)
plt.show()


# test gradient
'''
X = Input()
W = Variable()
X_ = np.array([[0.1, 0.1], [0.08, 0.08], [0.07, 0.07]])
W_ = np.array([[0.5, 0.5, 0.5]])
tar_ = np.array([[0.1, 0.1]])
res = Combine(X, W)
tar = Input('target')
cost = MSE(res, tar)
feed_dict = {X: X_, W: W_, tar: tar_}
graph = topological_sort(feed_dict)
forward_and_backward(graph)
cost_1 = cost.value
print('the gradient of X:', X.gradients[X])

X_[0, 0] = X_[0,0] + 1e-3
X.value = X_
cost_2 = forward_pass(cost, graph)
grad_check = (cost_2 - cost_1) / (1e-3)
print('\n=====================\ngradient for check', grad_check)
'''


