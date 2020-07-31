from mnnbox import *
import matplotlib.pyplot as plt
import unittest

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

class Test_MnnBox(unittest.TestCase):
    def _test_gradient_bn(self):
        x = Input()
        gamma = Variable('gamma')
        beta = Variable('beta')
        target = Input('target')
        u_input = np.random.uniform(size=(4, 100))
        s_input = np.random.uniform(size=(4, 100))
        x_ = np.stack([u_input, s_input], axis=-1)
        gamma_ = np.array([0.5, 0.5])
        beta_ = np.ones((100, 2)) * 2.
        target_ = np.stack([u_input, s_input], axis=-1)
        feed_dict = {x: x_, gamma: gamma_, beta: beta_, target: target_}
        out = BatchNormalization(1, x, gamma, beta)
        cost = MSE(out, target)
        graph = topological_sort(feed_dict)
        forward_and_backward(graph)
        loss = cost.value
        # print('\n---------------->the gradient of x\n', out.gradients[x][1, 1, 0])
        # print('\n---------------->the gradient of gamma\n', gamma.gradients[gamma][1])
        print('\n---------------->the gradient of beta\n', beta.gradients[beta][1, 0])

        # x_[1, 1, 0] = x_[1, 1, 0] + 0.0001
        # x.value = x_
        # gamma_[1] = gamma_[1] + 0.0001
        # gamma.value = gamma_
        beta_[1, 0] = beta_[1, 0] + 0.0001
        loss_ = forward_pass(cost, graph)
        grad = (loss_ - loss) / 0.0001
        # print('\n---------------->validate the gradient of x\n', grad)
        # print('\n---------------->validate the gradient of gamma\n', grad)
        print('\n---------------->validate the gradient of beta\n', grad)

    def _test_gradient_of_act(self):
        X = Input()
        W = Variable()
        u_input = np.ones((4, 100)) * 2.3
        s_input = np.ones((4, 100)) * 1.
        X_ = np.stack([u_input, s_input], axis=-1)
        u_output = np.ones((4, 100)) * 0.1
        s_output = np.ones((4, 100)) * 0.44
        target_ = np.stack([u_output, s_output], axis=-1)
        tar = Input('target')
        res = Activate(X)
        cost = MSE(res, tar)
        feed_dict = {X: X_, tar: target_}
        graph = topological_sort(feed_dict)
        forward_and_backward(graph)
        cost_1 = cost.value
        print('the gradient of X:', X.gradients[X][0, 0, 0])

        X_[0, 0, 0] = X_[0, 0, 0] + 1e-3
        X.value = X_
        cost_2 = forward_pass(cost, graph)
        grad_check = (cost_2 - cost_1) / (1e-3)
        print('\n=====================\ngradient for check', grad_check)


    def test_whole_nn(self):
        bs = 4
        X = Input('input')
        layer = [X]
        weight = []
        hidd = 5
        W = [Variable('weight_{}'.format(i)) for i in range(hidd)]
        Gamma = [Variable('gamma_{}'.format(i)) for i in range(hidd)]
        Beta = [Variable('beta_{}'.format(i)) for i in range(hidd)]
        for i in range(hidd):
            combine = Combine(layer[i], W[i])
            bn = BatchNormalization(1, combine, Gamma[i], Beta[i])
            activate = Activate(combine)
            layer.append(activate)
        target = Input('target')
        cost = MSE(layer[-1], target)

        u_input = np.ones((4, 100)) * 0.1
        s_input = np.ones((4, 100)) * 0.447
        X_ = np.stack([u_input, s_input], axis=-1)
        W_ = np.ones((100, 100)) * 0.5
        gamma_ = np.array([0.5, 0.5])
        beta_ = np.ones((100, 2)) * 2.
        target_ = np.stack([u_input, s_input], axis=-1)
        feed_dict = {X: X_, target: target_}
        for i in range(hidd):
            feed_dict[W[i]] = W_
            feed_dict[Gamma[i]] = gamma_
            feed_dict[Beta[i]] = beta_
        graph = topological_sort(feed_dict)

        train_ables = W + Gamma + Beta
        # forward_and_backward(graph)
        # loss = graph[-1].value
        # grad_w0 = W[0].gradients[W[0]][1, 1]
        # print('-------->', grad_w0)
        #
        # W__ = W_
        # W__[1, 1] = W_[1, 1] + 0.001
        # W[0].value = W__
        # loss_ = forward_pass(cost, graph)
        # grad_w0_ = (loss_ - loss) / 0.001
        # print('\n\n------->', grad_w0_)

        total_loss = []
        for i in range(10):
            forward_and_backward(graph)
            sgd_update(train_ables)
            loss = graph[-1].value
            total_loss.append(loss)

        print('\n\n====================Finish====================================')
        plt.figure()
        plt.plot(range(len(total_loss)), total_loss)
        plt.show()



if __name__ == '__main__':
    unittest.main()



