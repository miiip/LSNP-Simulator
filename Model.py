import numpy as np
class Model():
    def classification(self, w, b, test_x, test_y):
        output = np.zeros(test_y.shape)
        for i in range(len(test_x)):
            spikes = np.dot(test_x[i, :], w)+b
            index = np.argmax(spikes)
            output[i, index] = 1
        return output
    def accuracy_rate(self, output, y):
        k = 0
        for i in range(len(y)):
            if (y[i, :] == output[i, :]).all():
                k += (y[i, :] == output[i, :])
            else:
                pass
        acc_rate = k/len(y)
        return acc_rate[0]

    def loss(self, output, y):
        Loss = 0
        for i in range(len(y)):
            t_y = np.arange(len(y[i, :]))[(y[i, :] != 0)]
            t_output = np.arange(len(output[i, :]))[(output[i, :] != 0)]
            Loss += (t_y - t_output)**2
        loss = Loss/len(y)
        loss.tolist()
        return loss
