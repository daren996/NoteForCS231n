import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pylab as plt

# cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
# X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
#
# num_training = 500
# mask = list(range(num_training))
# X_train = X_train[mask]
# y_train = y_train[mask]
#
# num_test = 50
# mask = list(range(num_test))
# X_test = X_test[mask]
# y_test = y_test[mask]
#
# X_train = np.reshape(X_train, (X_train.shape[0], -1))
# X_test = np.reshape(X_test, (X_test.shape[0], -1))
# print(X_train.shape, X_test.shape)
#
# from cs231n.classifiers import KNearestNeighbor
#
# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)
#
# num_test = X_test.shape[0]
# num_train = X_train.shape[0]
# dists = np.zeros((num_test, num_train))
# for i in range(num_test):
#     dists[i] = np.sqrt(np.sum(np.square(X_train - X_test[i]), axis=1))
# print(dists.shape)

# print(np.bincount([1, 2, 1, 3, 2, 3, 2]))
# print(np.argmax(np.bincount([1, 2, 1, 3, 2, 3, 2])))

# a = np.array([1,2,3,2,4,3,3,2,3,3])
# print(a==2)  # [False  True False  True False False False  True False False]
# b = np.array([1,2,3,2,4,1,2,2,3,2])
# print(a==b)
# c = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
# d = np.array([[7,8,9],[10,11,12]])
# print(np.concatenate(c[0:2], axis=0))

plt.errorbar([1,2,3], [1,2,3], yerr=[1,2,3])
plt.show()
