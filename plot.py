from gene.dataset import *
import matplotlib.pyplot as plt


data, g_names, s_names = read_data(file='C:/Users/tianping/Desktop/last.csv')
x, y = preprocess(data, one_hot=False, nn=True)
plt.imshow(x[0].reshape([27, 27]), cmap='gray')
plt.show()
