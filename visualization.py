from gene.nn import conv2d, max_pool, dense
from vis.visualization import visualize_activation, visualize_saliency, visualize_cam
from vis.utils import utils
from keras.models import Sequential
from keras.layers import Flatten, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from gene.stat import *
from sklearn import manifold


class Visualization:
    """
    Visualize the input images.
    """
    def __init__(self, keep_prop=1.0, lr=1e-4):
        self.rate = 1.0 - keep_prop
        self.lr = lr

        self.model = None

    def build_network(self):
        self.model = Sequential()
        # conv_pool1
        self.model.add(conv2d(filters=12, kernel_size=(2, 4), padding='same', input_shape=(182, 4, 1),
                       vis=True, name='conv1_1'))
        # self.model.add(BatchNormalization(name='bn1_1'))
        self.model.add(Activation('relu'))
        self.model.add(max_pool(padding='same', vis=True))
        # reshape
        self.model.add(Flatten())
        # dense
        self.model.add(Dropout(rate=self.rate))
        self.model.add(dense(units=3, name='fc3_1', vis=True))
        self.model.add(dense(units=2, activation=None, vis=True, name='preds'))
        self.model.add(Activation('softmax'))
        # optimize
        self.model.compile(optimizer=Adam(lr=self.lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def set_weights(self, file='C:/Users/tianping/Desktop/tensorflow_data.npy'):
        tf_data = np.load(file).item()
        self.fix_one_layer(tf_data, 'conv_pool1/cov1_1/kernel:0', 'conv_pool1/cov1_1/bias:0', 'conv1_1')
        self.fix_one_layer(tf_data, 'dense/fc3_1/kernel:0', 'dense/fc3_1/bias:0', 'fc3_1')
        self.fix_one_layer(tf_data, 'dense/logits/kernel:0', 'dense/logits/bias:0', 'preds')

    def fix_one_layer(self, tf_data, w_name, b_name, layer_name):
        kernel_w = tf_data[w_name]
        kernel_b = tf_data[b_name]
        layer = self.model.get_layer(layer_name)
        layer.trainable = False
        layer.set_weights([kernel_w, kernel_b])

    def vis_activation(self, filter_index, layer_index='default', tv_weight=10):
        if layer_index == 'default':
            layer_index = utils.find_layer_idx(self.model, 'preds')
        img = visualize_activation(model=self.model,
                                   layer_idx=layer_index,
                                   filter_indices=filter_index,
                                   input_range=(0.0, 1.0),
                                   tv_weight=tv_weight)
        plt.figure()
        plt.subplot(121)
        plt.imshow(img[..., 0], cmap='jet')
        plt.subplot(122)
        plt.imshow(img[..., 0], cmap='gray')

        plt.show()

    def vis_saliency(self, filter_index, seed_input, grad_modifier='absolute', layer_index='default', plot=False):
        """
        Args:
            filter_index: The label (1 or 0) which represents "S" or "R".
            seed_input: The image with shape of (27, 27 ,1).
            grad_modifier: Default value is "absolute", it can be changed by "relu" or "negate".
            layer_index: If "default", it will choose "preds" layer, or you can pass an integer less than 5.
            plot: If True, it will plot a figure of class saliency map.(Default is False)

        Returns:
            An numpy.array with the shape of 283*24*3 of a RGB image.
        """
        if layer_index == 'default':
            layer_index = utils.find_layer_idx(self.model, 'preds')
        img = visualize_saliency(self.model,
                                 layer_idx=layer_index,
                                 seed_input=seed_input,
                                 filter_indices=filter_index,
                                 grad_modifier=grad_modifier)
        if plot:
            plt.figure()
            plt.imshow(img, cmap='jet')
            plt.show()

        return img

    def vis_cam(self, filter_index, seed_input, layer_index='default', plot=False):
        if layer_index == 'default':
            layer_index = utils.find_layer_idx(self.model, 'preds')
        img = visualize_cam(self.model,
                            layer_idx=layer_index,
                            filter_indices=filter_index,
                            seed_input=seed_input)
        if plot:
            plt.figure()
            plt.imshow(img, cmap='jet')
            plt.show()
        return img


def vis_filters(file='C:/Users/tianping/Desktop/tensorflow_data.npy', plot=True):
    tf_data = np.load(file).item()
    filters = tf_data['conv_pool1/cov1_1/kernel:0']
    height, width = filters.shape[:2]
    col = 4         # It can be set by yourself.
    row = filters.shape[-1] // col
    pad = 2         # It can also be set by yourself.
    img = np.zeros(shape=[height*row + (row-1)*pad, width*col + (col-1)*pad], dtype=filters.dtype)
    if plot:
        yy = 0
        num = 0
        for i in range(row):
            xx = 0
            for j in range(col):
                img[yy:(yy+height), xx:(xx+width)] = filters[..., 0, num]
                xx += width + pad
                num += 1
            yy += height + pad
        plt.imshow(img, cmap='gray')
        plt.show()
    return filters


def vis_saliency_part(x, y, coord, num=5):
    r = x[y[:, 0] == 1]
    s = x[y[:, 1] == 1]
    r_indices = np.random.choice(r.shape[0], num)
    s_indices = np.random.choice(s.shape[0], num)
    r_ = r[r_indices]
    s_ = s[s_indices]
    height = coord[1] - coord[0]
    width = coord[3] - coord[2]
    img = np.zeros([height*num + 3*(num-1), width*2 + 3])
    for i in range(num):
        for r_i, s_i in zip(r_, s_):
            img[(height*i + 3*i):(height*i+3*i + height), :width] = r_i[coord[0]:coord[1], coord[2]:coord[3], 0]
            img[(height*i + 3*i):(height*i+3*i + height), (width+3):] = s_i[coord[0]:coord[1], coord[2]:coord[3], 0]
    return img


def vis_fc(sess, layer, feed_dict, y, plot=True):
    x = sess.run(layer, feed_dict=feed_dict)
    tsne = manifold.TSNE()
    x_two_d = tsne.fit_transform(x)
    if plot:
        plt.figure()
        plt.scatter(x_two_d[y[:, 0] == 1, 0], x_two_d[y[:, 0] == 1, 1], c='red', label='R')
        plt.scatter(x_two_d[y[:, 1] == 1, 0], x_two_d[y[:, 1] == 1, 1], c='green', label='S')
        plt.legend()
        plt.show()


def plot_tsne(net, sess, x, y):
    feed_dict = {net.x: x, net.keep_prob: 1.0}
    high_d = sess.run(net.fc3_1, feed_dict=feed_dict)
    tsne = manifold.TSNE()
    low_d = tsne.fit_transform(high_d)
    plt.figure('T-sne')
    plt.plot(low_d[y[:, 0] == 1, 0], low_d[y[:, 0] == 1, 1], 'r.', label='R')
    plt.plot(low_d[y[:, 1] == 1, 0], low_d[y[:, 1] == 1, 1], 'b*', label='S')
    plt.legend()


def plot_train_curve(train, val, name):
    plt.figure(name)
    plt.plot(train, 'r', label='train')
    plt.plot(val, 'b', label='validation')
    plt.legend()
