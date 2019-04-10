import numpy as np
from tqdm import tqdm
from Gene_Classification.nn import *
from Gene_Classification.visualization import plot_train_curve, plot_tsne
from Gene_Classification.evaluate import confusion_matrix_and_auc


class CNN:
    """
    Args:
        train_x: Your training data's feature map with shape of (N1, D/6, 6*4, 1).N1 represents
            the number of your training data, D represents your training data's dimension.
        train_y: Your training data's labels with shape of (N1, 2).
        val_x: Your validation data's feature map with shape of (N2, D/6, 6*4, 1).N2 represents
            the number of your validation data, D represents your validation data's dimension.
        val_y: Your validation data's labels with shape of (N2, 2).
        batch_size: The number of training data in one iteration.
        epoch: The number of iteration.
    """
    def __init__(self, train_x, train_y, val_x, val_y, batch_size=32, epoch=200, ratio=1.0):
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_batch = train_x.shape[0] // self.batch_size + 1
        self.train_X = train_x
        self.train_y = train_y
        self.val_X = val_x
        self.val_y = val_y
        self.ratio = ratio

        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []
        self.x, self.y = None, None
        self.lr, self.keep_prob = None, None
        self.outputs, self.loss, self.accuracy = None, None, None
        self.op, self.current_index, self.merged = None, 0, None
        self.fc3_1, self.flatten = None, None

    def build(self):
        h, w, c = self.train_X.shape[1:]
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, h, w, c], name='x')    # black images
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')
        self.lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        with tf.name_scope('inputs'):
            inputs = self.x

        with tf.variable_scope('conv_pool1'):
            conv1_1 = conv2d(inputs, 12, kernel_size=(2, 2), padding='same', activation=None, name='cov1_1')    #12
            # bn1_1 = tf.layers.batch_normalization(conv1_1, name='bn1_1')
            # conv1_2 = conv2d(bn1_1, 12, kernel_size=(2, 2), padding='same', activation=None, name='cov1_2')
            # bn1_2 = tf.layers.batch_normalization(conv1_2)
            relu1_1 = tf.nn.relu(conv1_1)
            max_pool1_1 = max_pool(relu1_1, padding='same', name='max_pool1_1')
            # max_pool1_2 = max_pool(max_pool1_1, padding='same', name='max_pool1_2')
        # with tf.variable_scope('conv_pool2'):
        #     # dropout0 = tf.nn.dropout(max_pool1_1, keep_prob=self.keep_prob, name='dropout0')
        #     conv2_1 = conv2d(max_pool1_1, 16, kernel_size=(2, 2), padding='same', activation=None, name='cov2_1')
        #     bn2_1 = tf.layers.batch_normalization(conv2_1)
        #     relu2_1 = tf.nn.relu(bn2_1)
        #     max_pool2_1 = max_pool(relu2_1, padding='same', name='max_pool2_1')

        with tf.variable_scope('reshape'):
            new_shape = np.ceil(h/2).astype(np.int64)*np.ceil(w/2).astype(np.int64)*12
            self.flatten = tf.reshape(max_pool1_1, [-1, new_shape])

        with tf.variable_scope('dense'):
            dropout = tf.nn.dropout(self.flatten, keep_prob=self.keep_prob, name='dropout')
            self.fc3_1 = dense(dropout, 5, name='fc3_1')  #3

            logits = dense(self.fc3_1, 2, name='logits', activation=None)
            self.outputs = tf.nn.softmax(logits, name='outputs')

        with tf.name_scope('optimize'):
            total_right = tf.cast(tf.equal(tf.argmax(self.y, axis=1),
                                           tf.argmax(self.outputs, axis=1)), dtype=tf.float32)
            self.accuracy = tf.reduce_mean(total_right)
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(-tf.constant(np.array([1.0, self.ratio]).reshape([1, 2]),
                                                    dtype=tf.float32)*self.y*tf.log(self.outputs))
                        # 0.03*sum(reg_losses)
            # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits))
            self.op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

            self.merged = tf.summary.merge_all()

    def reset(self):
        index = list(range(self.train_X.shape[0]))
        np.random.shuffle(index)
        self.train_X = self.train_X[index]
        self.train_y = self.train_y[index]
        self.current_index = 0

    def next_batch(self):
        if self.current_index + self.batch_size <= self.train_X.shape[0]:
            batch_x = self.train_X[self.current_index:(self.current_index+self.batch_size)]
            batch_y = self.train_y[self.current_index:(self.current_index+self.batch_size)]
        else:
            batch_x = self.train_X[self.current_index:]
            batch_y = self.train_y[self.current_index:]
        self.current_index += self.batch_size
        return batch_x, batch_y

    def train(self, sess, merged=False):
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('G:/python_file/Gene_Classification/train/', sess.graph)
        val_writer = tf.summary.FileWriter('G:/python_file/Gene_Classification/val/', sess.graph)
        num = 0
        lr = 1e-3  # 1e-2
        for e in tqdm(range(self.epoch), desc='epoch'):
            self.reset()
            for _ in range(self.num_batch):
                batch_x, batch_y = self.next_batch()
                train_feed_dict = {self.x: batch_x, self.y: batch_y, self.lr: lr,
                                   self.keep_prob: 0.1}
                if merged:
                    train_result, _ = sess.run([self.merged, self.op],
                                               feed_dict=train_feed_dict)
                    val_result = self.evaluate(sess, self.val_X, self.val_y)
                    train_writer.add_summary(train_result, num)
                    val_writer.add_summary(val_result, num)
                else:
                    train_loss, train_acc, _ = sess.run([self.loss, self.accuracy, self.op],
                                                        feed_dict=train_feed_dict)
                    val_loss, val_acc, _ = self.evaluate(sess, self.val_X, self.val_y, merged=False)
                    self.train_loss.append(train_loss)
                    self.train_acc.append(train_acc)
                    self.val_loss.append(val_loss)
                    self.val_acc.append(val_acc)
                num += 1
            if e % 300 == 0:
                lr *= 0.5

    def evaluate(self, sess, x, y, merged=True):
        feed_dict = {self.x: x, self.y: y, self.keep_prob: 1.0}
        if merged:
            val_result = sess.run(self.merged, feed_dict=feed_dict)
            return val_result
        else:
            loss, accuracy, out = sess.run([self.loss, self.accuracy, self.outputs], feed_dict=feed_dict)
            return loss, accuracy, out


def save(sess, file='C:/Users/tianping/Desktop/tensorflow_data.npy'):
    data = {}
    for i in tf.trainable_variables():
        data[i.name] = sess.run(i)
    np.save(file, data)
    print('Saving the data to %s' % file)


def run(X_train, y_train, X_val, y_val, X_test, y_test, sample_names,
        batch_size=32, epoch=350, ratio=117/44.0, save_data=True):
    train_names, val_names, test_names = sample_names
    net = CNN(X_train, y_train, X_val, y_val, batch_size=batch_size, epoch=epoch, ratio=ratio)
    x = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)
    net.build()
    print('Model done')
    print('---------------------------------------------')
    print('---------------------------------------------')
    with tf.Session() as sess:
        net.train(sess)
        plot_train_curve(net.train_loss, net.val_loss, 'loss')
        plot_train_curve(net.train_acc, net.val_acc, 'accuracy')

        train_auc, train_acc, train_error = confusion_matrix_and_auc(net, sess,
                                                                     X_train, y_train,
                                                                     train_names,  'train')
        print('---------------------------------------------')
        print('---------------------------------------------')
        val_auc, val_acc, val_error = confusion_matrix_and_auc(net, sess,
                                                               X_val, y_val,
                                                               val_names, 'validation')
        print('---------------------------------------------')
        print('---------------------------------------------')
        _, _, test_error = confusion_matrix_and_auc(net, sess, X_test, y_test, test_names,  'test')
        if save_data:
            save(sess)
        plot_tsne(net, sess, x, y)
        error_names = (train_error, val_error, test_error)
    return train_acc, train_auc, val_acc, val_auc, error_names
