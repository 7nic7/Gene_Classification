import pandas as pd
import numpy as np
from sklearn import preprocessing


def read_data(file='C:/Users/tianping/Desktop/preprocess_gene.csv'):
    """
    Read 'gene' data from your work directory.

    Args:
        file: The gene data's location where you have put in your computer.(Default is my work directory)

    Returns:
        data: DataFrame.
    """
    data = pd.read_csv(file, index_col=0)
    # data = data.drop('ref', 1)
    sample_names = data.columns[0:]
    gene_names = data.index[1:]
    return data, gene_names, sample_names


def index_s_sort(origin_index, n_col):
    n_row = len(origin_index) // n_col
    new_index = np.zeros([n_row, n_col])
    count = 1
    for i in range(n_row):
        start_, end_ = n_col*i, n_col*i+n_col
        row_index = origin_index[start_:end_]
        if count % 2 == 0:      # 偶数行
            row_index = row_index[::-1]
        count += 1
        new_index[i, :] = row_index
    return new_index


def image_s_sort(gene_one, s_index):
    new_gene_shape = s_index.shape
    s_index = s_index.reshape([1, -1]).squeeze().astype('int')
    gene_one = gene_one[s_index].reshape(new_gene_shape)
    return gene_one


def preprocess(data, one_hot=True, nn=True, n_col=None):
    """
    Encode for the features and the labels of data.

    Args:
        data: DataFrame.
        one_hot: Boolean.If True,x will be encoded into the format of one-hot.
        nn: Boolean.If True, x will be encoded into the format of square matrix for the CNN.
         Otherwise,x will be used in classical methods.

    Returns:
        A tuple (x,y).
        x is tha data of gene with the shape (N, D/6, 6*4, 1).
        y is the labels of gene whit the shape (N, 2).
    """
    labels = data.ix[0, :]
    labels = np.array(labels)
    # y = np.zeros([labels.shape[0], 2])
    # y[labels=='R', 0] = 1
    # y[labels=='S', 1] = 1
    y = preprocessing.label_binarize(labels, ['XDR', 'MDR', 'DS'])  # 改
    y[:, 0] += y[:, 1]     # 改
    y = y[:, [0, 2]]   # 改
    print(pd.Series(labels).value_counts())
    gene = data.ix[1:, :]      # delete ref sample
    gene = np.array(gene).transpose([1, 0])
    n, gene_dim = gene.shape

    if one_hot:
        x = np.zeros([n, gene.shape[1], 4, 1])
        for i in range(n):
            x[i, :, :, 0] = preprocessing.label_binarize(gene[i], ['A', 'G', 'C', 'T'])
            # x[i, :, :, 0] = preprocessing.label_binarize(gene[i], ['1', '2', '3', '4'])
    else:
        gene[gene=='A'] = 1
        gene[gene=='G'] = 2
        gene[gene=='C'] = 3
        gene[gene=='T'] = 4
        gene = preprocessing.minmax_scale(gene)
        n_row = gene.shape[1] // n_col
        if nn:
            x = np.zeros([n, n_row, n_col, 1])
            s_index = index_s_sort(range(gene.shape[1]), n_col=n_col)
            for i in range(n):
                x[i, :, :, 0] = image_s_sort(gene[i].squeeze(), s_index)
            # print(x[0])
        else:
            # squeeze y's shape from 2d into 1d
            y_copy = np.zeros([y.shape[0], 1])
            y_copy[y[:, 1] == 1] = 1
            y = y_copy

            x = gene
            # IDEA: split a big image into some small pieces
            # labels = data.ix[0, 2:]
            # labels = np.array(labels).reshape([-1, 1])
            # y = preprocessing.label_binarize(labels, ['XDR', 'MDR', 'DS'])  # 改
            # y[:, 0] += y[:, 1]     # 改
            # y = y[:, [0, 2]]   # 改
            #
            # gene = read_data(file='C:/Users/tianping/Desktop/gene_clear.csv')
            # gene = np.array(gene)[:, start:end]
            # gene_dim, n = gene.shape
            # image_size = 28
            # one_image_own = image_size//4 * image_size
            # image_n = gene_dim // one_image_own
            # x = np.zeros([n*image_n, image_size, image_size, 1])
            # Y = np.zeros([n*image_n, 2])
            # count = 0
            # for i in range(n):
            #     one_image = preprocessing.label_binarize(gene[:, i], ['1', '2', '3', '4'])
            #     for j in range(image_n):
            #         # print(count)
            #         x[count, :, :, 0] = one_image[(j*one_image_own):((j+1)*one_image_own), :].reshape([image_size,
            #                                                                                            image_size])
            #
            #         Y[count, :] = y[i, :]
            #         count += 1
            # y = Y
    return x, y


def split_data(x, y=None, sample_names=None, train_ratio=0.8, val_ratio=0.1, random_state=None):
    """
    Split data into three parts which are training data, validation data and testing data.

    Args:
        x: Features of your data with the shape (N, D/6, 6*4, 1).
        y: Labels of your data with the shape (N, 2).
        sample_names: The names of samples.
        train_ratio: Float (0.0-1.0).Choose how big the training data are.
        val_ratio: Float (0.0-1.0).Choose how big the validation data are. Attention: train_ratio plus
            val_ratio should less than 1.0.
        random_state: Int. The seed of shuffle.

    Returns:
        x_train: shape (N1, D/6, 6*4, 1).
        y_train: shape (N1, 2).
        x_val: shape (N2, D/6, 6*4, 1).
        y_val: shape (N2, 2).
        x_test: shape (N3, D/6, 6*4, 1).
        y_test: shape (N3, 2).
    """
    assert train_ratio + val_ratio < 1.0
    num = x.shape[0]
    train_num = int(np.floor(train_ratio * num))
    val_num = int(np.floor(val_ratio * num))
    # test_num = num - train_num - val_num

    index = list(range(num))
    if random_state:
        np.random.seed(random_state)
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
        sample_names = sample_names[index]

    x_train, y_train = x[range(0, train_num)], y[range(0, train_num)]
    x_val, y_val = x[range(train_num, (train_num + val_num))], y[range(train_num, (train_num + val_num))]
    x_test, y_test = x[range((train_num + val_num), num)], y[range((train_num + val_num), num)]

    train_names = sample_names[range(0, train_num)]
    val_names = sample_names[range(train_num, (train_num + val_num))]
    test_names = sample_names[range((train_num + val_num), num)]
    sample_names = (train_names, val_names, test_names)
    return x_train, y_train, x_val, y_val, x_test, y_test, sample_names


def find_gene_name_by_index(features_name, index):
    names = np.array(features_name)
    names_mat = np.repeat(names, 4).reshape([-1, 4]).reshape([283, 24])
    print('The feature of index:%s is %s' % (index, names_mat[index[0], index[1]]))
