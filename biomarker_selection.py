from gene.dataset import read_data
import numpy as np
from sklearn import feature_selection
from scipy.stats import spearmanr


def delete_all_same(data):
    data = data.ix[~((data=='A').all(1) | (data=='G').all(1) | (data=='C').all(1) | (data=='T').all(1)), :]
    return data


def delete_by_corrcoef(data, delta):
    pass


# p_values < 0.05 : 10
def select_biomarker_chisq(X_train, y_train, X_val, y_val, X_test, y_test, gene_names):
    x = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    new_x = np.zeros([x.shape[0], x.shape[1]])
    new_y = np.argmax(y, axis=1)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            new_x[i, j] = np.argmax(x[i, j, :, 0])
    chi2_values, p_values = feature_selection.chi2(new_x, new_y)
    index = (p_values < 0.5)
    print('features number:', sum(p_values < 0.2))
    # print(gene_names)
    X_train = X_train[:, index]
    X_val = X_val[:, index]
    X_test = X_test[:, index]
    gene_names = gene_names[index]
    return X_train, y_train, X_val, y_val, X_test, y_test, gene_names


# p_values < 0.05 : 33
def select_biomarker_spearman(X_train, y_train, X_val, y_val, X_test, y_test, gene_names):
    x = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    new_x = np.zeros([x.shape[0], x.shape[1]])
    new_y = np.argmax(y, axis=1).reshape([-1, 1])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            new_x[i, j] = np.argmax(x[i, j, :, 0]) + 1
    rho, p_values = spearmanr(np.hstack([new_x, new_y]))
    rho, p_values = rho[:-1, -1], p_values[:-1, -1]
    index = (p_values < 0.5)
    print('features number:', sum(p_values < 0.05))
    X_train = X_train[:, index]
    X_val = X_val[:, index]
    X_test = X_test[:, index]
    gene_names = gene_names[index]
    return X_train, y_train, X_val, y_val, X_test, y_test, gene_names

if __name__ == '__main__':
    data, g_names, s_names = read_data(file='C:/Users/tianping/Desktop/gene_another(label).csv')
    print(data.shape)
    print(g_names.shape)
    print(s_names.shape)
    data = data.ix[~((data=='A').all(1) | (data=='G').all(1) | (data=='C').all(1) | (data=='T').all(1)), :]
    print(data.shape)
    data_copy = data
    data_copy = data_copy.drop('label')
    data_copy[data_copy=='A'] = 1
    data_copy[data_copy=='G'] = 2
    data_copy[data_copy=='C'] = 3
    data_copy[data_copy=='T'] = 4
    print(data_copy.shape)
    # print(data_copy)
    data_copy = data_copy.apply(lambda x: x.astype('float'))
    data_copy_np = np.array(data_copy)
    # r = np.corrcoef(data_copy_np)
    # for i in range(data_copy_np.shape[0]):
    a = data_copy_np[0, :]
    b = data_copy_np[1, :]
    print(np.cov(a, b)/np.sqrt(np.var(a) * np.var(b)))
