from Gene_Classification.CNN import *
from Gene_Classification.dataset import read_data, split_data, preprocess
from Gene_Classification.evaluate import *
from sklearn import model_selection
from Gene_Classification.visualization import plt
from Gene_Classification.biomarker_selection import select_biomarker_spearman
# import pandas as pd
# if you want to run the main.py, you should change the file path.
# through set parameter "file" in read_data.
# for example: data = read_data(file='C:/data.csv').
# and make sure that the format of the data should change to csv.
# then, you can click run.


def main(X_train, y_train, X_val, y_val, X_test, y_test, sample_names, ratio, kf=None, show=False):
    global total_err
    if kf:
        X = np.concatenate([X_train, X_val], axis=0)
        Y = np.concatenate([y_train, y_val], axis=0)
        kf = model_selection.StratifiedKFold(n_splits=kf, shuffle=True, random_state=42)
        train_auc_list, train_acc_list = [], []
        val_auc_list, val_acc_list = [], []
        train_names, val_names, test_names = sample_names
        train_val_names = np.concatenate([train_names, val_names], axis=0)
        tr_e, val_e, te_e = [], [], []
        for fold_index, (train_index, val_index) in enumerate(kf.split(X, np.argmax(Y, axis=1))):
            print('fold {}:'.format(fold_index))
            X_train, y_train, train_names_fold = X[train_index], Y[train_index], train_val_names[train_index]
            X_val, y_val, val_names_fold = X[val_index], Y[val_index], train_val_names[val_index]
            tf.reset_default_graph()
            # ratio = sum(y_train[:, 0] == 1) / sum(y_train[:, 1] == 1)
            sample_names_fold = (train_names_fold, val_names_fold, test_names)
            train_acc, train_auc, val_acc, val_auc, total_err_fold = run(X_train, y_train,
                                                                         X_val, y_val,
                                                                         X_test, y_test,
                                                                         sample_names_fold,
                                                                         batch_size=32, epoch=300,
                                                                         ratio=ratio)
            tr_e.append(total_err_fold[0])
            val_e.append(total_err_fold[1])
            te_e.append(total_err_fold[2])
            train_acc_list.append(train_acc)
            train_auc_list.append(train_auc)
            val_acc_list.append(val_acc)
            val_auc_list.append(val_auc)
            if show:
                plt.show()
        total_err = (np.hstack(tr_e).squeeze(), np.hstack(val_e).squeeze(), np.hstack(te_e).squeeze())

        print('k-fold:')
        print('---------------------------------------------------------')
        print('train_acc', train_acc_list, np.mean(train_acc_list))
        print('train_auc', train_auc_list, np.mean(train_auc_list))
        print('val_acc', val_acc_list, np.mean(val_acc_list))
        print('val_auc', val_auc_list, np.mean(val_auc_list))
    else:
        # ratio = sum(y_train[:, 0] == 1) / sum(y_train[:, 1] == 1)
        _, _, _, _, total_err = run(X_train, y_train, X_val, y_val, X_test, y_test, sample_names,
                                    batch_size=32, epoch=450, ratio=ratio)
        if show:
            plt.show()
    return total_err


if __name__ == '__main__':
    one_hot = False
    data, g_names, s_names = read_data(file='C:/Users/tianping/Desktop/GeneDeleteSameAndRLarge.csv')
    x, y = preprocess(data, one_hot=one_hot, nn=True, n_col=4)
    X_train, y_train, X_val, y_val, X_test, y_test, s_names = split_data(x, y, s_names,
                                                                         train_ratio=0.6,
                                                                         val_ratio=0.2)
    # if one_hot:
    #     X_train, y_train, X_val, y_val, X_test, y_test, g_names = \
    #         select_biomarker_spearman(X_train, y_train, X_val, y_val, X_test, y_test, g_names)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    print('Data done')

    main(X_train, y_train, X_val, y_val, X_test, y_test, s_names, ratio=5.0, show=True)
    # 重复，以便找到哪些样本的分类错误概率较高
    # errors_tr = []
    # errors_val = []
    # errors_te = []
    # for i in range(10):
    #     tf.reset_default_graph()
    #     errors = main(X_train, y_train, X_val, y_val, X_test, y_test, s_names,
    #                   ratio=3.0, show=False, kf=5)
    #     errors_tr.append(np.array(errors[0]))
    #     errors_val.append(np.array(errors[1]))
    #     errors_te.append(np.array(errors[2]))
    # errors_tr = np.hstack(errors_tr).squeeze()
    # errors_val = np.hstack(errors_val).squeeze()
    # errors_te = np.hstack(errors_te).squeeze()
    # print(pd.Series(errors_tr).value_counts())
    # print(pd.Series(errors_val).value_counts())
    # print(pd.Series(errors_te).value_counts())
