from sklearn import ensemble
from gene.dataset import *
from gene.evaluate import *
from sklearn import model_selection


data = read_data(file='C:/Users/tianping/Desktop/data.csv')

x, y = preprocess(data, one_hot=False, nn=False)
# x = x[:, :33]
X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.6, val_ratio=0.2)
X = np.concatenate([X_train, X_val])
Y = np.concatenate([y_train, y_val])

# Adaboost
kf = model_selection.KFold(n_splits=4)
train_auc_list, train_acc_list = [], []
val_auc_list, val_acc_list = [], []
for train_index, val_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    ada = ensemble.AdaBoostClassifier()
    ada.fit(X_train, y_train)
    train_pre = ada.predict(X_train)
    val_pre = ada.predict(X_val)
    train_auc, train_acc = confusion_matrix_and_auc_v2(y_train, train_pre, 'train')
    val_auc, val_acc = confusion_matrix_and_auc_v2(y_val, val_pre, 'validation')
    train_auc_list.append(train_auc)
    train_acc_list.append(train_acc)
    val_auc_list.append(val_auc)
    val_acc_list.append(val_acc)

print('k-fold:')
print('---------------------------------------------------------')
print('train_acc',train_acc_list, np.mean(train_acc_list))
print('train_auc', train_auc_list, np.mean(train_auc_list))
print('val_acc', val_acc_list, np.mean(val_acc_list))
print('val_auc', val_auc_list, np.mean(val_auc_list))

# # KNN
# kf = model_selection.KFold(n_splits=4)
# train_auc_list, train_acc_list = [], []
# val_auc_list, val_acc_list = [], []
# for train_index, val_index in kf.split(X):
#     X_train, y_train = X[train_index], Y[train_index]
#     X_val, y_val = X[val_index], Y[val_index]
#     knn = neighbors.KNeighborsClassifier(n_neighbors=5)
#     knn.fit(X_train, y_train)
#     pre_train = knn.predict(X_train)
#     pre_val = knn.predict(X_val)
#     train_auc, train_acc = confusion_matrix_and_auc_v2(y_train, pre_train, 'train')
#     val_auc, val_acc = confusion_matrix_and_auc_v2(y_val, pre_val, 'validation')
#
#     train_auc_list.append(train_auc)
#     train_acc_list.append(train_acc)
#     val_auc_list.append(val_auc)
#     val_acc_list.append(val_acc)
#
# print('k-fold:')
# print('---------------------------------------------------------')
# print('train_acc',train_acc_list, np.mean(train_acc_list))
# print('train_auc', train_auc_list, np.mean(train_auc_list))
# print('val_acc', val_acc_list, np.mean(val_acc_list))
# print('val_auc', val_auc_list, np.mean(val_auc_list))


# knn = neighbors.KNeighborsClassifier(n_neighbors=6)
# knn.fit(X, Y)
# te_pre = knn.predict(X_test)
# confusion_matrix_and_auc_v2(y_test, te_pre, 'test')

# # Random Forest
# kf = model_selection.KFold(n_splits=4)
# auc_kf, acc_kf = [], []
# for tr_index, val_index in kf.split(X):
#     X_train, y_train = X[tr_index], Y[tr_index]
#     X_val, y_val = X[val_index], Y[val_index]
#     rf = ensemble.RandomForestClassifier(n_estimators=300, class_weight={0: 1.0, 1: 117/44.0})
#
#     rf.fit(X_train, y_train)
#     tr_pre = rf.predict(X_train)
#     val_pre = rf.predict(X_val)
#
#     auc_tr, acc_tr = confusion_matrix_and_auc_v2(y_train, tr_pre, 'train')
#     auc_val, acc_val = confusion_matrix_and_auc_v2(y_val, val_pre, 'validation')
#     auc_kf.append(auc_val)
#     acc_kf.append(acc_val)
#
# print(auc_kf, np.mean(auc_kf))
# print(acc_kf, np.mean(acc_kf))
# rf = ensemble.RandomForestClassifier(n_estimators=10, class_weight={0: 1, 1: 1.0})
# rf.fit(X, Y)
# te_pre = rf.predict(X_test)
# confusion_matrix_and_auc_v2(y_test, te_pre, 'test')


# importance = rf.feature_importances_
# # print(importance)
# indices = np.argsort(importance)[::-1]
# features = data.columns[2:]
# # print(features)
# for f in range(36):
#     print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
#
# print(features[indices[1:36]])
