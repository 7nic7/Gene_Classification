from sklearn.metrics import confusion_matrix, auc, roc_curve, accuracy_score
import numpy as np


def confusion_matrix_and_auc(net, sess, x, y, sample_names, name):
    print(name+':')
    loss, accuracy, out = net.evaluate(sess, x, y, merged=False)
    print('- confusion matrix:')
    print(confusion_matrix(y_true=np.argmax(y, axis=1),
                           y_pred=np.argmax(out, axis=1)))
    x_coord, y_coord, _ = roc_curve(y_true=np.argmax(y, axis=1),
                                    y_score=out[np.arange(y.shape[0]), 1])
    auc_value = auc(x_coord, y_coord)
    print('%s data : loss:%s, accuracy:%s, auc:%s' % (name, loss, accuracy, auc_value))
    error_index = sample_names[np.argmax(y, axis=1) != np.argmax(out, axis=1)]
    print(error_index)
    # print(np.where(np.argmax(y, axis=1) != np.argmax(out, axis=1)))
    # print(out[error_index])
    return auc_value, accuracy, error_index


def confusion_matrix_and_auc_v2(y_true, y_pre, name):
    print(name+':')
    print('- confusion matrix:')
    confusion_m = confusion_matrix(y_true=y_true, y_pred=y_pre)
    print(confusion_m)
    x_coord, y_coord, _ = roc_curve(y_true=y_true, y_score=y_pre)
    auc_value = auc(x_coord, y_coord)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pre)
    print('%s data : accuracy:%s, auc:%s' % (name, accuracy, auc_value))
    y_true = np.squeeze(y_true.reshape([1, -1]))
    print(y_true)
    print(y_pre)
    print(np.where(y_true != y_pre))
    return auc_value, accuracy

