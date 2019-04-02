from Gene_Classification.visualization import *
from Gene_Classification.dataset import *
from Gene_Classification.biomarker_selection import select_biomarker_spearman

one_hot = True
data, g_names, s_names = read_data(file='C:/Users/tianping/Desktop/GeneDeleteSameAndRLarge.csv')
x, y = preprocess(data, one_hot=one_hot, nn=True, n_col=4)
X_train, y_train, X_val, y_val, X_test, y_test, s_names = split_data(x, y, s_names,
                                                                     train_ratio=0.6,
                                                                     val_ratio=0.2)
if one_hot:
    X_train, y_train, X_val, y_val, X_test, y_test, g_names = \
        select_biomarker_spearman(X_train, y_train, X_val, y_val, X_test, y_test, g_names)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
print('Start to visualization!')
vis = Visualization()
vis.build_network()
vis.set_weights(file='C:/Users/tianping/Desktop/tensorflow_data.npy')
print('Set weights done')
loss, accuracy = vis.model.evaluate(X_val, y_val)
pre_y = vis.model.predict(
    np.concatenate([X_train, X_val, X_test])
)
print('The loss of test data is %s, accuracy is %s' % (loss, accuracy))

# # 统计saliency map上的亮点
# right_index = np.argmax(pre_y, axis=1) == np.argmax(y, axis=1)
# x_right, y_right = x[right_index], y[right_index]
# r_index, s_index = (y_right[:, 0] == 1), (y_right[:, 1] == 1)
# ans = stat_saliency(x_right, y_right, vis, features_name[:-1],
#                     is_combine_rg=True)
# print(ans)

# # 只保留出现次数大于1的features的索引值
# remain = []
# for name, num in ans:
#     if num >= 1:
#         index = np.where(features_name == name)[0][0]
#         remain.append(index)
#     else:
#         break
# print(remain)

# print(y_test)
grads = vis.vis_saliency(np.argmax(y_test[-2], axis=0), X_test[-2], plot=False)
# map_part = vis_saliency_part(X_train, y_train, (245, 255, 7, 17), num=5)
print(grads)
print(grads.shape)


