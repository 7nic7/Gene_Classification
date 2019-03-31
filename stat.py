import numpy as np


def stat_saliency(x, y, vis, features_name, is_combine_rg=True):
    name_r = []
    name_g = []
    for x_i, y_i in zip(x, y):
        img = vis.vis_saliency(filter_index=np.argmax(y_i),
                               seed_input=x_i)
        img = img.reshape([len(features_name), 1, 3])
        r = img[..., 0]
        g = img[..., 1]
        name_r.extend(list(features_name[np.where(np.max(r, axis=1) != 0)]))
        name_g.extend(list(features_name[np.where(np.max(g, axis=1) != 0)]))
    if is_combine_rg:
        dic_rg = {}
        name_rg = name_r
        name_rg.extend(name_g)
        for item in name_rg:
            dic_rg[item] = dic_rg.get(item, 0) + 1

        return sorted(dic_rg.items(), key=lambda e: -e[1])
    else:
        dic_r = {}
        dic_g = {}
        for item in name_r:
            dic_r[item] = dic_r.get(item, 0) + 1

        for item in name_g:
            dic_g[item] = dic_g.get(item, 0) + 1

        return sorted(dic_r.items(), key=lambda e: -e[1]), sorted(dic_g.items(), key=lambda e: -e[1])
