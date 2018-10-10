import numpy as np
from sklearn.preprocessing import OneHotEncoder


def onehot(labels):
    enc = OneHotEncoder()
    labels = enc.fit_transform(labels)
    return labels.toarray()
# the following functions presume the imgs are black & white 


# shape should be (None,height,width,num_channels)
def vectors2imgs(vecs, shape):
    shape = (shape[1], shape[2])
    for i in range(vecs.shape[0]):
        vec = vecs[i, :]
        img = np.reshape(vec, newshape=shape, order='C')
        img = np.expand_dims(img, axis=0)
        if i == 0:
            imgs_out = img
        else:
            imgs_out = np.concatenate((imgs_out, img), axis=0)
    imgs_out = np.expand_dims(imgs_out, axis=3)
    return imgs_out


def imgs2vectors(imgs):
    length = imgs.shape[1] * imgs.shape[2]
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, 0]
        vec = np.reshape(img, newshape=(1, length), order='F')
        if i == 0:
            vecs_out = vec
        else:
            vecs_out = np.concatenate((vecs_out, vec), axis=0)
            
    return vecs_out


def split_train_test(x_train, y_train, num_per_sub):
    
    for i in range(int(max(y_train))):
        inds = np.where(y_train.ravel() == i+1)[0]
        
        x_train_group_original = x_train[inds, :]
        y_train_group_original = y_train[inds, :]
        train_inds = np.random.choice(x_train_group_original.shape[0], size=num_per_sub, replace=False)
        x_train_group = x_train_group_original[train_inds, :]
        y_train_group = y_train_group_original[train_inds, :]
        x_test_group = np.delete(x_train_group_original, train_inds, axis=0)
        y_test_group = np.delete(y_train_group_original, train_inds, axis=0)
        if i == 0:
            x_train_out = x_train_group
            y_train_out = y_train_group
            x_test_out = x_test_group
            y_test_out = y_test_group
        else:
            x_train_out = np.concatenate((x_train_out, x_train_group), axis=0)
            y_train_out = np.concatenate((y_train_out, y_train_group), axis=0)
            x_test_out = np.concatenate((x_test_out, x_test_group), axis=0)
            y_test_out = np.concatenate((y_test_out, y_test_group), axis=0)
        
    return x_train_out, y_train_out, x_test_out, y_test_out
