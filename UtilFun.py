import numpy as np
from sklearn.preprocessing import OneHotEncoder

def OneHot(labels):
    enc = OneHotEncoder()
    labels = enc.fit_transform(labels)
    return labels.toarray()
# the following functions presume the imgs are black & white 


# shape should be (None,height,width,num_channels)
def vectors2imgs(vecs,shape):
#    shape = (shape[0][1],shape[0][2])
    shape = (shape[1],shape[2])
    for i in range(vecs.shape[0]):
        vec = vecs[i,:]
        img = np.reshape(vec,newshape=shape,order='C')
        img = np.expand_dims(img,axis=0)
        if i == 0:
            imgs_out = img
        else:
            imgs_out = np.concatenate((imgs_out,img),axis=0)
    imgs_out = np.expand_dims(imgs_out,axis=3)
    return imgs_out


def imgs2vectors(imgs):
    length = imgs.shape[1] * imgs.shape[2]
    for i in range(imgs.shape[0]):
        img = imgs[i,:,:,0]
        vec = np.reshape(img,newshape=(1,length),order='F')
        if i == 0 :
            vecs_out = vec
        else:
            vecs_out = np.concatenate((vecs_out,vec),axis=0)
            
    return vecs_out

def split_train_test(X_train,Y_train,num_per_sub):
    
    for i in range(int(max(Y_train))):
        inds = np.where(Y_train.ravel()==i+1)[0]
        
        X_train_group_original = X_train[inds,:]
        Y_train_group_original = Y_train[inds,:]
        train_inds = np.random.choice(X_train_group_original.shape[0],size=num_per_sub,replace=False)
        X_train_group = X_train_group_original[train_inds,:]
        Y_train_group = Y_train_group_original[train_inds,:]
        X_test_group = np.delete(X_train_group_original,train_inds,axis=0)
        Y_test_group = np.delete(Y_train_group_original,train_inds,axis=0)
        if i == 0 :
            X_train_out = X_train_group
            Y_train_out = Y_train_group
            X_test_out = X_test_group
            Y_test_out = Y_test_group
        else :
            X_train_out = np.concatenate((X_train_out,X_train_group),axis=0)
            Y_train_out = np.concatenate((Y_train_out,Y_train_group),axis=0)
            X_test_out = np.concatenate((X_test_out,X_test_group),axis=0)
            Y_test_out = np.concatenate((Y_test_out,Y_test_group),axis=0)
        
    return X_train_out, Y_train_out, X_test_out, Y_test_out
        
