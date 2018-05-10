import tensorflow as tf
import scipy.io as sio
import numpy as np
from matplotlib.pyplot import imshow 
import h5py


def transform_imgs(imgs,shape):
    for i in range(imgs.shape[0]):
        img = imgs[i,:]
        img = np.reshape(img,newshape=shape,order="F")
        img = np.expand_dims(img,axis=0)
        if i == 0:
            imgs_out = img
        else:
            imgs_out = np.concatenate((imgs_out,img),axis=0)
    imgs_out = np.expand_dims(imgs_out,axis=3)
    return imgs_out

def transform_labels(labels):
    num_classes = np.max(labels)
    labels -= 1
    tensor_one_hot = tf.one_hot(labels,depth=num_classes)
    with tf.Session() as sess:
        one_hot = sess.run(tensor_one_hot)
    return np.squeeze(one_hot)
def split(images, labels, train_num,n_per_subject):
    for i in range(int(np.max(labels))) :
        ids = np.arange(i*n_per_subject,(i+1)*n_per_subject)
        training_id = np.random.choice(ids,train_num,replace=False)
        training = images[training_id,:,:]
        training_labels = labels[training_id]
        for j in range(train_num):
            if j == 0:
                ids = ids.tolist()
            ids.remove(training_id[j])
        testing_id = ids
        testing = images[testing_id,:]
        testing_labels = labels[testing_id]
        
        if i == 0 :
            training_all = training
            training_all_labels = training_labels
            testing_all = testing
            testing_all_labels = testing_labels
        else :
            training_all = np.concatenate((training_all,training),axis=0)
            training_all_labels = np.concatenate((training_all_labels,training_labels),axis=0)
            testing_all = np.concatenate((testing_all,testing),axis=0)
            testing_all_labels = np.concatenate((testing_all_labels,testing_labels),axis=0)
    return training_all, training_all_labels, testing_all, testing_all_labels
def shape_generater(string):
    if string == "AR":
        input_shape = [None,165,120,1]
        n = 14
        k = 100
    elif string =="ORL":
        input_shape = [None,112,92,1]
        n = 10
        k = 40
    elif string =="Yale":
        input_shape = [None,211,165,1]
        n = 11
        k = 15
    elif string == "GT":
        input_shape = [None,160,120,1]
        n = 15
        k = 50
    elif string == "FERET":
        input_shape = [None,150,100,1]
        n = 11
        k = 40
    elif string == "JAFFE":
        input_shape = [None,256,256,1]
        n = 20
        k = 10
    elif string =="UKmale":
        input_shape = [None,200,180,1]
        n = 20
        k = 113
    else :
        input_shape = [None,192,168,1]
        n = 64
        k = 38
    return input_shape, k, n
def load(string):
    string += ".mat"
    data= sio.loadmat(string)
    return data.get("X"), data.get("Y")
def create_labels(n_sub,n_sam):

    for i in range(n_sub):
        label = np.ones(shape=(n_sam,1))*(i+1)
        print(label)
        if i == 0 :
            labels = label
        else :
            labels = np.concatenate((labels,label))
    return labels



def change_order(imgs,n_sub,n_sam):
    imgss = np.zeros(shape=(165,211*165))
    count = 0
    for i in range(n_sub):
        for j in range(n_sam):
            ind = j * n_sub + i
            img = imgs[ind,:]
            
            imgss[count,:] = img
            count += 1
            
    return imgss
    
    
    
def example1():
    import Classifier as C
    import DimensionReductionApproaches as DRA
    import UtilFun as UF
    
    name = 'AR.npy'
    data = np.load(name)
    imgs = data[0]
    labels = data[1]
    # info = [(None,height,width), num of sub, num of sample per sub]
    info = data[2]
    
    X_train, Y_train, X_test, Y_test = split(imgs,labels,2,info[2])
    
    X_train = UF.imgs2vectors(X_train)
    X_test = UF.imgs2vectors(X_test)
    
    classifier = C.LinearDiscriminantClassifier(DRA.LinearDiscriminant.NLDA)
    classifier.Fit(X_train,Y_train)
    result = classifier.Classify(X_train,Y_train,X_test,Y_test)
    print(result)


def example2():
#    model = classifier(input_shape=input_shape,num_classes = int(np.max(labels)))
#    
#    classifier.Construct(model,layer_type="conv2d",shape=[15,15,10],padding="VALID")
#    classifier.Construct(model,layer_type="pooling",ksize=[4,4],pooling_type="avg",padding="VALID")
##    classifier.Construct(model,layer_type="conv2d",shape=[5,5,5],padding="VALID")
##    classifier.Construct(model,layer_type="pooling",ksize=[4,4],pooling_type="max",padding="VALID")
#    classifier.Construct(model,layer_type="flatten")
##    classifier.Construct(model,layer_type="fc",shape=200,last=False)
##    classifier.Construct(model,layer_type="dropout",keep_prob=0.7)
#    classifier.Construct(model,layer_type="fc",shape=int(num_classes),last=True)
#    
#    training_one_hot_labels = transform_labels(training_labels)
#    testing_one_hot_labels = transform_labels(testing_labels)
#    
#    #training
#    model.Fit(training,training_one_hot_labels)
#
#
#    #testing
#    correct_prediction = tf.equal(tf.argmax(model.out,1),tf.argmax(model.y,1))
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
#    
#    result = model.Run(accuracy,testing,testing_one_hot_labels)
    pass

def example3():
    
    import DimensionReductionApproaches as DRA
    import Classifier as C       
    
    #simulation
    
    # mean
    mu1 = np.zeros(100)
    mu2 = np.zeros(100) 
    mu2[0] = 5
    mu3 = np.zeros(100)
    mu3[1] = 5
    
    # variance
    sigma1 = np.zeros((100,100))
    sigma1[0,0] = 1
    sigma1[1,1] = 1
    for i in range(2,100):
        sigma1[i,i] = 0.1**2
    
    sigma2 = sigma1
    sigma3 = sigma1
    
    
    
    times = 10
    results1 = []
    results2 = []
    results3 = []
    for i in range(times):
            
        X_train_g1 = np.random.multivariate_normal(mean=mu1,cov=sigma1,size=(30,))
        X_train_g2 = np.random.multivariate_normal(mean=mu2,cov=sigma2,size=(30,))
        X_train_g3 = np.random.multivariate_normal(mean=mu3,cov=sigma3,size=(30,))
        
        Y_train_g1 = np.ones(shape=(30,1))
        Y_train_g2 = np.ones(shape=(30,1)) + 1
        Y_train_g3 = np.ones(shape=(30,1)) + 2
        
        X_train = np.concatenate((X_train_g1,np.concatenate((X_train_g2,X_train_g3),axis=0)),axis=0)
        Y_train = np.concatenate((Y_train_g1,np.concatenate((Y_train_g2,Y_train_g3),axis=0)),axis=0)        


        
        X_test_g1 = np.random.multivariate_normal(mean=mu1,cov=sigma1,size=(30,))
        X_test_g2 = np.random.multivariate_normal(mean=mu2,cov=sigma2,size=(30,))
        X_test_g3 = np.random.multivariate_normal(mean=mu3,cov=sigma3,size=(30,))
    
        Y_test_g1 = np.ones(shape=(30,1))
        Y_test_g2 = np.ones(shape=(30,1)) + 1
        Y_test_g3 = np.ones(shape=(30,1)) + 2
        
        X_test = np.concatenate((X_test_g1,np.concatenate((X_test_g2,X_test_g3),axis=0)),axis=0)
        Y_test = np.concatenate((Y_test_g1,np.concatenate((Y_test_g2,Y_test_g3),axis=0)),axis=0)
        
    
        classifier1 = C.LinearDiscriminantClassifier(DRA.LinearDiscriminant.PIRE)
        classifier1.Fit(X_train,Y_train)
        result1 = classifier1.Classify(X_train,Y_train,X_test,Y_test)
        
        
        classifier2 = C.LinearDiscriminantClassifier(DRA.LinearDiscriminant.NLDA)
        classifier2.Fit(X_train,Y_train)
        result2 = classifier2.Classify(X_train,Y_train,X_test,Y_test)
        
        classifier3 = C.LinearDiscriminantClassifier(DRA.LinearDiscriminant.DRLDA)
        classifier3.Fit(X_train,Y_train)
        result3 = classifier3.Classify(X_train,Y_train,X_test,Y_test)
        
        results1.append(result1[0])        
        results2.append(result2[0])
        results3.append(result3[0])
        

    print(np.mean(results3))
    
if __name__ == "__main__":
    example1()