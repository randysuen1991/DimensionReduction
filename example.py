import numpy as np



    
# This example use the linear discriminant analysis to classify face data.
def example1():
    import sys
    if 'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Classifier-and-Regressor' not in sys.path :
        sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Classifier-and-Regressor')
    import Classifier as C
    import DimensionReductionApproaches as DRA
    import UtilFun as UF
    
    
    import time as t
    
    name = 'Yale.npy'
    data = np.load(name)
    imgs = data[0]
    labels = data[1]
    
    
    results1 = []
    
    for i in range(1000):
    
    
        t1 = t.time()
        X_train, Y_train, X_test, Y_test = UF.split_train_test(imgs,labels,2)
        t2 = t.time()
        print(t2-t1)
    
    
    
        X_train = UF.imgs2vectors(X_train)
        X_test = UF.imgs2vectors(X_test)
        t3 = t.time()
        print(t3-t2)
    
    
        classifier = C.LinearDiscriminantClassifier(DRA.LinearDiscriminant.NLDA)
        classifier.Fit(X_train,Y_train)
        result = classifier.Classify(X_train,Y_train,X_test,Y_test)
        t4 = t.time()
        
        print(t4-t3)
        results1.append(result[0])
    
        
    print(np.mean(results1))
# This example use deep convolution neural network to classify face data.
def example2():
    import tensorflow as tf
    import sys
    if 'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Neural-Network' not in sys.path :
        sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Neural-Network')
    import NeuralNetworkModel as NNM
    import NeuralNetworkUnit as NNU
    import NeuralNetworkLoss as NNL
    import UtilFun as UF
    
    
    data = np.load('Yale.npy')
    imgs = data[0]
    labels = data[1]
    info = data[2]
    img_size = info[0][1:3]
    
    
    
    X_train, Y_train, X_test, Y_test = UF.split_train_test(imgs,labels,2)
    Y_train = UF.OneHot(Y_train)
    Y_test = UF.OneHot(Y_test)
    
    
    
    
    model = NNM.NeuralNetworkModel(dtpe=tf.float32,img_size=img_size)
    #shape=(5,5,3) means the kernel's height=5 width=5 num of ker=3
    model.Build(NNU.ConvolutionUnit(dtype=tf.float32,shape=(5,5,3),transfer_fun=tf.tanh))
    model.Build(NNU.AvgPooling(dtype=tf.float32,shape=(1,4,4,1)))
    model.Build(NNU.Dropout(keep_prob=0.5))
    model.Build(NNU.Flatten())
    model.Build(NNU.NeuronLayer(hidden_dim=10,dtype=tf.float32))
    model.Build(NNU.SoftMaxLayer())
    model.Fit(X_train,Y_train,loss_fun=NNL.NeuralNetworkLoss.CrossEntropy,show_graph=True,num_epochs=1000)



# This example use the linear discriminant analysis to classify normal distributed data.
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