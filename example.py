import tensorflow as tf
import numpy as np





    
# This example use the linear discriminant analysis to classify face data.
def example1():
    import Classifier as C
    import DimensionReductionApproaches as DRA
    import UtilFun as UF
    
    name = 'GT.npy'
    data = np.load(name)
    imgs = data[0]
    labels = data[1]

    
    X_train, Y_train, X_test, Y_test = UF.split_train_test(imgs,labels,2)
    
    X_train = UF.imgs2vectors(X_train)
    X_test = UF.imgs2vectors(X_test)
    
    classifier = C.LinearDiscriminantClassifier(DRA.LinearDiscriminant.PIRE)
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
#    training_one_hot_labels = UF.transform_labels(training_labels)
#    testing_one_hot_labels = UF.transform_labels(testing_labels)
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