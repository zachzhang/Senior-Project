from convnet import MLP
from convnet import DNN,HiddenLayer

import theano.tensor as T
import cPickle, gzip
from numpy import *
import theano
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

from tsne import tsne

#class Theano_KNN():

#    def __init__(self):

def embed_network():

    X = loadtxt("mnist2500_X.txt");
    #Y = loadtxt("tsne_points.txt");
    #Y = loadtxt('tsne_20.txt')
    #Y = loadtxt('tsne_40.txt')
    Y = loadtxt('tsne_60.txt')

    labels = loadtxt("mnist2500_labels.txt");

    test_data_X  = X[2300:]
    test_data_Y  = Y[2300:]
    test_labels = labels[2300:]

    X = X[:2300]
    Y = Y[:2300]
    labels = labels[:2300]

    training_x = theano.shared( X )
    training_y = theano.shared( Y )

    nn_test = DNN([28*28,100,100,80,80,60] , dropout = .2)
    x = T.matrix()
    y = T.matrix()
    y_hat = nn_test.apply_dropout(x)

    loss = T.mean(T.sqr(y_hat - y))

    params = nn_test.params
    gparams = [T.grad(loss,param) for param in params]

    lr = .0035
    updates = [ (param, param - lr* gparam) for param, gparam in zip(params,gparams)  ]
    rms_updates = RMSprop(loss,params,lr=lr)


    index = T.lscalar()

    epochs = 600
    batch_size =200

    train_model_rms = theano.function(
        inputs=[index],
        outputs= loss,
        updates=rms_updates,
        givens={
            x: training_x[index * batch_size: (index + 1) * batch_size],
            y: training_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict = theano.function(
        inputs=[x],
        outputs= nn_test.dropout_predict(x),
    )

    num_batches = int(floor(X.shape[0] / batch_size))
    for j in range(epochs):
        errors = 0
        for i in range(num_batches):
            #t =  train_model(i)
            t = train_model_rms(i)
            errors = errors + t
        print errors / num_batches

    print "Finished Training"


    print "Embedding Data"
    #Data with dimensionality reduction
    Y_hat = predict(X)

    validation  = predict(test_data_X)
    l2 =  mean((validation - test_data_Y)**2)

    print l2

    #Knn fit to the reduced data
    knn_sne = KNeighborsClassifier(n_neighbors=7)

    #Knn fit to unperturbed data
    knn = KNeighborsClassifier(n_neighbors=7)

    #Knn fit to perfect tsne data
    knn_test = KNeighborsClassifier(n_neighbors=7)

    print "Fitting Embedded KNN"
    knn_sne.fit(Y_hat, labels)

    print "Fitting normal KNN"
    knn.fit(X, labels)

    print "Fitting Validation KNN"
    knn_test.fit(Y,labels)

    error_sne = 0
    error = 0
    error_valid = 0

    embeded = predict(test_data_X)
    print "Beginning Testing"
    for i in range(test_data_X.shape[0]):
        sne_y_hat = knn_sne.predict(embeded[i])
        y_hat = knn.predict(test_data_X[i])
        y_valid = knn_test.predict(test_data_Y[i])

        if sne_y_hat != test_labels[i]:
            error_sne = error_sne +1
        if y_hat != test_labels[i]:
            error = error +1
        if y_valid != test_labels[i]:
            error = error +1

    print "Embedded KNN :  " + str( float(test_data_X.shape[0] - error_sne) / test_data_X.shape[0] )
    print "Normal KNN :  " + str( float(test_data_X.shape[0] - error) / test_data_X.shape[0] )
    print "Validation KNN :  " + str( float(test_data_X.shape[0] - error_valid) / test_data_X.shape[0] )

    '''
    plt.figure()
    plt.scatter(Y[:,0],Y[:,1])

    plt.figure()
    plt.scatter(Y_hat[:,0],Y_hat[:,1])

    plt.show()
    '''

if __name__ == '__main__':
    embed_network()