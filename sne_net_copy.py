#from convnet import DNN,HiddenLayer

import theano.tensor as T
from numpy import *
import theano
import scipy.spatial.distance as dist
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
#from tsne import tsne


#import sys
#sys.path.insert(0, '/Users/zachzhang/Arcade-Learning-Environment-0.5.1/doc/examples/ConvNet')

from convnet import *


def get_rmsprop_update(ma , gparams , decay):
    ma = [  decay * m + (1-decay)*T.sqr(grad)  for m,grad in zip(ma,gparams)]
    return ma



def RMSprop(cost, params, lr, rho=0.9,     lamb = .00 , epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g - p*lamb))

    return updates


def theano_pdist(X,Y):

    translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))
    sqr_dist = (abs(translation_vectors) ** 2.).sum(2) ** (1. / 2.)

    return sqr_dist

def gauss_kernel(X,sigma):

    translation_vectors = X.reshape((X.shape[0], 1, -1)) - X.reshape((1, X.shape[0], -1))
    sqr_dist = (abs(translation_vectors) ** 2.).sum(2) ** (1. / 2.)

    return T.exp(-sqr_dist / sigma**2)



def test_minibatch_with_dropout():

    X = loadtxt("mnist2500_X.txt");
    #Y = loadtxt("tsne_points.txt");
    #Y = loadtxt('tsne_20.txt')
    Y = loadtxt('tsne_40.txt')
    #Y = loadtxt('tsne_60.txt')
    #Y = loadtxt('tsne_150.txt')

    labels = loadtxt("mnist2500_labels.txt");

    test_data_X  = X[2300:]
    test_data_Y  = Y[2300:]
    test_labels = labels[2300:]

    X = X[:2300]
    Y = Y[:2300]
    labels = labels[:2300]

    n=20
    pca = PCA(n_components=n)
    pca.fit(X)

    training_x = theano.shared( X )
    training_y = theano.shared( Y )

    nn_test = DNN([28*28,400,200,40] , dropout = .3)
    x = T.matrix()
    y = T.matrix()
    y_hat = nn_test.apply_dropout(x)

    loss = T.mean(T.sqr(y_hat - y))

    params = nn_test.params
    gparams = [T.grad(loss,param) for param in params]

    lr = theano.shared(.005)

    #rms_updates = [ (param, param - lr* gparam) for param, gparam in zip(params,gparams)  ]
    rms_updates = RMSprop(loss,params,lr)


    index = T.lscalar()

    epochs = 2000
    batch_size =500

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
        print(errors / num_batches , j)

        if j > 1200:
            lr.set_value(.001)

    print("Finished Training")


    print("Embedding Data")
    #Data with dimensionality reduction
    Y_hat = predict(X)

    validation  = predict(test_data_X)
    l2 =  mean((validation - test_data_Y)**2)

    print(l2)

    #Knn fit to the reduced data
    knn_sne = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to the PCA data
    knn_pca = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to unperturbed data
    knn = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to perfect tsne data
    knn_test = KNeighborsClassifier(n_neighbors=3)

    print("Fitting Embedded KNN")
    knn_sne.fit(Y_hat, labels)

    print("Fitting PCA KNN")
    knn_pca.fit(pca.transform(X), labels)

    print("Fitting normal KNN")
    knn.fit(X, labels)

    print("Fitting Validation KNN")
    knn_test.fit(Y,labels)

    error_sne = 0
    error = 0
    error_valid = 0
    error_pca = 0


    embeded_pca = pca.transform(test_data_X)
    embeded_sne = predict(test_data_X)

    print("Beginning Testing")
    for i in range(test_data_X.shape[0]):
        sne_y_hat = knn_sne.predict(embeded_sne[i])
        pca_y_hat = knn_pca.predict(embeded_pca[i])

        y_hat = knn.predict(test_data_X[i])
        y_valid = knn_test.predict(test_data_Y[i])

        if sne_y_hat != test_labels[i]:
            error_sne = error_sne +1
        if y_hat != test_labels[i]:
            error = error +1
        if y_valid != test_labels[i]:
            error = error +1
        if pca_y_hat != test_labels[i]:
            error_pca = error_pca +1

    print("Embedded KNN :  " + str( float(test_data_X.shape[0] - error_sne) / test_data_X.shape[0] ))
    print("PCA KNN :  " + str( float(test_data_X.shape[0] - error_pca) / test_data_X.shape[0] ))

    print("Normal KNN :  " + str( float(test_data_X.shape[0] - error) / test_data_X.shape[0] ))
    print("Validation KNN :  " + str( float(test_data_X.shape[0] - error_valid) / test_data_X.shape[0] ))

    '''
    plt.figure()
    plt.scatter(Y[:,0],Y[:,1])

    plt.figure()
    plt.scatter(Y_hat[:,0],Y_hat[:,1])

    plt.show()
    '''


def test_cnn(fn , epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout):

    f = open('/nfshome/zazhang/Senior-Project/results/'+ fn,'w')

    X = loadtxt("mnist2500_X.txt");
    #Y = loadtxt("tsne_points.txt");
    #Y = loadtxt('tsne_20.txt')
   # Y = loadtxt('tsne_40.txt')
   # Y = loadtxt('tsne_60.txt')
    Y = loadtxt('tsne_10.txt')

    labels = loadtxt("mnist2500_labels.txt");

    test_data_X  = X[2300:]
    test_data_Y  = Y[2300:]
    test_labels = labels[2300:]

    X = X[:2300]
    Y = Y[:2300]
    labels = labels[:2300]

    X_ = reshape(X, (X.shape[0],1,28,28))

    n=20
    pca = PCA(n_components=n)
    pca.fit(X)

    training_x = theano.shared( X_ )
    training_y = theano.shared( Y )

    image_shape = (28,28)
    #conv_arc =  [( 5, 15,2),( 5, 10,2)]
    #mlp_params = [100,60]

    conv_net = ConvNet(mlp_params,conv_arc,image_shape , conv_act = conv_act, dense_act = dense_act,dropout= dropout)
    
    print(conv_net.mlp.arc)

    x = T.tensor4()
    y = T.matrix()
    y_hat = conv_net.apply_dropout(x)

    loss = T.mean(T.sqr(y_hat - y))

    params = conv_net.params

    lr = theano.shared(.01)
    rms_updates = RMSprop(loss,params,lr)

    index = T.lscalar()


    #epochs = 5
    #batch_size =500

    train_model_rms = theano.function(
        inputs=[index],
        outputs= loss,
        updates=rms_updates,
        givens={
            x: training_x[index * batch_size: (index + 1) * batch_size,:,:,:],
            y: training_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict = theano.function(
        inputs=[x],
        outputs= conv_net.predict_dropout(x),
    )

    best_so_far = 10000
    last_improvement = 1
    
    patience = 10

    num_batches = int(floor(X.shape[0] / batch_size))
    
    for j in range(epochs):
        errors = 0
        for i in range(num_batches):
            t = train_model_rms(i)

            errors = errors + t
    
        avg_error = errors / num_batches
        
        print(avg_error , j)

        f.write(str(avg_error) + "  " + str(j) + "\n" )

        if avg_error < best_so_far:
            best_so_far = avg_error
            last_improvement = 0
        else:
            last_improvement+=1

        if last_improvement > patience:
            lr.set_value(  lr.get_value() * .75 )
            last_improvement = 0
            print("NEW LEARNING RATE = " + str(lr.get_value()))
        #if j > 1200:
        #    lr.set_value(.001)

    print("Finished Training")


    print("Embedding Data")
    
    
    print(reshape(X,(X.shape[0],1,28,28) ).shape)
    #Data with dimensionality reduction
    Y_hat = predict(reshape(X,(X.shape[0],1,28,28) )  )


    validation  = predict(reshape(test_data_X , ( test_data_X.shape[0],1,28,28) ) )

    l2 =  mean((validation - test_data_Y)**2)

    print(l2)

    #Knn fit to the reduced data
    knn_sne = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to the PCA data
    knn_pca = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to unperturbed data
    knn = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to perfect tsne data
    knn_test = KNeighborsClassifier(n_neighbors=3)

    print("Fitting Embedded KNN")
    knn_sne.fit(Y_hat, labels)

    print("Fitting PCA KNN")
    knn_pca.fit(pca.transform(X), labels)

    print("Fitting normal KNN")
    knn.fit(X, labels)

    print("Fitting Validation KNN")
    knn_test.fit(Y,labels)

    error_sne = 0
    error = 0
    error_valid = 0
    error_pca = 0


    embeded_pca = pca.transform(test_data_X)
    embeded_sne = predict(reshape(test_data_X , ( test_data_X.shape[0],1,28,28) )  )

    print("Beginning Testing")
    for i in range(test_data_X.shape[0]):
        sne_y_hat = knn_sne.predict(embeded_sne[i])
        pca_y_hat = knn_pca.predict(embeded_pca[i])

        y_hat = knn.predict(test_data_X[i])
        y_valid = knn_test.predict(test_data_Y[i])

        if sne_y_hat != test_labels[i]:
            error_sne = error_sne +1
        if y_hat != test_labels[i]:
            error = error +1
        if y_valid != test_labels[i]:
            error = error +1
        if pca_y_hat != test_labels[i]:
            error_pca = error_pca +1

    print("Embedded KNN :  " + str( float(test_data_X.shape[0] - error_sne) / test_data_X.shape[0] ))
    print("PCA KNN :  " + str( float(test_data_X.shape[0] - error_pca) / test_data_X.shape[0] ))

    print("Normal KNN :  " + str( float(test_data_X.shape[0] - error) / test_data_X.shape[0] ))
    print("Validation KNN :  " + str( float(test_data_X.shape[0] - error_valid) / test_data_X.shape[0] ))


    f.write("L2 error during trianing: " + str(best_so_far) + "\n")
    f.write("L2 error on test data:  " + str(l2) + "\n")
    f.write("Embedded KNN :  " + str( float(test_data_X.shape[0] - error_sne) / test_data_X.shape[0] ) + "\n")
    f.write("PCA KNN :  " + str( float(test_data_X.shape[0] - error_pca) / test_data_X.shape[0] ) + "\n")
    f.write("Normal KNN :  " + str( float(test_data_X.shape[0] - error) / test_data_X.shape[0] ) + "\n")
    f.write("Validation KNN :  " + str( float(test_data_X.shape[0] - error_valid) / test_data_X.shape[0] ) + "\n")


def relu(input_):

    return T.switch(input_ > 0, input_, input_ * .05)


if __name__ == '__main__':
    #epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout

    epochs = 350
    batch_size = 500
    conv_arc = [( 5, 15,2),( 5, 15,2)]
    mlp_params = [100,10]
    conv_act = T.tanh
    dense_act = T.tanh
    dropout = .0


    #test_cnn('baseline_test.txt',epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout)
    #test_cnn('baseline_test.txt',epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,.2)

    #best so far .945 error
    #params - conv arc [( 5, 15,2),( 5, 15,2)] ,desnse arc[100,10], all relu, .2 dropout

    #test_cnn('more_dropout.txt',epochs, batch_size, conv_arc,mlp_params, relu,relu,.1)
    #test_cnn('less_dropout.txt',epochs, batch_size, conv_arc,mlp_params, relu,relu,.3)

    #conv_arc = [( 5, 20,2),( 5, 20,2)]
    #mlp_params = [200,10]

    test_cnn('smoother_relu.txt',epochs, batch_size, conv_arc,mlp_params, relu,relu,.2)

    '''
    #Actual Baseline
    test_cnn('baseline_test.txt',epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout)

    dropout = .1

    #Baseline test- std activations + .1 dropout
    test_cnn('dropout_test.txt',epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout)

    dropout = .2

    #more dropout
    test_cnn('more_dropout.txt',epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout)

    dropout =.1
    conv_act = relu
    dense_act = relu

    #all relu
    test_cnn('all_relu.txt' ,epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout)

    conv_act = T.tanh

    #just desne relue
    test_cnn('only_dense_relu.txt' , epochs, batch_size, conv_arc,mlp_params, conv_act,dense_act,dropout)
    '''
