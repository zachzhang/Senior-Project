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
from sklearn.decomposition import PCA
from tsne import tsne

class SNE():
    
    def __init__(self,X, n,m,sigma):
        self.Y = random.randn(n,m)
        self.X = X
        self.sigma = sigma

    def grad_kl(self,P,Q):
        Y = self.Y
        grad_y = zeros(Y.shape)
        
        diff = P-Q
        coef = diff + transpose(diff)
        a = P.shape[0]
    
        for i in range(a):
            pwisedif = Y[i] - Y
            yi_grad = dot(coef[i],pwisedif)
            grad_y[i] = yi_grad
    
        return grad_y
        
    def train(self,k,lr):
        
        P = self.gauss_kernel(self.X,self.sigma)
        Y = self.Y
        sigma = self.sigma
        
        for i in range(k):     
            Q = self.gauss_kernel( Y,sigma)
            grad = self.grad_kl(P,Q)
            Y = Y - lr* grad
            print self.cost(P,Q)
            
        self.Y =Y
    
    def cost(self,P,Q):
        KL = sum( P * log(P / Q))
        return KL
     
    def gauss_kernel(self,D,sigma):
        A = dist.squareform(dist.pdist(D,'sqeuclidean'))
        K = exp(-A**2 / (2 * sigma**2))
        K_sum = sum(K,axis=1)
        K = transpose(K/K_sum)
        
        return K
        


def get_rmsprop_update(ma , gparams , decay):
    ma = [  decay * m + (1-decay)*T.sqr(grad)  for m,grad in zip(ma,gparams)]
    return ma

def test_rmsprop():
    
    X = loadtxt("mnist2500_X.txt",dtype='float64');
    Y = loadtxt("tsne_points.txt",dtype = 'float64');
    
    X_ = preprocessing.scale(X)
    
    training_x = theano.shared( X_ )
    training_y = theano.shared( Y )
    
    nn_test = DNN([28*28,100,2])
    x = T.matrix()
    y = T.matrix()
    y_hat = nn_test.apply(x)

    loss = T.mean(T.sqr(y_hat - y))
    
    params = nn_test.params
    gparams = [T.grad(loss,param) for param in params]

    grad_ma = [theano.shared(zeros(p.get_value().shape)) for p in params ] 
    decay = .9
    
    rms_update = get_rmsprop_update(grad_ma, gparams, decay)

    #updates = [ (param, param -  gparam * (1 - decay) * T.sqr(gparam)) for param, gparam in zip(params,gparams)  ]
    updates = [ (param, param - gparam/ T.sqrt(rms) ) for param, gparam,rms in zip(params,gparams,rms_update)  ]
    
    ma_updates = [ (ma,new_ma) for ma, new_ma in zip(grad_ma,rms_update) ]
    
    updates = updates #+ ma_updates
    
    index = T.lscalar()
    
    batch_size = 250

    test_func = theano.function(
                inputs=[index],
            outputs= gparams,
            #updates=updates,
            givens={
                #x: training_x[index * batch_size: (index + 1) * batch_size],
                x: random.randn(batch_size,28*28),
                y: training_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
    test = test_func(0)
    
    init_W  = params[0].get_value()    
    
    epochs = 1
    train_model = theano.function(
        inputs=[index],
        outputs= loss,
        updates=updates,
        givens={
            x: training_x[index * batch_size: (index + 1) * batch_size],
            y: training_y[index * batch_size: (index + 1) * batch_size]
        }
    )
       
    
    for j in range(epochs):
        for i in range(3):
            test = train_model(i)
            print test

    final_W  = params[0].get_value()    
    
    print init_W - final_W
    
def dnn_test():
    path = '/Users/zachzhang/DeepLearningTutorials/SNE_Network/'

    X = loadtxt(path +"mnist2500_X.txt")
    Y = loadtxt(path +"tsne_points.txt")

    x = T.vector()
    y = T.vector()
    
    mlp = DNN([28*28,10,2])
    
    y_hat = mlp.apply(x)
    
    cost = T.mean(T.sqr(y_hat - y))
    
    in_grad = T.grad(cost,mlp.params[0])
    
    test_func = theano.function(
            inputs = [x,y],
            outputs = in_grad
        )
    
    print test_func(X[0],Y[0])

def test_single():
    
    X = loadtxt("mnist2500_X.txt");
    Y = loadtxt("tsne_points.txt");
    
    #X_ = preprocessing.scale(X)
    
    training_x = theano.shared( asarray(X, dtype=theano.config.floatX) )
    training_y = theano.shared( asarray(Y, dtype=theano.config.floatX))
    
    nn_test = DNN([28*28,10,2])
    
    index = T.lscalar()

    x = T.vector()
        
    y = T.vector()
    y_hat = nn_test.apply(x)
    
    loss = T.mean(T.sqr(y_hat - y))
    
    params = nn_test.params
    gparams = [T.grad(loss,param) for param in params]

    lr = .1
    updates = [ (param, param - lr* gparam) for param, gparam in zip(params,gparams)  ]
        
    epochs = 10
    batch_size = 250
        
    '''
    train_model = theano.function(
        inputs=[index],
        outputs= loss,
        updates=updates,
        givens={
            x: training_x[index * batch_size: (index + 1) * batch_size],
            y: training_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    '''
    train_model = theano.function(
        inputs=[x,y],
        outputs= loss,
        updates= updates,
        
    )
    
    
    test_func = theano.function(
            inputs=[x,y],
            outputs= gparams,
            #updates=updates,
            #givens={
            #    x: training_x[index],
            #    y: training_y[index]
            #}
        )
    error = 0
    for j in range(epochs):
        for i in range(X.shape[0]):
            error = error + train_model(X[i],Y[i])
        print error/ X.shape[0]
        error = 0

def RMSprop(cost, params, lr=0.001, rho=0.9,     lamb = .00 , epsilon=1e-6):
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

def test_minibatch():
    
    X = loadtxt("mnist2500_X.txt");
    #Y = loadtxt("tsne_points.txt");
    Y = loadtxt('tsne_20.txt')
    labels = loadtxt("mnist2500_labels.txt");

    test_data_X  = X[2000:]
    test_data_Y  = Y[2000:]
    test_labels = labels[2000:]

    X = X[:2000]
    Y = Y[:2000]
    labels = labels[:2000]

    n=20
    pca = PCA(n_components=n)
    pca.fit(X)

    training_x = theano.shared( X )
    training_y = theano.shared( Y )
    
    nn_test = DNN([28*28,40,20])
    x = T.matrix()
    y = T.matrix()
    y_hat = nn_test.apply(x)
    
    loss = T.mean(T.sqr(y_hat - y))
    
    params = nn_test.params
    gparams = [T.grad(loss,param) for param in params]

    lr = .002
    updates = [ (param, param - lr* gparam) for param, gparam in zip(params,gparams)  ]
    rms_updates = RMSprop(loss,params,lr=lr)


    index = T.lscalar()
    
    epochs = 700
    batch_size = 50
        
    train_model = theano.function(
        inputs=[index],
        outputs= loss,
        updates=updates,
        givens={
            x: training_x[index * batch_size: (index + 1) * batch_size],
            y: training_y[index * batch_size: (index + 1) * batch_size]
        }
    )

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
        outputs= y_hat,
        #givens={
        #    x: training_x#training_x[index * batch_size: (index + 1) * batch_size],
        #}
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
    knn_sne = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to the PCA data
    knn_pca = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to unperturbed data
    knn = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to perfect tsne data
    knn_test = KNeighborsClassifier(n_neighbors=3)

    print "Fitting Embedded KNN"
    knn_sne.fit(Y_hat, labels)

    print "Fitting PCA KNN"
    knn_pca.fit(pca.transform(X), labels)

    print "Fitting normal KNN"
    knn.fit(X, labels)

    print "Fitting Validation KNN"
    knn_test.fit(Y,labels)

    error_sne = 0
    error_pca = 0
    error = 0
    error_valid = 0


    embeded_sne = predict(test_data_X)
    embeded_pca = pca.transform(test_data_X)

    print "Beginning Testing"
    for i in range(test_data_X.shape[0]):
        sne_y_hat = knn_sne.predict(embeded_sne[i])
        pca_y_hat = knn_pca.predict(embeded_pca [i])
        y_hat = knn.predict(test_data_X[i])
        y_valid = knn_test.predict(test_data_Y[i])


        if sne_y_hat != test_labels[i]:
            error_sne = error_sne +1
        if y_hat != test_labels[i]:
            error = error +1
        if y_valid != test_labels[i]:
            error_valid = error_valid +1
        if pca_y_hat != test_labels[i]:
            error_pca = error_pca +1

    print "Embedded Using SNE KNN :  " + str( float(test_data_X.shape[0] - error_sne) / test_data_X.shape[0] )

    print "Embedded Using PCA KNN :  " + str( float(test_data_X.shape[0] - error_pca) / test_data_X.shape[0] )

    print "Normal KNN :  " + str( float(test_data_X.shape[0] - error) / test_data_X.shape[0] )
    print "Validation KNN :  " + str( float(test_data_X.shape[0] - error_valid) / test_data_X.shape[0] )

    '''
    plt.figure()
    plt.scatter(Y[:,0],Y[:,1])

    plt.figure()
    plt.scatter(Y_hat[:,0],Y_hat[:,1])

    plt.show()
    '''

def theano_pdist(X,Y):

    translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))
    sqr_dist = (abs(translation_vectors) ** 2.).sum(2) ** (1. / 2.)

    return sqr_dist

def gauss_kernel(X,sigma):

    translation_vectors = X.reshape((X.shape[0], 1, -1)) - X.reshape((1, X.shape[0], -1))
    sqr_dist = (abs(translation_vectors) ** 2.).sum(2) ** (1. / 2.)

    return T.exp(-sqr_dist / sigma**2)


def unsupervised_sne_net():

    X = loadtxt("mnist2500_X.txt");

    labels = loadtxt("mnist2500_labels.txt");

    test_data_X  = X[2300:]
    test_labels = labels[2300:]

    X = X[:2300]
    labels = labels[:2300]

    n=20
    pca = PCA(n_components=n)
    pca.fit(X)

    training_x = theano.shared( X )

    nn_test = DNN([28*28,300,300,150] , dropout = .3)
    x = T.dmatrix()

    y_hat = nn_test.apply_dropout(x)

    sigma = 2
    P = gauss_kernel(x,sigma)
    Q = gauss_kernel(y_hat,sigma)

    KL = T.sum(P * T.log(P / Q))

    lr = .001


    params = nn_test.params
    gparams = [T.grad(KL,param) for param in params]
    updates = [ (param, param - lr* gparam) for param, gparam in zip(params,gparams)  ]
    #rms_updates = RMSprop(KL,params,lr=lr)

    train_model_rms = theano.function(
        inputs=[x],
        outputs= KL,
        updates=updates,
    )

    epochs = 10

    print "Starting Training"
    for i in range(epochs):
        print train_model_rms(X)

def test_minibatch_with_dropout():

    X = loadtxt("mnist2500_X.txt");
    #Y = loadtxt("tsne_points.txt");
    #Y = loadtxt('tsne_20.txt')
    #Y = loadtxt('tsne_40.txt')
    #Y = loadtxt('tsne_60.txt')
    Y = loadtxt('tsne_150.txt')

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

    nn_test = DNN([28*28,300,300,200,150] , dropout = .3)
    x = T.matrix()
    y = T.matrix()
    y_hat = nn_test.apply_dropout(x)

    loss = T.mean(T.sqr(y_hat - y))

    params = nn_test.params
    gparams = [T.grad(loss,param) for param in params]

    lr = .01
    updates = [ (param, param - lr* gparam) for param, gparam in zip(params,gparams)  ]
    rms_updates = RMSprop(loss,params,lr=lr)


    index = T.lscalar()

    epochs = 1000
    batch_size =400

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
        print errors / num_batches , j

    print "Finished Training"


    print "Embedding Data"
    #Data with dimensionality reduction
    Y_hat = predict(X)

    validation  = predict(test_data_X)
    l2 =  mean((validation - test_data_Y)**2)

    print l2

    #Knn fit to the reduced data
    knn_sne = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to the PCA data
    knn_pca = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to unperturbed data
    knn = KNeighborsClassifier(n_neighbors=3)

    #Knn fit to perfect tsne data
    knn_test = KNeighborsClassifier(n_neighbors=3)

    print "Fitting Embedded KNN"
    knn_sne.fit(Y_hat, labels)

    print "Fitting PCA KNN"
    knn_pca.fit(pca.transform(X), labels)

    print "Fitting normal KNN"
    knn.fit(X, labels)

    print "Fitting Validation KNN"
    knn_test.fit(Y,labels)

    error_sne = 0
    error = 0
    error_valid = 0
    error_pca = 0


    embeded_pca = pca.transform(test_data_X)
    embeded_sne = predict(test_data_X)

    print "Beginning Testing"
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

    print "Embedded KNN :  " + str( float(test_data_X.shape[0] - error_sne) / test_data_X.shape[0] )
    print "PCA KNN :  " + str( float(test_data_X.shape[0] - error_pca) / test_data_X.shape[0] )

    print "Normal KNN :  " + str( float(test_data_X.shape[0] - error) / test_data_X.shape[0] )
    print "Validation KNN :  " + str( float(test_data_X.shape[0] - error_valid) / test_data_X.shape[0] )

    '''
    plt.figure()
    plt.scatter(Y[:,0],Y[:,1])

    plt.figure()
    plt.scatter(Y_hat[:,0],Y_hat[:,1])

    plt.show()
    '''

def dropout_test():
    hid = HiddenLayer(10,5)

    x = T.matrix()
    h = hid.apply(x)
    mask = theano.shared(.2 * ones(5))


    test_func = theano.function(
        inputs =[x],
        outputs = mask*hid.apply(x)
    )

    test_func2 = theano.function(
        inputs =[x],
        outputs = hid.apply(x)
    )

    test = random.randn(2,10)

    print test_func(test)
    print test_func2(test)


if __name__ == '__main__':
    #[ X,Y,labels] = tsne.run_tsne()
    #dropout_test()
    #theano_pdist()

    #unsupervised_sne_net()

    test_minibatch_with_dropout()

    #test_rmsprop()
    #test_minibatch()
    #test_single()
    #dnn_test()
