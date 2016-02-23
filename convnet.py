import os
import sys
import time

from numpy import *
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import matplotlib.pyplot as plt

import cPickle, gzip

import base64
import pickle

"""
As far as I can tell this works
"""

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

class HiddenLayer():
    
    def __init__(self,in_,out_,act_func = T.nnet.sigmoid ):
        
        init_var = .1
        #W = theano.shared(init_var*random.randn(out_,in_))
        W = theano.shared(init_var*random.randn(in_,out_))
        b = theano.shared(zeros(out_))
        self.act_func = act_func
        self.params = [W,b]

    def apply(self,x):
        
        W = self.params[0]
        b = self.params[1]
        
        #a  = T.dot(W,x) + b
        a  = T.dot(x,W) + b
        
        if(self.act_func is not None):
            return self.act_func(a)

        return a



class DNN():

    def relu(self,input_):
        return T.switch(input_ > 0, input_, 0.05*input_)

    def __init__(self,arc,act_func = T.nnet.sigmoid ,output_act = None ,dropout= 0):
        self.arc = arc
        self.layers = []
        self.params =[]
        self.dropout = dropout

        for i in range(len(arc) -1):
            H = HiddenLayer(arc[i],arc[i+1],act_func = act_func)
            self.layers.append(H)
            self.params += H.params

        self.layers[-1].act_func = None

    def apply(self,x):
        h = x
        for l in self.layers:
            h = l.apply(h)
        return h

    def apply_dropout(self,x):
        srng = theano.tensor.shared_randomstreams.RandomStreams(100)

        #inital input
        h = x

        #for each hidden layer compute activation with dropout
        for i in range(len(self.layers ) -1):
            l = self.layers[i]
            h = l.apply(h)
            mask = srng.binomial(n=1, p=1-self.dropout, size=h.shape)
            h =  h * T.cast(mask, theano.config.floatX)

        h = self.layers[-1].apply(h)

        return h

    def dropout_predict(self,x):

        arc = self.arc
        #inital input
        h = x
        #for each hidden layer compute activation with dropout
        for i in range(len(self.layers ) -1):
            l = self.layers[i]
            h = l.apply(h)

            mask = theano.shared((1-self.dropout) * ones(arc[i+1]))

            h =  h * T.cast(mask, theano.config.floatX)

        h = self.layers[-1].apply(h)

        return h


class ConvNet():

    #CHANGE IMAGE SIZE TO BE TUPLE (X,Y)
    def __init__(self,mlp_params, conv_layers,image_size,stack_size = 1,batch_size =1,dropout=0,dense_act = T.nnet.sigmoid ,
                 conv_act = T.tanh):
        self.layers = []
        self.params = []
        self.image_size = image_size
        self.stack_size = stack_size
        self.dropout = dropout


        #self.batch_size = batch_size

        stack_ = 1
        for i in range(len(conv_layers)):
            p = conv_layers[i]
            if i ==0:
                conv = ConvLayer(p[0],p[1],p[2],stack_size,act_func = conv_act)
                self.layers.append(conv)
                self.params += conv.params
            else:
                stack_ = conv_layers[i-1][1]
                conv = ConvLayer(p[0],p[1],p[2],stack_,act_func = conv_act)
                self.layers.append(conv)
                self.params += conv.params
        
        response_size_x  = image_size[0]
        response_size_y  = image_size[1]

        num_responses = 1
        for p in conv_layers:    
            response_size_x = (response_size_x - p[0]+1 ) / p[2]
            response_size_y = (response_size_y - p[0]+1 ) / p[2]

            #num_responses = num_responses * p[1]
                  
        mlp_inputs = response_size_x * response_size_y * conv_layers[-1][1]#

        #self.mlp = DNN([mlp_inputs,mlp_params[0],mlp_params[1]],dropout=dropout) #MLP(mlp_inputs,mlp_params[0],mlp_params[1])#,output_act = T.nnet.softmax)
        self.mlp = DNN([mlp_inputs] + mlp_params,dropout=dropout,act_func = dense_act) #MLP(mlp_inputs,mlp_params[0],mlp_params[1])#,output_act = T.nnet.softmax)
        self.params += self.mlp.params

    def activate(self,X):
        h = X#.reshape( (self.b,self.stack_size,self.image_size[0],self.image_size[1] ) )

        for layer in self.layers:
            h = layer.apply(h)

        return T.flatten(h,2) , h


    def apply(self,X):
        h = X
        
        for layer in self.layers:
            h = layer.apply(h)

        t = T.flatten(h,2)
        
        y = self.mlp.apply(t)
        return y

    def apply_dropout(self,X):

        srng = theano.tensor.shared_randomstreams.RandomStreams(100)

        h = X

        for layer in self.layers:
            h = layer.apply(h)
            mask = srng.binomial(n=1, p=1-self.dropout, size=h.shape)
            h =  h * T.cast(mask, theano.config.floatX)

        t = T.flatten(h,2)

        y = self.mlp.apply_dropout(t)
        return y


    def predict_dropout(self,X):
        h = X

        for layer in self.layers:
            h = layer.apply(h)
            mask = theano.shared((1-self.dropout) * h.shape)
            h =  h * T.cast(mask, theano.config.floatX)

        t = T.flatten(h,2)

        y = self.mlp.dropout_predict(t)
        return y

class ConvLayer():

    def relu(self,input_):
        return T.switch(input_ > 0, input_, 0.05*input_)
        
    def __init__(self,kernel_size, num_maps, poolsize ,stack_size ,act_func = T.tanh):
        
        rng = random.RandomState(23455) 
        
        self.W = theano.shared( random.randn(num_maps, stack_size , kernel_size,kernel_size) )
                
        b_values = zeros(num_maps)
        self.b = theano.shared(value=b_values, borrow=True) 
              
        self.params = [self.W, self.b]
        self.poolsize = poolsize
        self.kernel_size = kernel_size
        self.num_maps = num_maps
        self.stack_size = stack_size
    
        self.act_func = act_func

    def apply(self,X):
                
        conv_out = conv.conv2d(
                input = X,
                filters = self.W,
            )
        
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=(self.poolsize,self.poolsize),
            ignore_border=True
        )
        
        h = self.act_func(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x') )
        #h = self.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        return h



    
def test_convnet():
    print "Loading Images ..."
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    X = train_set[0]
    labels = train_set[1]
    Y = zeros((labels.shape[0],10))  
    
    for i in range(Y.shape[0]):
        label = labels[i]
        Y[i,label] = 1.0 
    
    x = T.vector()
    y = T.vector()
        
    conv_arc =  [( 5, 2,2),( 5, 2,2)] 
    mlp_params = [20,10]

    conv_net = ConvNet(mlp_params,conv_arc,(28,28))    
    
    y_hat = conv_net.apply(x)
    neg_log = T.mean(-T.log( T.dot(y_hat,y.T) ) )
  
    params = conv_net.params
    
    gparams = [T.grad(neg_log,param) for param in params]

    lr = .03
    updates = [ (param, param - lr* (gparam ) ) for param, gparam in zip(params,gparams)  ]
    
    train_model = theano.function(
            inputs = [x,y],
            outputs = [neg_log],
            updates = updates
        )

    predict = theano.function(
            inputs = [x],
            outputs = y_hat
        )
    
    epochs = 2
   
    
    print "Beginning Training..."
    
    total = 0.0
    for j in range(epochs):
        for i in range(X.shape[0]):
            
            x_in = X[i,:]
            y_ = Y[i,:]
            total = total + train_model(x_in,y_)[0]
            #print train_model(x_in,y_)[0]
            if i % 1000 == 999 :
                print total/1000.0
                total = 0.0  
        
    X_test = test_set[0]
    Y_test = test_set[1]
    errors = 0.0
    for i in range(X_test.shape[0]):
        x_in = X_test[i,:]
        y_hat = argmax(predict(x_in))
        y = Y_test[i]
        print str(y) + "  "  + str(y_hat)
        
        if y != y_hat:
            errors = errors +1
            
    print "Error Rate ..."
    print (X_test.shape[0]-errors) / float(X_test.shape[0])
    
    '''
    for i in range(10):
        x_in = reshape(X[i,:],[1,1,28,28])
        print str(argmax(predict(x_in))) + "   " + str(labels[i])
        print labels[i]
    '''

def test_convnet_batch_no_stack():
    print "Loading Images ..."
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X = train_set[0]
    labels = train_set[1]
    Y = zeros((labels.shape[0],10))

    for i in range(Y.shape[0]):
        label = labels[i]
        Y[i,label] = 1.0

    x  = T.tensor4()

    y = T.matrix()

    conv_arc =  [( 5, 6,2),( 5, 4,2)]
    mlp_params = [20,10]

    conv_net = ConvNet(mlp_params,conv_arc,(28,28),batch_size=3,stack_size=1)

    test_h = theano.function(
        inputs = [x],
        outputs = conv_net.activate(x)
    )

    test_h_single = theano.function(
        inputs = [x],
        outputs = conv_net.layers[0].apply(x)
    )

    y_hat =  T.nnet.softmax(conv_net.apply(x))
    neg_log = T.mean(-T.log( T.sum(y_hat*y ,axis= 1 )) )

    params = conv_net.params

    gparams = [T.grad(neg_log,param) for param in params]

    lr = .1
    #updates = [ (param, param - lr* (gparam ) ) for param, gparam in zip(params,gparams)  ]
    updates = RMSprop(neg_log, params , lr = lr)
    train_model = theano.function(
            inputs = [x,y],
            outputs = neg_log,
            updates = updates
        )

    predict = theano.function(
            inputs = [x],
            outputs = y_hat
        )

    epochs = 10


    print "Beginning Training..."
    batch_size = 500
    n_batchs = int(floor(X.shape[0] / batch_size))


    for j in range(epochs):

        total = 0.0

        for i in range(n_batchs):

            x_in = X[i*batch_size : (i+1)*batch_size,:]
            y_ = Y[i*batch_size : (i+1)*batch_size,:]
            x_in = reshape(x_in,[batch_size,1,28,28])

            total = total + train_model(x_in, y_)

            #print i

        print total / n_batchs


    X_test = test_set[0]
    Y_test = test_set[1]

    test_stack = reshape(X_test,[X_test.shape[0], 1,28,28])
    Y_hat = argmax(predict(test_stack),axis = 1)

    errors = 0
    for i in range(10000):

        print str(Y_hat[i]) + "    " + str(Y_test[i])
        if Y_hat[i] != Y_test[i]:
            errors = errors +1

    print (float(10000 - errors)) / 10000

def test_convnet_batch():
    print "Loading Images ..."
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X = train_set[0]
    labels = train_set[1]
    Y = zeros((labels.shape[0],10))

    for i in range(Y.shape[0]):
        label = labels[i]
        Y[i,label] = 1.0

    #x = T.vector()
    x  = T.tensor4()

    y = T.matrix()

    conv_arc =  [( 5, 10,2),( 5, 10,2)]
    mlp_params = [100,10]

    conv_net = ConvNet(mlp_params,conv_arc,(28,28),batch_size=3)


    test_h = theano.function(
        inputs = [x],
        outputs = conv_net.activate(x)
    )

    test_h_single = theano.function(
        inputs = [x],
        outputs = conv_net.layers[0].apply(x)
    )

    #batch size = 3 stack =4 image size  = 28
    test_stack = zeros((3,4,28,28))
    test_stack[0,:,:,:] = repeat(reshape(X[0],[1,28,28]) ,4,axis=0)
    test_stack[1,:,:,:] = repeat(reshape(X[1],[1,28,28]) ,4,axis=0)
    test_stack[2,:,:,:] = repeat(reshape(X[2],[1,28,28]) ,4,axis=0)

    h = test_h(test_stack)
    h2 = test_h_single(test_stack)

    y_hat =  T.nnet.softmax(conv_net.apply(x))
    neg_log = T.mean(-T.log( T.sum(y_hat*y ,axis= 1 )) )

    params = conv_net.params

    gparams = [T.grad(neg_log,param) for param in params]

    lr = .01
    updates = [ (param, param - lr* (gparam ) ) for param, gparam in zip(params,gparams)  ]

    train_model = theano.function(
            inputs = [x,y],
            outputs = neg_log,
            updates = updates
        )

    predict = theano.function(
            inputs = [x],
            outputs = y_hat
        )

    epochs = 1


    print "Beginning Training..."

    total = 0.0
    for j in range(epochs):
        #for i in range(X.shape[0] -2 ):
        for i in range(20000):

            x_in = X[i,:]
            y_ = Y[i,:]

            test_stack = zeros((1,4,28,28))
            test_stack[0,:,:,:] = repeat(reshape(X[i],[1,28,28]) ,4,axis=0)
            #test_stack[1,:,:,:] = repeat(reshape(X[i+1],[1,28,28]) ,4,axis=0)
            #test_stack[2,:,:,:] = repeat(reshape(X[i+2],[1,28,28]) ,4,axis=0)

            test_y = array( [ Y[i,:] ])# , Y [i+1,:] , Y[i+2,:] ] )

            total = total + train_model(test_stack,test_y)

            #print train_model(x_in,y_)[0]
            if i % 1000 == 999 :
                print total/1000.0
                total = 0.0


    X_test = test_set[0]
    Y_test = test_set[1]

    test_stack = zeros((X_test.shape[0],4 ,28,28))
    for i in range(X_test.shape[0]):
        test_stack[i,:,:,:] = repeat(reshape(X_test[i],[1,28,28]) ,4,axis=0)

    Y_hat = argmax(predict(test_stack),axis = 1)

    errors = 0
    for i in range(10000):

        print str(Y_hat[i]) + "    " + str(Y_test[i])
        if Y_hat[i] != Y_test[i]:
            errors = errors +1

    print (float(10000 - errors)) / 10000

    '''
    errors = 0.0
    for i in range(X_test.shape[0]):
        x_in = X_test[i,:]
        y_hat = argmax(predict(x_in))
        y = Y_test[i]
        print str(y) + "  "  + str(y_hat)

        if y != y_hat:
            errors = errors +1

    print "Error Rate ..."
    print (X_test.shape[0]-errors) / float(X_test.shape[0])
    '''
    '''
    for i in range(10):
        x_in = reshape(X[i,:],[1,1,28,28])
        print str(argmax(predict(x_in))) + "   " + str(labels[i])
        print labels[i]
    '''
        
if __name__ == '__main__':
    #test_mlp()

    #test_convnet_batch()
    test_convnet_batch_no_stack()
    #load_cifar()
    #conv_net_cifar()
    
    
    