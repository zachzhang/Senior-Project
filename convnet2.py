import os
import sys
import time

from numpy import *
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import base64
import pickle

"""
As far as I can tell this works
"""

class HiddenLayer():
    
    def __init__(self,in_,out_,act_func = T.nnet.sigmoid ,dropout = None):

        self.dropout = dropout

        init_var = .1
        #W = theano.shared(init_var*random.randn(out_,in_))
        W = theano.shared(init_var*random.randn(in_,out_))
        b = theano.shared(zeros(out_))
        self.act_func = act_func
        self.params = [W,b]

    def apply(self,x):
        
        W = self.params[0]
        b = self.params[1]
        
        a  = T.dot(x,W) + b

        if(self.act_func is not None):
            return self.act_func(a)

        return a

class DNN():
    
    def relu(self,input_):
        return T.switch(input_ > 0, input_, 0.05*input_)

    def __init__(self,arc,act_func = T.nnet.sigmoid ,output_act = None ,dropout= None):
        self.arc = arc
        self.layers = []
        self.params =[]
        self.dropout = dropout
        for i in range(len(arc) -1):
            H = HiddenLayer(arc[i],arc[i+1],dropout = dropout)
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


      
class MLP():
    
    def relu(self,input_):
        return T.switch(input_ > 0, input_, 0.05*input_)
        
    
    def __init__(self,x,h,y,act_func = T.nnet.sigmoid ,output_act = None ):
        self.arc = [x,h,y]
        
        init_var = .01
        
        W_in = theano.shared(init_var*random.randn(h,x))
        b_in = theano.shared(zeros(h))
        
        W_out = theano.shared(init_var*random.randn(y,h))
        b_out = theano.shared(zeros(y))

        self.params = [W_in,b_in,W_out,b_out]
        self.act_func = act_func
        #self.act_func = self.relu
        self.output_act = output_act
        
    def apply(self,X):
        W_in = self.params[0]
        b_in = self.params[1]
        W_out = self.params[2]
        b_out = self.params[3]

        h = self.act_func( T.dot(W_in,X) + b_in )
        a = T.dot(W_out,h) + b_out
        
        if(self.output_act is not None):
            y = self.output_act(a)
        else:
            y = a
        
        return y


class ConvNet():
    """
    conv_layers is a list of tuples that each specify a conv layer initialization
    """
    def __init__(self,mlp_params, conv_layers):
        self.layers = []
        self.params = []
        
        for p in conv_layers:
            conv = ConvLayer(p[0],p[1],p[2])
            self.layers.append(conv)
            self.params += conv.params
          
        self.mlp = MLP(mlp_params[0],mlp_params[1],mlp_params[2],output_act = T.nnet.softmax)
        self.params += self.mlp.params
        
    def apply(self,X):
        h = X
        for layer in self.layers:
             h = layer.apply(h)
        
        t = T.flatten(h)
        
        y = self.mlp.apply(t)
        return y
        
class ConvLayer():
    '''
    input_  -> input to this layer in the form of a 3 tensor ( image_size x,image_size y, num outputmaps  )
    image_size -> a tuple of the  (length , width) of the input images 
    
    :type filter_shape: tuple or list of length 4
    :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)
    '''
    def relu(self,input_):
        return T.switch(input_ > 0, input_, 0.05*input_)
        
    
    def __init__(self,image_size, filter_shape ,poolsize):
        
        rng = random.RandomState(23455) 
        
        fan_in = prod(filter_shape[1:])
        fan_out = (filter_shape[0] * prod(filter_shape[2:]) /
                   prod(poolsize))
        
        W_bound = sqrt(6. / (fan_in + fan_out))
        
        self.W = theano.shared(
            asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
                
        b_values = zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)  
              
        self.params = [self.W, self.b]
        self.poolsize = poolsize
        self.filter_shape = filter_shape
        self.image_size = image_size
    
    def apply(self,X):
        
        conv_out = conv.conv2d(
                input = X,
                filters = self.W,
                filter_shape=self.filter_shape,
                image_shape=self.image_size
            )
        
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )
        
        #h = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        h = self.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        return h
     

        
#if __name__ == '__main__':
    #test_mlp()
    #test_convnet()
    #load_cifar()
    #conv_net_cifar()
    
    
    