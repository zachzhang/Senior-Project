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
        
        init_var = .01
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


    #def relu(self,input_):
    #    return T.switch(input_ > 0, input_, 0.05*input_)

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
            mask = theano.shared((1-self.dropout))
            h =  h * T.cast(mask, theano.config.floatX)

        t = T.flatten(h,2)

        y = self.mlp.dropout_predict(t)
        return y


class ConvLayer():

    #def relu(self,input_):
    #    return T.switch(input_ > 0, input_, 0.05*input_)
        
    def __init__(self,kernel_size, num_maps, poolsize ,stack_size ,act_func = T.tanh):
        
        init_var = .01

        rng = random.RandomState(23455) 
        
        self.W = theano.shared( init_var * random.randn(num_maps, stack_size , kernel_size,kernel_size) )
                
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



    
