from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy import misc

import sys
import os
import time
import pdb

#========================#
#    Model Components    #
#========================#
def conv2D(
    net,
    kernel_size,
    strides, 
    output_channel,
    initializer = tf.contrib.layers.variance_scaling_initializer(), 
    padding = "SAME", 
    scope = None
    ):
    
    """
    2D Convolution in CNN.
    
    Args:
    (1) net            : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) kernel_size    : Kernel Size of the Convolution. (i.e. 3)
    (3) strides        : Stride of the Convolution. (i.e. 2)
    (4) output_channel : Output Channel of the Convolution. (i.e. 64)
    (5) initializer    : Initialization Strategy of the weights and biases variables.
    (6) padding        : Convolution Padding type. ('SAME'/'VALID')
    (7) scope          : Scope name.
    
    Returns:
    (1) net : An output tensor after convolution.
    """
    
    input_channel = net.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        # Define Weights Variable
        weights = tf.get_variable( name        = "weights", 
                                   shape       = [kernel_size, kernel_size, input_channel, output_channel],
                                   dtype       = tf.float32,
                                   initializer = initializer)
        # Define Biases Variable				
        biases = tf.get_variable( name        = "biases", 
                                  shape       = [output_channel],
                                  dtype       = tf.float32,
                                  initializer = initializer)
        
        tf.add_to_collection("weights", weights)
        
        # Convolution  
        net = tf.nn.conv2d( input   = net, 
                            filter  = weights,
                            strides = [1, strides, strides, 1],
                            padding = padding,
                            name    = 'conv')
    
        # Add Biases
        net = tf.nn.bias_add(net, biases)
        
    return net
	
def max_pooling(
    net,
    kernel_size,
    strides,
    padding = "SAME"
    ):
    
    """
    Max polling with indices.
    
    Args:
    (1) net         : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) kernel_size : Kernel Size of the Convolution. (i.e. 2)
    (3) strides     : Stride of the Convolution. (i.e. 2)
    (4) padding     : Pooling Padding type. ('SAME'/'VALID')
    
    Return:
    (1) net         : An output tensor after max pooling.
    (2) indices     : An indices of the input net. You can see the below web-site for more detail introduction.
    (3) input_shape : The shape before pooling. It is used for the unpooling afterward.
    """
    
    input_shape = net.get_shape().as_list()
    
    # Detail : https://www.tensorflow.org/api_docs/python/tf/nn/max_pool_with_argmax
    net, indices = tf.nn.max_pool_with_argmax( input   = net,
                                               ksize   = [1, kernel_size, kernel_size, 1],
                                               strides = [1, strides, strides, 1],
                                               padding = padding,
                                               name    = 'max_pool')
    return net, indices, input_shape

def max_unpooling(
    net,
    output_shape,
    indices
    ):
    
    """
    Max unpolling by indices and output_shape.
    
    Args:
    (1) net          : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    (2) output_shape : The shape to be restored to.
    (3) indices      : An indices of the max pooling. More detail in "max_pooling"
    
    Return:
    (1) net : An output tensor after max unpooling.
    """
    
    input_shape = net.get_shape().as_list()
    
    # Calculate indices for batch, height, width and channel
    meshgrid = tf.meshgrid(tf.range(input_shape[1]), tf.range(input_shape[0]), tf.range(input_shape[2]), tf.range(input_shape[3]))
    b = tf.cast(meshgrid[1], tf.int64)
    h = indices // (output_shape[2] * output_shape[3])
    w = indices // output_shape[3] - h * output_shape[2]
    c = indices - (h * output_shape[2] + w) * output_shape[3]
    
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, h, w, c]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    net = tf.scatter_nd(indices, values, output_shape)
        
    return net

def batch_norm(
    net,
    is_training,
    scope
    ):
    
    """
    Batch Normalization.
    
    Args:
    (1) net : Input. An 4D tensor. Shape=[Batch_Size, Image_Height, Image_Width, Image_Depth]
    
    Return:
    (1) net : An output tensor after max unpooling.
    """
    
    # Detail : https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    net = tf.contrib.layers.batch_norm(
                inputs                 = net, 
                decay                  = 0.95,
                center                 = True,
                scale                  = True,
                epsilon                = 0.001,
                activation_fn          = None,
                param_initializers     = None,
                param_regularizers     = None,
                updates_collections    = tf.GraphKeys.UPDATE_OPS,
                is_training            = is_training,
                reuse                  = None,
                variables_collections  = None,
                outputs_collections    = None,
                trainable              = True,
                batch_weights          = None,
                fused                  = False,
                #data_format           = DATA_FORMAT_NHWC,
                zero_debias_moving_mean= False,
                scope                  = scope,
                renorm                 = False,
                renorm_clipping        = None,
                renorm_decay           = 0.99)
    
    return net

#============#
#    Model   #
#============#
def ID_A061054( # name it with "ID_{StudentID}"
    net,
        is_training,
        initializer,
        class_num,
    scope = None
    ):
    
    with tf.variable_scope(scope):
	with tf.variable_scope("layer1_1"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 12, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer1_1_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
         with tf.variable_scope("layer1_2"):
            net = dilated_conv2D( net,
                          kernel_size    = 3,
                          rate       = 6, 
                          output_channel = 64,
                          initializer    = initializer, 
                          padding        = "SAME", 
                          scope          = "conv3x3")
            net = batch_norm(net, is_training ,scope = "layer1_2_bn")                           
            net = tf.nn.relu(features = net, name = "ReLU")	    
            net, indices1, unpool_shape1 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
            # Apply Dropout (if is_training is False, dropout is not applied)
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
            #net = tf.nn.dropout(net, keep_prob=dropout)
	with tf.variable_scope("layer2_1"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 6, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer2_1_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
	with tf.variable_scope("layer2_2"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer2_2_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
	    net, indices2, unpool_shape2 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
	with tf.variable_scope("layer3_1"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer3_1_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
	with tf.variable_scope("layer3_2"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer3_2_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
	with tf.variable_scope("layer3_3"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer3_3_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
	    net, indices3, unpool_shape3 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)
	with tf.variable_scope("layer4_1"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer4_1_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)

	with tf.variable_scope("layer4_2"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer4_2_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)

	with tf.variable_scope("layer4_3"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer4_3_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
	   net, indices4, unpool_shape4 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)

	with tf.variable_scope("layer5_1"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer5_1_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)

	with tf.variable_scope("layer5_2"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = 64,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            net = batch_norm(net, is_training ,scope = "layer5_2_bn")                       
            net = tf.nn.relu(features = net, name = "ReLU")	
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)

	with tf.variable_scope("layer5_3"):
	    net = dilated_conv2D( net,
                    kernel_size    = 3,
                    rate       = 1, 
                    output_channel = class_num,
                    initializer    = initializer, 
                    padding        = "SAME", 
                    scope          = "conv3x3")                       
            #net = batch_norm(net, is_training ,scope = "layer5_3_bn")                       
            #net = tf.nn.relu(features = net, name = "ReLU")	
	   #net, indices5, unpool_shape5s = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
            #net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.5), lambda: net)

        #net = net
    return net

def ID_0550225(
    net, 
    is_training,
    initializer,
    class_num,
    scope = None
    ):
    with tf.variable_scope(scope):
        with tf.variable_scope("layer0"):
            net = conv2D( net,
                          kernel_size    = 3,
                          strides        = 1, 
                          output_channel = 64,
                          initializer    = initializer, 
                          padding        = "SAME", 
                          scope          = "conv3x3")
                        
            net = batch_norm(net, is_training, scope = "layer0_bn")
                            
            net = tf.nn.relu(features = net, name = "ReLU")		
                            
        with tf.variable_scope("layer1"): # With Pooling
            net = conv2D( net,
                          kernel_size    = 3,
                          strides        = 1, 
                          output_channel = 64,
                          initializer    = initializer, 
                          padding        = "SAME", 
                          scope          = "conv3x3")
                        
            net = batch_norm(net, is_training, scope = "layer1_bn")
                            
            net = tf.nn.relu(features = net, name = "ReLU")
    
            net, indices0, unpool_shape0 = max_pooling(net, kernel_size = 2, strides = 2, padding = "SAME")
                                                        
        with tf.variable_scope("layer2"): # With Unpooling
            net = max_unpooling(net, output_shape = unpool_shape0, indices = indices0)
            
            net = conv2D( net,
                          kernel_size    = 3,
                          strides        = 1, 
                          output_channel = 64,
                          initializer    = initializer, 
                          padding        = "SAME", 
                          scope          = "conv3x3")
                        
            net = batch_norm(net, is_training, scope = "layer2_bn")
                            
            net = tf.nn.relu(features = net, name = "ReLU")										
            
            net = tf.cond(is_training, lambda: tf.layers.dropout(net, 0.9), lambda: net)
            
        with tf.variable_scope("layer3"):
            net = conv2D( net,
                          kernel_size    = 3,
                          strides        = 1, 
                          output_channel = class_num,
                          initializer    = initializer, 
                          padding        = "SAME", 
                          scope          = "conv3x3")
                        
            net = tf.nn.relu(features = net, name = "ReLU")
    return net


# Here are some different types of convolution.
# Hopefully, you can be easier to construct your designed model
# but not limited by the unfamiliarity with the tensorflow
#-------------------------------
# (1) redisual bottleneck block
# (2) dilated convolution
# (3) depthwise block
#-------------------------------
def residual_bottleneck_block(
    net,
    kernel_size,
    strides,
    output_channel,
    internal_channel,
    initializer,
    is_training,
    padding,
    scope
    ):
    
    with tf.variable_scope(scope):
        
        input_channel = net.get_shape().as_list()[-1]
        
        # -- shortcut --
        if input_channel != output_channel:
            shortcut = conv2D( net,
                               kernel_size    = 1,
                               strides        = 1, 
                               output_channel = output_channel,
                               initializer    = initializer, 
                               padding        = padding, 
                               scope          = "shortcut_conv")
                               
            shortcut = batch_norm(shortcut, is_training, scope = "shortcut_bn")
        else:
            shortcut = net
        
        # -- dimension reduction --
        net = conv2D( net,
                      kernel_size    = 1,
                      strides        = 1, 
                      output_channel = internal_channel,
                      initializer    = initializer, 
                      padding        = padding, 
                      scope          = "1x1_reduction")
        
        net = batch_norm(net, is_training, scope = "1x1_reduction_bn")
        
        net = tf.nn.relu(features = net, name = "ReLU")
        
        # -- convolution --
        net = conv2D( net,
                      kernel_size    = kernel_size,
                      strides        = strides, 
                      output_channel = internal_channel,
                      initializer    = initializer, 
                      padding        = padding, 
                      scope          = "conv")

        net = batch_norm(net, is_training, scope = "conv_bn")
        
        net = tf.nn.relu(features = net, name = "ReLU")
        
        # -- dimension recovery --
        net = conv2D( net,
                      kernel_size    = 1,
                      strides        = 1, 
                      output_channel = output_channel,
                      initializer    = initializer, 
                      padding        = padding, 
                      scope          = "1x1_recovery")              
                      
        net = batch_norm(net, is_training, scope = "1x1_recovery_bn")
        
        # -- add shortcut --
        net = tf.add(net, shortcut)
        
        net = tf.nn.relu(features = net, name = "ReLU")
        
    return net
    
def dilated_conv2D(
    net,
    kernel_size,
    rate, 
    output_channel,
    initializer = tf.contrib.layers.variance_scaling_initializer(), 
    padding = "SAME", 
    scope = None
    ):
    
    input_channel = net.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        # Define Weights Variable
        weights = tf.get_variable( name        = "weights", 
                                   shape       = [kernel_size, kernel_size, input_channel, output_channel],
                                   dtype       = tf.float32,
                                   initializer = initializer)
        # Define Biases Variable				
        biases = tf.get_variable( name        = "biases", 
                                  shape       = [output_channel],
                                  dtype       = tf.float32,
                                  initializer = initializer)
        
        tf.add_to_collection("weights", weights)
        
        # Dilated Convolution 
        # https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
        net = tf.nn.atrous_conv2d( value   = net, 
                                   filters = weights, 
                                   rate    = rate, 
                                   padding = padding,
                                   name    = "dilated_conv")     
    
        # Add Biases
        net = tf.nn.bias_add(net, biases)
        
    return net
    
def depthwise_block(
    net,
    kernel_size,
    strides, 
    output_channel,
    initializer = tf.contrib.layers.variance_scaling_initializer(), 
    padding = "SAME", 
    scope = None
    ):
    
    input_channel = net.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        # -- depthwise convolution --
        # Define Weights Variable
        weights = tf.get_variable( name        = "weights", 
                                   shape       = [kernel_size, kernel_size, input_channel, 1],
                                   dtype       = tf.float32,
                                   initializer = initializer)
        # Define Biases Variable				
        biases = tf.get_variable( name        = "biases", 
                                  shape       = [input_channel],
                                  dtype       = tf.float32,
                                  initializer = initializer)
        
        tf.add_to_collection("weights", weights)
        
        # Depthwise Convolution 
        # https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
        net = tf.nn.depthwise_conv2d( input   = net, 
                                      filter  = weights, 
                                      strides = [1, strides, strides, 1], 
                                      padding = padding,
                                      rate    = [1, 1],
                                      name    = "depthwise_conv") 
    
        # Add Biases
        net = tf.nn.bias_add(net, biases)
        
        net = batch_norm(net, is_training, scope = "depthwise_conv_bn")
        
        net = tf.nn.relu(features = net, name = "ReLU")
        
        # -- 1x1 convolution --
        net = conv2D( net,
                      kernel_size    = 1,
                      strides        = 1, 
                      output_channel = output_channel,
                      initializer    = initializer, 
                      padding        = padding, 
                      scope          = "1x1_conv")

        net = batch_norm(net, is_training, scope = "1x1_conv_bn")
        
        net = tf.nn.relu(features = net, name = "ReLU")
        
    return net
  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
