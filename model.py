import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from mode.config import *
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import TFOptimizer
from tensorflow.keras import backend as K
import tensorflow as tf
#from mutli_gpu import ParallelModel

import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
class ParallelModel(Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """
    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        super(ParallelModel, self).__init__() # Thanks to @greatken999 for fixing bugs
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)
    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)
    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)
    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}
        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])
        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)
        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # If outputs are numbers without dimensions, add a batch dim.
                def add_dim(tensor):
                    """Add a dimension to tensors that don't have any."""
                    if K.int_shape(tensor) == ():
                        return KL.Lambda(lambda t: K.reshape(t, [1, 1]))(tensor)
                    return tensor
                outputs = list(map(add_dim, outputs))
                # Concatenate
                merged.append(KL.Concatenate(axis=0, name=name)(outputs))
        return merged



arg = command_arguments()
learning_rate = arg.learning_rate
learning_decay_rate = arg.learning_decay_rate

img_size = (128,64,1) # 640 * 512 grayscale img with 1 channel
dr_rate = 0.6 # never mind
leakyrelu_alpha = 0.3
'''def customLoss(yTrue,yPred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    true = tf.slice(yTrue,[0,0,0,0],size=[16,640,512,2])
    pre = tf.slice(yPred,[0,0,0,0],size=[16,640,512,2])
    loss = bce(true, pre)
    return loss
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    true = tf.slice(yTrue,[0,0,0,0],size=[16,640,512,3])
    pre = tf.slice(yPred,[0,0,0,0],size=[16,640,512,3])
    loss = cce(true, pre)
    return loss'''
def weighted_loss(yTrue,yPred,margin=1):

    result =  yTrue * -tf.math.log(yPred) * (1/4)
    loss = tf.reduce_sum(result)

    return loss
def contrastive_loss(yTrue,yPred,margin = 1):
    result = 0
    for i in range(4):
        cut1 = tf.slice(yTrue,[i,0,0,0],[1,32,32,3])
        cut2 = tf.slice(yTrue,[i,32,0,0],[1,32,32,3])
        a = tf.add(cut1,cut2)
        x = tf.reduce_sum(a)
        sum_ = tf.get_static_value(x)
        
        if sum_ == 0.0:
            label = 0
        else:
            label = 1
        c1 = tf.slice(yPred,[i,0,0,0],[1,32,32,3])
        c2 = tf.slice(yPred,[i,32,0,0],[1,32,32,3])

        r = tf.subtract(c1, c2)
        d = tf.math.reduce_sum(tf.math.square(r), axis=1, keepdims=True)
        d_sqrt = tf.sqrt(d)
        
        loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d
        
        loss = 0.5 * tf.reduce_mean(loss)
        result += loss
    return result 

def btrflynet(pretrained_weights,input_size = img_size):
    
    '''input_ = Input((64,32,1))
    inputs_anterior = tf.slice(input_,[0,0,0,0],[4,32,32,1])
    inputs_posterior = tf.slice(input_,[0,32,0,0],[4,32,32,1])'''
    inputs_anterior = Input((32,32,1))
    inputs_posterior = Input((32,32,1))
    
    up_conv1 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs_anterior)
    up_conv1 = BatchNormalization()(up_conv1)
    up_conv1 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv1)
    up_conv1 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(up_conv1)
    up_conv1 = BatchNormalization()(up_conv1)    
    up_conv1 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv1)
    up_pool1 = MaxPooling2D(pool_size=(2, 2))(up_conv1)
    
    up_conv2 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(up_pool1)
    up_conv2 = BatchNormalization()(up_conv2)
    up_conv2 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv2)
    up_pool2 = MaxPooling2D(pool_size=(2, 2))(up_conv2)
    
    up_conv3 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(up_pool2)
    up_conv3 = BatchNormalization()(up_conv3)
    up_conv3 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv3)
    up_pool3 = MaxPooling2D(pool_size=(2, 2))(up_conv3)
    
    
    
    down_conv1 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs_posterior)
    down_conv1 = BatchNormalization()(down_conv1)
    down_conv1 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv1)
    down_conv1 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(down_conv1)
    down_conv1 = BatchNormalization()(down_conv1)    
    down_conv1 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv1)
    down_pool1 = MaxPooling2D(pool_size=(2, 2))(down_conv1)
    
    down_conv2 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(down_pool1)
    down_conv2 = BatchNormalization()(down_conv2)
    down_conv2 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv2)
    down_pool2 = MaxPooling2D(pool_size=(2, 2))(down_conv2)
    
    down_conv3 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(down_pool2)
    down_conv3 = BatchNormalization()(down_conv3)
    down_conv3 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv3)
    down_pool3 = MaxPooling2D(pool_size=(2, 2))(down_conv3)
    
    
    
    concat_middle1 = concatenate([up_pool3,down_pool3], axis = -1)
    middle1 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_middle1)
    middle1 = BatchNormalization()(middle1)
    middle1 = LeakyReLU(alpha=leakyrelu_alpha)(middle1) 
    middle1_pool = MaxPooling2D(pool_size=(2, 2))(middle1)
    
    middle2 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(middle1_pool)
    middle2 = BatchNormalization()(middle2)
    middle2 = LeakyReLU(alpha=leakyrelu_alpha)(middle2) 
    middle2_pool = MaxPooling2D(pool_size=(2, 2))(middle2)
    
    middle3 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(middle2_pool)
    middle3 = BatchNormalization()(middle3)
    middle3 = LeakyReLU(alpha=leakyrelu_alpha)(middle3) 

    middle3_us = Conv2D(1024, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(middle3))
    middle3_us = BatchNormalization()(middle3_us)   
    middle3_us = LeakyReLU(alpha=leakyrelu_alpha)(middle3_us)
    concat_middle3 = concatenate([middle2,middle3_us], axis = -1)
    
    middle4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_middle3)
    middle4 = BatchNormalization()(middle4)
    middle4 = LeakyReLU(alpha=leakyrelu_alpha)(middle4)
    
    middle4_us = Conv2D(512, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(middle4))
    middle4_us = BatchNormalization()(middle4_us)   
    middle4_us = LeakyReLU(alpha=leakyrelu_alpha)(middle4_us)
    concat_middle4 = concatenate([middle1,middle4_us], axis = -1)
    
    
    up_conv4_us = Conv2D(512, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(concat_middle4))
    up_conv4_us = BatchNormalization()(up_conv4_us)   
    up_conv4_us = LeakyReLU(alpha=leakyrelu_alpha)(up_conv4_us)
    concat_up_conv4 = concatenate([up_conv3,up_conv4_us], axis = -1)
    
    up_conv5 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_up_conv4)
    up_conv5 = BatchNormalization()(up_conv5)
    up_conv5 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv5)
    
    up_conv6_us = Conv2D(256, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up_conv5))
    up_conv6_us = BatchNormalization()(up_conv6_us)   
    up_conv6_us = LeakyReLU(alpha=leakyrelu_alpha)(up_conv6_us)
    concat_up_conv6 = concatenate([up_conv2,up_conv6_us], axis = -1)
    
    up_conv7 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_up_conv6)
    up_conv7 = BatchNormalization()(up_conv7)
    up_conv7 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv7)
    
    up_conv8_us = Conv2D(128, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(up_conv7))
    up_conv8_us = BatchNormalization()(up_conv8_us)   
    up_conv8_us = LeakyReLU(alpha=leakyrelu_alpha)(up_conv8_us)
    concat_up_conv8 = concatenate([up_conv1,up_conv8_us], axis = -1)
    
    up_conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_up_conv8)
    up_conv9 = BatchNormalization()(up_conv9)
    up_conv9 = LeakyReLU(alpha=leakyrelu_alpha)(up_conv9)
    
    
    down_conv4_us = Conv2D(512, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(concat_middle4))
    down_conv4_us = BatchNormalization()(down_conv4_us)   
    down_conv4_us = LeakyReLU(alpha=leakyrelu_alpha)(down_conv4_us)
    concat_down_conv4 = concatenate([down_conv3,down_conv4_us], axis = -1)
    
    down_conv5 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_down_conv4)
    down_conv5 = BatchNormalization()(down_conv5)
    down_conv5 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv5)
    
    down_conv6_us = Conv2D(256, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(down_conv5))
    down_conv6_us = BatchNormalization()(down_conv6_us)   
    down_conv6_us = LeakyReLU(alpha=leakyrelu_alpha)(down_conv6_us)
    concat_down_conv6 = concatenate([down_conv2,down_conv6_us], axis = -1)
    
    down_conv7 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_down_conv6)
    down_conv7 = BatchNormalization()(down_conv7)
    down_conv7 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv7)
    
    down_conv8_us = Conv2D(128, 4, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(down_conv7))
    down_conv8_us = BatchNormalization()(down_conv8_us)   
    down_conv8_us = LeakyReLU(alpha=leakyrelu_alpha)(down_conv8_us)
    concat_down_conv8 = concatenate([down_conv1,down_conv8_us], axis = -1)
    
    down_conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(concat_down_conv8)
    down_conv9 = BatchNormalization()(down_conv9)
    down_conv9 = LeakyReLU(alpha=leakyrelu_alpha)(down_conv9)
    
    up_conv10 = Conv2D(3, 1, activation = 'softmax')(up_conv9)
    down_conv10 = Conv2D(3, 1, activation = 'softmax')(down_conv9)

    
    output_1 = tf.concat([up_conv10, down_conv10], axis=1, name = 'contrastive_loss')
    output_2 = tf.concat([up_conv10, down_conv10], axis=1, name = 'weighted_loss')
    
    model = Model(inputs = [inputs_anterior,inputs_posterior], outputs = [output_1,output_2])  

    model.summary()
    model.compile(optimizer = Adam(lr = learning_rate, decay = learning_decay_rate), 
              loss={
                  'tf_op_layer_contrastive_loss': contrastive_loss,
                  'tf_op_layer_weighted_loss': weighted_loss},
              loss_weights={
                  'tf_op_layer_contrastive_loss': 1.,
                  'tf_op_layer_weighted_loss': 1.}, 
              metrics = ['accuracy'])
    if(pretrained_weights != ""):
        print("********************************************")
        print("********************************************")    
        print("***********using pretrained model***********")
        print("********************************************")
        print("********************************************")
        model.load_weights(pretrained_weights)
    return model
