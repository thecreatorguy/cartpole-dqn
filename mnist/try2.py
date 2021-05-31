
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.run_functions_eagerly(True)

def batchsize(tensor: tf.Tensor):
    return tensor.shape[0]


@tf.function
def MSE(trueVal: tf.Tensor, predVal: tf.Tensor):
    # print(tf.reduce_sum(tf.square(trueVal - predVal), (1)))   trueval shape = (batchsize, 10) -> (batchsize) -> mean of this
    return tf.reduce_mean(tf.reduce_sum(tf.square(trueVal - predVal), 1), 0)# / batchsize(trueVal) -> x -> [[x]] (1,1)

#! cross entropy loss with softmax activation


@tf.function
def sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))
    #! look into gradient vanishing/exploding

@tf.function
def softmax(x):
    ex = tf.exp(x)
    total = tf.reduce_sum(ex, 1, keepdims=True)
    return ex / total

@tf.function
def relu(x):
    return tf.maximum(x, 0)

def one_hot_digit(x):
    compare = [0] * 10
    compare[x] = 1
    return tf.constant(compare, dtype=tf.dtypes.float32)


def preprocess_images(images: tf.Tensor) -> tf.Tensor:
    images = images / 255  # from int to float
    return tf.reshape(images, (batchsize(images), 28*28))


def preprocess_labels(labels: tf.Tensor) -> tf.Tensor:
    return [one_hot_digit(int(x)) for x in labels.numpy()]

class Dense(tf.Module):
    def __init__(self, outLen, activation):
        super().__init__()
        self.outLen = outLen
        self.b = tf.Variable(tf.random.uniform((1, outLen), -1, 1))
        self.w = None
        self.activation = activation

    @tf.function
    def __call__(self, x: tf.Tensor):
        if self.w is None:
            # todo: look into how this works
            #! xavier he's method?
            self.w = tf.Variable(tf.random.uniform((x.shape[1], self.outLen), -1, 1))
        z = tf.matmul(x, self.w) + self.b
        print(z)
        return self.activation(z)

class MnistClassifier(tf.Module):
    def __init__(self):
        super().__init__(name='MnistClassifier')

        # Hyper Parameters
        input_size = 28*28
        hidden_layers = 1
        hidden_height = 30
        output_height = 10

        self.layers = [Dense(hidden_height, relu) for _ in range(hidden_layers)]
        self.layers.append(Dense(output_height, softmax))
    
    @tf.function
    def forward_pass(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @tf.function
    def __call__(self, x):
        return tf.argmax(self.forward_pass(x)[0,])

    @tf.function
    def train_step(self, batch: tf.Tensor, expected: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = MSE(self.forward_pass(batch), expected)
        grads = tape.gradient(loss, self.trainable_variables)
        return loss, grads
        

def test():
    batch_size = 100

    fw = tf.summary.create_file_writer('./logs/try2/' + sys.argv[1])
    fw.set_as_default()
    
    mc = MnistClassifier()
    opt = tf.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    opt = tf.optimizers.Adam()

    step = 0
    ds_train = tfds.load('mnist', split='train', batch_size=batch_size)
    ds_train = ds_train.shuffle(ds_train.cardinality() * 2, reshuffle_each_iteration=True)
    ds_train = ds_train.repeat(100)
    # print(ds_train.cardinality())
    # return
    for batch in ds_train:
        images = preprocess_images(batch['image'])
        labels = preprocess_labels(batch['label'])
        print()
        print(step)
        
        loss, grads = mc.train_step(images, labels)
        # print('loss', loss)
        
        # print('grads:', grads)
        # print(mc.trainable_variables)
        if math.isnan(loss):
            print(grads)
            print(mc.trainable_variables)
            return
        opt.apply_gradients(zip(grads, mc.trainable_variables))
        # print('variables:', mc.trainable_variables)
        # print('min', tf.reduce_min(tf.abs(mc.layers[0].w)))
        
        # return

        step += batch_size
        tf.summary.scalar('loss_mse_train', loss, step=step)
    
    
    ds_test = tfds.load('mnist', split='test', batch_size=1)
    correct = 0
    for row in ds_test:
        images = preprocess_images(row['image'])
        labels = preprocess_labels(row['label'])
        if mc(images) == row['label'].numpy():
            correct += 1

    tf.summary.scalar('loss_mse_test_avg', correct / tf.cast(ds_test.cardinality(), tf.dtypes.float32), step=0)

if __name__ == '__main__':
    test()