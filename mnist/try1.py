"""
This is my first attempt at using TensorFlow 2 to solve the MNIST handwritten digit classifier. I used
this resource: http://neuralnetworksanddeeplearning.com/chap1.html
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def each_max(t: tf.Tensor):
    return [tf.argmax(x) for x in t.numpy()]

def loss_class(val: tf.Tensor, expected: tf.Tensor):
    compare = val.numpy()
    compare.fill(0)
    # print(compare)
    compare[0, expected] = 1
    # print(compare)
    # tf.nn.l2_l

    return tf.reduce_sum(tf.square(val - compare)) 

class MnistClassifier(tf.keras.models.Sequential):

    def __init__(self, hiddenLayers, hiddenlength: int):
        super().__init__()

        act = tf.keras.activations.sigmoid

        self.add(tf.keras.Input(shape=(1, 28*28)))
        # self.model.add(tf.keras.layers.Flatten(input_shape=inputShape))
        # self.model.add(tf.keras.layers.Reshape((inputShape[0] * inputShape[1],), input_shape=inputShape))
        # print(self.model.output_shape)
        for _ in range(hiddenLayers):
            self.add(tf.keras.layers.Dense(hiddenlength, activation=act))
        self.add(tf.keras.layers.Dense(10, activation=act))
       
    # def classify(self, image: tf.Tensor):
    #     # print(tf.reshape(image, (28*28, )))
    #     # return self.model(image)
        
    #     ret = self.model(tf.reshape(image, (1, 28*28)))

    #     # print(each_max(ret))
    #     # print([tf.reduce_max(x) for x in ret.numpy()])
    #     return ret

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    image = image / 255  # from int to float
    return tf.reshape(image, (1, 28*28))

def test():
    ds_train = tfds.load('mnist', split='train', shuffle_files=True)
    mc = MnistClassifier(1, 30)

    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    fw = tf.summary.create_file_writer('./logs/try2/' + sys.argv[1])
    fw.set_as_default()
    count = 0
    print('training')
    for row in ds_train:
        image = normalize_image(row['image'])
        label = row['label']
        with tf.GradientTape() as tape:
            loss = loss_class(mc(image), label)
        grads = tape.gradient(loss, mc.trainable_variables)
        opt.apply_gradients(zip(grads, mc.trainable_variables))
        
        count += 1
        if (count % 100 == 0):
            tf.summary.scalar('loss_mse_train', loss, step=count)
    
    print()
    print('testing:')
    ds_test = tfds.load('mnist', split='test', shuffle_files=True)
    for row in ds_test.take(5):
        image = normalize_image(row['image'])
        label = row['label']
        out = mc(image)
        print("predicted: {0}, received: {1}".format(tf.argmax(out[0]).numpy(), label.numpy()))


if __name__ == '__main__':
    test()

