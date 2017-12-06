#This is an implementation of digit recognition in the MNIST data set using tensorflow library. This
#code achieves this with the use of a neural network having three hidden layers.

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

n_hidden1 = 1000
n_hidden2 = 800
n_hidden3 = 500

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

W1 = tf.Variable(tf.random_normal([784,n_hidden1],stddev=0.1))
b1 = tf.Variable(tf.zeros([1,n_hidden1]))
W2 = tf.Variable(tf.random_normal([n_hidden1,n_hidden2],stddev=0.1))
b2 = tf.Variable(tf.zeros([1,n_hidden2]))
W3 = tf.Variable(tf.random_normal([n_hidden2,n_hidden3],stddev=0.1))
b3 = tf.Variable(tf.zeros([1,n_hidden3]))
W_out = tf.Variable(tf.random_normal([n_hidden3,10],stddev=0.1))
b_out = tf.Variable(tf.zeros([1,10]))

layer1 = tf.add(tf.matmul(x,W1),b1)
layer1 = tf.nn.relu(layer1)
layer2 = tf.add(tf.matmul(layer1,W2),b2)
layer2 = tf.nn.relu(layer2)
layer3 = tf.add(tf.matmul(layer2,W3),b3)
layer3 = tf.nn.relu(layer3)
output = tf.add(tf.matmul(layer3,W_out),b_out)
output = tf.nn.sigmoid(output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = output))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)
tf.reset_default_graph()

'''
# ******************Training and saving**************************
for i in range(5000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict = {x:batch[0], y_: batch[1]})
	if i%100 == 0:
		print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
print("Training completed\n")
saver = tf.train.Saver([W1,b1,W2,b2,W3,b3,W_out,b_out])
save_path = saver.save(sess,"/home/siddhant/Courses/deep_learning/tensorflow/weight_3layers.ckpt")
print("saved in", save_path)
print("Saved\n")
'''


#***************Restore and Predict***************
saver= tf.train.Saver([W1,b1,W2,b2,W3,b3,W_out,b_out])
saver.restore(sess, "weight_3layers.ckpt")
print("Restored")

predicted = tf.argmax(output,1) + 1
print(sess.run(cost, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
print('predicted = ',predicted.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
print('original = ',tf.argmax(y_,1).eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
print('accuracy = ',sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

	

sess.close()