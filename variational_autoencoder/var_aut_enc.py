import tensorflow as tf 
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
train_X = mnist.train.images
train_Y = mnist.train.labels
test_X = mnist.test.images
test_Y = mnist.test.labels

input_dim = 28
n_input = input_dim * input_dim

#Necessary Functions
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def mul_mat(X,W,b):
	return tf.matmul(X,W) + b

#placeholder
X = tf.placeholder(tf.float32, shape = ([None, n_input]))

#encoder

latent_dim = 20
hidden_dim = 500     #2/3 rd of 784

#layer 1
W_enc = weight_variable([n_input,hidden_dim])
b_enc = bias_variable([1,hidden_dim])
#activation 1
h_enc = tf.nn.tanh(mul_mat(X,W_enc,b_enc))  #tanh used since it removes vanishing gradient problem

#layer 2

#mean
W_mean = weight_variable([hidden_dim,latent_dim])
b_mean = bias_variable([1,latent_dim])
h_mean = mul_mat(h_enc, W_mean, b_mean)

#std
W_std = weight_variable([hidden_dim, latent_dim])
b_std = bias_variable([1,latent_dim])
h_std = mul_mat(h_enc, W_std, b_std)

noise = tf.random_normal([1,latent_dim])

#latent variable(passed to decoder)
z = h_mean + np.dot(noise,tf.exp(0.5*h_std))

#decoder

#layer 1
W_dec = weight_variable([latent_dim,hidden_dim])
b_dec = bias_variable([1,hidden_dim])
h_dec = tf.nn.tanh(mul_mat(z,W_dec,b_dec))

#layer 2 - recnstruct data
W_rec = weight_variable([hidden_dim,n_input])   #rec - reconstruct
b_rec = bias_variable([1,n_input])
# Reconstructed image...Sigmoid used to produce output between 0 and 1
rec_out = tf.nn.sigmoid(mul_mat(h_dec, W_rec, b_rec))


#Loss Function
# Here loss function has 2 components: (1) log_likelihood which tells us
# about how effectively the decoder has learned to reconstruct
# an input image x given its latent representation z (2) #KL Divergence
#If the encoder outputs representations z that are different 
#than those from a standard normal distribution, it will receive 
#a penalty in the loss. This regularizer term means 
#‘keep the representations z of each digit sufficiently diverse’. 
#If we didn’t include the regularizer, the encoder could learn to cheat
#and give each datapoint a representation in a different region of Euclidean space. 

log_likelihood = tf.reduce_sum( X*tf.log(rec_out+1e-9) + (1-X)*tf.log(1-rec_out+1e-9), axis = 1 )

KL_term = -0.5*tf.reduce_sum(1 + 2*h_std - tf.pow(h_mean, 2) - tf.pow(h_std, 2), axis = 1)

#Loss 
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_iterations = 1000000
recording_interval = 1000
#store values for 3 terms to plot later
variational_lower_bound_array = []
log_likelihood_array = []
KL_term_array = []
iteration_array = [i*recording_interval for i in range(int(num_iterations/recording_interval))]
'''
# Train
with tf.Session() as sess:
	with tf.device('/gpu:0'):	
		sess.run(init)

		for i in range(num_iterations):
			# Training in batches of 200
			x_batch = np.round(mnist.train.next_batch(200)[0])  #np.round makes mnist binary

			sess.run(optimizer, feed_dict = {X: x_batch})

			if(i % recording_interval == 0):
				vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
				print("Iteration: ",i,"   Loss: ", vlb_eval)
				variational_lower_bound_array.append(vlb_eval)
				log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict = {X: x_batch})))	
				KL_term_array.append(np.mean(KL_term.eval(feed_dict = {X: x_batch})))

	#The saver.save command must be kept inside sess but outside the device
	#"/gpu:0" since it is assigned to work with "/cpu:0"
	print("Training completed\n")
	saver = tf.train.Saver([W_enc,b_enc,W_mean,b_mean,W_std,b_std,W_dec,b_dec,W_rec,b_rec ])		
	save_path = saver.save(sess,"/home/sid/deep_learning/codes/variational_autoencoder/weights.ckpt")
	#print("saved in", save_path)
	print("Saved\n")
'''

#plt.figure()
#for the number of iterations we had 
#plot these 3 terms
#plt.plot(iteration_array, variational_lower_bound_array)
#plt.plot(iteration_array, KL_term_array)
#plt.plot(iteration_array, log_likelihood_array)
#plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
#plt.title('Loss per iteration')

#Test 
# import os
# load_model = False
# if load_model:
saver = tf.train.Saver([W_enc,b_enc,W_mean,b_mean,W_std,b_std,W_dec,b_dec,W_rec,b_rec ])		
#	saver.restore(sess, "weights.ckpt")
#saver.restore(sess, os.path.join(os.getcwd(), "Trained Bernoulli VAE"))



num_pairs = 10
image_indices = np.random.randint(0, 200, num_pairs)
#Lets plot 10 digits
for pair in range(num_pairs):
    #reshaping to show original test image
	with tf.Session() as sess:
		#with tf.device('/cpu:0'):	
			sess.run(init)
		    
		    #Always restore weights inside the session
			saver.restore(sess, "weights.ckpt")	
			
			x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_input))
			plt.figure()
			x_image = np.reshape(x, (28,28))
			plt.subplot(121)
			plt.imshow(x_image)
		    #reconstructed image, feed the test image to the decoder
			x_reconstruction = rec_out.eval(feed_dict={X: x})
		    #reshape it to 28x28 pixels
			x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
		    #plot it!
			plt.subplot(122)
			plt.imshow(x_reconstruction_image)	

			plt.show()