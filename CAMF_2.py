from __future__ import print_function, division

import scipy

import matplotlib

matplotlib.use('Agg')

from keras import backend as KB
from keras.datasets import mnist
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import InputLayer,Input, Dense, Reshape, Flatten, Dropout, Concatenate,Convolution2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import label_binarize


class CAMF():
	def __init__(self, weights, img_height = 64, img_width = 64, w = True):
		# Input shape
		self.channels = 1
		self.img_height = img_height                
		self.img_width = img_width        
		self.img_shape = (self.img_height, self.img_width, self.channels)
		self.n_residual_blocks = 4
		optimizer = Adam(lr = 0.0005)
		self.data_loader = DataLoader()
		self.detector = self.build_detector()
		plot_model(self.detector, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
		self.weights = w
		if self.weights == True:
			self.detector.load_weights(weights)
			print('weight loaded')
		self.detector.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	        

	def build_detector(self):

		def d_block(layer_input, filters, strides=1, bn=True):

			d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
            
			d = LeakyReLU(alpha=0.2)(d)
			return d


		def residual_block(layer_input, filters,name = None):
			"""Residual block described in paper"""
			d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
			d = BatchNormalization(momentum=0.8)(d)
			d = LeakyReLU(alpha=0.2)(d)
			d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name = 'conv_'+name)(d)
			d = BatchNormalization(momentum=0.8)(d)
			d = Add()([d, layer_input])
			d = LeakyReLU(alpha=0.2)(d)
                        #d = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(d)
			return d



	        # Input img
		inp = Input(shape=self.img_shape)

		'''d1 = d_block(d0, self.df)
		d2 = d_block(d1, self.df, strides=2)
		d3 = d_block(d2, self.df*2)
		d4 = d_block(d3, self.df*2, strides=2)
		d5 = d_block(d4, self.df*4)
		d6 = d_block(d5, self.df*4, strides=2)
		d7 = d_block(d6, self.df*8)
		d8 = d_block(d7, self.df*8, strides=2)
		d9 = Dense(self.df*16, name = 'dense_d9')(d8)
		d10 = LeakyReLU(alpha=0.2)(d9)
		validity = Dense(1, activation='sigmoid')(d10)'''
	


		r = residual_block(inp, 64, name = 'r0')
		for i in range(self.n_residual_blocks - 1):
			r = residual_block(r, 64, name = 'r'+str(i+1))

		# Post-residual block
		d1 = Conv2D(3, kernel_size=3, strides=1, kernel_regularizer=regularizers.l2(0.01),padding='same')(r)
		#d1 = d_block(r, 32, strides=2)
		d1 = Add()([d1, inp])
		d1 = d_block(d1, 32, strides = 2)
		d1 = GlobalAveragePooling2D()(d1)
		#validity = Dense(3, activation='sigmoid')(d1)
		validity = Dense(2, activation='sigmoid')(d1)
                
		return Model(inp, validity)

		


	def train(self, epochs, batch_size=128, test_interval=20):
		start_time_1 = datetime.datetime.now()
		loss = []
		accuracy = []
		test_loss = []
		test_accuracy = []
		tfsess_ = tf.Session()
		KB.set_session(tfsess_)
		sess_ = KB.get_session()
		sess_.run([tf.local_variables_initializer(),
					tf.global_variables_initializer()])
		elapsed_time = start_time_1
		for epoch in range(epochs):
			# Sample images and their conditioning counterparts
			X_train, y_train = self.data_loader.load_data(samples_per_class = 1000,patch_size = 256, mode = 'train')
			X_test, y_test = self.data_loader.load_data(samples_per_class = 400, patch_size = 256, mode = 'test')
			start_time = datetime.datetime.now()
			y_train = tf.one_hot(y_train, depth=2)
			y_test = tf.one_hot(y_test, depth=2)
			
			y_train = sess_.run(y_train)
			y_test  = sess_.run(y_test)
			elapsed_time = datetime.datetime.now() - elapsed_time
			print ("Training_Testing Data generated time: %s" % (elapsed_time))
			#X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, random_state=42)
			print("Training data:", X_train.shape[0],"Testing data:", X_test.shape[0], "Batch_size: 128")
			datagen = ImageDataGenerator()
			datagen.fit(X_train)
                        
			#print(len(X_train))
			batches = 0
			for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=128):
				
				
				
				history = self.detector.fit(x_batch, y_batch, verbose=0)
				batches += 1
				if batches >= X_train.shape[0] / batch_size:
				# we need to break the loop by hand because
				# the generator loops indefinitely
					break
			
			loss.append(np.array(history.history['loss'])[0])
			accuracy.append(np.array(history.history['acc'])[0])
			# Plot the progress
			
			
			
			if epoch % test_interval == 0:
				datagen_test = ImageDataGenerator()
				datagen_test.fit(X_test)
				batches = 0
				for x_batch, y_batch in datagen.flow(X_test, y_test, batch_size=batch_size):
					test_loss_acc = self.detector.evaluate(x_batch, y_batch, verbose=0)
					batches += 1
					if batches >= X_train.shape[0] // batch_size:
					# we need to break the loop by hand because
					# the generator loops indefinitely
						break
					elapsed_time = datetime.datetime.now() - start_time
				print ("Epoch %d time: %s,\nTraining--- loss: %f, acc: %f" % (epoch, elapsed_time, np.array(history.history['loss']), np.array(history.history['acc'])))
				print ("Validating--- loss: %f, acc: %f" % (test_loss_acc[0], test_loss_acc[1]))



						
				test_loss.append(test_loss_acc[0])
				test_accuracy.append(test_loss_acc[1])
				
				self.detector.save_weights('./2class_256dectector_weights'+str(epoch)+'.h5')
				plt.figure(100)
				plt.plot(loss)
				plt.plot(accuracy)
				plt.savefig('Train_256.png')
				plt.figure(200)
				plt.plot(test_loss)
				plt.plot(test_accuracy)
				plt.savefig('Test_128.png')
				data = [loss, test_loss, accuracy, test_accuracy]
        
				with open('log_2_256class.csv', 'w') as writeFile:
					writer = csv.writer(writeFile)
					writer.writerows(data)
		print("Training Completed in ")
		#elapsed_time = elapsed_time - start_time_1
		#print(elapsed_time, '\n Testing...')
		#self.test(epoch)				
		KB.clear_session()


	def test(self,name):
		imgs, labels = self.data_loader.load_data(samples_per_class = 2048, patch_size = 64, mode = 'test')
		
		#labels = sess_.run(tf.one_hot(labels, depth=2))
		
		datagen = ImageDataGenerator()
		datagen.fit(imgs)
		n_classes = 2       
		#print(len(X_train))
		batches = 0
		pred_labels = []
		l = []
		for x_batch, y_batch in datagen.flow(imgs, labels, batch_size=128):
			pred_ = self.detector.predict(x_batch)
			#Sprint (pred_)
			batches += 1
			for i in range(128):
				pred_labels.append(pred_[i])
				l.append(y_batch[i])

			if batches >= imgs.shape[0] // 128:
				# we need to break the loop by hand because
				# the generator loops indefinitely
				break
		
		max_pred = np.argmax(np.asarray(pred_labels), axis=-1)
		max_pred_ = np.max(pred_labels, axis=-1)
		max_labels = l #np.argmax(l, axis=-1)
		#print(pred_labels, max_pred_, max_labels)
		cnf_mat = confusion_matrix(max_labels , max_pred)
		print("Confusion Matrix:\n", cnf_mat)
		#y = label_binarize(np.asarray(l), classes=[0, 1])
		#plot_roc(pred_labels, y)
		#print(y.shape[1])
		# Compute ROC curve and ROC area for each class
		np.savetxt(name+'.csv', np.asarray([np.asarray(l),np.asarray(pred_labels)[:,1]]), delimiter=",")
		fpr, tpr, thresholds = roc_curve(np.asarray(l),np.asarray(pred_labels)[:,1])
		roc_auc = auc(fpr, tpr)

        
		fig = plt.figure()
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
		#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([-0.05, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
		plt.show()
		fig.savefig('./'+name+'.png')        
       


    











if __name__ == '__main__':
    #camf = CAMF(img_height = 256, img_width = 256)
    #camf.train(epochs=70, batch_size=128, test_interval=1)
    for i in [20,5,11,28,6,44,33,13,41,35,45,32,38,63,37]:
        camf = CAMF(img_height = 64, img_width = 64, weights = '2class_dectector_weights'+str(i)+'.h5')
        camf.test(name = 'ROC_64_4096_w'+str(i) )
    

