from kaggleSrc.lung_data_process import *
import tensorflow as tf
import time
import sys
import argparse

def conv3d(x, W):
    """
            tf.nn.conv3d(input, filter, strides, padding, name=None)
            input:  Shape [batch, in_depth, in_height, in_width, in_channels].
            filter:  A Tensor. Must have the same type as input.
            Shape [filter_depth, filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
            strides: A list of ints that has length >= 5. 1-D tensor of length 5.
            The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
            padding: A string from: "SAME", "VALID". The type of padding algorithm to use. "SAME" ---0
            name: A name for the operation (optional).
    """
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    """
            tf.nn.max_pool3d(input, ksize, strides, padding, name=None)
            input: A Tensor Shape [batch, depth, rows, cols, channels] tensor to pool over.
            ksize: A list of ints that has length >= 5. 1-D tensor of length 5.
            The size of the window for each dimension of the input tensor. Must have ksize[0] = ksize[4] = 1.
            strides: A list of ints that has length >= 5. 1-D tensor of length 5.
            The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
            padding: A string from: "SAME", "VALID". The type of padding algorithm to use. "SAME" ---0
            name: A name for the operation (optional).
    """
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x,keep_rate):
	weights={'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
		'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
		'W_fc':tf.Variable(tf.random_normal([54080,1024])),
		'out':tf.Variable(tf.random_normal([1024, 2]))
	}
	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
		'b_conv2':tf.Variable(tf.random_normal([64])),
		'b_fc':tf.Variable(tf.random_normal([1024])),
		'out':tf.Variable(tf.random_normal([2]))
	}
	x = tf.reshape(x,shape=[-1,SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
	#x=tf.reshape(x,shape=[-1,IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT,1])
	conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool3d(conv1)

	conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool3d(conv2)

	fc = tf.reshape(conv2,[-1, 54080])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)
	output = tf.matmul(fc, weights['out'])+biases['out']
	return output

def train_convolutional_neural_network(All_label_data,All_Unlabel_data,Eposhs=200,isTrain=True):
	x = tf.placeholder('float')
	y = tf.placeholder('float')
	keep_rate=tf.placeholder('float')
	prediction = convolutional_neural_network(x,keep_rate)
	prediction_softmax = tf.nn.softmax(prediction)
	softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
	cost = tf.reduce_mean( softmax_cross_entropy_with_logits)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
	correct = tf.equal(tf.argmax(prediction_softmax, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		if isTrain==True:
			print("train begining ::: ")
			sess.run(tf.global_variables_initializer()) # initial all variable
			train_data = All_label_data[:-len(All_label_data)// 8,:]
			validation_data = All_label_data[-len(All_label_data)// 8:,:]
			print("train data length is : ",len(train_data))
			print("validata data length is : ",len(validation_data))
			checkpoint_steps=50
			for i_epoch in range(Eposhs):
				epoch_loss = 0
				for data  in train_data:
					X = data[0]
					Y = data[1]
					if(i_epoch % 10 == 0 or (i_epoch+1) ==Eposhs):
						p,p_s,p_s_c = sess.run([prediction,prediction_softmax,softmax_cross_entropy_with_logits],feed_dict={x: X, y: Y,keep_rate:0.8})
						print("prediction is : %s\nprediction_softmax is :%s\nsoftmax_cross_entropy_with_logits is :%s\n"%(p,p_s,p_s_c))
					_, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y,keep_rate:0.8})
					print("cost is :",c)
					epoch_loss += c
				print('Epoch :', i_epoch+1,'loss:',epoch_loss)
				# save the model
				if (i_epoch+1) % checkpoint_steps == 0 or (i_epoch+1) == Eposhs:
					timeFlag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
					modelPath = '/home/wangbing/scripts/lung_cnn_model/'+'model-'+timeFlag
					if(os.path.exists(modelPath)!=True):
  						os.makedirs(modelPath)
					saver.save(sess,modelPath+'/model',global_step=i_epoch+1)
				test_accuracy = sess.run(accuracy,feed_dict={x: [i[0] for i in validation_data], y: [i[1] for i in validation_data],keep_rate:1.0})
				print('Accuracy:', test_accuracy)
			print('Done. Finishing accuracy:')
			print('Accuracy:', sess.run(accuracy,feed_dict={x: [i[0] for i in validation_data], y: [i[1] for i in validation_data],keep_rate:1.0}))
			print('train the model is over')
		else:
			# predict test data
			print ("begin test the model")
			# 在指定的路径下 找到 最新的模型
			modelList = os.listdir('/home/wangbing/scripts/lung_cnn_model')
			modelList.sort()
			latestModelPath = modelList[-1]
			latestMode = '/home/wangbing/scripts/lung_cnn_model'+'/'+latestModelPath
			print("the latest model is : ",latestMode)
			ckpt = tf.train.get_checkpoint_state(latestMode)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print("no model")
			# test all Data
			print("used saved model :%s test All_label_data\n"%(latestMode))
			for data in All_label_data:
				X = data[0]
				Y = data[1]
				p,p_s,p_s_c = sess.run([prediction,prediction_softmax,softmax_cross_entropy_with_logits],feed_dict={x: X, y: Y,keep_rate:0.8})
				print("prediction is : %s\nprediction_softmax is :%s\nsoftmax_cross_entropy_with_logits is :%s\n"%(p,p_s,p_s_c))
			all_data_accuracy = sess.run(accuracy,feed_dict={x: [i[0] for i in All_label_data], y: [i[1] for i in All_label_data],keep_rate:1.0})
			print("all_data_accuracy is : ",all_data_accuracy)
def main():
	img_size=50
	slice_count=20
	data_dir='/home/wangbing/stage1/'
	label_file_path='/home/wangbing/stage1_labels/stage1_labels.csv'
	All_label_data=None
	All_unlabel_data=None
	All_label_data,All_unlabel_data = dataProcess(data_dir,label_file_path,img_size,slice_count)
#	train_convolutional_neural_network(All_label_data,All_unlabel_data,120,True)
	train_convolutional_neural_network(All_label_data,All_unlabel_data,120,False)
if __name__ == '__main__':
	main()

