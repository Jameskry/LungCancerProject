# -*- coding: utf-8 -*-
"""
model_3
这个程序是用来训练3D CNN 模型 对 肺结节进行分类的模型
利用 tensorflow 来完成 model_3
目前利用的数据是 luna16 的 annotations.csv  为 正样本 和 annotations_excluded.csv 为负样本
利用下采样的方法：从负样本中 抽样出和正样本同样多的数据
"""

import sys
import os
sys.path.append('../')
import numpy as np
import pandas as pd
import random as random
import time
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
import src.config as config
import src.helper as helper

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')
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
def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
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

def ConvolutionalNeuralNetwork3D(x=None,keepRate=None):
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([8000, 512])),
               'out': tf.Variable(tf.random_normal([512, 2]))
               }
    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([512])),
              'out': tf.Variable(tf.random_normal([2]))
              }
    oneDirLength = int(config.luna6DataParamDict['noduleSemidiameter'])*2+1
    x = tf.reshape(x,shape=[-1,oneDirLength,oneDirLength,oneDirLength,1])
    conv1 = tf.nn.relu(conv3d(x,weights['W_conv1'])+biases['b_conv1'])
    mpool1 = maxpool3d(conv1)
    conv2 = tf.nn.relu(conv3d(mpool1,weights['W_conv2'])+biases['b_conv2'])
    mpool2 = maxpool3d(conv2)
    fc = tf.reshape(mpool2,shape=[-1,8000])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keepRate)
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def noduleClassModel3Fun(trainDataX=None,trainDataY=None,isTest=False,allEpoch=200):
    """
    这个是3DCNN 模型，用来训练 nodule 进行分类的模型
    :return: 
    """
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keepRate = tf.placeholder('float')
    predictY = ConvolutionalNeuralNetwork3D(x=x,keepRate=keepRate)
    predictYSoftmax = tf.nn.softmax(predictY)
    predictYCrossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictY, labels=y)
    predictYMeanCost = tf.reduce_mean(predictYCrossEntropy)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(predictYMeanCost)
    correct_prediction = tf.equal(tf.argmax(predictYSoftmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    saver = tf.train.Saver()
    session = tf.Session()
    if(isTest == True):
        pass
    else:
        helper.printFun(strInfo="train begining :::", hasTime=True)
        session.run(tf.global_variables_initializer())  # initial all variable
        X_train, X_test, y_train, y_test = train_test_split(trainDataX, trainDataY, test_size=0.2)
        for iEpoch in range(allEpoch):
            iEpochLoss = 0
            minBatchLength = math.ceil(len(X_train)/int(config.noduleClassModel3['modelTrainMinBatchSize']))
            for index in range(0,len(X_train),minBatchLength):
                tempX = X_train[index:index+minBatchLength]
                tempY = y_train[index:index+minBatchLength]
                _train_step, _predictYMeanCost = session.run([train_step, predictYMeanCost], feed_dict={x: tempX, y: tempY, keepRate: 0.8})
                iEpochLoss += _predictYMeanCost
            strInfo = "iEpoch is %s ; iEpochLoss is %s "%(iEpoch,iEpochLoss)
            helper.printFun(strInfo=strInfo,hasTime=True)
            # save the model
            if (iEpoch + 1) % int(config.noduleClassModel3['modelCheckpointSteps']) == 0 or (iEpoch + 1) == allEpoch:
                modelSavePath = os.path.join(config.noduleClassModel3['modelSavedPath'],'model-'+helper.getTimeStr())
                if(os.path.exists(modelSavePath)==False):
                    os.makedirs(modelSavePath)
                saver.save(session, os.path.join(modelSavePath,'model'), global_step=iEpoch + 1)
            _predictY,_predictYSoftmax,test_accuracy = session.run([predictY,predictYSoftmax,accuracy],feed_dict={x: X_test, y: y_test,keepRate: 1.0})
            strInfo = "iEpoch is %s ;predictY is %s \n predictYSoftmax is %s \n ;test data accuracy is : %s"%(iEpoch,_predictY,_predictYSoftmax,test_accuracy)
            helper.printFun(strInfo=strInfo,hasTime=True)
        predictYSoftmax_value,test_accuracy = session.run([predictYSoftmax,accuracy], feed_dict={x: X_test, y: y_test,keepRate: 1.0})
        strInfo = "%s  is done ; test data accuracy is : %s\npredictYSoftmax_value is %s \n y_test is : %s" % (allEpoch, test_accuracy,predictYSoftmax_value,y_test)
        helper.printFun(strInfo=strInfo, hasTime=True)

def dataNormalization(rawData=None):
    minValue = int(config.luna6DataParamDict['voxelValueMin'])
    maxValue = int(config.luna6DataParamDict['voxelValueMax'])
    rawData = (rawData-minValue/(maxValue-minValue))
    rawData[rawData<0]=0
    rawData[rawData>1]=1
    return rawData

def dataProcessFun():
    positiveNoduleNpyFileList = np.load(config.luna6DataParamDict['noduleSavaBasePath'] + '/' + 'positiveNoduleNpyFileList.npy')
    negativeNoduleNpyFileList = np.load(config.luna6DataParamDict['noduleSavaBasePath'] + '/' + 'negativeNoduleNpyFileList.npy')
    # 因为负样本太多了，进行下采样，使正负样本保持同样的大小
    # 另外的处理的办法是 进行 正样本的数据扩增（后续添加这个功能）
    if (negativeNoduleNpyFileList.shape[0] > positiveNoduleNpyFileList.shape[0]):
        negativeNoduleNpyFileListRandomSampleN = random.sample(negativeNoduleNpyFileList.tolist(),positiveNoduleNpyFileList.shape[0])
    else:
        negativeNoduleNpyFileListRandomSampleN = negativeNoduleNpyFileList
    tempAllNodulesX = []
    tempAllNodulesY = []
    for i_file in positiveNoduleNpyFileList:
        tempNodule = np.load(i_file)
        shapeOneDir = int(config.luna6DataParamDict['noduleSemidiameter'])*2 +1
        if(tempNodule.shape == (shapeOneDir,shapeOneDir,shapeOneDir)):
            tempAllNodulesX.append(dataNormalization(np.load(i_file)))
            tempAllNodulesY.append(np.array([1,0])) # [1,0]--y  表示正样本
    for i_file in negativeNoduleNpyFileListRandomSampleN:
        tempNodule = np.load(i_file)
        shapeOneDir = int(config.luna6DataParamDict['noduleSemidiameter'])*2 +1
        if (tempNodule.shape == (shapeOneDir, shapeOneDir, shapeOneDir)):
            tempAllNodulesX.append(dataNormalization(np.load(i_file)))
            tempAllNodulesY.append(np.array([0,1])) #[0,1]--y  表示负样本
    return tempAllNodulesX,tempAllNodulesY

def dataProcessFunAddDataAugmentation():
    positiveNoduleNpyFileList = np.load(os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                                     config.luna6DataParamDict['positiveNoduleFileNpy']))
    newAddPositiveNoduleNpyFileList = np.load(os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                                           config.luna6DataParamDict['newPositiveNoduleNpyFileListInExcludeFile']))
    newNegativeNoduleNpyFileList = np.load(os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                                        config.luna6DataParamDict['newNegativeNoduleNpyFileListAfterReduceNewPositive']))
    # 因为负样本太多了，进行下采样，使正负样本保持同样的大小
    # 对于正样本来说，进行是数据扩增技术，也就是对正样本的 nodule ,进行了 np.transpose ??
    # 这个还没有用到：
    # 目前只是 从 annotation_exclude 中 diameter_mm != -1 的这一部分 由原来的认为是 负样本，现在当成正样本了


    positiveNoduleList = np.append(positiveNoduleNpyFileList,newAddPositiveNoduleNpyFileList)
    negativeNoduleList = newNegativeNoduleNpyFileList
    if (negativeNoduleList.shape[0] > positiveNoduleList.shape[0]):
        negativeNoduleNpyFileListRandomSampleN = random.sample(negativeNoduleList.tolist(),positiveNoduleList.shape[0])
    else:
        negativeNoduleNpyFileListRandomSampleN = negativeNoduleList
    tempAllNodulesX = []
    tempAllNodulesY = []
    for i_file in positiveNoduleList:
        tempNodule = np.load(i_file)
        shapeOneDir = int(config.luna6DataParamDict['noduleSemidiameter'])*2 +1
        if(tempNodule.shape == (shapeOneDir,shapeOneDir,shapeOneDir)):
            tempAllNodulesX.append(dataNormalization(np.load(i_file)))
            tempAllNodulesY.append(np.array([1,0])) # [1,0]--y  表示正样本
    for i_file in negativeNoduleNpyFileListRandomSampleN:
        tempNodule = np.load(i_file)
        shapeOneDir = int(config.luna6DataParamDict['noduleSemidiameter'])*2 +1
        if (tempNodule.shape == (shapeOneDir, shapeOneDir, shapeOneDir)):
            tempAllNodulesX.append(dataNormalization(np.load(i_file)))
            tempAllNodulesY.append(np.array([0,1])) #[0,1]--y  表示负样本
    return tempAllNodulesX, tempAllNodulesY


def dataProcessFunAddDataAugmentationUsedNpTransposeToPositiveNodule():
    positiveNoduleNpyFileList = np.load(os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                                     config.luna6DataParamDict['positiveNoduleFileNpy']))
    newAddPositiveNoduleNpyFileList = np.load(os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                                           config.luna6DataParamDict[
                                                               'newPositiveNoduleNpyFileListInExcludeFile']))
    newNegativeNoduleNpyFileList = np.load(os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                                        config.luna6DataParamDict[
                                                            'newNegativeNoduleNpyFileListAfterReduceNewPositive']))
    # 因为负样本太多了，进行下采样，使正负样本保持同样的大小
    # 对于正样本来说，进行是数据扩增技术，也就是对正样本的 nodule ,进行了 np.transpose
    # 1:从 annotation_exclude 中 diameter_mm != -1 的这一部分 由原来的认为是 负样本，现在当成正样本了
    # 2:最 positiveNoduleNpyFileList + newAddPositiveNoduleNpyFileList 这些正样本进行 转置 然后作为新的正样本


    positiveNoduleList = np.append(positiveNoduleNpyFileList, newAddPositiveNoduleNpyFileList)
    negativeNoduleList = newNegativeNoduleNpyFileList
    if (negativeNoduleList.shape[0] > positiveNoduleList.shape[0]*2):
        negativeNoduleNpyFileListRandomSampleN = random.sample(negativeNoduleList.tolist(), positiveNoduleList.shape[0]*2)
    else:
        negativeNoduleNpyFileListRandomSampleN = negativeNoduleList
    tempAllNodulesX = []
    tempAllNodulesY = []
    for i_file in positiveNoduleList:
        tempNodule = np.load(i_file)
        shapeOneDir = int(config.luna6DataParamDict['noduleSemidiameter']) * 2 + 1
        if (tempNodule.shape == (shapeOneDir, shapeOneDir, shapeOneDir)):
            tempNoduelNormalization = dataNormalization(np.load(i_file))
            tempAllNodulesX.append(tempNoduelNormalization)
            tempAllNodulesY.append(np.array([1, 0]))  # [1,0]--y  表示正样本
            #利用对这些原来的正样本 进行转置 然后做也作为 新的正样本
            tempAllNodulesX.append(np.transpose(tempNoduelNormalization))
            tempAllNodulesY.append(np.array([1, 0]))  # [1,0]--y  表示正样本

    for i_file in negativeNoduleNpyFileListRandomSampleN:
        tempNodule = np.load(i_file)
        shapeOneDir = int(config.luna6DataParamDict['noduleSemidiameter']) * 2 + 1
        if (tempNodule.shape == (shapeOneDir, shapeOneDir, shapeOneDir)):
            tempAllNodulesX.append(dataNormalization(np.load(i_file)))
            tempAllNodulesY.append(np.array([0, 1]))  # [0,1]--y  表示负样本
    return tempAllNodulesX, tempAllNodulesY

def main():
    #allNodulesX , allNodulesY = dataProcessFun() # 原来是正负样本
    #allNodulesX, allNodulesY = dataProcessFunAddDataAugmentation() # 将 负样本一部分转换为正样本
    allNodulesX, allNodulesY = dataProcessFunAddDataAugmentationUsedNpTransposeToPositiveNodule() # 利用转置对正样本进行转置
    noduleClassModel3Fun(allNodulesX,allNodulesY,isTest=False,allEpoch=20)


if __name__ == '__main__':
	main()