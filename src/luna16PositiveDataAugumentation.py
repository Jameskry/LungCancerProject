# -*- coding: utf-8 -*-
"""
这个脚本用来对 luna16的用于model3都 annotation中的正样本进行数据扩增的。
model3 的正样本是 annotation.csv;负样本是 annotation_exclude.csv 
原来的做法是：对负样本进行 欠采样
现在要进行正样本的数据扩增；
具体做法是：
1：先将 annotation_exclude 中的 （其中： diameter_mm 为 -1 有 30513  和   不是 -1 的有 4679）
    将 diameter_mm 不是 -1 的也当成正样本
    生成新的 newPositiveNoduleNpyFileListInExcludeFile.npy ; 这个里面保存 annotation_exclude 的 diameter_mm != -1
    生成新的 newNegativeNoduleNpyFileListAfterReduceNewPositive.npy;
    这个里面保存 negativeNoduleNpyFileList 减去后 newPositiveNoduleNpyFileListInExcludeFile 后的结节
    -->: newPositiveNoduleNpyFileListInExcludeFile + newNegativeNoduleNpyFileListAfterReduceNewPositive = negativeNoduleNpyFileList
2：由第一步（addPositiveFun）的完成： 现在正样本包括：positiveNoduleNpyFileList.npy + newPositiveNoduleNpyFileListInExcludeFile.npy
                  负样本包括 ： newNegativeNoduleNpyFileListAfterReduceNewPositive.npy
    这一步要完成的是： 对 正样本数据进行扩增操作了
    

"""
import os
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import src.config as config


def addPositiveFun():
    """
    1：先将 annotation_exclude 中的 （其中： diameter_mm 为 -1 有 30513  和   不是 -1 的有 4679）
    将 diameter_mm 不是 -1 的也当成正样本
    生成新的 newPositiveNoduleNpyFileListInExcludeFile.npy ; 这个里面保存 annotation_exclude 的 diameter_mm != -1
    生成新的 newNegativeNoduleNpyFileListAfterReduceNewPositive.npy;
    这个里面保存 negativeNoduleNpyFileList 减去后 newPositiveNoduleNpyFileListInExcludeFile 后的结节
    -->: newPositiveNoduleNpyFileListInExcludeFile + newNegativeNoduleNpyFileListAfterReduceNewPositive = negativeNoduleNpyFileList
    :return: 
    """
    pPath = config.luna6DataParamDict['noduleSavaBasePath']
    negativeSampleNpy = np.load(os.path.join(pPath,'negativeNoduleNpyFileList.npy'))
    newPositiveList = []
    newNegativeList = []
    for iNegative in negativeSampleNpy:
        fileName = iNegative[:iNegative.rfind('.')]
        justNoduleName = fileName.split('/')[-1]
        noduleClassFlag = justNoduleName.split(config.luna6DataParamDict['noduleSaveFileNameSepFlag'])[-1]
        if(noduleClassFlag != '-1.0'):
            newPositiveList.append(iNegative)
        else:
            newNegativeList.append(iNegative)
    newPositiveList = np.array(newPositiveList)
    newNegativeList = np.array(newNegativeList)
    newPositiveListFileName = os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                           config.luna6DataParamDict['newPositiveNoduleNpyFileListInExcludeFile'])

    np.save(newPositiveListFileName,newPositiveList)
    print("the file path si : %s\nthe shape is : %s"%(newPositiveListFileName,newPositiveList.shape))
    newNegativeListFileName = os.path.join(config.luna6DataParamDict['noduleSavaBasePath'],
                                           config.luna6DataParamDict['newNegativeNoduleNpyFileListAfterReduceNewPositive'])
    np.save(newNegativeListFileName,newNegativeList)
    print("the file path si : %s\nthe shape is : %s" % (newNegativeListFileName, newNegativeList.shape))
    print("the shape of negativeSampleNpy is %s"%(negativeSampleNpy.shape))



def saveTransformNodule(noduleFileList=None):
    for oneNodule in noduleFileList:
        originalNodule = np.load(oneNodule)
        originalNoduleTransform = np.transpose(originalNodule)
        filePathList = oneNodule.split('/')
        newNoduleFilePath = '/'.join(filePathList[:-1])
        newNoduleFileName = 'T' + config.luna6DataParamDict['noduleSaveFileNameSepFlag'] + filePathList[-1]
        np.save(os.path.join(newNoduleFilePath, newNoduleFileName), originalNoduleTransform)
def positiveNoduleTranspose():
    noduleBasePath = config.luna6DataParamDict['noduleSavaBasePath']
    onePositiveNoduleList = np.load(os.path.join(noduleBasePath,config.luna6DataParamDict['positiveNoduleFileNpy']))
    twoPositiveNoduleList = np.load(os.path.join(noduleBasePath,
                                                 config.luna6DataParamDict['newPositiveNoduleNpyFileListInExcludeFile']))
    # 在 nodule 原本保存的相同路径下面，新建一个文件，保存转置过的nodule:
    # 转置的nodule 文件
    saveTransformNodule(onePositiveNoduleList)
    saveTransformNodule(twoPositiveNoduleList)



def main():
    #addPositiveFun()
    positiveNoduleTranspose()

if __name__ == '__main__':
    main()







