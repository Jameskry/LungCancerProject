# -*- coding: utf-8 -*-
"""
这个程序主要是一些公用的函数
"""
import os
import sys
import numpy
import pandas
import time

def getAllSepcialEndFileFromThePath(basePath=None,fileEndFlag=None):
    """
    返回值：dict ： key is fileName and value is the absolute Path of the file
           os.walk(top, topdown=True, onerror=None, followlinks=False)
           可以得到一个三元tupple(dirpath, dirnames, filenames),
           第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
                dirpath 是一个string，代表目录的路径，
                dirnames 是一个list，包含了dirpath下所有子目录的名字。
                filenames 是一个list，包含了非目录文件的名字。
                这些名字不包含路径信息，如果需要得到全路径，需要使用os.path.join(dirpath, name).
                # print "parent is : "+ parent # parent 该文件的目录名称
                # print "filename is : " + filename # filename 该文件的名称
                # print "the full name of the file is : " + os.path.join(parent, filename) # 由路径名称+文件名称构成 绝对路径
                """
    if(basePath==None):
        return None
    allFileDict = {}
    for parent, dirnames, filenames in os.walk(basePath):
        for filename in filenames:
            if(filename.endswith(fileEndFlag)):
                allFileDict[filename]=os.path.join(parent, filename)
    return allFileDict

def printFun(strInfo=None,hasTime=None):
    if(hasTime == True):
        print('time : %s \n%s'%(getTimeStr(),strInfo))

def getTimeStr():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
