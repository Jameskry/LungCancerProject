# -*- coding: utf-8 -*-
"""
这个脚本用来处理：mhd文件，然后将每个CT scan 保存为图片

"""
import  os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pylab import *
import SimpleITK as sitk




def pltShowMhdFile():
    dataBasePath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleData/sunset01'
    mhdFileName = '1.3.6.1.4.1.14519.5.2.1.6279.6001.898642529028521482602829374444.mhd'
    absFilePath = os.path.join(dataBasePath,mhdFileName)
    full_image_info = sitk.ReadImage(absFilePath)
    full_scan = sitk.GetArrayFromImage(full_image_info)
    for index in range(0,full_scan.shape[0],10):
        plt.imshow(full_scan[index],cmap='gray')
        fileName = mhdFileName[:mhdFileName.rfind('.')]+'_'+str(index)
        plt.axis('off')
        plt.savefig('/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/testPngFile/'+fileName+'.jpg')

def generatePicture():


    t = arange(0.0, 2.0, 0.01)
    s = sin(2 * pi * t)
    plot(t, s, linewidth=1.0)

    xlabel('time (s)')
    ylabel('voltage (mV)')
    title('About as simple as it gets, folks')
    grid(True)
    savefig('/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/testPngFile/test.jpg')





def mhdFileShowAndSaveToPng():
    # 这个 dataBasePath 目录下面保存的 images 保存的 每个nodule 对应的 ct scan 每个images 保存三张ct
    dataBasePath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleData/kaggleTutorial/stage1NoduleMaskOutput'
    annotationsFile = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/CSVFILES/annotations.csv'
    annoDF = pd.read_csv(annotationsFile)
    imagesFileList = glob(dataBasePath+'/'+'images*.npy')
    for i_images in imagesFileList:
        images = np.load(i_images)
        indexOftheInAnnoDF = i_images[i_images.rfind('_')+1:i_images.rfind('.')]
        indexOftheInAnnoDF = int(indexOftheInAnnoDF)
        annoDF_index_values= annoDF.iloc[indexOftheInAnnoDF:indexOftheInAnnoDF+1,:].values
        a = annoDF_index_values[0]
        b = a.astype('str')
        c = b.tolist()
        d = '_'.join(c)
        for i in range(images.shape[0]):
            e = d+'_'+str(i)
            print(e)
            plt.imshow(images[i],cmap='gray')
            plt.axis('off')
            #plt.savefig('%s.png'%(e))
            plt.title(e)
            plt.show()
def main():
    #pltShowMhdFile()
    generatePicture()


if __name__ == '__main__':
    main()