# -*- coding: utf-8 -*-
"""
这个程序是用于：从 mhd 文件中根据annotations.csv 和 annotations_excluded.csv 找到结节的数据，保存起来
用来训练结节分类模型

"""
import sys
import os
import pandas as pd
import numpy as np
import array
sys.path.append('../') # 为了在 console 中找到 src
import src.config as config
import src.helper as helper
import SimpleITK as sitk
import scipy.ndimage

def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

def write_meta_header(filename, meta_dict):
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType', 'NDims', 'BinaryData',
            'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
            'TransformMatrix', 'Offset', 'CenterOfRotation',
            'AnatomicalOrientation',
            'ElementSpacing',
            'DimSize',
            'ElementType',
            'ElementDataFile',
            'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n' % (tag, meta_dict[tag])
    f = open(filename, 'w')
    f.write(header)
    f.close()

def dump_raw_data(filename, data):
    """ Write the data into a raw format file. Big endian is always used. """
    # Begin 3D fix
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    # End 3D fix
    rawfile = open(filename, 'wb')
    a = array.array('f')
    for o in data:
        a.fromlist(list(o))
    # if is_little_endian():
    #    a.byteswap()
    a.tofile(rawfile)
    rawfile.close()

def write_mhd_file(mhdfile, data, dsize):
    assert (mhdfile[-4:] == '.mhd')
    meta_dict = {}
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False'
    meta_dict['ElementType'] = 'MET_FLOAT'
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
    write_meta_header(mhdfile, meta_dict)

    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd + '/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    dump_raw_data(data_file, data)

def save_nodule(nodule_crop, name_index):
    np.save(str(name_index) + '.npy', nodule_crop)
    write_mhd_file(str(name_index) + '.mhd', nodule_crop, nodule_crop.shape[::-1])

def getManyNodulesFromOneMhdFile(OneMhdFileManyNodules=None,positiveOrNegative=None):
    if(len(OneMhdFileManyNodules['filePath'].unique())!=1):
        print("seriesuid has many mhd file error : errorInfo : %s"%(OneMhdFileManyNodules['seriesuid']))
    mhdFilePath = OneMhdFileManyNodules['filePath'].unique()[0]
    full_image_info = sitk.ReadImage(mhdFilePath)
    full_scan = sitk.GetArrayFromImage(full_image_info)
    origin = np.array(full_image_info.GetOrigin())[::-1]  # get [z, y, x] origin
    old_spacing = np.array(full_image_info.GetSpacing())[::-1]  # get [z, y, x] spacing
    image, new_spacing = resample(full_scan, old_spacing)
    # seriesuid,coordX,coordY,coordZ,diameter_mm,filePath
    noduleNpyFileList = [] # 用力存放 保存的 结节 文件 路径信息（表示正负样本时候用到）
    for index,nodule in OneMhdFileManyNodules.iterrows():
        nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
        # Attention: Z, Y, X
        v_center = np.rint((nodule_center - origin) / new_spacing)
        v_center = np.array(v_center, dtype=int)
        window_size = int(config.luna6DataParamDict['noduleSemidiameter'])
        # This will give you the volume length = 9 + 1 + 9 = 19
        # Why the magic number 19, I found that in "LUNA16/annotations.csv",
        # the 95th percentile of the nodules' diameter is about 19.
        # This is kind of a hyperparameter, will affect your final score.
        # Change it if you want.
        zyx_1 = v_center - window_size  # Attention: Z, Y, X
        zyx_2 = v_center + window_size + 1
        #         print('Crop range: ')
        #         print(zyx_1)
        #         print(zyx_2)
        # This will give you a [19, 19, 19] volume
        img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
        # save the nodule
        if(positiveOrNegative==True):
            noduleSaveFilePath  = os.path.join(config.luna6DataParamDict['nodulePositiveSaveBasePath'],nodule.seriesuid)
        else:
            noduleSaveFilePath  = os.path.join(config.luna6DataParamDict['noduleNegativeSaveBasePath'],nodule.seriesuid)
        fileNameList = np.array([nodule.coordZ, nodule.coordY, nodule.coordX,nodule.diameter_mm])
        fileNameList = fileNameList.astype(str)
        noduleSaveFileName = config.luna6DataParamDict['noduleSaveFileNameSepFlag'].join(fileNameList)
        if(os.path.exists(noduleSaveFilePath)==False):
            os.makedirs(noduleSaveFilePath)
        np.save(os.path.join(noduleSaveFilePath,noduleSaveFileName)+'.npy',img_crop)
        noduleNpyFileList.append(os.path.join(noduleSaveFilePath,noduleSaveFileName)+'.npy')
    return noduleNpyFileList

def getAllNoduleFromFileByAnnotationsFile(annotations=None,positiveOrNegative=None):
    annotationsDropNa = annotations.dropna()
    allNodulesNpyFileList = []
    for seriesuid_i in annotationsDropNa['seriesuid'].unique():
        OneMhdFileManyNodules = annotationsDropNa[annotationsDropNa['seriesuid']==seriesuid_i]
        oneMhdFileManyNodulesNpyFilesList = getManyNodulesFromOneMhdFile(OneMhdFileManyNodules=OneMhdFileManyNodules,positiveOrNegative=positiveOrNegative)
        allNodulesNpyFileList.extend(oneMhdFileManyNodulesNpyFilesList)
    return allNodulesNpyFileList

def main():
    mhdFileDict = helper.getAllSepcialEndFileFromThePath(config.luna6DataParamDict['mhdFileBasePath'],'.mhd')
    # mhdFileDict 包含所以的mhd文件：key:value ; fileName:AbsoluteFilePath
    annotations = pd.read_csv(
        os.path.join(config.luna6DataParamDict['anaFileBasePath'],config.luna6DataParamDict['annoFileName'])
    )
    annotations_excluded = pd.read_csv(
        os.path.join(config.luna6DataParamDict['anaFileBasePath'],config.luna6DataParamDict['annoExcFileName'])
    )
    # lambda
    # if else --> expression1 if A else expression2  ;如果A为True，条件表达式的结果为expression1，否则为expression2
    annotations['filePath']=annotations['seriesuid'].map(
        lambda fileName:mhdFileDict[fileName+'.mhd'] if fileName+'.mhd' in mhdFileDict else None
    )
    annotations_excluded['filePath']=annotations_excluded['seriesuid'].map(
        lambda fileName: mhdFileDict[fileName + '.mhd'] if fileName + '.mhd' in mhdFileDict else None
    )
    positiveNoduleNpyFileList = getAllNoduleFromFileByAnnotationsFile(annotations=annotations,positiveOrNegative=True)
    negativeNoduleNpyFileList = getAllNoduleFromFileByAnnotationsFile(annotations=annotations_excluded,positiveOrNegative=False)
    np.save(config.luna6DataParamDict['noduleSavaBasePath']+'/'+'positiveNoduleNpyFileList.npy',positiveNoduleNpyFileList)
    np.save(config.luna6DataParamDict['noduleSavaBasePath']+'/'+'negativeNoduleNpyFileList.npy',negativeNoduleNpyFileList)




if __name__ == '__main__':

    main()
