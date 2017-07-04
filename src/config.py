# -*- coding: utf-8 -*-
"""
这个程序是用来存放 路径和更好参数信息
"""

luna6DataParamDict={
    "mhdFileBasePath":"/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data",
    "anaFileBasePath":"/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/CSVFILES",
    "annoFileName":"annotations.csv",
    "annoExcFileName":"annotations_excluded.csv",
    "noduleSavaBasePath":"/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleDataNoduleSaveBasePath",
    "nodulePositiveSaveBasePath":"/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleDataNoduleSaveBasePath/positiveNodules",
    "noduleNegativeSaveBasePath":"/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleDataNoduleSaveBasePath/negativeNodules",
    "noduleSaveFileNameSepFlag":"_@#$_",
    "noduleSemidiameter":"9",
    "voxelValueMax":"400",
    "voxelValueMin":"-1000",
}
noduleClassModel3={
    "modelCheckpointSteps":"30",
    "modelSavedPath":"/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleDataNoduleSaveBasePath/trainedNoduleClassModel3",
    "modelTrainMinBatchSize":"30"
}