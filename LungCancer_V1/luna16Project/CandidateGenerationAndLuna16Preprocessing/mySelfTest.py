# -*- coding: utf-8 -*-
import pandas as pd
from LungCancer_V1.luna16Project.CandidateGenerationAndLuna16Preprocessing.UNETForCandidatePointGeneration import *
from LungCancer_V1.luna16Project.CandidateGenerationAndLuna16Preprocessing.config import annotationsFilePath







def getCands(mhdFileDataName,annotations = None):
    if(annotations == None):
        annotations = pd.read_csv(annotationsFilePath)
    tmpFileAna = annotations[annotations.seriesuid.isin([mhdFileDataName])]
    # the return values is numpy.ndarray
    return tmpFileAna.get_values()


def main():
    dataBasePath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data'
    mhdFileDataPath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleData/sunset01/'
    mhdFileDataName = '1.3.6.1.4.1.14519.5.2.1.6279.6001.898642529028521482602829374444'
    annotationsFilePath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/CSVFILES/annotations.csv'
    outputDataFilePath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/luna16ExampleData_testOutPut/'




if __name__ == '__main__':
    main()
