aPath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/CSVFILES/annotations.csv'
cPath = '/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/CSVFILES/candidates.csv'

import pandas as pd
import numpy as np
import csv
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

aCSV = readCSV(aPath)
cPath = readCSV(cPath)

index =0
aa = 0
for cC in cPath[1:]:
    if (int(cC[4])==1):
        aa = aa + 1
print (aa)

annotations_excluded='/Users/wangbing/MyProjects/LungCancer/lungData/luna16Data/evaluationScript/annotations/annotations_excluded.csv'

annotations_excludedFiles = readCSV(annotations_excluded)
index = 0
for i in annotations_excludedFiles[1:]:
    tmp = i[len(i)-1]
    if(tmp != '-1'):
        for j in aCSV[1:]:
            if(i[0]==j[0] and i[1]==j[1] and i[2]==j[2] and i[3]==j[3]):
                print (i)
    # print (tmp,type(tmp))
#     if(tmp=='-1'):
#         index = index + 1
#
# print (index)

data = pd.read_csv(annotations_excluded)


