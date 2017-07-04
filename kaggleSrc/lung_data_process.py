#  the script to load data and process data and store processed data and rload
import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

global  IMG_SIZE_PX
IMG_SIZE_PX = 50
global SLICE_COUNT
SLICE_COUNT = 20
global DATA_DIR
DATA_DIR = ''
global LABEL_FILE_PATH
LABEL_FILE_PATH=''




# step one :  set the params
def setParam(img_size=50,slice_count=20,data_dir='',label_file_path=''):
	global  IMG_SIZE_PX
	global SLICE_COUNT
	global DATA_DIR
	global LABEL_FILE_PATH
	IMG_SIZE_PX=img_size
	SLICE_COUNT=slice_count
	DATA_DIR=data_dir
	LABEL_FILE_PATH=label_file_path

# step two : read data from files and process(normalize)

# read data from file and process the data

# step three : store the data

# step four : rload the data from alread stored data


#
def checkDataShape(data):
	shape = np.array(data).shape
	count =1
	for i in shape:
		count = count*i
	if(count==IMG_SIZE_PX*IMG_SIZE_PX*SLICE_COUNT):
		return True
	else:
		return False

def preprocessData():
	patients=os.listdir(DATA_DIR)
	# index_col=0 : labels_id_cancer key is id (id col is 0),so get_value by id
	labels_id_cancer = pd.read_csv(LABEL_FILE_PATH,index_col=0)

	All_labeled_data=[]
	ALL_unlabeled_data=[]
	for iNum,patient in enumerate(patients):
		try:
			# the patient has label
			label = labels_id_cancer.get_value(patient,'cancer')
			image_data,label_data= process_data(patient,label,True,img_px_size=IMG_SIZE_PX,hm_slices=SLICE_COUNT)
			if(checkDataShape(image_data)==True):
				All_labeled_data.append([image_data,label_data])
		except KeyError as e:
			# the patients has not label and the data is unlabeled_data
			print('the patient : %s is unlabeled data!'%(patient))
			image_data = process_data(patient,label,False,img_px_size=IMG_SIZE_PX,hm_slices=SLICE_COUNT)
			if(checkDataShape(image_data)==True):
				ALL_unlabeled_data.append(image_data)
	return (np.array(All_labeled_data),np.array(ALL_unlabeled_data))

def storeDataToFile(data,filePath):
	np.save(filePath,data)

def reloadDataFromFile(filePath):
	return np.load(filePath)


def chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]
def mean(a):
	return sum(a) / len(a)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
def normalize(image):
	image[image==-2000]=0
	image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	image[image>1] = 1.
	image[image<0] = 0.
	return image

def process_data(patient,label,label_flag=True,img_px_size=50, hm_slices=20):

	global DATA_DIR
	patient_path = DATA_DIR + patient
	slices = [dicom.read_file(patient_path + '/' + s) for s in os.listdir(patient_path)]
	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
	new_slices = []
	slices = [cv2.resize(np.array(normalize(each_slice.pixel_array)),(img_px_size,img_px_size)) for each_slice in slices]

	chunk_sizes = math.ceil(len(slices) / hm_slices)
	for slice_chunk in chunks(slices, int(chunk_sizes)):
		slice_chunk = list(map(mean, zip(*slice_chunk)))
		new_slices.append(slice_chunk)

	if len(new_slices) == hm_slices-1:
		new_slices.append(new_slices[-1])

	if len(new_slices) == hm_slices-2:
		new_slices.append(new_slices[-1])
		new_slices.append(new_slices[-1])

	if len(new_slices) == hm_slices+2:
		new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
		del new_slices[hm_slices]
		new_slices[hm_slices-1] = new_val
	if len(new_slices) == hm_slices+1:
		new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
		del new_slices[hm_slices]
		new_slices[hm_slices-1] = new_val

	if(label_flag==True):
		# the patient has label
		if label == 1: label=np.array([0,1])
		elif label == 0: label=np.array([1,0])
		return np.array(new_slices),label
	else:
		# the patient don't has label
		return np.array(new_slices)



def dataProcess(data_dir,label_file_path,img_size,slice_count):
	img_size=img_size
	slice_count=slice_count
	data_dir=data_dir
	label_file_path=label_file_path
	setParam(img_size,slice_count,data_dir,label_file_path)
	ALL_LABEL_DATA=None
	ALL_UNLABEL_DATA=None
	if(os.path.exists(data_dir+'labeledData.npy') and os.path.exists(data_dir+'unlabeledData.npy')):
		print("file already exists")
		ALL_LABEL_DATA=reloadDataFromFile(data_dir+'labeledData.npy')
		ALL_UNLABEL_DATA=reloadDataFromFile(data_dir+'unlabeledData.npy')
	else:
		print('file not exists')
		ALL_LABEL_DATA,ALL_UNLABEL_DATA=preprocessData()
		storeDataToFile(ALL_LABEL_DATA,data_dir+'labeledData.npy')
		storeDataToFile(ALL_UNLABEL_DATA,data_dir+'unlabeledData.npy')
	return ALL_LABEL_DATA,ALL_UNLABEL_DATA


def main():
	data_dir='../stage1/'
	label_file_path='../stage1_labels/stage1_labels.csv'
	img_size=50
	slice_count=20
	label,unlabeled = dataProcess(data_dir,label_file_path,img_size,slice_count)
	print(len(label))
	print(len(unlabeled))

if __name__ == '__main__':
	main()
