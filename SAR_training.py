import numpy as np
from netCDF4 import Dataset
import keras
import os
import h5py

#Parameter functions
def MP(waves):
    MaxP = np.zeros(len(waves))
    for i in range(len(MaxP)):
        MaxP[i] = np.max(waves[i])
    return MaxP

def WW(waves):
    Width = np.zeros(len(waves))
    for i in range(len(Width)):
        arr = waves[i,np.where(waves[i]>0.01*MaxP[i])[0]]
        Width[i] = len(arr)
    return Width

def LES(waves):
    LE_slope = np.zeros(len(waves))
    for i in range(len(LE_slope)):
        arr = np.where(waves[i]>0.125*MaxP[i])[0][0]
        LE_slope[i] = np.where(waves[i]==np.max(waves[i]))[0][0]-arr
    return LE_slope

def TES(waves):
    TE_slope = np.zeros(len(waves))
    for i in range(len(TE_slope)):
        arr = np.where(waves[i]>0.125*MaxP[i])[0][-1]
        TE_slope[i] =arr-np.where(waves[i]==np.max(waves[i]))[0][0]
    return TE_slope

def peakiness(waves):
    peaky = np.zeros(len(waves))
    for i in range(len(waves)):
        #maximum = np.max(waves[i])
        #maximum_bin=waves[i,np.where(waves[i]==maximum)[0][0]]
        peaky[i] = np.max(waves[i])/np.mean(waves[i])
    return peaky

#Data
path = '/cpnet/li2_cpdata/SATS/RA/S3A/L2/THEMATIC/BC005/SI/'
directory = os.listdir(path+'032/')
print(len(directory))
print(directory[176])
SAR_data = np.array([[]])
'''
for i in range(120,200):
    wave = Dataset(path+directory[i]+'/enhanced_measurement.nc')['waveform_20_ku']
    flags = Dataset(path+directory[i]+'/enhanced_measurement.nc')['surf_type_class_20_ku']
    flags = flags[:]
    indexes = np.where(flags<3)[0]
    if i == 176:
        print(SAR_data.shape)
    if i == 120:
        SAR_data = wave[indexes]
    else:
        SAR_data = np.vstack((SAR_data,wave[indexes]))
    if i == 176:
        print(SAR_data.shape)
print(SAR_data.shape)
'''
June_data = np.load('/home/spowell/SAR_Kmedoids/Data/No_Outliers/June_data_ParsNormal2.npy')
SAR_data = June_data

print(June_data.shape)
#print(np.max(abs(June_data[:,0])),np.median(June_data[:,0]))
#pathlab = '/home/spowell/SAR_Kmedoids/kmed_labels_param2_normal.npy'
#labels = np.load(pathlab)
#print(labels.shape)
mat_file = h5py.File('/home/spowell/SAR_Kmedoids/June-October_clusters', 'r')
data = mat_file['June_data_kmedoids']['c40']
labels = data[0,:]
print(labels.shape)

leads = np.zeros(len(labels))
leads = leads + 2
leads[np.where(labels==12)] = 1
leads[np.where(labels==28)] = 1
leads[np.where(labels==18)] = 1
leads[np.where(labels==8)] = 1
leads[np.where(labels==34)] = 1
#leads[np.where(labels==33)] = 1
leads[np.where(labels==30)] = 0
leads[np.where(labels==16)] = 0
leads[np.where(labels==4)] = 0
leads[np.where(labels==10)] = 0
leads[np.where(labels==22)] = 0
where_lead = np.where(leads==1)[0]
where_non_lead = np.where(leads==0)[0]
print(leads[where_lead].shape,leads[where_non_lead].shape)

import random
#sample non_leads
non_lead_data = SAR_data[where_non_lead]
sample_indices = random.sample(range(len(non_lead_data)), len(where_lead))
sampled_non_leads = non_lead_data[sample_indices]
sampled_leads = SAR_data[where_lead]
training_data = np.concatenate((sampled_non_leads,sampled_leads))
training_labels = np.concatenate((np.zeros(len(where_lead)),np.ones(len(where_lead))))

#shuffle data so leads and non leads are interspersed
permutation = np.random.permutation(len(training_labels))
training_data = training_data[permutation]
training_labels = training_labels[permutation]
print(training_data.shape)

##########################################
#model
##########################################
batch_size = 128
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(training_data.shape[1],)),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(1,activation = 'sigmoid')
    ])

model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])

history=model.fit(training_data[:580000], training_labels[:580000],batch_size=batch_size, 
                  epochs=16,validation_data=(training_data[580000:], training_labels[580000:]))

#####################################
#plotting
'''
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']

# Extract validation metrics
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

pathplot =  '/home/spowell/SAR_Kmedoids/Training/'
np.save(pathplot+'val_loss',val_loss)
import matplotlib.pyplot as plt
plt.plot(train_loss)
'''
############################################
#predictions
path2 = '/cpnet/li2_cpdata/SATS/RA/S3B/L2/THEMATIC/BC005/SI/'
directory_test = os.listdir(path+'033/')
print(len(directory_test))
print(directory[242])
new_data = Dataset(path+'032/'+directory[242]+'/enhanced_measurement.nc')['waveform_20_ku']
print(new_data.shape)
MaxP = MP(new_data)
Width = WW(new_data)
LE_slope = LES(new_data)
TE_slope = TES(new_data)
peaky = peakiness(new_data)

Total = np.vstack((MaxP,Width,LE_slope,TE_slope,peaky))
new_data= Total.T
print(new_data.shape)
mean = np.mean(new_data, axis=0)
std_dev = np.std(new_data, axis=0)

# Apply z-score normalization
new_data_norm = (new_data - mean) / std_dev
predictions = model.predict(new_data_norm)
binary_predictions = (predictions > 0.5).astype(int)
print(binary_predictions.shape)
pathsave = '/home/spowell/SAR_Kmedoids/Predictions/' 
np.save(pathsave+'June_model_predictions_0609.npy',binary_predictions)
print(np.where(binary_predictions==1)[0].shape)

y = os.listdir(pathsave)
print(y)
