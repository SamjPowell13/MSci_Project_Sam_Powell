from netCDF4 import Dataset
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import os

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

#DATA
path = '/cpnet/li2_cpdata/SATS/RA/S3A/L2/THEMATIC/BC005/SI/031/'
directory = os.listdir(path)
print(len(directory)," files")
#print(directory[176])
SAR_data = np.array([[]])
for i in range(len(directory)):
    wave = Dataset(path+directory[i]+'/enhanced_measurement.nc')['waveform_20_ku']
    flags = Dataset(path+directory[i]+'/enhanced_measurement.nc')['surf_type_class_20_ku']
    flags = flags[:]
    indexes = np.where((flags<3)&(flags>0))[0] #just sea ice and lead
    #if i == 176:
        #print(SAR_data.shape)
    if i == 0:
        SAR_data = wave[indexes]
    else:
        #print(wave[indexes].shape[1])
        if wave[indexes].shape[1] == 256:
            SAR_data = np.vstack((SAR_data,wave[indexes]))
    #if i == 176:
        #print(SAR_data.shape)
print("Shape:",SAR_data.shape)
#np.save('/home/spowell/SAR_Kmedoids/June_data_120-200.npy',SAR_data)
#test = np.load('/home/spowell/SAR_Kmedoids/June_data.npy')
#print(test.shape)
#Full Echoes
####################################################################
'''
# Initialize KMedoids model with 40 clusters
kmedoids = KMedoids(n_clusters=40, random_state=0)

# Fit the model to the data
kmedoids.fit(SAR_data)

# Get the cluster labels
cluster_labels = kmedoids.labels_
print(cluster_labels.shape)
savepath = '/home/spowell/SAR_Kmedoids/'
np.save(savepath+'kmed_labels',cluster_labels)
check = np.load(savepath+'kmed_labels.npy')
print(check.shape)
'''
###################################################################

#Parameters
#################################################################
MaxP = MP(SAR_data)
Width = WW(SAR_data)
LE_slope = LES(SAR_data)
TE_slope = TES(SAR_data)
peaky = peakiness(SAR_data)

Total = np.vstack((MaxP,Width,LE_slope,TE_slope,peaky))
Total= Total.T
print(Total.shape)

# Compute the mean and standard deviation along each dimension (column)
mean = np.mean(Total, axis=0)
std_dev = np.std(Total, axis=0)

# Apply z-score normalization
normalized_Total = (Total - mean) / std_dev
print("normalised_total:",normalized_Total[0:5],normalized_Total.shape)
index_keep = np.where(np.all(abs(normalized_Total)<3,axis = 1))
normalized_Total = normalized_Total[index_keep]
print("final shape:",normalized_Total.shape)
np.save('/home/spowell/SAR_Kmedoids/Data/May_data_ParsNormal.npy',normalized_Total)
#j = 2
#print(Total[j],MaxP[j],Width[j],LE_slope[j],TE_slope[j],peaky[j])

# Initialize KMedoids model with 10 clusters
#kmedoids = KMedoids(n_clusters=10, init= 'heuristic')
'''
kmeans = KMeans(n_clusters=10,init = 'k-means++')

print("fitting")
# Fit the model to the data
#kmedoids.fit(normalized_Total)
cluster_labels = kmeans.fit_predict(normalized_Total)

# Get the cluster labels
#cluster_labels = kmedoids.labels_
print(cluster_labels.shape)
savepath = '/home/spowell/SAR_Kmedoids/'
np.save(savepath+'kmean_labels_param_normal_160-180.npy',cluster_labels)
check = np.load(savepath+'kmean_labels_param_normal_160-180.npy')
print(check.shape)
'''
print("done")
########################################################