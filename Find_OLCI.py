import numpy as np
path = '/home/spowell/SAR_Kmedoids/Data/'
data = np.load(path+'July_data_ParsNormal.npy')
print(data.shape)
data = data[369662:371074]
print(data.shape)
index_keep = np.where(np.all(abs(data)<3,axis = 1))
print(index_keep[0].shape)
no_outliers_indices = index_keep[0]
np.save(path+'0630_no_outliers_indices.npy',no_outliers_indices)