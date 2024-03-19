import numpy as np
path = '/home/spowell/SAR_Kmedoids/Data/'
data = np.load(path+'June_data_ParsNormal.npy')
print(data.shape)
index_keep = np.where(np.all(abs(data)<3,axis = 1))[0]
print(index_keep.shape)
new_data = data[index_keep]
###################################################
#for finding indexes for plotting
start = 843315
end = 854799
data2 = data[start:end]
print(data2.shape)
k = np.where((index_keep>start)&(index_keep<end))[0]
print(k)
no_outliers = index_keep[k]-start
print(data2.shape,no_outliers,no_outliers.shape)
np.save(path+'no_outliers_indices.npy',no_outliers)
####################################################
print(new_data.shape)
np.save(path+'October_data_ParsNormal2.npy',new_data)
