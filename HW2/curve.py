import math
import hdf5storage
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import matplotlib.patches as patches
import matplotlib.path as path
from scipy.interpolate import interp1d


### pos ###
# indy_20170124_01 = hdf5storage.loadmat('../indy_20170124_01.mat')
# target_pos = indy_20170124_01['target_pos']
# t= indy_20170124_01['t']
# spikes = indy_20170124_01['spikes'][0,2]
# data = hdf5storage.loadmat('../trial_angles.mat')
# trial_angles = data['trial_angles']
# data = hdf5storage.loadmat('../index.mat')
# index = data['index']
#
# rate = np.zeros((1,trial_angles.shape[1]))
# for i in range(trial_angles.shape[1]):
#     count = 0
#     for j in range(spikes.shape[0]):
#         if spikes[j][0]>=t[index[i][0]-1] and spikes[j][0]<=t[index[i+1][0]-1]:
#             count += 1
#     # rate[0,i] = count/ (t[index[i+1][0]-1]- t[index[i][0]-1])
#     rate[0, i] = count
#
# curve_data = np.concatenate((trial_angles,rate))
# unique_angles= np.arange(0,360,45)
# rate_new = np.zeros((1,unique_angles.shape[0]))
# std = np.zeros((1,unique_angles.shape[0]))
# for i in range(unique_angles.shape[0]):
#     if i==2:
#         angle_index = np.where((trial_angles[0]>=80)&(trial_angles[0]<=100))
#     elif i==6:
#         angle_index = np.where((trial_angles[0] >= 260) & (trial_angles[0] <= 280))
#     else:
#         angle_index = np.where(trial_angles[0] == unique_angles[i])
#     rate_new[0,i] = np.mean(rate[0,angle_index])
#     std[0,i] = np.std(rate[0,angle_index])
# rate_new[0][0] = rate_new[0][-1]
# std[0][0] = std[0][-1]
# curve_data_new = np.concatenate((unique_angles.reshape(1,unique_angles.shape[0]),rate_new))
#
# xnew = np.linspace(0,315,100)
# a=unique_angles.reshape(1,unique_angles.shape[0])
# func = interp1d(a[0],rate_new[0],kind='cubic')
# ynew = func(xnew)
#
# plt.errorbar(a[0],rate_new[0],yerr=std-1,fmt='',ecolor='k',color='b',elinewidth=1,capsize=4, ls='none')
# plt.plot(xnew,ynew)
# plt.ylabel('Spike count')
# plt.xlabel('Direction of movement')
# plt.title('Tuning position')
# plt.show()

### velocity ###
indy_20170124_01 = hdf5storage.loadmat('../indy_20170124_01.mat')
target_pos = indy_20170124_01['target_pos']
t= indy_20170124_01['t']
spikes = indy_20170124_01['spikes'][0,2]
data = hdf5storage.loadmat('../indy_20170124_01/velocity_angles.mat')
trial_angles = data['velocity_angles']
data = hdf5storage.loadmat('../indy_20170124_01/velocity_index.mat')
index = data['index']
unique_angles= np.arange(45,405,45)
per_velocity_rate = [[] for index in range(8)]
count_index = np.zeros((1,unique_angles.shape[0]))
velocity_rate = np.zeros((1,unique_angles.shape[0]))
for i in range(index.shape[1]-1):
    count = 0
    for j in range(spikes.shape[0]):
        if spikes[j][0]>=t[index[0][i]-1] and spikes[j][0]<=t[index[0][i+1]-2]:
            count += 1
    velocity_rate[0, np.where(unique_angles == trial_angles[0][index[0][i+1]-2])] += (count/ (t[index[0][i+1]-2]- t[index[0][i]-1]))
    count_index[0, np.where(unique_angles == trial_angles[0][index[0][i+1]-2])] += 1
    per_velocity_rate[np.where(unique_angles == trial_angles[0][index[0][i+1]-2])[0][0]].append((count/ (t[index[0][i+1]-2]- t[index[0][i]-1])))

velocity_rate = velocity_rate/count_index
io.savemat('../indy_20170124_01/velocity_rate.mat', {'velocity_rate': velocity_rate})
per_velocity_rate = np.array(per_velocity_rate)
io.savemat('../indy_20170124_01/per_velocity_rate.mat', {'per_velocity_rate': per_velocity_rate})
# velocity_rate = hdf5storage.loadmat('../indy_20170124_01/velocity_rate.mat')
# velocity_rate = velocity_rate['velocity_rate']
# per_velocity_rate = hdf5storage.loadmat('../indy_20170124_01/per_velocity_rate.mat')
# per_velocity_rate = per_velocity_rate['per_velocity_rate']

std = np.zeros((1,unique_angles.shape[0]))
for i in range(unique_angles.shape[0]):
    std[0,i] = np.std(np.array(per_velocity_rate[0][i]))

xnew = np.linspace(0,315,100)
a=unique_angles.reshape(1,unique_angles.shape[0])
func = interp1d(a[0]-45,velocity_rate[0],kind='cubic')
ynew = func(xnew)

plt.errorbar(a[0]-45,velocity_rate[0],yerr=std, fmt='',ecolor='k',color='b',elinewidth=1,capsize=4, ls='none')
plt.plot(xnew,ynew)
plt.ylabel('Spike count')
plt.xlabel('Direction of movement')
plt.title('Tuning velocity')
plt.show()
