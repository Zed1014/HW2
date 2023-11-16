import math
import hdf5storage
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io


indy_20170124_01 = hdf5storage.loadmat('../indy_20170124_01.mat')
target_pos = indy_20170124_01['target_pos']
t= indy_20170124_01['t']
spikes = indy_20170124_01['spikes'][0,2]
data = hdf5storage.loadmat('../trial_angles.mat')
trial_angles = data['trial_angles']
data = hdf5storage.loadmat('../trial_target_pos.mat')
trial_target_pos = data['trial_target_pos']

data = hdf5storage.loadmat('../index.mat')
index = data['index']

data1=[]
for i in range(trial_target_pos.shape[0]-1):
    start = np.min(np.where(spikes > round(t[index[i]-1][0][0],5))[0])
    end = np.min(np.where(spikes > round(t[index[i+1]-1][0][0],5))[0])
    data1.append(spikes[start-5:end] - round(spikes[start][0],5))

position = np.zeros([len(data1),len(max(data1,key = lambda x: len(x)))])
length = np.zeros((len(data1),))
for i,j in enumerate(data1):
    position[i][0:len(j)] = j.ravel()
    length[i] = len(j)
io.savemat('position1.mat', {'position': position})
io.savemat('length.mat', {'length': length})

position = hdf5storage.loadmat('position1.mat')
position = position['position']
length = hdf5storage.loadmat('length.mat')
length = length['length']

# # 找出索引
indices = []
for i in range(trial_angles.shape[1]):
    if trial_angles[0,i]>=315 and trial_angles[0,i]<=360:
    # if trial_angles[0, i] == 360:
        indices.append(i)
data2 = position[indices,:]
length2=length[0][indices]

bin = 0.02
PSTH_data = np.zeros((data2.shape[0],400))
for i in range(PSTH_data.shape[0]):
    for j in range(int(length2[i])):
            PSTH_data[i,math.ceil((data2[i][j]+3) / bin)]=PSTH_data[i,math.ceil((data2[i][j]+3) / bin)]+1

fig1 = plt.figure()
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.eventplot(data2, colors='black', lineoffsets=1,linelengths=1,linewidths=0.4)
plt.ylabel('Trial')
plt.xlabel('Time(s)')
plt.show()

fig2 = plt.figure()
PSTH_data1 = np.sum(PSTH_data,axis=0)
plt.bar(np.arange(-3,5,0.02),PSTH_data1,width=0.1)
plt.show()




