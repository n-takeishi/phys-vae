import h5py
import numpy as np

with h5py.File('./Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])
labels = labels.astype(np.float32)
images = images.astype(np.float32) / 255.0

data = np.transpose(images[np.where(labels==6.0)], (0,3,1,2))

# save data
np.save('./data_all.npy', data)
print('saved data: mean(x)={:.3e}'.format(np.mean(data)))
print(data.shape)
