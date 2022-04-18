import matplotlib.pyplot as plt
import h5py
import numpy as np
from copy import deepcopy

def handle_one_img(img):
    b = np.log(img[1])
    # b[img[1] > 999] = np.min(b)
    new_img = np.zeros([img.shape[1], img.shape[2],3])
    new_img[:,:,0] = img[0]*1.0/np.max(img[0])
    # new_img[:,:,1] = b * 1.0/np.max(b)
    return (new_img.clip(0,1) * 255).astype(np.uint8)

f = h5py.File("test.h5", "r")
a = f["5"]
print(np.sum(a[0]))
print(a.attrs["nhit"])
a = handle_one_img(a)
f.close()

plt.imshow(a)
plt.show()
