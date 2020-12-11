# Plot
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

# GIF
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from tools import create_time_serie, cos_sum, tabulate, get_time_series
from gramian_angular_field import transform

from scipy.io import loadmat
import imageio

# Jupyter
# %matplotlib inline

# Global iteration
iteration = 0

if __name__ == "__main__":

    data = loadmat('/home/Desktop/data1000.mat')['data']

    plt.figure(0)
    sampling = 10
    for i in range(0,data.shape[1]):
        time_serie1 = data[::sampling,i]
        size_time_serie1 = time_serie1.size
        gaf1, phi, r, scaled_time_serie = transform(time_serie1)

        print('processing ' + str(i))
        imageio.imwrite('~/Clutering/dataset/GAFs/pixel_' + (format(i, '05d')) + '.jpg', gaf1)

        # plt.figure
        # if i%100==0:
        #     plt.imshow(gaf1)
        #     plt.show()

from PIL import Image
import glob
image_list = []
for filename in glob.glob('/home/Clutering/dataset/GAFs/*.jpg'): #assuming gif
    im=Image.open(filename)
    im = im.resize((28,28))
    image_list.append(im)


nz = len(image_list)
[nx,ny] = image_list[0].size
Stack =  np.zeros((nz,nx,ny,1))
import matplotlib.pyplot as plt
for i in range(0,len(image_list)):
    img = np.array(image_list[i])
    Stack[i,:,:,0] = img

data = loadmat('/home/Desktop/data1000.mat')['data']
M = np.zeros((Stack.shape[0], 844))
for i in range(0,Stack.shape[0]):
    M[i,:] = data[::,i]

sil = []
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
kmax = 5
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(M)
    labels = kmeans.labels_
    sil.append(silhouette_score(M, labels, metric = 'l1'))
    print(str(k)+' - '+ str(sil[k-2]))

plt.plot(range(2, kmax+1),sil,'o-')
plt.show

