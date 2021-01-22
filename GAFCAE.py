from keras.models import Model
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
import numpy as np

def autoencoderConv2D(input_shape=(28, 28, 1), filters=[32, 64, 128, 20]):
    input_img = Input(shape=input_shape)
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

    x = Conv2D(filters[2], 5, strides=3, padding=pad3, activation='relu', name='conv3')(x)

    x = Flatten()(x)
    encoded = Dense(units=filters[3], name='embedding')(x)
    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

autoencoder, encoder = autoencoderConv2D()
autoencoder.summary()
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

import matplotlib.pyplot as plt
plt.imshow(Stack[1,:,:,0])
plt.show()
# plt.show(block = True)


pretrain_epochs = 200
batch_size = 256
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(Stack, Stack, batch_size=batch_size, epochs=pretrain_epochs)
autoencoder.save_weights('conv_ae_weights.h5')

autoencoder.load_weights('conv_ae_weights.h5')


from keras.engine.topology import Layer, InputSpec
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

ym = encoder.predict(Stack)

## how many clusters
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


sil = []
kmax = 10

# # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k,max_iter=10).fit(ym)
    labels = kmeans.labels_
    sil.append(silhouette_score(ym, labels, metric = 'l1'))
    print(str(k)+' - '+ str(sil[k-2]))
#
# plt.plot(range(2, kmax+1),sil,'o-')
# plt.show

n_clusters = 6 % obtained by seperately minimizing
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer='adam', loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)


from scipy.io import loadmat
data = loadmat('/home/Desktop/data1000.mat')['data']
M = np.zeros((Stack.shape[0], 844))
for i in range(0,Stack.shape[0]):
    M[i,:] = data[::,i]

# y_pred = kmeans.fit_predict(encoder.predict(Stack))

y_pred = kmeans.fit_predict(ym)
y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])


loss = 0
index = 0

update_interval = 140
index_array = np.arange(Stack.shape[0])

#%%

tol = 0.001 # tolerance threshold to stop training

#%% md

### Start training

#%%

# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


import metrics
maxiter = 2000
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(Stack, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        print('Iter %d' % (ite))
          # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, Stack.shape[0])]
    loss = model.train_on_batch(x=Stack[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= Stack.shape[0] else 0

model.save_weights('conv_DEC_model_final.h5')

#%% md

### Load the clustering model trained weights

#%%

model.load_weights('conv_DEC_model_final.h5')

#%% md

### Final Evaluation

#%%

# Eval.
q = model.predict(Stack, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)

import scipy.io
scipy.io.savemat("results1000.mat", mdict={'y_pred': y_pred, 'y_pred_last': y_pred_last})
