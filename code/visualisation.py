import numpy as np
import os
from keras_vggface.vggface import VGGFace
from glob import glob
from train_BCE_FL import read_img
import tensorflow as tf
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from collections import defaultdict
from glob import glob
from random import choice, sample
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, \
    Subtract, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
# from tqdm import tqdm_notebook
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def baseline_model(classes):
    input = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x = base_model(input)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.02)(x)
    out = Dense(classes, activation="softmax")(x)

    model = Model(input, out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    #     model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.00003))
    #     model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.00001))
    model.summary()

    return model


# visualisation data
# investigate the features of image after fed into VVFace model

stat_id = 1200
end_id = 1500
# get all data path
train_folders_path = "../input/train/"
all_images = glob(train_folders_path + "*/*/*.jpg")

# get labels
# labels = [(i.split("/")[2]).split("\\")[1] + (i.split("/")[2]).split("\\")[2] for i in all_images]  # per person
labels = [(i.split("/")[2]).split("\\")[1] for i in all_images]
labels = np.array(labels[stat_id:end_id])
# labels = np.array(labels)
labels_unique = np.unique(labels)
labels_count = len(labels_unique)
le = preprocessing.LabelEncoder()
transform_le = le.fit_transform(labels)

# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# get feed to VGGFace
features = []
# define model
base_model = VGGFace(model='resnet50', include_top=False)
for image_id in tqdm(range(len(all_images[stat_id:end_id]))):
    # for image_id in tqdm(range(stat_id,end_id)):
    tmp_raw_data = read_img(all_images[image_id])
    reshaped_data = np.reshape(tmp_raw_data, (1, 224, 224, 3)).astype(np.float32)
    tensor_data = tf.convert_to_tensor(reshaped_data)
    feature = base_model(tensor_data)
    # feature = GlobalMaxPool2D()(feature)
    feature_np = np.array(feature)
    features.append(np.reshape(feature_np, feature_np.shape[3]))

features = np.array(features)


# pca
pca = TSNE(n_components=2)
x_pcaed = pca.fit_transform(features)
plt.scatter(x_pcaed[:, 0], x_pcaed[:, 1], c=transform_le, cmap=cmap)
plt.legend()
plt.show()

# cum precent plot
# percent = np.array(pca.explained_variance_ratio_)/np.sum(np.array(pca.explained_variance_ratio_))
# cum_precent = np.cumsum(percent)
# plt.plot(range(50), cum_precent)
# plt.show()


# train model, classification, family as class
X = np.zeros((end_id, 224, 224, 3))
for image_id in tqdm(range(len(all_images[stat_id:end_id]))):
    X[image_id, :, :, :] = read_img(all_images[image_id])
y = np.reshape(transform_le, (-1, 1))
# label one hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
y_onehot = enc.fit_transform(np.reshape(labels, (-1, 1))).toarray()
y_onehot = np.reshape(y_onehot, (y_onehot.shape[0], 1, 1, y_onehot.shape[1]))

# train val set split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=0)


model = baseline_model(labels_count)

history = model.fit(X_train, y_train,
                    batch_size=16,
                    use_multiprocessing=False,
                    validation_data=(X_test, y_test),
                    epochs=66, verbose=2,
                    workers=1,
                    steps_per_epoch=30, validation_steps=20)

