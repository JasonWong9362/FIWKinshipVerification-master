import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tqdm import tqdm
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
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import layers


val_famillies_list = ["F09", "F04", "F08", "F06", "F02"]


def get_train_val(familly_name):
    train_file_path = "../input/train_relationships.csv"
    train_folders_path = "../input/train/"
    # train_file_path = "./input/train_pairs_new.xlsx"
    # train_folders_path = "./input/train_new/"
    val_famillies = familly_name

    all_images = glob(train_folders_path + "*/*/*.jpg")
    all_images = [x.replace('\\', '/') for x in all_images]
    train_images = [x for x in all_images if val_famillies not in x]  # train data path set
    val_images = [x for x in all_images if val_famillies in x]  # val data path set

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
    #     relationships = pd.read_excel(train_file_path)

    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]

    return train, val, train_person_to_images_map, val_person_to_images_map


def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)


def read_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img = np.array(img).astype(np.float64)
    return preprocess_input(img, version=2)


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        labels = np.array(labels).astype(np.float64)
        yield [X1, X2], labels


def contrastive_loss(y_true, y_pred, margin=10):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y_true, y_pred.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(y_pred)
    squaredMargin = K.square(K.maximum(margin - y_pred, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def cosine_distance(x, y):
    xy = K.dot(x, y)
    x_nor = K.l2_normalize(x, axis=-1)
    y_nor = K.l2_normalize(y, axis=-1)

    return -K.mean(x * y, axis=-1, keepdims=True)


# for binary_crossentropy & focal loss
def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False, pooling='max')  # 'resnet50'

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # x1 = GlobalMaxPool2D()(x1)
    # x2 = GlobalAvgPool2D()(x2)

    # feature fusion
    # x3 = Subtract()([x1, x2])
    # x3 = Multiply()([x3, x3])
    #
    # x1_ = Multiply()([x1, x1])
    # x2_ = Multiply()([x2, x2])
    # x4 = Subtract()([x1_, x2_])
    #
    # x5 = Multiply()([x1, x2])
    #
    # x = Concatenate(axis=-1)([x3, x4, x5])

    x = Concatenate(axis=-1)([x1, x2])

    # MLP
    x = Dense(512, activation="relu")(x)  # length of vector x
    x = Dropout(0.02)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.02)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    # binary_crossentropy
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    # focal_loss 1
    # model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.00003))

    # focal_loss 2
    # model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model


# for contrastive_loss; feature extractor
def contrastive_loss_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False, pooling='max')  # 'resnet50'

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)  # feature 1
    x2 = base_model(input_2)  # feature 2



    # calculate cos distance
    # distance = layers.Lambda(euclidean_distance)([x1, x2])
    distance = 1 - cosine_similarity(x1, x2)

    model = Model(inputs=[input_1, input_2], outputs=distance)

    model.compile(loss=contrastive_loss, optimizer="adam")

    model.summary()

    return model


def chunker(seq, size=8):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


# ===============================================================================================================
# model construction
model = contrastive_loss_model()

# train
train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[0])
file_path = "../save_model/contrastive_loss.h5"
# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.3, patience=30, verbose=1)
#     reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.2, patience=20, verbose=1)
callbacks_list = [checkpoint, reduce_on_plateau]

history = model.fit(gen(train, train_person_to_images_map, batch_size=8),
                    use_multiprocessing=False,
                    validation_data=gen(val, val_person_to_images_map, batch_size=8),
                    epochs=50, verbose=2,
                    workers=1, callbacks=callbacks_list,
                    steps_per_epoch=300, validation_steps=200)

# ===============================================================================================================
# for i in range(len(val_famillies_list)):
#     train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[i])
#     file_path = "focal_loss.h5"
#     checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.3, patience=30, verbose=1)
#     #     reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.2, patience=20, verbose=1)
#     callbacks_list = [checkpoint, reduce_on_plateau]
#
#     history = model.fit(gen(train, train_person_to_images_map, batch_size=8),
#                         use_multiprocessing=False,
#                         validation_data=gen(val, val_person_to_images_map, batch_size=8),
#                         epochs=50, verbose=2,
#                         workers=1, callbacks=callbacks_list,
#                         steps_per_epoch=300, validation_steps=200)

# ===============================================================================================================
# # test
# # *** config ***
# model_params_path = "../save_model/contrastive_loss.h5"
# save_path = "../submission/contrastive_loss/"  # config depend on loss function
#
# test_path = "../input/test-private-faces/"
# all_lists = glob("../input/test-private-lists/" + "*.csv")
# all_labels = glob("../input/test-private-labels/" + "*.csv")
# all_relationships = [l.split('\\')[1].split('.')[0] for l in all_lists]
# # define model
# model = contrastive_loss_model()
# model.load_weights(model_params_path)
# for i, relation in enumerate(all_relationships):
#     list_df = pd.read_csv(all_lists[i], header=0)
#     labels = pd.read_csv(all_labels[i])
#     preds = []
#     submission = pd.DataFrame()
#
#     # tmp = chunker(list_df.values)
#     # for batch in tqdm(tmp[:2]):
#     for batch in tqdm(chunker(list_df.values)):
#         X1 = [x[0] for x in batch]
#         X1 = np.array([read_img(test_path + x) for x in X1])
#
#         X2 = [x[1] for x in batch]
#         X2 = np.array([read_img(test_path + x) for x in X2])
#
#         pred = model.predict([X1, X2]).ravel().tolist()
#         preds += pred
#
#     preds = np.array(preds)
#
#     # binary
#     # threshold = 0.5
#     # preds = preds > threshold
#     # preds = preds.astype(int)
#
#     # labels = labels.values
#     # labels = np.reshape(labels, labels.shape[0])
#
#     # save submission
#     save_path_tmp = save_path + relation + ".csv"
#     submission['prediction'] = preds
#     submission.to_csv(save_path_tmp, index=False)
#
#     # print result
#     # print(classification_report(y_true=labels, y_pred=preds))


# ===============================================================================================================
# previous test script
# test_path = "./test/test-faces/"
#
# submission = pd.read_csv('./test/test_pairs.csv')
#
#
# def chunker(seq, size=32):
#     return (seq[pos:pos + size] for pos in range(0, len(seq), size))
#
#
# preds_for_sub = np.zeros(submission.shape[0])
#
# for i in tqdm_notebook(range(n_val_famillies_list)):
#     file_path = f"./vgg_face_{i}.h5"
#     model.load_weights(file_path)
#     # Get the predictions
#     predictions = []
#
#     for batch in tqdm_notebook(chunker(submission.image_pairs.values)):
#         X1 = [x.split("-")[0] for x in batch]
#         X1 = np.array([read_img(test_path + x) for x in X1])
#
#         X2 = [x.split("-")[1] for x in batch]
#         X2 = np.array([read_img(test_path + x) for x in X2])
#
#         pred = model.predict([X1, X2]).ravel().tolist()
#         predictions += pred
#
#     preds_for_sub += np.array(predictions) / n_val_famillies_list
#
# submission['score'] = preds_for_sub
# submission.to_csv("predictions.csv", index=False)
