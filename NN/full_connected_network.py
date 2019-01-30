# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import PIL
# from sklearn.model_selection import train_test_split

# imageBigDataORL = np.zeros((401, 112, 92))

# def data_prep_ORL():
#     for people in range(1,40):
#         for face in range(1,10):
#             path = './NN/ORL/s%d_%d.bmp' % (people, face)
#             oriImage = PIL.Image.open(path)
#             imageArray = np.array(oriImage)
#             # print(imageArray.shape)
#             # print(imageBigDataORL.shape)
#             imageBigDataORL[(people - 1) * 10 + face] = imageArray
            


# data_prep_ORL()
# print(imageBigDataORL)     

import tensorflow as tf
import numpy as np
import ssl

tf.enable_eager_execution()
ssl._create_default_https_context = ssl._create_unverified_context

class DataLoader():
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images                                 # np.array [55000, 784]
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)   # np.array [55000] of int32
        self.eval_data = mnist.test.images                                   # np.array [10000, 784]
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)     # np.array [10000] of int32
        print(self.train_data.shape, self.train_labels.shape)

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]
    
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def predict(self, inputs):
        logits = self(inputs)
        return tf.argmax(logits, axis=-1)
    
num_batches = 1000
batch_size = 50
learning_rate = 0.001
model = MLP()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        if(batch_index == 1):
            print(y_logit_pred)
        # print(y.shape)
        # print(y_logit_pred.shape)
        # print(X.dtype)
        # print(y.dtype)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        # print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# #REFERENCE:https://tf.wiki/zh/models.html#mlp
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import PIL
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# tf.enable_eager_execution()

# imageBigDataORL = np.zeros((400, 112*92))
# YBigData = np.zeros(400)

# def data_prep_ORL():
#     for people in range(1,41):
#         for face in range(1,11):
#             path = './NN/ORL/s%d_%d.bmp' % (people, face)
#             oriImage = PIL.Image.open(path)
#             imageArray = np.asarray(oriImage, dtype=np.int32)
# #             print(imageArray)
#             imageVec = np.reshape(imageArray, imageArray.shape[0] * imageArray.shape[1])
#             # print(imageArray.shape)
#             # print(imageBigDataORL.shape)
# #             print((people - 1) * 10 + face)
#             imageBigDataORL[(people - 1) * 10 + face - 1] = imageVec
#             YBigData[(people - 1) * 10 + face - 1] = people
# #             print(imageBigDataORL)

# def get_batch(size):
#     index = np.random.randint(0, np.shape(X_train)[0], size)
#     return X_train[index, :], Y_train[index]

# class simplePerceptron(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=40)

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         return x

#     def predict(self, inputs):
#         logits = self(inputs)
#         return tf.argmax(logits, axis=-1)
# data_prep_ORL()
# # print(YBigData)
# # print(imageBigDataORL)
# X_train, X_test, Y_train, Y_test = train_test_split(imageBigDataORL, YBigData, test_size=0.2, random_state=42)
# # Y_train.reshape((1, Y_train.shape[0]))
# # Y_test.reshape((1, Y_test.shape[0]))
# # enc = OneHotEncoder(handle_unknown='ignore')
# # Y_train_reshaped = np.reshape(Y_train, (Y_train.shape[0], 1))
# # enc.fit(Y_train_reshaped)
# # onehot_Y_train_reshaped = enc.transform(Y_train_reshaped).toarray()
# # Y_test_reshaped = np.reshape(Y_test, (Y_test.shape[0], 1))
# # onehot_Y_test_reshaped = enc.transform(Y_test_reshaped).toarray()
# # onehot_Y_test = np.reshape(onehot_Y_test_reshaped, (one))
# # print(X_train.shape)
# # print(X_test.shape)
# # print(Y_train)
# # print(Y_test)
# # print(Y_train.T)
# # print(np.reshape(Y_train, (Y_train.shape[0], 1)))
# # Y_train_reshaped = np.reshape(Y_train, (Y_train.shape[0], 1))
# # print(Y_train_reshaped)
# # enc = OneHotEncoder(handle_unknown='ignore')
# # enc.fit(Y_train_reshaped)
# # onehot_Y_train_reshaped = enc.transform(Y_train_reshaped).toarray()
# # Y_test_reshaped = np.reshape(Y_test, (Y_test.shape[0], 1))
# # onehot_Y_test_reshaped = enc.transform(Y_test_reshaped).toarray()
# # print(Y_train_reshaped.shape)
# # print(onehot_Y_train_reshaped.shape)
# # print(onehot_y_train_reshaped)
# num_batches = 1000
# batch_size = 50
# learning_rate = 0.001
# model = simplePerceptron()
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# for batch_index in range(num_batches):
#     X, y = get_batch(batch_size)
#     with tf.GradientTape() as tape:
#         y_logit_pred = model(tf.convert_to_tensor(X))
#         print(y.shape)
#         print(y_logit_pred.shape)
#         print(X.dtype)
#         print(y.dtype)
#         # print(y_logit_pred.dtype)
#         loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
#         print("batch %d: loss %f" % (batch_index, loss.numpy()))
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))