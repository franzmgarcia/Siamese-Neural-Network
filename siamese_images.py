import random 

import numpy as np 
import tensorflow as tf 

def par_entrenamiento(inputs: np.ndarray, labels: np.ndarray):

    classes = 10

    digit_indices = [np.where(labels == i) [0] for i in range (classes)]

    pairs = list()
    labels = list()

    n = min([len(digit_indices[d]) for d in range (classes)]) -1

    for d in range(classes):
        for i in range(n):
            z1,z2 = digit_indices[d][i],digit_indices[d][i+1]
            pairs += [[inputs[z1], inputs[z2]]]
            inc = random.randrange(1,classes)
            dn = (d+inc) % classes
            z1,z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[inputs[z1], inputs[z2]]]
            labels += [1,0]
    
    return np.array(pairs), np.array(labels, dtype=np.float32)

def create_base_network():

    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64,activation='relu'),
    ])

(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

train_pairs, tr_labels = par_entrenamiento(x_train,y_train)
test_pairs, test_labels = par_entrenamiento(x_test,y_test)

base_network = create_base_network()

input_a = tf.keras.layers.Input(shape=input_shape)

encoder_a = base_network(input_a)

input_b = tf.keras.layers.Input(shape=input_shape)
encoder_b = base_network(input_b)

l1_dist = tf.keras.layers.Lambda(
    lambda embeddings: tf.keras.backend.abs(embeddings[0] - embeddings[1])) \
    ([encoder_a, encoder_b])

flattened_weighted_distance = tf.keras.layers.Dense(1, activation='sigmoid') \
    (l1_dist)


model = tf.keras.models.Model([input_a, input_b], flattened_weighted_distance)

# Train
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit([train_pairs[:, 0], train_pairs[:, 1]], tr_labels,
          batch_size=128,
          epochs=20,
          validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels))










