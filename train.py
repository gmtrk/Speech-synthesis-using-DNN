import numpy as np
import os
import string
import pandas as pd
from keras import models, layers, optimizers

if __name__ == "__main__":
    samples = []
    labels_fpath = "E:\pythor\labels\labels.csv"
    df = pd.read_csv("labels.csv", sep=';', usecols=['sentence','path'])
    samples = df.sentence
    paths = df.path + ".npy"

    print(samples[13175])

    characters = string.printable
    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    token_index.update({"ą": 101,
                        "ć": 102,
                        "ę": 103,
                        "ł": 104,
                        "ń": 105,
                        "ó": 106,
                        "ś": 107,
                        "ż": 108,
                        "ź": 109,
                        "Ą": 110,
                        "Ć": 111,
                        "Ę": 112,
                        "Ł": 113,
                        "Ń": 114,
                        "Ó": 115,
                        "Ś": 116,
                        "Ż": 117,
                        "Ź": 118,
                        })
    max_len = 150
    sample_len = 13572
    results = np.zeros((sample_len, max_len, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        if i == sample_len:
            break
        print(i)
        print(sample)
        for j, characters in enumerate(sample[:max_len]):
            index = token_index.get(characters)
            results[i, j, index] = 1


    spec_path = "E:\pythor/spektogramy"
    database = []
    for path, _, fnames in os.walk(spec_path):
        for file in fnames:
                    file_path = os.path.join(spec_path, file)
                    spectogram = np.load(file_path) / 255
                    database.append(spectogram)
    database = np.array(database)
    database = database[..., np.newaxis]
    print(database.shape) #(13175, 256, 862, 1)
    print(results.shape)
    #wejściowy (150, 119)
    #wyjsciowy (256, 862, 1)
    #TF_GPU_ALLOCATOR = cuda_malloc_async
    #nowy 256,313,1

    print(database.max())
    print(database.min())

    model = models.Sequential(name='Synteza')
    model.add(layers.Input(shape=(150, 119)))
    model.add(layers.Conv1D(119, 3, activation='relu', padding='same'))
    # model.add(layers.Conv1D(119, 2, activation='relu'))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    # model.add(layers.Conv1D(64, 2, activation='relu'))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    # model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.Conv1D(16, 3, activation='relu'))
    #model.add(layers.Conv1D(16, 2, activation='relu'))
    model.add(layers.Flatten())  # 2304
    model.add(layers.Dense(2304))  # bottleneck
    model.add(layers.Dense(150))
    model.add(layers.Dense(2304))
    model.add(layers.Reshape((6, 6, 64)))
    model.add(layers.Conv2DTranspose(64, (6, 7)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (7, 9), strides=2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (8, 10), strides=2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (5, 9), strides=1, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (1, 5), strides=4, activation='relu'))
    model.add(layers.BatchNormalization())
    #model.add(layers.Conv2DTranspose(64, (1, 3), strides=2, activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(1, 1, activation="sigmoid"))

    model.summary()
    x_train = results[500:6000]
    y_train = database[500:6000]
    x_val = results[:500]
    y_val = database[:500]

    learning_rate = 0.005
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    model.fit(x_train, y_train,
               batch_size=4,
               epochs=20,
               validation_data=(x_val, y_val),
                shuffle=True)

    model.save('E:\pythor/speechmodel7')
