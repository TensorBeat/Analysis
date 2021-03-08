import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import models
from keras import layers
from datetime import datetime
from sklearn.decomposition import PCA

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#Doing Feature Extraction for all songs we want, 
def get_data():
    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:

        for filename in os.listdir(f'./genres/{g}'):
            songname = f'./genres/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

def split_and_scale_data():
    data = pd.read_csv('data.csv')
    data.head()
    data = data.drop(['filename'],axis=1)
    data.head()
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    # print(len(y))
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test, X, data

def train(X_train, X_test, y_train, y_test, saveModel):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    if saveModel:
        now = datetime.today().strftime('%m/%d/%Y')
        saveString = "SavedModels/" + now
        model.save(saveString)
        print("Model saved to:", saveString)

    history = model.fit(X_train,
                        y_train,
                        epochs=20,
                        batch_size=128)

    test_loss, test_acc = model.evaluate(X_test,y_test)

    # predictions = model.predict(X_test)
    # np.argmax(predictions[0])
    
def component_analysis(scaled_data, frame):
    pca = PCA()
    pca.fit(scaled_data.T)
    pca_data = pca.transform(scaled_data.T)

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1,len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel("Principal Component")
    plt.title("Scree Plot")
    plt.show()
    features = list(frame.columns.values)
    pca_df = pd.DataFrame(pca_data, columns=labels)
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title("PCA Graph")
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.show()

X_train, X_test, y_train, y_test, X, frame = split_and_scale_data()
# train(X_train, X_test, y_train, y_test, False)
component_analysis(X, frame)
