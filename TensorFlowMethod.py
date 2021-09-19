import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Load the data
train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")

train_data_x = train_data.iloc[:,1:]
train_data_y = train_data.iloc[:,0]
train_data_x = train_data_x.values.reshape(-1,28,28)
test_data_x = test_data.values.reshape(-1,28,28)

print(train_data_x[0].shape)

train_data_x = np.expand_dims(train_data_x,axis=-1)/255.0
test_data_x = np.expand_dims(test_data_x,axis=-1)/255.0

print(test_data_x[0].shape)

#Split the Data for Testing and Training
X_train, X_test, y_train, y_test = train_test_split(train_data_x,train_data_y,
                                                    test_size=0.2,train_size=0.8,
                                                    shuffle=True,random_state=40)

#Create the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()

#Compile the model
model.compile(
    optimizer='adam',
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

#Train the neural network
model.fit(X_train,y_train,epochs=30, validation_data=(X_test,y_test),shuffle=True,use_multiprocessing=True)

#Create a submission file with the trained neural network
with open("submit.csv","w") as submitFile:
    count = 1
    submitFile.write("ImageId,Label\n")
    prediction = model.predict(test_data_x)
    for i in prediction:
        submitFile.write(str(count) + "," + str(np.argmax(i)) + "\n")
        count += 1
