from data_loader import DataLoader, DataType
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dl = DataLoader()
train, test = dl.load_reviews()
train_x = np.squeeze(np.array([v for v,l in train]))
train_y = np.array([l for v,l in train])
test_x = np.squeeze(np.array([v for v,l in test]))
test_y = np.array([l for v,l in test])

model = Sequential()
model.add(Dense(units=30, activation='relu', input_dim=500))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=30, batch_size=10)
loss_and_metrics = model.evaluate(train_x, train_y, batch_size=128)
print(loss_and_metrics)
