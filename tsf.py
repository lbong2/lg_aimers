import pandas as pd
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

train_df = pd.read_csv('./open/train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

x_train_all, x_val, y_train_all, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)



x_test = pd.read_csv('./open/test.csv').drop(columns=['ID'])





model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(56,)))
model.add(Dense(32, input_shape=(64,)))
model.add(Dense(14))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

history = model.fit(x_train_all, y_train_all, epochs=50, validation_data=(x_val, y_val))

pred_x = model.predict(x_val)
preds = model.predict(x_test)

y_val = y_val.to_numpy()
print(lg_nrmse(y_val, pred_x))

submit = pd.read_csv('./open/sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv('./submit.csv', index=False)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0, 100)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss', 'val_loss'])
plt.show()
