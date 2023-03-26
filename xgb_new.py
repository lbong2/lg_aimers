from xgboost import XGBRegressor # model
import xgboost as xgb
from xgboost import plot_importance # 중요변수 시각화
from sklearn.model_selection import train_test_split # train/test
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # model 평가
import pandas as pd
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def lg_nrmse2(gt, preds, num):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    #for idx in range(0,14): # ignore 'ID'
    rmse = metrics.mean_squared_error(gt[:], preds[:], squared=False)
    nrmse = rmse/np.mean(np.abs(gt[:]))
    score = np.sum(nrmse)
    if 1 <= num <= 8:
        score *= 1.2
    # score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
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

sum = 0
for i in range(1, 15):
    model = XGBRegressor()
    model.fit(x_train_all, y_train_all.loc[:, "Y_%02d" % i])

    # preds = model.predict(x_test)
    y_pred = model.predict(x_val)
    y_val_tmp = y_val.loc[:, "Y_%02d" % i].to_numpy()
    print(f"{i} : ", lg_nrmse2(y_val_tmp, y_pred, i))
    sum += lg_nrmse2(y_val_tmp, y_pred, i)


    #ax = plot_importance(model)
    #ax.figure.set_size_inches(12, 15)
    #ax.figure.savefig('Y_%02d.png' % i)

print(sum)
'''
submit = pd.read_csv('./open/sample_submission.csv')
    for idx, col in enumerate(submit.columns):
        if col=='ID':
            continue
        submit[col] = preds[:,idx-1]
    print('Done.')
'''