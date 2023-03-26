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
random.seed(42)
train_df = pd.read_csv('./open/train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

x_train_all, x_val, y_train_all, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

x_test = pd.read_csv('./open/test.csv').drop(columns=['ID'])
pd.set_option('display.max_columns', None)  ## 모든 열을 출력한다.

train_x = train_x.drop(columns=['X_04'])
print()


plt.violinplot(train_x['X_45'])
plt.xlabel('45')

plt.show()
