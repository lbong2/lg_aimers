from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV# train/test
import pandas as pd
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return -score

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
x_test = pd.read_csv('./open/test.csv').drop(columns=['ID'])

train_x = train_x.drop(columns=['X_04', 'X_23', 'X_47', 'X_48', 'X_34', 'X_35', 'X_36', 'X_37'])
x_test = x_test.drop(columns=['X_04', 'X_23', 'X_47', 'X_48', 'X_34', 'X_35', 'X_36', 'X_37'])
train_x = train_x.drop([24093])
train_y = train_y.drop([24093])
train_x['X_10'] = train_x['X_10'].replace(0, 3.0)
train_x['X_11'] = train_x['X_11'].replace(0, 0.5)

x_test['X_10'] = x_test['X_10'].replace(0, 3.0)
x_test['X_11'] = x_test['X_11'].replace(0, 0.5)

train_x['X_60'] = train_x.index / train_x.shape[0]
x_test['X_60'] = x_test.index / x_test.shape[0]

train_x['X_41_1'] = 0.
train_x[train_x['X_41'] > 21.27].X_41_1 = 1
train_x[train_x['X_41'] < 21.12].X_41_1 = -1

x_test['X_41_1'] = 0.
x_test[x_test['X_41'] > 21.27].X_41_1 = 1
x_test[x_test['X_41'] < 20.12].X_41_1 = -1

train_x['X_42_1'] = 0.
train_x[train_x['X_42'] > 21.18].X_42_1 = 1
train_x[train_x['X_42'] < 20.95].X_42_1 = -1

x_test['X_42_1'] = 0.
x_test[x_test['X_42'] > 21.18].X_42_1 = 1
x_test[x_test['X_42'] < 20.95].X_42_1 = -1

train_x['X_43_1'] = 0.
train_x[train_x['X_43'] > 21.34].X_43_1 = 1
train_x[train_x['X_43'] < 21.07].X_43_1 = -1

x_test['X_43_1'] = 0.
x_test[x_test['X_43'] > 21.34].X_43_1 = 1
x_test[x_test['X_43'] < 21.07].X_43_1 = -1

train_x['X_44_1'] = 0.
train_x[train_x['X_44'] > 21.28].X_44_1 = 1
train_x[train_x['X_44'] < 21.03].X_44_1 = -1

x_test['X_44_1'] = 0.
x_test[x_test['X_44'] > 21.28].X_44_1 = 1
x_test[x_test['X_44'] < 21.03].X_44_1 = -1

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
x_train_all, x_val, y_train_all, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# learning_rate=0.1, max_depth=14, num_leaves=31, min_child_samples=43
scorer = {'custom': make_scorer(lg_nrmse, greater_is_better=True)}
lg = MultiOutputRegressor(lgb.LGBMRegressor(subsample=0.6, random_state=42, num_leaves=60, n_estimators=100,
                                            min_child_samples=5, metric='rmse', max_depth=20, learning_rate=0.05,
                                            colsample_bytree=0.7))
'''
parameters = {
            'estimator__num_leaves': [60, 80, 100],
            'estimator__min_child_samples': [5, 10, 15, 20],
            'estimator__max_depth': [-1, 10, 15, 20],
            'estimator__learning_rate': [0.05],
            'estimator__random_state': [42],
            'estimator__subsample': [0.6, 0.65, 0.7, 0.75],
            'estimator__colsample_bytree': [0.7, 0.75],
            #'estimator__metrics':['rmse'],
            'estimator__n_estimators': [100, 150, 200]
              }

lgb = MultiOutputRegressor(lgb.LGBMRegressor())
lg_grid = RandomizedSearchCV(lgb, parameters, n_iter=30, scoring=scorer, refit='custom', verbose=2)

result = lg_grid.fit(train_x, train_y)

print('best parameters : ', result.best_params_)
print('best score : ', result.best_score_)

'''
lgmodel = lg.fit(train_x, train_y)

lg_y_pred = lgmodel.predict(x_val)
lg_preds = lgmodel.predict(x_test)


lg_cv = cross_validate(lg, train_x, train_y, scoring=scorer, cv=5)
print('cv lg_rmse:', lg_cv['test_custom'].mean())


submit = pd.read_csv('./open/sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = lg_preds[:,idx-1]
print('Done.')
submit.to_csv('./newsubmit.csv', index=False)
#feature_imp = pd.DataFrame(sorted(zip(lgmodel.estimators_[0].feature_importances_, train_x.columns)), columns=['Value','Feature'])
