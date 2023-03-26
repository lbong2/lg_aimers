
import xgboost as xgb
from xgboost import plot_importance # 중요변수 시각화
from sklearn.model_selection import train_test_split # train/test
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import pandas as pd
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

mas = MaxAbsScaler()
mms = MinMaxScaler()
sss = StandardScaler()
rbs = RobustScaler()
nor = Normalizer()
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

#####

train_x = train_x.drop(columns=['X_04', 'X_23', 'X_47', 'X_48'])
x_test = x_test.drop(columns=['X_04', 'X_23', 'X_47', 'X_48'])


train_x = train_x.drop([24093])
train_y = train_y.drop([24093])

train_x[['X_13']] = mms.fit_transform(train_x[['X_13']])
x_test[['X_13']] = mms.fit_transform(x_test[['X_13']])

train_x[['X_45']] = nor.fit_transform(train_x[['X_45']])
x_test[['X_45']] = nor.fit_transform(x_test[['X_45']])

train_x[['X_46']] = sss.fit_transform(train_x[['X_46']])
x_test[['X_46']] = sss.fit_transform(x_test[['X_46']])

train_x[['X_49']] = sss.fit_transform(train_x[['X_49']])
x_test[['X_49']] = sss.fit_transform(x_test[['X_49']])

train_x[['X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56']] = rbs.fit_transform(train_x[['X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56']])
x_test[['X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56']] = rbs.fit_transform(x_test[['X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56']])

condition = train_x.X_10 > 0

tmp_list = train_x[condition]['X_10'].to_list()
tmp_list2 = train_x['X_10'].to_list()
for i in range(len(tmp_list2)):
    if tmp_list2[i] == 0:
        tmp_list2[i] = random.choice(tmp_list)
train_x.X_10 = tmp_list2

condition = train_x.X_11 > 0

tmp_list = train_x[condition]['X_11'].to_list()
tmp_list2 = train_x['X_11'].to_list()
for i in range(len(tmp_list2)):
    if tmp_list2[i] == 0:
        tmp_list2[i] = random.choice(tmp_list)
train_x.X_11 = tmp_list2

condition = x_test.X_10 > 0

tmp_list = x_test[condition]['X_10'].to_list()
tmp_list2 = x_test['X_10'].to_list()
for i in range(len(tmp_list2)):
    if tmp_list2[i] == 0:
        tmp_list2[i] = random.choice(tmp_list)
x_test.X_10 = tmp_list2

condition = x_test.X_11 > 0

tmp_list = x_test[condition]['X_11'].to_list()
tmp_list2 = x_test['X_11'].to_list()
for i in range(len(tmp_list2)):
    if tmp_list2[i] == 0:
        tmp_list2[i] = random.choice(tmp_list)
x_test.X_11 = tmp_list2


print(train_x.columns)
'''
train_x = train_x[[
                'X_01', 'X_02', 'X_05', 'X_06', # PCB 누름량
                'X_03', 'X_07', 'X_08', 'X_09', 'X_10', 'X_11',# 방열 재료
                'X_12', # 커넥터 위치 좌표
                'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', # 안테나 패드
                'X_19', 'X_20', 'X_21', 'X_22', 'X_30', 'X_31', 'X_32', 'X_33', # 스크류 삽입
                'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 'X_29', # 커넥터 핀 치수
                'X_34', 'X_35', 'X_36', 'X_37', # 스크류 토크
                'X_38', 'X_39', 'X_40', # 하우징 PCB
                'X_41', 'X_42', 'X_43', 'X_44', 'X_45', # 레이돔, 레이돔 기울기
                'X_46', # 본드 소모량
                'X_49', # cal 전 시간
                'X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56' # RF 납 량
]]
x_test = x_test[[
                'X_01', 'X_02', 'X_05', 'X_06', # PCB 누름량
                'X_03', 'X_07', 'X_08', 'X_09', 'X_10', 'X_11',# 방열 재료
                'X_12', # 커넥터 위치 좌표
                'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', # 안테나 패드
                'X_19', 'X_20', 'X_21', 'X_22', 'X_30', 'X_31', 'X_32', 'X_33', # 스크류 삽입
                'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 'X_29', # 커넥터 핀 치수
                'X_34', 'X_35', 'X_36', 'X_37', # 스크류 토크
                'X_38', 'X_39', 'X_40', # 하우징 PCB
                'X_41', 'X_42', 'X_43', 'X_44', 'X_45', # 레이돔, 레이돔 기울기
                'X_46', # 본드 소모량
                'X_49', # cal 전 시간
                'X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56' # RF 납 량
]]
'''
#####
x_train_all, x_val, y_train_all, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)




params = {'max_depth': 9,
          'min_child_weight': 6,
          'eta': 0.05,
          'subsample': 1,
          'colsample_bytree': 0.8,
          'objective': 'reg:squarederror',
          'eval_metric': 'rmse'
          }
'''gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6, 12)
    for min_child_weight in range(2, 8)
]
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(5,11)]
]'''

dtrain = xgb.DMatrix(x_train_all, label=y_train_all)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test)

vali_test = xgb.DMatrix(x_val)
eval_list = [(dval, 'validation')]
##### Parameters Learning_rate
'''min_rmse = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=5,
            metrics=['rmse'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\trmse {} for {} rounds\n".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = eta
print("Best params: {}, rmse: {}".format(best_params, min_rmse))

#####
'''
##### Parameters subsample and colsample_bytree
'''
min_rmse = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\trmse {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample,colsample)
print("Best params: {}, {}, rmse: {}".format(best_params[0], best_params[1], min_rmse))
#####
'''
##### Parameters max_depth and min_child_weight
'''min_rmse = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # 매개변수 업데이트
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # CV 실행
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'rmse'},
        early_stopping_rounds=10
    )
    # 업데이트 최고의 MAE
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth, min_child_weight)
print("최고의 매개변수: {}, {}, rmse: {}".format(best_params[0], best_params[1], min_rmse))
'''
#####
#model = XGBRegressor()
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=999, early_stopping_rounds=10, evals=eval_list)

#cv_result = xgb.cv(params=params, dtrain=dtrain, num_boost_round=999, seed=42, nfold=5, metrics={'rmse'}, early_stopping_rounds=10)
#print(cv_result)



print(f'Best mse: {model.best_score}\nBest iter: {model.best_iteration + 1}')
preds = model.predict(dtest)


y_pred = model.predict(vali_test)
y_val = y_val.to_numpy()
sum = 0

for i in range(1, 15):

    print(f"{i} : ", lg_nrmse2(y_val[:, i-1], y_pred[:, i-1], i))
    sum += lg_nrmse2(y_val[:, i-1], y_pred[:, i-1], i)

print("sum: ", sum)
print("lg_nrsme: ", lg_nrmse(y_val, y_pred))

submit = pd.read_csv('./open/sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')
submit.to_csv('./newsubmit.csv', index=False)

ax = plot_importance(model)
ax.figure.set_size_inches(12, 15)
ax.figure.savefig('ex1.png')
plt.show()
