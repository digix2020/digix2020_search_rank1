import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_pickle("./data/train_data.pkl")
test_df = pd.read_pickle("./data/test_data_B.pkl")

feature_name = [i for i in train_df.columns if 'feature' in i]
for i in tqdm(feature_name):
	if train_df[i].std()==0:
		feature_name.remove(i)
print(len(feature_name))
target = 'label'

params = {
		'boosting_type': 'gbdt',
		'objective': 'regression',
		'metric': 'rmse',
		'boost_from_average' : True,
		'train_metric': True, 
		'feature_fraction_seed' : 1,
		'learning_rate': 0.05,
		'is_unbalance': False,  #当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
		'num_leaves': 256,  # 一般设为少于2^(max_depth)
		'max_depth': -1,  #最大的树深，设为-1时表示不限制树的深度
		'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
		'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
		'subsample': 1,  # Subsample ratio of the training instance.
		'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
		'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
		'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
		'subsample_for_bin': 200000,  # Number of samples for constructing bin
		'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
		'reg_alpha': 2.99,  # L1 regularization term on weights
		'reg_lambda': 1.9,  # L2 regularization term on weights
		'nthread': 12,
		'verbose': 0,
	}

fold = 5
preds = 0
skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)

for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df[target])):

	print('fold:{}'.format(i + 1))

	X_train = train_df.iloc[trn_idx].reset_index(drop=True)
	X_valid = train_df.iloc[val_idx].reset_index(drop=True)

	y_train = X_train[target]
	y_valid = X_valid[target]

	weight = (y_train + 1) / 5
	lgb_train = lgb.Dataset(X_train[feature_name], y_train, weight = weight)
	lgb_eval = lgb.Dataset(X_valid[feature_name], y_valid, reference=lgb_train)

	gbm = lgb.train(params, lgb_train, num_boost_round=20000, valid_sets=(lgb_eval), early_stopping_rounds=500, verbose_eval=500)

	preds += gbm.predict(test_df[feature_name], num_iteration=gbm.best_iteration) / fold

res = test_df[['query_id','doc_id']].reset_index(drop=True)
res['predict_label'] = preds
res.columns = ['queryid','documentid','predict_label']
res.to_csv("./result/submission_lgb.csv",index=False)