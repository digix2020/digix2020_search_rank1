import pandas as pd

lgb = pd.read_csv('submission_lgb.csv')
lgb_rank = pd.read_csv('submission_lgbrank.csv')

res = lgb[['queryid','documentid']].reset_index(drop=True)
res['predict_label'] = lgb['predict_label'] * 0.8 + lgb_rank['predict_label'] * 0.2
res.columns = ['queryid','documentid','predict_label']
res.to_csv("./submission.csv",index=False)
