import glob
import pandas as pd

folder = "val/fangus_charbinary_lgbm_poisson/"
path = glob.glob(folder+"seed*.csv")
print(len(path))

for i, p in enumerate(path):
    df = pd.read_csv(p)
    df['deal_probability'] = df['deal_probability'].clip(0.0, 1.0) 
    dftest = pd.read_csv(p.replace("val/", "result/"))
    if i == 0:
        oof = df
        test = dftest
    else:
        oof = oof.merge(df, how="left", on="item_id")
        test = test.merge(dftest, how="left", on="item_id")
        
oof_id = oof.item_id
oof.drop("item_id", axis=1, inplace=True)
pd.DataFrame({"item_id":oof_id, "deal_probability": oof.mean(axis=1).values}).to_csv(folder+"bagging.csv", index=False)

test_id = test.item_id
test.drop("item_id", axis=1, inplace=True)
pd.DataFrame({"item_id":test_id, "deal_probability": test.mean(axis=1).values}).to_csv(folder.replace("val/", "result/")+"bagging.csv", index=False)