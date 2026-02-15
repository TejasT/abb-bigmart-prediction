import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# load data
train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
test_ids = test[["Item_Identifier", "Outlet_Identifier"]].copy()

train["source"] = "train"
test["source"] = "test"
df = pd.concat([train, test], ignore_index=True)

# fix fat content labels
df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"})

# impute missing item weight with mean per product
weight_mean = df.groupby("Item_Identifier")["Item_Weight"].mean()
df["Item_Weight"] = df.apply(lambda row: weight_mean[row["Item_Identifier"]] if pd.isna(row["Item_Weight"]) else row["Item_Weight"], axis=1)
df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].median())

# impute missing outlet size with mode per outlet type
size_mode = df.groupby("Outlet_Type")["Outlet_Size"].apply(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
df["Outlet_Size"] = df.apply(lambda row: size_mode[row["Outlet_Type"]] if pd.isna(row["Outlet_Size"]) or row["Outlet_Size"] == "" else row["Outlet_Size"], axis=1)

# broad item type from identifier prefix
df["Item_Type_Broad"] = df["Item_Identifier"].str[:2].map({"FD": "Food", "DR": "Drinks", "NC": "Non-Consumable"})

# fix zero visibility with mean per item type
vis_mean = df.groupby("Item_Type")["Item_Visibility"].mean()
df.loc[df["Item_Visibility"] == 0, "Item_Visibility"] = df.loc[df["Item_Visibility"] == 0, "Item_Type"].map(vis_mean)

# mrp,outlet type interaction features
for otype in df["Outlet_Type"].unique():
    df[f"MRP_x_{otype.replace(' ', '_')}"] = df["Item_MRP"] * (df["Outlet_Type"] == otype).astype(int)

# label encode categoricals
cat_cols = ["Item_Type", "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Type_Broad"]
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# split back
train_df = df[df["source"] == "train"].drop(columns=["source"])
test_df = df[df["source"] == "test"].drop(columns=["source", "Item_Outlet_Sales"])
train_df = train_df.copy()
test_df = test_df.copy()

# kfold target encoding: outlet sales per mrp
train_df["Outlet_Sales_Per_MRP"] = np.nan
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(train_df):
    fold_train = train_df.iloc[train_idx]
    spm = (fold_train["Item_Outlet_Sales"] / fold_train["Item_MRP"]).groupby(fold_train["Outlet_Identifier"]).mean()
    train_df.iloc[val_idx, train_df.columns.get_loc("Outlet_Sales_Per_MRP")] = train_df.iloc[val_idx]["Outlet_Identifier"].map(spm)

global_spm = (train_df["Item_Outlet_Sales"] / train_df["Item_MRP"]).mean()
train_df["Outlet_Sales_Per_MRP"] = train_df["Outlet_Sales_Per_MRP"].fillna(global_spm)

full_spm = (train_df["Item_Outlet_Sales"] / train_df["Item_MRP"]).groupby(train_df["Outlet_Identifier"]).mean()
test_df["Outlet_Sales_Per_MRP"] = test_df["Outlet_Identifier"].map(full_spm)
test_df["Outlet_Sales_Per_MRP"] = test_df["Outlet_Sales_Per_MRP"].fillna(global_spm)

# kfold target encoding: item sales mean
train_df["Item_Sales_Mean"] = np.nan
global_item_mean = train_df["Item_Outlet_Sales"].mean()
for tr_idx, va_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(train_df):
    fold_y = train_df.iloc[tr_idx]["Item_Outlet_Sales"]
    fold_ids = train_df.iloc[tr_idx]["Item_Identifier"]
    im = fold_y.groupby(fold_ids).mean()
    train_df.iloc[va_idx, train_df.columns.get_loc("Item_Sales_Mean")] = train_df.iloc[va_idx]["Item_Identifier"].map(im)
train_df["Item_Sales_Mean"] = train_df["Item_Sales_Mean"].fillna(global_item_mean)

full_item_mean = train_df["Item_Outlet_Sales"].groupby(train_df["Item_Identifier"]).mean()
test_df["Item_Sales_Mean"] = test_df["Item_Identifier"].map(full_item_mean)
test_df["Item_Sales_Mean"] = test_df["Item_Sales_Mean"].fillna(global_item_mean)

# prepare features
X = train_df.drop(columns=["Item_Identifier", "Item_Outlet_Sales", "Item_Weight", "Item_Fat_Content"])
y = train_df["Item_Outlet_Sales"]
X_test = test_df.drop(columns=["Item_Identifier", "Item_Weight", "Item_Fat_Content"])

print(f"Train: {len(X)}, Test: {len(X_test)}")
print(f"Features: {list(X.columns)}\n")

# model setup
model_names = ["LGB", "CB", "ET", "Ridge_feat"]
cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

def make_models(seed=42):
    return [
        LGBMRegressor(n_estimators=600, max_depth=8, learning_rate=0.007, subsample=1.0, colsample_bytree=1.0, min_child_weight=3, reg_alpha=1.0, reg_lambda=2.0, num_leaves=20, random_state=seed, verbose=-1, objective="tweedie", tweedie_variance_power=1.7),
        CatBoostRegressor(iterations=320, depth=6, learning_rate=0.05, subsample=0.5, l2_leaf_reg=1.0, random_seed=seed, verbose=0, cat_features=cat_feature_indices, loss_function="Tweedie:variance_power=1.8"),
        ExtraTreesRegressor(n_estimators=1000, max_depth=8, min_samples_leaf=7, max_features=1.0, random_state=seed, n_jobs=2),
    ]

# seed averaged stacking
seeds = [42, 123, 456]
all_oof = []
all_test_pred = []

for seed in seeds:
    print(f"=== Seed {seed} ===")
    oof = np.zeros((len(X), 4))
    kf_stack = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_i, (tr_idx, va_idx) in enumerate(kf_stack.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]

        models = make_models(seed)
        for m_i, model in enumerate(models):
            model.fit(X_tr, y_tr)
            oof[va_idx, m_i] = model.predict(X_va)
        del models; gc.collect()

        # ridge on scaled features
        scaler = StandardScaler()
        ridge = Ridge(alpha=1)
        ridge.fit(scaler.fit_transform(X_tr), y_tr)
        oof[va_idx, 3] = ridge.predict(scaler.transform(X_va))

        print(f"  Fold {fold_i+1}/5 done")

    # meta learner
    meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    meta.fit(oof, y)
    rmse = root_mean_squared_error(y, meta.predict(oof))
    print(f"  CV RMSE: {rmse:.2f}  weights={dict(zip(model_names, meta.coef_.round(4)))}")
    all_oof.append(meta.predict(oof))

    # full data predictions for test
    test_oof = np.zeros((len(X_test), 4))
    models = make_models(seed)
    for m_i, model in enumerate(models):
        model.fit(X, y)
        test_oof[:, m_i] = model.predict(X_test)

    scaler = StandardScaler()
    ridge = Ridge(alpha=1)
    ridge.fit(scaler.fit_transform(X), y)
    test_oof[:, 3] = ridge.predict(scaler.transform(X_test))

    all_test_pred.append(meta.predict(test_oof))
    del models; gc.collect()

# average across seeds
avg_oof = np.mean(all_oof, axis=0)
avg_test = np.mean(all_test_pred, axis=0)
print(f"\nSeed-averaged CV RMSE: {root_mean_squared_error(y, avg_oof):.2f}")

predictions = np.clip(avg_test, 0, None)
submission = test_ids.copy()
submission["Item_Outlet_Sales"] = predictions
submission.to_csv("submission.csv", index=False)

print(f"Submission shape: {submission.shape}")
print(f"Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")
print("Saved to submission.csv")
