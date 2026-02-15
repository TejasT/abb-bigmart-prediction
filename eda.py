"""basic data exploration and plots."""
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train = pd.read_csv("train_v9rqX0R.csv")

# --- 2x2 EDA grid ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. target is right-skewed -> motivates tweedie loss
axes[0, 0].hist(train["Item_Outlet_Sales"], bins=40, color="blue", alpha=0.85)
axes[0, 0].set_title("Target is right-skewed")
axes[0, 0].set_xlabel("Sales (target)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].axvline(train["Item_Outlet_Sales"].median(), color="black", linestyle="--", alpha=0.5)
axes[0, 0].text(train["Item_Outlet_Sales"].median() + 100, axes[0, 0].get_ylim()[1] * 0.9, "median", fontsize=8)

# 2. mrp vs sales with regression line
corr = train["Item_MRP"].corr(train["Item_Outlet_Sales"])
axes[0, 1].scatter(train["Item_MRP"], train["Item_Outlet_Sales"], alpha=0.12, s=4, c="blue")
coeffs = P.polyfit(train["Item_MRP"], train["Item_Outlet_Sales"], deg=1)
x_line = np.linspace(train["Item_MRP"].min(), train["Item_MRP"].max(), 100)
axes[0, 1].plot(x_line, P.polyval(x_line, coeffs), color="red", linewidth=2)
axes[0, 1].set_title(f"MRP vs Sales (r = {corr:.2f})")
axes[0, 1].set_xlabel("Item MRP")
axes[0, 1].set_ylabel("Sales")

# 3. same mrp, different outlet type = different sales -> motivates interaction features
type_colors = {"Grocery Store": "red", "Supermarket Type1": "blue",
               "Supermarket Type2": "orange", "Supermarket Type3": "green"}
for otype, c in type_colors.items():
    mask = train["Outlet_Type"] == otype
    short = otype.replace("Supermarket ", "SM ")
    axes[1, 0].scatter(train.loc[mask, "Item_MRP"], train.loc[mask, "Item_Outlet_Sales"],
                       alpha=0.18, s=4, color=c, label=short)
axes[1, 0].legend(fontsize=7, markerscale=4, loc="upper left")
axes[1, 0].set_title("MRP vs Sales by outlet type")
axes[1, 0].set_xlabel("MRP")
axes[1, 0].set_ylabel("Sales")

# 4. errors grow with mrp -> motivates tweedie loss (heteroscedasticity)
pred = P.polyval(train["Item_MRP"], coeffs)
resid = np.abs(train["Item_Outlet_Sales"] - pred)

axes[1, 1].scatter(train["Item_MRP"], resid, alpha=0.1, s=3, c="gray")
bins = pd.cut(train["Item_MRP"], bins=12)
mean_err = resid.groupby(bins).mean()
centers = [interval.mid for interval in mean_err.index]
axes[1, 1].plot(centers, mean_err.values, "o-", color="red", linewidth=2, markersize=5, label="avg |error| per bin")
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_title("Error vs MRP")
axes[1, 1].set_xlabel("MRP")
axes[1, 1].set_ylabel("Abs Error")

fig.suptitle("EDA", fontsize=13, y=1.005)
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
print("saved eda_plots.png")
