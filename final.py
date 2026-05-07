# %% [markdown]
# <h1 align="center"><b>Gold Price Prediction Project (1970–2026 → Future Forecast)</b></h1>

# %% [markdown]
# ## Project Overview
# This project focuses on predicting gold prices using historical data from 1970 to 2026. The dataset contains daily gold prices, and machine learning models are used to analyze trends and forecast future values.
# 
# ---
# 
# ## Objectives
# - Analyze historical gold price trends  
# - Build machine learning models for prediction  
# - Improve accuracy using time-series features  
# - Forecast future gold prices  
# 
# ---

# %% [markdown]
# ## Import Libraries

# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Load Dataset

# %%
df=pd.read_csv(r"C:\Users\LENOVO\Desktop\MAKBIG\DATASETS\gold_price_1970_2026_daily.csv")
df

# %% [markdown]
# ## Dataset Shape

# %%
df.shape

# %%
df.columns

# %% [markdown]
# ## Dataset Information

# %%
df.info()

# %% [markdown]
# ## Statistical Summary

# %%
df.describe()

# %% [markdown]
# ## Display Dataset

# %%
df.head(10)

# %%
df.tail(10)

# %% [markdown]
# ## Duplicates

# %%
df.duplicated().sum()

# %%
df.drop_duplicates(inplace=True)

# %% [markdown]
# ## Missing Values

# %%
# Missing values
df.isnull().sum()

# %%
df.dropna(inplace=True)

# %%
df.isnull().sum()

# %% [markdown]
# ## Outlier Detection using Boxplot

# %%
sns.boxplot(df)
plt.show()

# %% [markdown]
# ## Feature Engineering

# %%
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# %%
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# %%
df.drop(['Date'], axis=1, inplace=True)

# %% [markdown]
# ## Close Price Trend Over Years

# %%
df.plot.line(x='Year',y='Close',use_index=True)

# %%
df_2025=df[df['Year']==2025]
df_2025

# %%
df_2025_test=df_2025.Close
df_2025_test

# %%
df_2025=df_2025.drop('Close',axis=1)

# %%
df1 = df[df['Year'] != 2025]

# %%
df1

# %%
x = df1.drop('Close', axis=1)
y = df1['Close']

# %%
y

# %% [markdown]
# ## XGBoost Predicted Close Prices for 2025

# %%
from xgboost import XGBRegressor
xg=XGBRegressor(n_estimators=100,learning_rate=0.05)
model1=xg.fit(x,y)
y1_pred=model1.predict(df_2025)
y1_pred

# %%
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
print("R2:", r2_score(df_2025_test, y1_pred))
print("MAE:", mean_absolute_error(df_2025_test, y1_pred))
print("RMSE:", np.sqrt(mean_squared_error(df_2025_test, y1_pred)))

# %%
print("Train Score:", model1.score(x, y))
print("Test Score:", model1.score(df_2025, df_2025_test))

# %% [markdown]
# ## Random Forest Predicted Close Prices for 2025

# %%
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=300,max_depth=15)
model2=rf.fit(x,y)
y2_pred=model2.predict(df_2025)
y2_pred

# %%
print("R2:", r2_score(df_2025_test, y2_pred))
print("MAE:", mean_absolute_error(df_2025_test, y2_pred))
print("RMSE:", np.sqrt(mean_squared_error(df_2025_test, y2_pred)))

# %%
print("Train Score:", model2.score(x, y))
print("Test Score:", model2.score(df_2025, df_2025_test))

# %% [markdown]
# ## RobustScaler for Linear Regression

# %%
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x)
x_test = scaler.transform(df_2025)

# %% [markdown]
# ## Linear Regression Predicted Close Prices for 2025

# %%
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model3=lr.fit(x,y)
y3_pred=model3.predict(df_2025)
y3_pred

# %%
print("R2:", r2_score(df_2025_test, y3_pred))
print("MAE:", mean_absolute_error(df_2025_test, y3_pred))
print("RMSE:", np.sqrt(mean_squared_error(df_2025_test, y3_pred)))

# %%
print("Train Score:", model3.score(x, y))
print("Test Score:", model3.score(df_2025, df_2025_test))

# %%
df2=df
df2

# %% [markdown]
# ## Lag and Rolling Feature Engineering

# %%
df2['lag1'] = df2['Close'].shift(1)
df2['lag7'] = df2['Close'].shift(7)
df2['lag30'] = df2['Close'].shift(30)
df2['rolling_mean_7'] = df2['Close'].rolling(7).mean()
df2['rolling_mean_30'] = df2['Close'].rolling(30).mean()
df2['rolling_std_7'] = df2['Close'].rolling(7).std()
df2

# %%
df2.dropna(inplace=True)

# %%
features = ['lag1','lag7','lag30','Year','Month','Day','rolling_mean_7','rolling_mean_30','rolling_std_7']

X2 = df[features]
y2 = df['Close']


# %% [markdown]
# ## Train-Test Split

# %%
train = df2[df2['Year'] < 2025]
test = df2[df2['Year'] >= 2025]

X2_train = train[features]
y2_train = train['Close']

X2_test = test[features]
y2_test = test['Close']


# %%
df2['Date'] = pd.to_datetime(df2[['Year','Month','Day']])
df2 = df2.sort_values('Date')
df2

# %% [markdown]
# ## 365-Day Future Forecast using XGBoost

# %%
model4=XGBRegressor(n_estimators=300,learning_rate=0.05)
model4.fit(X2_train,y2_train)

# %%
future_days = 365

last_data = df.copy()
future_predictions = []

for i in range(future_days):

    last_row = last_data.iloc[-1]

    next_date = last_row['Date'] + pd.Timedelta(days=1)

    new_row = {}

    # Lag features
    new_row['lag1'] = last_row['Close']
    new_row['lag7'] = last_data['Close'].iloc[-7]
    new_row['lag30'] = last_data['Close'].iloc[-30]

    # Rolling features
    new_row['rolling_mean_7'] = last_data['Close'].tail(7).mean()
    new_row['rolling_mean_30'] = last_data['Close'].tail(30).mean()
    new_row['rolling_std_7'] = last_data['Close'].tail(7).std()

    # Date features
    new_row['Year'] = next_date.year
    new_row['Month'] = next_date.month
    new_row['Day'] = next_date.day

    # Keep same feature order as training
    X_new = pd.DataFrame([new_row])[features]

    pred = model4.predict(X_new)[0]

    # Store prediction
    new_row['Close'] = pred
    new_row['Date'] = next_date

    future_predictions.append(new_row)

    # Append back for recursive forecasting
    last_data = pd.concat(
        [last_data, pd.DataFrame([new_row])],
        ignore_index=True
    )

# %%
future_df = pd.DataFrame(future_predictions)
future_df

# %% [markdown]
# ## Future Gold Price Trend (2027)

# %%
plt.figure(figsize=(14,6))
plt.plot(future_df['Date'],future_df['Close'])
plt.title("Gold Price Prediction for 2027")
plt.show()


