from sklearn.datasets import load_boston, load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, TheilSenRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from joblib import dump, load

boston = load_boston()
#diabetes = load_diabetes()
#X = boston['data']
Y = boston['target']
column_names = []
i = 1
for v in list(boston["feature_names"]):
    l = "x" + str(i) + "-" + v
    column_names.append(l)
    i = i+1
df_boston = pd.DataFrame(boston.data,columns=column_names)
df_boston["Y"] = Y

# Preprocessing
X = df_boston.iloc[:,:-1].values
Y = df_boston["Y"].values # array
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.35, random_state=26, shuffle=True) # Migliri

# Training X_train -------------------------
#reg = RandomForestRegressor()  # instanzia la classe
#reg = GradientBoostingRegressor()
#reg = LinearRegression()
#reg.fit(X_train,Y_train) # allenamento
reg = Pipeline([
    ("std", StandardScaler())  ,          
    #("poly",PolynomialFeatures(degree=3,interaction_only=False)),
	("reg", Lasso())
 	#("reg", ElasticNet())
    #("reg", TheilSenRegressor())
    #("reg", LinearRegression())
])
reg.fit(X_train,Y_train)
#print(np.round(reg.named_steps["reg"].coef_,3),reg.named_steps["reg"].intercept_)
#---------------------------------
dump(reg, 'mypipeline.joblib')

regload = load("mypipeline.joblib")
# Training Error
Y_hat_train = regload.predict(X_train)
err_sklearn_rmse = np.sqrt(mean_squared_error(Y_train,Y_hat_train))
err_sklearn_mae = mean_absolute_error(Y_train,Y_hat_train)
err_mae = np.sum(np.abs(Y_hat_train -Y_train))/len(Y_hat_train) # mean absolute error ()
err_rmse = np.sqrt(np.sum(np.power(Y_hat_train -Y_train,2))/len(Y_hat_train)) # mean squared error ()
print("Training sklaern mse ", err_sklearn_rmse)
print("Training  err mae", np.round(err_mae,3))
print("Training err rmse ", np.round(err_rmse,3)) 

# Testing Error
Y_hat_test = regload.predict(X_test)
err_sklearn_rmse = np.sqrt(mean_squared_error(Y_test,Y_hat_test))
err_sklearn_mae = mean_absolute_error(Y_test,Y_hat_test)
err_mae = np.sum(np.abs(Y_hat_test -Y_test))/len(Y_hat_test) # mean absolute error ()
err_rmse = np.sqrt(np.sum(np.power(Y_hat_test -Y_test,2))/len(Y_hat_test)) # mean squared error ()

print("Test sklaern mse ", err_sklearn_rmse)
print("Test  err mae", np.round(err_mae,3))
print("Test err rmse ", np.round(err_rmse,3)) 


# Figure Test Results
xaxis = np.linspace(1,len(Y_train),len(Y_train),dtype=np.int)
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=xaxis, y=Y_train,mode='lines+markers', name="Y Train")
)
fig.add_trace(
    go.Scatter(x=xaxis, y=Y_hat_train,mode='lines+markers', name="Y Hat Train")
)
fig.update_layout(hovermode="x", title="Train Results", xaxis_title="Samples",yaxis_title="Price")
fig.write_html("Train.html")


# Figure Test Results
xaxis = np.linspace(1,len(Y_test),len(Y_test),dtype=np.int)
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=xaxis, y=Y_test,mode='lines+markers', name="Y Test")
)
fig.add_trace(
    go.Scatter(x=xaxis, y=Y_hat_test,mode='lines+markers', name="Y Hat Test")
)
fig.update_layout(hovermode="x", title="Test Results", xaxis_title="Samples",yaxis_title="Price")
fig.show()
#fig.write_html("Test.html")