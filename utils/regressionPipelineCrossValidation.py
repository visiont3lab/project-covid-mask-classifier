from sklearn.datasets import load_boston, load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, TheilSenRegressor, RANSACRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from joblib import dump, load
from sklearn.model_selection import KFold,cross_val_score
import pandas as pd
import numpy as np

# LINK: https://visiont3lab.github.io/tecnologie_data_science/docs/regression/regression_choice.html

#df = pd.read_csv("https://raw.githubusercontent.com/Frenz86/machine-learning-course/main/ShareRaceByCity.csv")
#df.to_csv("data/ShareRaceByCity.csv")
df = pd.read_csv("data/ShareRaceByCity.csv")
print(df.head())

# Filtro  Remove null values
df_clean = df[df["share_white"]!="(X)"]

# Cleaning and settings info type
dfnew_dict = {
 'Geographic area' : df_clean['Geographic area'],
  #'City': df_clean['City'],
 'share_white': np.array(df_clean['share_white'], dtype=np.float),
 'share_black' : np.array(df_clean['share_black'], dtype=np.float),
 'share_native_american' : np.array(df_clean['share_native_american'], dtype=np.float),
 'share_asian' : np.array(df_clean['share_asian'], dtype=np.float),
 'share_hispanic' : np.array(df_clean['share_hispanic'], dtype=np.float),
}
dfnew = pd.DataFrame(data=dfnew_dict)

ln = list(df["City"].unique())
city_names = [ln[e] for e in [100,330,34,67,89]]
#dfnew = dfnew[(dfnew["City"]==city_names[0]) | (dfnew["City"]==city_names[1]) | (dfnew["City"]==city_names[2])]
#dfnew = dfnew[(dfnew["City"]==city_names[0]) & (dfnew["share_hispanic"]>0.7)]

# Preprocessing
df_clean_mod = pd.get_dummies(dfnew, prefix_sep='-', drop_first=False)
print(df_clean_mod.head())

# Starting 
Y = df_clean_mod.pop("share_hispanic")
X = df_clean_mod.values
#Y = df_clean_mod["share_hispanic"].values

# Compariamo/Scegliamo il modello
models = {}
models["Linear"]       = LinearRegression()
models["Lasso"]        = Lasso()
models["Ridge"]        = Ridge()
models["Theilsen"]     = TheilSenRegressor()
models["ElasticNet"]   = ElasticNet()
models["DecisionTree"]  = DecisionTreeRegressor()
models["RandomForest"]  = RandomForestRegressor()
folds   = 5 # 10 = 10%, 5 = 20% for testing
#metric  = "neg_mean_squared_error"
metric = "neg_root_mean_squared_error"
model_results = []
model_names   = []
for model_name in models:
    model   = models[model_name]
    pipe = Pipeline([
        ("std", StandardScaler())  ,          
        ("model", model)
    ])
    k_fold  = KFold(n_splits=folds, random_state=25,shuffle=True)
    results = cross_val_score(pipe, X, Y, cv=k_fold, scoring=metric)
    model_results.append(results)
    model_names.append(model_name)
    print("{}: {}, {}".format(model_name, np.round(results.mean(), 3), np.round(results.std(), 3)))

fig = go.Figure()
for name,res in zip(model_names,model_results):    
    fig.add_trace(go.Box(y=res,name=name, boxpoints='all'))
fig.write_html("CompareBoxPlot.html")
fig.show()

# Train Test (NOT REQUIRED ANYMORE)
# X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.35, random_state=26, shuffle=True) # Migliri

# TIP: It might be a good idea shuffle data first

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
reg.fit(X,Y)
print(np.round(reg.named_steps["reg"].coef_,3),reg.named_steps["reg"].intercept_)
dump(reg, 'mypipeline.joblib')
#---------------------------------

regload = load("mypipeline.joblib")
# Training Error
Y_hat_train = regload.predict(X)
err_sklearn_rmse = np.sqrt(mean_squared_error(Y,Y_hat))
err_sklearn_mae = mean_absolute_error(Y,Y_hat)
err_mae = np.sum(np.abs(Y_hat -Y))/len(Y_hat) # mean absolute error ()
err_rmse = np.sqrt(np.sum(np.power(Y_hat -Y,2))/len(Y_hat)) # mean squared error ()
print("Overall sklaern mse ", err_sklearn_rmse)
print("Overall  err mae", np.round(err_mae,3))
print("Overall err rmse ", np.round(err_rmse,3)) 

# Figure Results
xaxis = np.linspace(1,len(Y_hat),len(Y_hat),dtype=np.int)
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=xaxis, y=Y,mode='lines+markers', name="Y Test")
)
fig.add_trace(
    go.Scatter(x=xaxis, y=Y_hat,mode='lines+markers', name="Y Hat Test")
)
fig.update_layout(hovermode="x", title="Test Results", xaxis_title="Samples",yaxis_title="Price")
fig.write_html("Test.html")
fig.show()
