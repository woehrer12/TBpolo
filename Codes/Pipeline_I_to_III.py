########### PARTS I to III
import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests
import json
import time
import csv

# Load the data, define test set and train set
#os.chdir("C:\\Users\\Utilisateur\\Desktop\\Crypto\\Data")

now = time.time()
_5year = 157784630
_6year = 189341556
now_5year = now - _5year
now_6year = now - _6year

url = "https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=" + str(now_6year) + "&end=" + str(now) + "&period=7200"

resp = requests.get(url=url)
data = resp.json()

csv_path = 'USDT_BTC_Poloniex_20022015_21122020_7200.csv'
with open(csv_path, 'w', newline='') as f:
	dict_writer = csv.DictWriter(f, data[0].keys())
	dict_writer.writeheader()
	dict_writer.writerows(data)
	print("CSV file saved at path {}".format(csv_path))


# print(data)

df = pd.DataFrame(data)

# df = pd.read_csv('./Data/USDT_BTC_Poloniex_20022015_21122020_7200.csv')
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by = 'date')

# Keep only 5 basic information + the date
df = df[['close', 'date', 'high', 'low', 'open', 'volume']]

# Define limit between train set and testset
start_validation = '2018-12-21 12:00:00'
start_test = '2019-12-21 12:00:00'
stoploss = 0.05
takeprofit = 0.1

# Creating subfolders if they don't exist ...
if not os.path.exists('./Data'):
        os.makedirs('./Data')
if not os.path.exists('./Models'):
        os.makedirs('./Models')
if not os.path.exists('./Results'):
        os.makedirs('./Results')

#################### I - Define standard functions ###############################
def compute_sma(df, window, colname):
    '''Computes Simple Moving Average column on a dataframe'''
    df[colname] = df['close'].rolling(window=window, center=False).mean()
    return(df)

def compute_rsi(df, window, colname):
    '''Computes RSI column for a dataframe. http://stackoverflow.com/a/32346692/3389859'''
    series = df['close']
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[window - 1]] = np.mean(u[:window])
    u = u.drop(u.index[:(window - 1)])
    # first value is sum of avg losses
    d[d.index[window - 1]] = np.mean(d[:window])
    d = d.drop(d.index[:(window - 1)])
    rs = u.ewm(com=window - 1,ignore_na=False,
               min_periods=0,adjust=False).mean() / d.ewm(com=window - 1, ignore_na=False,
                                            min_periods=0,adjust=False).mean()
    df[colname] = 100 - 100 / (1 + rs)
    df[colname].fillna(df[colname].mean(), inplace=True)
    return(df)

#################### II - Build the function to compute variables #################
def compute_variables1(df):
    print("Let's compute predictive variables : ")
    df["date"] = pd.to_datetime(df["date"])
    df['bodysize'] = df['close'] - df['open']
    df['shadowsize'] = df['high'] - df['low']
    for window in [3, 8, 21, 55, 144, 377]: # several Fibonacci numbers
        print(window)
        df = compute_sma(df, window, colname = 'sma_{}'.format(window))
        df = compute_rsi(df, window, colname = 'rsi_{}'.format(window))
        df["Min_{}".format(window)] = df["low"].rolling(window).min()
        df["Max_{}".format(window)] = df["high"].rolling(window).max()
        df["volume_{}".format(window)] = df["volume"].rolling(window).mean()
        df['percentChange_{}'.format(window)] = df['close'].pct_change(periods = window)
        df['RelativeSize_sma_{}'.format(window)] = df['close'] / df['sma_{}'.format(window)]
        df['Diff_{}'.format(window)] = df['close'].diff(window)

    # (a) Add modulo 10, 100, 1000, 500, 50
    df["Modulo_10"] = df["close"].copy() % 10
    df["Modulo_100"] = df["close"].copy() % 100
    df["Modulo_1000"] = df["close"].copy() % 1000
    df["Modulo_500"] = df["close"].copy() % 500
    df["Modulo_50"] = df["close"].copy() % 50
    # (b) Add weekday and day of the month
    df["WeekDay"] = df["date"].dt.weekday
    df["Day"] = df["date"].dt.day
    df.dropna(inplace=True)
    return(df)

df = compute_variables1(df)
df.to_csv('./Data/DatasetWithVariables.csv', index = False)

#################### III - Compute the output #######################################
def check_outcome(df, line, stoploss, takeprofit):
    '''0 means we reached stoploss
    1 means we reached takeprofit
    -1 means still in between'''
    price0 = df["close"].iloc[line]
    upper_lim = price0*(1+takeprofit)
    down_lim = price0*(1-stoploss)
    for i in range(line, df["close"].size):
        if df["low"].iloc[i] < down_lim :
            return(0)
        elif df["high"].iloc[i] > upper_lim :
            return(1)
    return(-1)

def compute_result(df, stoploss, takeprofit):
    df['result'] = 0
    for i in range(df["close"].size):
        if i%500 == 0:
            print(i, '/', df.shape[0])
        df['result'].iloc[i] = check_outcome(df, i, stoploss, takeprofit)
    return(df)

df = compute_result(df, stoploss, takeprofit)
df = df[df['result']>=0] # Only keep observations where we also have the result
df.to_csv('./Data/DatasetWithVariablesAndY_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)

#################### IV - Apply PCA and save results ##############################
# First we define the trainset, validation set, testset. This is important in this step to avoid causality issues.
k = df.shape[0]
trainset = df[:int(k*.5)] #df[df['date'] < start_validation]
validation_set = df[int(k*.5):int(k*.75)] #df[(df['date'] >= start_validation) & (df['date'] < start_test)]
testset = df[int(k*.75):] #df[df['date'] > start_test]

trainset.to_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
validation_set.to_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
testset.to_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)

# Display the splitting
plt.plot(pd.to_datetime(trainset['date']), trainset['close'], c = 'orange')
plt.plot(pd.to_datetime(validation_set['date']), validation_set['close'], c = 'b')
plt.plot(pd.to_datetime(testset['date']), testset['close'], c = 'g')
plt.title('Repartition between trainset, validation set, and test set')
plt.show()

# (i) Scale the variables
scale_fct = StandardScaler()
scale_fct.fit(trainset.drop('date', 1).drop('result', 1))
pk.dump(scale_fct, open('./Models/scaler.pkl','wb'))

# (ii) Apply PCA
pca = PCA(n_components=trainset.shape[1] - 2) # remove the result and the date
pca.fit(scale_fct.transform(trainset.drop('date', 1).drop('result', 1)))
pk.dump(pca, open('./Models/pca.pkl',"wb"))

# (iii) Scale PCA components (this accelerates training process in Deep Learning)
pca_scaler = StandardScaler()
pca_scaler.fit(pca.transform(scale_fct.transform(trainset.drop('date', 1).drop('result', 1))))
pk.dump(pca_scaler, open('./Models/pca_scaler.pkl','wb'))

# (iv) Save ready-to-use versions (i.e. datasets after applying scalers and PCA)
trainset_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(trainset.drop('date', 1).drop('result', 1)))))
validation_set_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(validation_set.drop('date', 1).drop('result', 1)))))
testset_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(testset.drop('date', 1).drop('result', 1)))))

trainset_final.to_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
validation_set_final.to_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
testset_final.to_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)

















from keras.models import Sequential
from keras.layers import Dense
import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

list_nPCs = [10, 15, 20, 25, 30, 35, 40]

# (a) Load previously built datasets : we just need train sets and validation sets here
trainset_final = pd.read_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
trainset = pd.read_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

validation_set_final = pd.read_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
validation_set = pd.read_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

# (b) Build and train several models different amounts of PCs
for nPCs in list_nPCs:
    print(nPCs)
    X = trainset_final.iloc[:, :nPCs]
    y = trainset["result"]

    # Build model and train it
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
    #Second, third and fourth  hidden Layers
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    #Fitting the data to the training dataset
    classifier.fit(X,y, batch_size=500, epochs=75, verbose =1)

    # pk.dump(classifier, open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit),"wb"))
    classifier.save_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
# (c) Test onto the testset : we compare all models and store results in a csv file
accuracies, nPCs_list = [], []
for nPCs in list_nPCs:
    print(nPCs)
     # Build model and train it
    clf = Sequential()
    #First Hidden Layer
    clf.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
    #Second, third and fourth  hidden Layers
    clf.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

    #Output Layer
    clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    # with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    #     clf = pk.load(f)
    clf.load_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
    # Compute predictions on testset
    preds = (clf.predict(validation_set_final.iloc[:, :nPCs]) > 0.5)*1

    # Assess accuracy on Bullish predictions only (because we will only perform Bullish trades IRL) : we prioritize selectivity
    validation_set1 = validation_set[preds == 1].copy()
    accuracies.append(np.mean(preds == list(validation_set1['result'])))
    nPCs_list.append(nPCs)

recap = pd.DataFrame({'nPCs' : list(nPCs_list), 'Accuracy' : (list(accuracies))})
recap.to_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
print(recap)







import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import ttest_1samp

################### (a) Load previously built datasets
fees = 0.00125 # transaction fees : 0.125% for example

nTrades_mini = 50 # minimal number of trades we want over the test set: this is for second approach

trainset_final = pd.read_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
trainset = pd.read_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

validation_set_final = pd.read_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
validation_set = pd.read_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

testset_final = pd.read_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
testset = pd.read_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))


def compute_earnings_loss(stoploss, takeprofit, fees):
    '''Compute earnings and loss with given fees, stoploss, takeprofit'''
    win = (1-fees)*(1+takeprofit)*(1-fees) -1
    loss = (1-fees)*(1-stoploss)*(1-fees) -1
    return(win, loss)

def predict_and_backtest_bullish(df, df_final, model, stoploss, takeprofit, fees, plotting = True):
    '''This functin takes the test set as input (in both shapes) + the model, computes predictions and  probabilities, then compute the earnings according to the fees. Finally it can plot the strategy'''
    # Compute predictions on testset
    df['preds'] = (clf.predict(df_final.iloc[:, :nPCs]) > 0.5)*1
    df['proba1'] = clf.predict(df_final.iloc[:, :nPCs])

    # keep only the timesteps in which the model predicts a bullish trend
    testset1 = df[df['preds'] == 1].copy()

    # Compute earnings column
    a = compute_earnings_loss(stoploss, takeprofit, fees)
    testset1['EarningsBullish'] = (testset1['preds'] == testset1['result'])*a[0] + (testset1['preds'] != testset1['result'])*a[1]

    if plotting:
        # Now plot our trading strategy
        plt.plot(pd.to_datetime(testset1['date']), np.cumsum(testset1['EarningsBullish']))
        plt.title('Approach over the test set \n ROI = {} %'.format(100*np.mean(testset1['EarningsBullish'])))
        plt.xlabel('Date')
        plt.xlabel('Cumulative Earnings')
        plt.show()

        # Display the entry points
        plt.plot(pd.to_datetime(df['date']), df['close'])
        plt.scatter(pd.to_datetime(testset1['date']), testset1['close'], c = (testset1['EarningsBullish']>0))
        plt.title('Entry points \n Yellow = Win, Blue = Loss')
        plt.show()

    return(testset1)

#################### (b) Basic strategy : pick the best model and bet on bullish trends over the testset
recap = pd.read_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]

# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf = Sequential()
#First Hidden Layer
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
#Second, third and fourth  hidden Layers
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

#Output Layer
clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf.load_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
# Compute predictions on testset

testset1 = predict_and_backtest_bullish(testset, testset_final, clf, stoploss, takeprofit, fees, plotting = True)

# Assess the performance by comparing to if we always traded bullish blindly over the period
a = compute_earnings_loss(stoploss, takeprofit, fees)
testset_benchmark = testset.copy()
testset_benchmark['EarningsBullish'] = (testset['result'] == 1)*a[0] + (testset['result'] == 0)*a[1]
avg_return_benchmark = np.mean(testset_benchmark['EarningsBullish'])

# Now let's look at our approach's performance and std
p_value = ttest_1samp(testset1['EarningsBullish'], popmean = avg_return_benchmark)[1]
print('Our model has an average ROI of {} %, while trading blindly bullish over the same period yielded a ROI of {} %, when we perform statistical testing of difference there is a p-value of {}.'.format(100*np.mean(testset1['EarningsBullish']), 100*avg_return_benchmark, p_value))

################### (c) More evolved strategy : look for the threshold to limit to the cases where p>a (for the best model)
def table_recap(df, stoploss, takeprofit, nPCs, columnA = 'proba1', columnB = 'EarningsBullish'):
    ''' Summarize the strategy by steps of 0.05, depending on which column (i.e. strategy)
    we choose'''
    recap = pd.DataFrame(np.zeros((int(10), 0)))
    recap['stoploss'] = stoploss
    recap['takeprofit'] = takeprofit
    recap['nPCs'] = nPCs
    recap['Min'] = [0.5 + k*0.05 for k in range(10)]
    recap['Max'] = 1
    recap['ROI%'] = 0
    recap['nTrades'] = 0
    for i in range(len(recap['Min'])):
        min, max = recap['Min'].iloc[i], recap['Max'].iloc[i]
        df2 = df[(df[columnA] > min) & (df[columnA] < max)]
        recap['ROI%'].iloc[i] = 100 * np.mean(df2[columnB])
        recap['nTrades'].iloc[i] = df2.shape[0]

    return(recap)

# (i) Identify best threshold
# Load best model
recap = pd.read_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]
# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf = Sequential()
#First Hidden Layer
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
#Second, third and fourth  hidden Layers
clf.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
clf.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

#Output Layer
clf.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
#     clf = pk.load(f)
clf.load_weights('./Models/DL_model_{}PC_stoploss{}_takeprofit{}.h5'.format(nPCs, stoploss, takeprofit))
# Compute predictions on testset

# Compute predictions on validation_set
validation_set = predict_and_backtest_bullish(validation_set, validation_set_final, clf, stoploss, takeprofit, fees, plotting = False)

# Compute recapitulative table
recap = table_recap(validation_set, stoploss, takeprofit, nPCs)
recap.to_csv('./Results/Recapitulative_result_stoploss{}_takeprofit{}_{}PCs.csv'.format(stoploss, takeprofit, nPCs), index = False)

# Pick the most profitable
recap = recap.sort_values('ROI%', ascending = False)
recap = recap[recap['nTrades'] > nTrades_mini]
print(recap)
min, max = recap['Min'].iloc[0], recap['Max'].iloc[0]


# (ii) Now that we have identified the best threshold, filter the predictions
# ...Finally : we plot our strategy
testset2 = predict_and_backtest_bullish(testset, testset_final, clf, stoploss, takeprofit, fees, plotting = False)
testset2 = testset2[(testset2['proba1'] > min) & (testset2['proba1'] < max)].copy()

# Now plot our trading strategy
plt.plot(pd.to_datetime(testset2['date']), np.cumsum(testset2['EarningsBullish']))
plt.title('Approach n°2 over the test set \n ROI = {} %'.format(100*np.mean(testset2['EarningsBullish'])))
plt.xlabel('Date')
plt.xlabel('Cumulative Earnings')
plt.show()

# Display the entry points
plt.plot(pd.to_datetime(testset['date']), testset['close'])
plt.scatter(pd.to_datetime(testset2['date']), testset2['close'], c = (testset2['EarningsBullish']>0))
plt.title('Entry points \n Yellow = Win, Blue = Loss')
plt.show()

# (iii) Assess the performance by comparing to if we always traded bullish over the period
a = compute_earnings_loss(stoploss, takeprofit, fees)
testset_benchmark = testset.copy()
testset_benchmark['EarningsBullish'] = (testset['result'] == 1)*a[0] + (testset['result'] == 0)*a[1]
avg_return_benchmark = np.mean(testset_benchmark['EarningsBullish'])

# Now let's look at our approach's performance and std
p_value = ttest_1samp(testset2['EarningsBullish'], popmean = avg_return_benchmark)[1]
print('Our model has an average ROI of {} %, while trading blindly bullish over the same period yielded a ROI of {} %, when we perform statistical testing of difference there is a p-value of {}.'.format(100*np.mean(testset2['EarningsBullish']), 100*avg_return_benchmark, p_value))