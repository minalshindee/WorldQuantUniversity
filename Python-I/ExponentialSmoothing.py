import pandas as pd
import matplotlib.pyplot as plt


def exponential_smoothing(df):

    df['pred_es'] = 0

    i = 0
    while i < len(df):
        # print (df.iloc[i]['Adj Close'])
        i = i + 1

        x1 = df.iloc[i - 5]['Adj Close'] * 1
        x2 = df.iloc[i - 4]['Adj Close'] * 4
        x3 = df.iloc[i - 3]['Adj Close'] * 9
        x4 = df.iloc[i - 2]['Adj Close'] * 16
        x5 = df.iloc[i - 1]['Adj Close'] * 25

        df.ix[i - 1, 'pred_es'] = (x1 + x2 + x3 + x4 + x5) / (1 + 4 + 9 + 16 + 25)
    return df

def linear_regression(df):
    df['pred_lr'] = 0
    from sklearn import linear_model
    import numpy as np

    regr = linear_model.LinearRegression()
    x = np.array(range(len(df)))
    X = x.reshape(len(df),1)
    y = np.array(df['Adj Close'].values)
    #print(X,y)
    #print(X.shape,y.shape)
    regr.fit(X, y)
    for i in range(len(df)):
        df.ix[i,'pred_lr'] = regr.predict(i)

    #print(df)

    return df


def main():
    df = pd.read_csv('^GSPC.csv')
    df = exponential_smoothing(df)
    df = linear_regression(df)
    print(df)
    df.plot('Date', ['Adj Close', 'pred_es','pred_lr'])
    plt.show()

main()
