import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import datetime as dt
import urllib.request
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class LSTM_Model:

    @classmethod
    def LSTM_Pred(self, tick):

        data_source = 'alphavantage'

        if data_source == 'alphavantage':
            # ====================== Loading Data from Alpha Vantage ==================================

            api_key = '7TONQ8CM5PXZ4YEO'

            # TCS stock market prices
            ticker = tick

            # JSON file with all the stock market data for TCS
            url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" % (
                ticker, api_key)

            # Save data to this file
            file_to_save = 'stock_market_data-%s.csv' % ticker

            # If you haven't already saved data,
            # Go ahead and grab the data from the url
            # And store date, low, high, volume, close, open values to a Pandas DataFrame
            if not os.path.exists(file_to_save):
                with urllib.request.urlopen(url_string) as url:
                    data = json.loads(url.read().decode())
                    # extract stock market data
                    data = data['Time Series (Daily)']
                    df = pd.DataFrame(
                        columns=['Date', 'Low', 'High', 'Close', 'Open'])
                    for k, v in data.items():
                        date = dt.datetime.strptime(k, '%Y-%m-%d')
                        data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                                    float(v['4. close']), float(v['1. open'])]
                        df.loc[-1, :] = data_row
                        df.index = df.index + 1
                print('Data saved to : %s' % file_to_save)
                df.to_csv(file_to_save)

            # If the data is already there, just load it from the CSV
            else:
                print('File already exists. Loading data from CSV')
                df = pd.read_csv(file_to_save)

        file_name = "stock_market_data-" + tick + ".csv"
        df = pd.read_csv(file_name)

        df = df[['Date', 'Open', 'Close', 'Low', 'High']]

        df = df.sort_values('Date')

        high_prices = df.loc[:, 'High']
        low_prices = df.loc[:, 'Low']
        df["Mid Prices"] = (high_prices + low_prices) / 2.0

        df.drop("Date", axis=1, inplace=True)

        df1 = df
        df = df.values

        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)

        def build_model(layers):
            model = Sequential()

            model.add(LSTM(
                input_shape=(None, layers[0]),
                units=50,
                return_sequences=True))
            model.add(Dropout(0.2))

            model.add(LSTM(
                100,
                return_sequences=False))
            model.add(Dropout(0.2))

            model.add(Dense(
                units=1))
            model.add(Activation('linear'))

            start = time.time()
            model.compile(loss='mse', optimizer='rmsprop')
            print('compilation time : ', time.time() - start)
            return model

        def load_data(stock, seq_len):
            amount_of_features = 5
            data = stock
            sequence_length = seq_len + 1
            result = []
            for index in range(len(data) - sequence_length):
                result.append(data[index: index + sequence_length])

            result = np.array(result)
            row = round(0.75 * result.shape[0])
            train = result[:int(row), :]
            x_train = train[:, :-1]
            y_train = train[:, -1][:, -1]
            x_test = result[int(row):, :-1]
            y_test = result[int(row):, -1][:, -1]

            x_train = np.reshape(
                x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
            x_test = np.reshape(
                x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

            return [x_train, y_train, x_test, y_test]

        window = 5
        X_train, y_train, X_test, y_test = load_data(df, window)

        x_latest = df[-5:]

        x_latest = np.reshape(x_latest, (1, x_latest.shape[0], window))

        model = build_model([5, window, 1])

        model.fit(
            X_train,
            y_train,
            batch_size=512,
            epochs=15,
            validation_split=0.1,
            verbose=1)

        trainScore = model.evaluate(X_train, y_train, verbose=0)

        # Predictions

        p = model.predict(X_test)

        mse = mean_squared_error(y_test, p)

        p = np.reshape(p, p.shape[0]).tolist()
        y_test = np.reshape(y_test, y_test.shape[0]).tolist()

        p_latest = model.predict(x_latest)
        p.append(p_latest[0, 0])

        def inverse_minmax(x, maxval, minval):
            return (x * (maxval - minval) + minval)

        maxval = max(df1['Mid Prices'])
        minval = min(df1['Mid Prices'])

        p = [inverse_minmax(y, maxval, minval) for y in p]
        y_test = [inverse_minmax(y, maxval, minval) for y in y_test]

        def delete_stock_data(file_name):
            if os.path.exists(file_name):
                os.remove(file_name)
                print("File removed successfully")
            else:
                print("The file does not exist")

        delete_stock_data(file_name)

        return (p, y_test, p[-1], mse)


# ob = LSTM_Model()
# p, y, tomorrow, mse = ob.LSTM_Pred("TCS")

# print("Predictions->", p)
# print()
# print("Tomorrow prediction-> ", tomorrow)
# print("MSE->", mse)
