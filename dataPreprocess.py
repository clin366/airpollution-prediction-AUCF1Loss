import pandas as pd
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score 
from sklearn import preprocessing
import argparse
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.linear_model import LogisticRegressionCV
import datetime as dt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
import argparse

class DataSplit(object):
    def __init__(self, file_path, season, final_year):
        self.file_path = file_path 
        self.season = season
        self.final_year = final_year 

    def readData(self):
        first_year = pd.read_csv(self.file_path)
        first_year_part = first_year.drop(['ozone.1', 'ozone.2', 'ozone.3', 'ozone.4', 'DATE'], axis = 1)
        first_year_part.index = first_year_part['date'].apply(lambda x: datetime.strptime(x[:-2] + '20' + x[-2:], '%m/%d/%Y').date())
        return first_year_part

    # select the cold months and warm months from the data
    def isColdWarm(self, row):
        dt = row['date']
        # get month, day
        fields = dt.split("/")
        if (len(fields) < 3):
            return row

        month = int(fields[0])
        day = int(fields[1])

        if (month >= 4) and (month <= 10):
            if ((month == 4) and (day < 15)):
                row['isCold']=True
                return row
            if ((month == 10) and (day > 14)):
                row['isCold'] = True
                return row

            row['isWarm'] = True
            return row
        else:
            row['isCold'] = True
            return row

    # choose the final year, the year range in the training, validation and testing data
    def selectData(self, first_year_part, season, final_year):
        # season = self.season
        # final_year = self.final_year
        # one-one-year split
        start = first_year_part.index.searchsorted(dt.date(2007, 1, 1))
        # start = first_year_part.index.searchsorted(dt.date(final_year-2, 1, 1))
        # all_year split
        # start = first_year_part.index.searchsorted(dt.date(2007, 1, 1))
        # print out the year for three years
        # end = first_year_part.index.searchsorted(dt.date(final_year, 1, 1))
        end = first_year_part.index.searchsorted(dt.date(final_year+1, 1, 1))
        first_year_part= first_year_part.ix[start:end]

        first_year_part = first_year_part.drop(['lung irritation'], axis = 1)
        first_year_part = first_year_part[-first_year_part['O3_M8_SCH1'].isnull()]
        first_year_part = first_year_part.fillna(first_year_part.mean())
    #     first_year_part['pollution'] = first_year_part.O3_M8_SCH1.apply(lambda x: 1 if x > 50 else 0)
        first_year_part['pollution'] = first_year_part.O3_M8_SCH1
    #     first_year_part['pollution'] = first_year_part.O3_M8_SCH1.apply(lambda x: 1 if x > 70 else 0)
        first_year_part = first_year_part.drop(['O3_M8_SCH1'], axis = 1)

        # print(first_year_part.head())
        # sys.exit("========Above is first_year_part_head=========")
        # get first diff terms
        # first_year_diff = first_year_part.drop(['pollution', 'date'], axis = 1).diff()
        # first_year_diff.columns = [i + '_diff' for i in first_year_diff.columns]
        # first_year_diff = first_year_diff.fillna(first_year_diff.mean())
        # first_year_part = pd.concat([first_year_part, first_year_diff], axis=1, sort=False)

        data = first_year_part.copy()
        data['isCold']=False
        data['isWarm']=False
        data = data.apply(self.isColdWarm, axis=1)
        data = data.drop(['date'], axis = 1)
        # data_warm = data[data.isWarm == True]
        if season == 'winter':
            data_warm = data[data.isCold == True]
        else:
            data_warm = data[data.isWarm == True]
        data_warm = data_warm.drop(['isCold', 'isWarm'], axis = 1)
        return data_warm

    # Given the testing data range(i.e. which year is the testing year), do time series trainTestSplit
    def trainTestSplit(self, air_data, test_start, test_end, final_start, final_end, single_feature = False, shift_period = 0):
        np.random.seed(0)

        # shift all the search terms data
        # print(air_data.columns)
        pollution_labels = air_data.pollution
        air_data = air_data.drop(['pollution'], axis = 1)

        # shift one day to make prediction, for the existing search terms 
        # air_data = air_data.shift(1)
        # use current days feature
        air_data.fillna(air_data.mean(), inplace = True)

        air_data['pollution'] = pollution_labels
        # delete previous pollution
        # air_data['previous_pollution'] = air_data.pollution.shift(shift_period)
        #append the first previous_pollution to zero
        for i in range(shift_period):
            air_data.iloc[i, air_data.columns.get_loc('previous_pollution')] = 0

        start = air_data.index.searchsorted(test_start)
        end = air_data.index.searchsorted(test_end)

        final_start = air_data.index.searchsorted(final_start)
        final_end = air_data.index.searchsorted(final_end)

        air_data_test = air_data.ix[start:end]
        air_data_train = air_data.ix[:start]
        air_data_final = air_data.ix[final_start:final_end]

        y_train = np.array(air_data_train['pollution'])
        y_test = np.array(air_data_test['pollution'])
        y_final = np.array(air_data_final['pollution'])
        #
        if single_feature:
            X_train = air_data_train['previous_pollution']
            X_test = air_data_test['previous_pollution']
            X_final = air_data_final['previous_pollution']
        else:
            X_train = air_data_train.drop(['pollution'], axis = 1)
            X_test = air_data_test.drop(['pollution'], axis = 1)
            X_final = air_data_final.drop(['pollution'], axis = 1)

        
        return X_train, X_test, X_final, y_train, y_test, y_final

    def generateTrainTest(self):
        air_data_raw = self.readData()
        air_data = self.selectData(air_data_raw.copy(), season = self.season, final_year=self.final_year)
        single_feature = False
        test_start = dt.date(self.final_year-1, 1, 1)
        test_end = dt.date(self.final_year,1,1)
        final_start = dt.date(self.final_year, 1, 1)
        final_end = dt.date(self.final_year+1, 1, 1)
        X_train, X_test, X_final, y_train, y_test, y_final = self.trainTestSplit(air_data, test_start, test_end, final_start, final_end, single_feature = single_feature)
        return X_train, X_test, X_final, y_train, y_test, y_final

    # season = 'summer'
    # air_data = selectData(air_data_raw.copy(), season = season, final_year=final_year)
    # X_train,X_test, y_train, y_test = trainTestSplit(air_data, test_start, test_end, single_feature = single_feature, shift_period = shift_period)
    # arima_res = predARIMA(X_train, X_test)
    # exog_res = predARIMAExog(X_train, X_test)

    # print("===========Run Result=======")
    # print(arima_res)
    # print(exog_res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ROC-AUC of air pollution prediction.')
    # Required file path
    parser.add_argument('-f','--file', type=str,
                    help='Path to the air pollution data')
    parser.add_argument('-fo','--file_out', type=str,
                    help='Path to the CSV stat data')
    args = parser.parse_args()
    if args.file:
        main(args.file, args.file_out)
    else:
        print("Input file path:-f")

# if __name__ == "__main__":
#     main()


