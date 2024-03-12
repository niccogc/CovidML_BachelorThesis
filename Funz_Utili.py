
from pandas import read_csv
import pandas as pd
import numpy as np


# This function converts a dataframe into a time series matrix by extracting specified 
#numbers of days before and after each timestamp.
# It outputs a matrix where each row corresponds to a shifted version of the original data, 
#and each column represents a variable at a specific time.
def Data_to_time_series_Matrix(data, days_before, days_after, indexes, removenan):
    nv = 1 if type(data) is list else data.shape[1]
    dt = pd.DataFrame(data)
    mdl, names = list(), list()
    for i in range(days_before, 0, -1):
        mdl.append(dt.shift(i))
        names += ('var%d (t - %d)' % (j + 1, i) for j in range(nv))
    for i in range(0, days_after):
        mdl.append(dt.shift(-i))
        if i == 0:
            names += ('var%d (t)' % (j+1) for j in range(nv))
        else:
            names += ('var%d (t +%d)' % (j+1,i) for j in range(nv))
    agg = pd.concat(mdl, axis = 1)
    agg.columns = names
    agg.index = indexes
    if removenan:
        agg.dropna(inplace=True)
    return agg

# This function creates a DataFrame 
#by reordering the columns of the input DataFrame and dropping specified columns.

def Create_DataFrame(Delete_column, Num_Regione, Column_index_1, Column_index_2):
    DatFrame = read_csv('Csv_saved/df_r='+str(Num_Regione)+'.csv', index_col=0)
    cols = list(DatFrame)
    cols = cols[Column_index_1:Column_index_2] + cols[0:Column_index_1] + cols[Column_index_2:]
    DatFrame = DatFrame.reindex(columns=cols)
    DatFrame.drop(DatFrame.columns[Delete_column], axis=1, inplace=True)
    return DatFrame


# This function normalizes a DataFrame by scaling its values within a specified fraction of the dataset.

def Normalization_DataFrame(DataFrame, Fraction):
    values = DataFrame.values.astype('float32')
    Max_list = np.array([abs(values[:int(Fraction * values.shape[0]), i]).max() for i in range(values.shape[1])])
    Max_list[Max_list == 0] = 1
    scaled = np.concatenate((values[:, :4]/Max_list[:4], (values[:, -6:]/Max_list[-6:]+1)/2), axis=1)
    df1 = pd.DataFrame(scaled, columns=DataFrame.columns)
    val = {df1.columns[-1]: -1, df1.columns[-2]: -1, df1.columns[-3]: -1, df1.columns[-4]: -1,
           df1.columns[-5]: -1, df1.columns[-6]: -1}
    df1.fillna(value=val, inplace=True)
    values = df1.values
    return values

# This function removes specified variables from a DataFrame based on the number of variables to keep, 
#days before, and prediction days.

def Var_Keep(DataFrame, Var_Keep, D_Before, D_Prediction):
    n_var = int(len(DataFrame.columns) / (D_Before + D_Prediction))
    lis = []
    for j in range(D_Prediction):
        lis = lis + [i for i in range(n_var * (D_Before + j) + Var_Keep, n_var * (D_Before + j + 1))]
    DataFrame.drop(DataFrame.columns[lis], axis=1, inplace=True)
    return DataFrame

# This function implements target variables by summing specified variables 
#over certain days and averaging them to create target values.

def Target_Implementation(DataFrame, Num_variables, D_Before, D_Predictions, D_To_Mediate):
    if D_Predictions%D_To_Mediate != 0:
        print('ATTENTION: Days out are not a multiple of (medium days)')
    Var = DataFrame.columns[Num_variables * D_Before:].tolist()
    for j in range(int(D_Predictions / D_To_Mediate) - 1):
        stringa = ['target' + str(j+1)]
        DataFrame[stringa] = DataFrame.iloc[:, Num_variables * D_Before + j * D_To_Mediate: Num_variables * D_Before + (j + 1) * D_To_Mediate].sum(axis=1) / D_To_Mediate
    stringa = ['Target' + str(int(D_Predictions / D_To_Mediate))]
    DataFrame[stringa] = DataFrame.iloc[:, Num_variables * D_Before + (D_Predictions - D_To_Mediate):].sum(axis=1) / D_To_Mediate
    DataFrame.drop(Var, axis=1, inplace=True)
    return DataFrame

# This function splits region data into training and testing sets based on a specified fraction.
# It further divides the data into days before and prediction days, reshaping them into 3D arrays.

def Create_Train_XY_and_Test_XY(DataFrame, D_Before, D_Prediction, N_Regioni, Train_Fraction, N_Var):

    leng = len(pd.read_csv('Csv_saved/df_r=0.csv')) - D_Before - D_Prediction + 1
    msk = np.arange(0, leng) < Train_Fraction * leng
    msk = list(msk)
    msk = msk * N_Regioni
    msk = np.array(msk)

    Train = DataFrame[msk].values
    Test = DataFrame[~msk].values

    print(Train.max())
    print(Test.max())

    Train_X, Train_Y = Train[:, :D_Before * N_Var], Train[:, D_Before * N_Var:]
    Test_X, Test_Y = Test[:, :D_Before * N_Var], Test[:, D_Before * N_Var:]

    Train_X = Train_X.reshape((Train_X.shape[0], 1, Train_X.shape[1]))
    Train_Y = Train_Y.reshape((Train_Y.shape[0], 1, Train_Y.shape[1]))
    Test_X = Test_X.reshape((Test_X.shape[0], 1, Test_X.shape[1]))
    Test_Y = Test_Y.reshape((Test_Y.shape[0], 1, Test_Y.shape[1]))

    return Train_X, Train_Y, Test_X, Test_Y

# This function creates target variables by summing specified variables over a certain number of days and averaging them.

def Target (DataFrame, N_Var, D_Before, N_Prediction, N_cicl):
    Next_var = DataFrame.columns[N_Var * D_Before:].to_list()

    if N_Prediction % N_cicl != 0:
        print('ATTENTION: n_out is not a multiple of n_cl!')
    stringa = 'target' + str(N_cicl)
    DataFrame[stringa] = DataFrame.iloc[:, N_Var * D_Before + (N_cicl - 1) * int(N_Prediction / N_cicl):].sum(axis=1) / (
            N_Prediction - int(N_Prediction / N_cicl) * (N_cicl - 1))
    for i in range(N_cicl - 1):
        stringa = 'target' + str(i + 1)
        DataFrame[stringa] = DataFrame.iloc[:,
                             N_Var * D_Before + i * int(N_Prediction / N_cicl):N_Var * D_Before + (i + 1) * int(N_Prediction / N_cicl)].sum(
            axis=1) / int(N_Prediction / N_cicl)
    if N_cicl != 1:
        cols = DataFrame.columns.tolist()
        cols = cols[:-N_cicl] + cols[-(N_cicl - 1):] + cols[-N_cicl:-(N_cicl - 1)]
        DataFrame = DataFrame[cols]
    DataFrame.drop(Next_var, axis=1, inplace=True)
    return DataFrame
