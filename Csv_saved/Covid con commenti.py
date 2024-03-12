import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Funz_Utili as fu
import pandas as pd
from pandas import read_csv
#import matplotlib.ticker as ticker
#import pickle
#import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

n_in = 10
frac = 0.8
n_out = 7
n_cl = 7


# prendo il dataframe, estraggo n_in righe e li metto nella prima riga, poi estraggo altri n_in valori shiftati
# di un giorno e li metto nella seconda riga
# mi escono dimensione del dataframe - n_in + 1 righe (dropna + shift), con (n_in + n_out)*variabili colonne
def series_to_supervised(data,ind_data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]     #data.shape[1] numero colonne
    df = pd.DataFrame(data)                                 #crea dataframe
    cols, names = list(), list()                            #definisce due liste vuote
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):                            #n_in giorni indietro che guardo
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.index = ind_data
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg





#bho
#Regions = pickle.load( open( "Regions.pkl", "rb" ))


df = read_csv('Csv_saved/df_r='+str(0)+'.csv', index_col=0) #legge il csv e mette la colonna 0 come indice
cols = list(df)        #matrice dal datafarame
cols = cols[2:3] + cols[0:2] + cols[3:]
#riordino le parti [0:2] vuol dire elementi da 0 a 2
#il nuovo ordinamento è così
#'terapia_intensiva', 'nuovi_positivi', 'ricoverati_con_sintomi',
#'isolamento_domiciliare', 'dimessi_guariti', 'deceduti', 'tamponi',
#'retail_and_recreation_percent_change_from_baseline',
#'grocery_and_pharmacy_percent_change_from_baseline',
#'parks_percent_change_from_baseline',
#'transit_stations_percent_change_from_baseline',
#'workplaces_percent_change_from_baseline',
#'residential_percent_change_from_baseline'
df = df.reindex(columns=cols)
li = [1, 4, 6]
df.drop(df.columns[li], axis=1, inplace=True)
#elimina colonne del dataframe
values = df.values
values = values.astype('float32')
#values è un N-dimensional array in questo caso una matrice
index_d = df.index
#gli indici sono le date

#NORMALIZATION
Max_list = []
#prendo il valore assoluto del massimo di una frazione(frac) di dati in una colonna di values per ogni colonna
#e creo il vettore
M_lis = [abs(values[:int(frac*values.shape[0]), i]).max() for i in range(values.shape[1])]
#sostituisco gli elementi = 0 con 1
for i in range(len(M_lis)):
    if M_lis[i] == 0:
        M_lis[i] = 1
#salvo il massimo delle terapie intensive su max list
Max_list.append(M_lis[0])
#concateno le prime 4 colonne divise per il loro Max e le altre (rielaborate in modo da essere tra 0 e 1, perchè ci son valori negativi)
scaled = np.concatenate((values[:, :4]/M_lis[:4], (values[:, -6:]/M_lis[-6:]+1)/2), axis=1)
scaled_DF = pd.DataFrame(scaled, columns=df.columns)
#definisco un dizionario che associa colonne a valori, e dico di rimuovere il nan in quelle colonne da -1 a -6
val = {scaled_DF.columns[-1]: -1, scaled_DF.columns[-2]: -1, scaled_DF.columns[-3]: -1, scaled_DF.columns[-4]: -1,
       scaled_DF.columns[-5]: -1, scaled_DF.columns[-6]: -1}
scaled_DF = scaled_DF.fillna(value=val)
scaled = scaled_DF.values
reframed = series_to_supervised(scaled,index_d, n_in, n_out)

#droppa le colonne del dataset messo a matrice temporale relative a day out per tutte le variabili tranne la prima
#TIENE SOLO LA PRIMA VARIABILE PER TUTTI I TEMPI DI DAY OUT (ciclo parte da +1)
#INVARIATO DAY IN
n_var = len(df.columns)
lis = []
for j in range(n_out):
#lis prende tutti i valori delle variabili TRANNE LA PRIMA per ogni day out, e poi per quello dopo ecc...
    lis = lis + [i for i in range(n_var * (n_in + j) + 1, n_var * (n_in + j + 1))]
reframed.drop(reframed.columns[lis], axis=1, inplace=True)

for r in range(1,20):
    df1 = read_csv('Csv_saved/df_r='+str(r)+'.csv',index_col=0)
    cols = list(df1)
    cols = cols[2:3] + cols[0:2] + cols[3:]
    df1=df1.reindex(columns=cols)
    df1.drop(df1.columns[li], axis=1, inplace=True)
    #df1.drop(df1.columns[7:], axis=1, inplace=True)
    values = df1.values
    index_d = df1.index
    values = values.astype('float32')

    # NORMALIZATION
    M_lis = [abs(values[:int(frac * values.shape[0]), i]).max() for i in range(values.shape[1])]
    for i in range(len(M_lis)):
        if M_lis[i] == 0:
           M_lis[i] = 1
    Max_list.append(M_lis[0])
    scaled=np.concatenate((values[:,:4]/M_lis[:4],(values[:,-6:]/M_lis[-6:]+1)/2), axis=1)
    scaled_DF = pd.DataFrame(scaled, columns=df.columns)
    val = {scaled_DF.columns[-1]: -1, scaled_DF.columns[-2]: -1, scaled_DF.columns[-3]: -1, scaled_DF.columns[-4]: -1,
           scaled_DF.columns[-5]: -1, scaled_DF.columns[-6]: -1}
    scaled_DF = scaled_DF.fillna(value=val)
    scaled = scaled_DF.values
    reframed1 = series_to_supervised(scaled, index_d, n_in, n_out)
    reframed1.drop(reframed1.columns[lis], axis=1, inplace=True)
    reframed = pd.concat([reframed, reframed1])

#lista di tutte le colonne var t => 0
Next_var=reframed.columns[n_var*n_in:].to_list()

if n_out%n_cl != 0:
    print('ATTENTION: n_out is not a multiple of n_cl!')
stringa = 'target' + str(n_cl)
#sum(axis=1) somma la riga
#fa la stessa cosa, ultima colonna
reframed[stringa]= reframed.iloc[:,n_var*n_in+(n_cl-1)*int(n_out/n_cl):].sum(axis=1)/(n_out-int(n_out/n_cl)*(n_cl-1))

#per ogni i sommo le variabili su nout/ncl giornie li divido per il numero di giorni e gli assegno un valore di target,
#poi proseguo al gruppo di giorni successivo e medio
#se ncl = ncout , considero sempre un giorno alla volta (non si fanno medie)
for i in range(n_cl-1):
    stringa = 'target' + str(i+1)
    reframed[stringa]= reframed.iloc[:,n_var*n_in+i*int(n_out/n_cl):n_var*n_in+(i+1)*int(n_out/n_cl)].sum(axis=1)/int(n_out/n_cl)

if n_cl != 1:
    cols = reframed.columns.tolist()
    cols = cols[:-n_cl] + cols[-(n_cl-1):] + cols[-n_cl:-(n_cl-1)]
    reframed = reframed[cols]


reframed.drop(Next_var, axis=1, inplace=True)

#i miei dati sono matrici di giorni righe e variabili colonne, per ogni regione messe una sopra l'altra

#quanti giorni nel dataframe senza quelli che prendiamo n-in e n-out
len_series = len(df)-n_in-n_out+1
#crea una maschera di lunghezza della seria con frac % degli elementi True e il resto false
msk = np.arange(0, len_series) < frac * len_series
msk = list(msk)
#ne mette 20 uguali a quella creata prima in fila
msk = msk * 20
msk = np.array(msk)

#definisce i train , prima 0.8 porzione dei dati e test 0.2 porz dati
train = reframed[msk].values
test = reframed[~msk].values

# split into input and outputs
train_X, train_y = train[:, :n_var*n_in], train[:, n_var*n_in:]
test_X, test_y = test[:, :n_var*n_in], test[:, n_var*n_in:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1],))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1],))

#return_sequence --> Boolean. Whether to return the last output. in the output sequence, or the full sequence. Default: False.
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(train_X.shape[1], train_X.shape[2])))
# units in layers sta per la dimensione dell'hidden state vector nella cella LSTM (quanti "neuroni")
#c'è una cella LSTM ripetuta uguale nel tempo un numero di volte timestep
#mettiamo 2 celle LSTM "una sopra l'altra"
#input shape vuole sapere i time steps e le features che passo
model.add(tf.keras.layers.LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
#dense(n_cl) mi dice che la dimensione dell output è n_cl (numero cicli)
#dense è il layer che ho messo per ultimo che mi da l'output, dense perchè i neuroni sono FULLY CONNECTED con matrice pesi e bias
#i neuroni del dense sono quelli che guardo per il risultato (credo)
model.add(tf.keras.layers.Dense(n_cl, activation='sigmoid'))
#compile definisce la loss function, la metrica ecc ecc, non allena la rete ma la crea e basta
# mse è mean squared error (differenza al quadrato) mediata sulle variabili
#Optimizer è l'algoritmo che mi dice come ottimizzare e gli do un certo passo il learning rate
model.compile(loss='mse', optimizer=Adam(lr=0.00001))
#.fit train il modello
#shuffle false perchè i batch sono correlati e ordinati! non vogliamo mischiarli
#verbose mi dice come voglio vedere i training data
#batch size mi dice quanti batch guarda prima di aggiornare i parametri
#epochs mi dice quante volte ripasso tutto il dataset
history = model.fit(train_X, train_y, epochs=100, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)
