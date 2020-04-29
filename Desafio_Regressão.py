# Desafio da Quarentena Dados - Valendo um

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import math
import numpy as np

#   Bibliotecas para machine learning       # procurar SQLearning
from sklearn.metrics import mean_squared_error          # Analise de "qualidade" da Inteligencia Artificial (IA)
from sklearn.model_selection import train_test_split    # Separar a amostra em treino e teste
from sklearn.dummy import DummyRegressor                # Modelo que retorna a média/mediana
from sklearn.svm import LinearSVR                       # Uma forma de inteligencia artificial para regressão Lienar
from sklearn.svm import SVR                             # Forma mais robusta e pesada das bilbiotecas usadas (não é necessariamente a melhor)
from sklearn.tree import DecisionTreeRegressor          # Arvore de Decisão, forma mais rápida de treinar uma IA

from sklearn.svm import NuSVR                           # Novo método pra testar

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor



from sklearn.neural_network import MLPRegressor         # Multi-layer Perceptron regressor

# Formatações Gerais
pd.options.display.float_format = '{:,.2f}'.format  # 2 casas decimais
sns.set_style("whitegrid")                          # Gráficos com Grid
np.random.seed(43267)       # Fixa a geração dos termos aleatórios

# Verificar o tempo gasto
import time
start = time.time()

# Links para os arquivos
# Dados para treino: https://raw.githubusercontent.com/tgcsantos/quaretenadados/master/DADOS_TREINO.csv
# Dados para Tesre:  https://raw.githubusercontent.com/tgcsantos/quaretenadados/master/DADOS_TESTE.csv
dados_treino = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DADOS_TREINO.csv')
dados_teste = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DADOS_TESTE.csv')
dados_desafio = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DESAFIOQT.csv')
print(f'Dados do Treino: \t{len(dados_treino)}')
print(f'Dados do Teste: \t{len(dados_teste)}')
print(f'Dados do Desafio: \t{len(dados_desafio)}')


# sns.pairplot(dados_treino)
# plot.show()

coluna_label = 'NU_NOTA_LC'
coluna_features = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
x_treino = dados_treino[coluna_features].to_numpy()
y_treino = dados_treino[coluna_label].to_numpy()
x_teste = dados_teste[coluna_features].to_numpy()
y_teste = dados_teste[coluna_label].to_numpy()



################################            Treino da inteligencia              ################################
resultados  = {}    # Criar um Dicionário para armazenar a qualidade dos testes


print('\nModelo - RandomForestRegressor')                     # RandomForestRegressor
a = time.process_time()
modelo_svr = RandomForestRegressor()
modelo_svr = modelo_svr.fit(x_treino, y_treino)
predicoes_svr = modelo_svr.predict(x_teste)
qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
del modelo_svr, predicoes_svr
resultados['RandomForestRegressor \t\t']=qualidade_svr0
print(f'Tempo gasto: {time.process_time()- a} s')

# print('\nModelo - KNeighborsRegressor')                     # KNeighborsRegressor
# a = time.process_time()
# modelo_svr = KNeighborsRegressor()
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['KNeighborsRegressor \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Lasso')                     # Lasso
# a = time.process_time()
# modelo_svr = Lasso(alpha=2.0)
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['Lasso \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - BayesianRidge')                     # BayesianRidge
# a = time.process_time()
# modelo_svr = BayesianRidge(tol=0.0001)
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['BayesianRidge: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - MLPRegressor')                     # MLPRegressor
# a = time.process_time()
# modelo_svr = MLPRegressor()
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['MLPRegressor: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - NuSVR')                     # NuSVR
# a = time.process_time()
# modelo_svr = NuSVR()
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['NuSVR: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
#
# print('\nModelo - SVR')                     # SVR - Sigmoid
# a = time.process_time()
# modelo_svr = SVR(kernel='sigmoid')
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['Sigmoid: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - SVR')                     # SVR - precomputed
# a = time.process_time()
# modelo_svr = SVR(kernel='precomputed')
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['Precomputed: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')

# print('\nModelo - SVR')                     # SVR - precomputed - poly
# a = time.process_time()
# modelo_svr = SVR(kernel='poly')
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['Poly: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
















# print('\nModelo - Linear SVR')              # Linear SVR
# a = time.process_time()
# modelo_svrl = LinearSVR(max_iter=1000)
# modelo_svrl = modelo_svrl.fit(x_treino, y_treino)
# predicoes_svrl = modelo_svrl.predict(x_teste)
# qualidade_svrl0 = mean_squared_error(y_teste, predicoes_svrl)
# del modelo_svrl, predicoes_svrl
# resultados['lin. SVR0: \t']=qualidade_svrl0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - SVR')                     # SVR
# a = time.process_time()
# modelo_svr = SVR()
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['SVR0: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Ávore de Decisão')          # Ávore de Decisão
# a = time.process_time()
# modelo_dt = DecisionTreeRegressor()
# modelo_dt = modelo_dt.fit(x_treino, y_treino)
# predicoes_dt = modelo_dt.predict(x_teste)
# qualidade_dt0 = mean_squared_error(y_teste, predicoes_dt)
# del modelo_dt, predicoes_dt
# resultados['DT0: \t\t']=qualidade_dt0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Falso (média)')             # Média
# a = time.process_time()
# modelo_falso = DummyRegressor()
# modelo_falso = modelo_falso.fit(x_treino, y_treino)
# predicoes_falsas = modelo_falso.predict(x_teste)
# qualidade_media0 = mean_squared_error(y_teste, predicoes_falsas)
# del modelo_falso, predicoes_falsas
# resultados['Media0: \t']=qualidade_media0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Falso (mediana)')           # Mediana
# a = time.process_time()
# modelo_falso = DummyRegressor(strategy="median")
# modelo_falso = modelo_falso.fit(x_treino, y_treino)
# predicoes_falsas = modelo_falso.predict(x_teste)
# qualidade_mediana0 = mean_squared_error(y_teste, predicoes_falsas)
# del modelo_falso, predicoes_falsas
# resultados['Mediana0: \t']=qualidade_mediana0
# print(f'Tempo gasto: {time.process_time()- a} s')





# ##################################       Remover as notas abaxido de 100      ######################################
# del x_treino, y_treino
# f = 100 # filto do valor mínimo
# notas = dados_treino[ (dados_treino.NU_NOTA_LC > f) &         # Fazer em Apenas um comando
#                             (dados_treino.NU_NOTA_CH > f) &
#                             (dados_treino.NU_NOTA_CN > f) &
#                             (dados_treino.NU_NOTA_MT > f) &
#                             (dados_treino.NU_NOTA_REDACAO > f)]
#
# x_treino = notas[coluna_features].to_numpy()
# y_treino = notas[coluna_label].to_numpy()
#
# # sns.pairplot(notas)
# # plot.show()
#
# print('\nDados após remover notas abaixo de 100 (supondo que são Outliers')
# print(f'foram removidas {len(dados_treino) - len(notas)} linhas com notas abaixo de 100')
# print(f'Dados do Treino = {len(x_treino)} e {len(y_treino)}')
# print(f'Dados do Teste = {len(x_teste)} e {len(y_teste)}')


# print('\nModelo - Linear SVR')              # Linear SVR
# a = time.process_time()
# modelo_svrl = LinearSVR(max_iter=1000)
# modelo_svrl = modelo_svrl.fit(x_treino, y_treino)
# predicoes_svrl = modelo_svrl.predict(x_teste)
# qualidade_svrl0 = mean_squared_error(y_teste, predicoes_svrl)
# del modelo_svrl, predicoes_svrl
# resultados['lin. SVR1: \t']=qualidade_svrl0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - SVR')                     # SVR
# a = time.process_time()
# modelo_svr = SVR()
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# resultados['SVR1: \t\t']=qualidade_svr0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Ávore de Decisão')          # Ávore de Decisão
# a = time.process_time()
# modelo_dt = DecisionTreeRegressor()
# modelo_dt = modelo_dt.fit(x_treino, y_treino)
# predicoes_dt = modelo_dt.predict(x_teste)
# qualidade_dt0 = mean_squared_error(y_teste, predicoes_dt)
# del modelo_dt, predicoes_dt
# resultados['DT1: \t\t']=qualidade_dt0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Falso (média)')             # Média
# a = time.process_time()
# modelo_falso = DummyRegressor()
# modelo_falso = modelo_falso.fit(x_treino, y_treino)
# predicoes_falsas = modelo_falso.predict(x_teste)
# qualidade_media0 = mean_squared_error(y_teste, predicoes_falsas)
# del modelo_falso, predicoes_falsas
# resultados['Media1: \t']=qualidade_media0
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - Falso (mediana)')           # Mediana
# a = time.process_time()
# modelo_falso = DummyRegressor(strategy="median")
# modelo_falso = modelo_falso.fit(x_treino, y_treino)
# predicoes_falsas = modelo_falso.predict(x_teste)
# qualidade_mediana0 = mean_squared_error(y_teste, predicoes_falsas)
# del modelo_falso, predicoes_falsas
# resultados['Mediana1: \t']=qualidade_mediana0
# print(f'Tempo gasto: {time.process_time()- a} s')


# Nova forma de imprimir os resultados
print('\nAvaliação de desempenho (usando todos os dados):')
for k, v in resultados.items():
    print(f'{k} \tPontuação = {v:.2f}, Raiz = {math.sqrt(v):.2f}')