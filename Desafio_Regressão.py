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

# Formatações Gerais
pd.options.display.float_format = '{:,.2f}'.format  # 2 casas decimais
sns.set_style("whitegrid")                          # Gráficos com Grid
np.random.seed(43267)       # Fixa a geração dos termos aleatórios

# Verificar o tempo gasto
import time
start = time.time()


# https://github.com/tgcsantos/quaretenadados/blob/master/DADOS_TREINO.csv?raw=true
# https://raw.githubusercontent.com/tgcsantos/quaretenadados/master/DADOS_TREINO.csv
dados_treino = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DADOS_TREINO.csv')
dados_teste = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DADOS_TESTE.csv')
dados_desafio = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DESAFIOQT.csv')
print(f'Dados do Treino: {dados_treino.shape}')
print(f'Dados do Teste: {dados_teste.shape}')
print(f'Dados do Desafio: {dados_desafio.shape}')



coluna_label = 'NU_NOTA_LC'
coluna_features = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']

x_treino = dados_treino[coluna_features].to_numpy()
y_treino = dados_treino[coluna_label].to_numpy()
x_teste = dados_teste[coluna_features].to_numpy()
y_teste = dados_teste[coluna_label].to_numpy()



################################            Treino da inteligencia              ################################
print('\nModelo - Linear SVR')              # Linear SVR
modelo_svrl = LinearSVR(max_iter=1000)
modelo_svrl = modelo_svrl.fit(x_treino, y_treino)
predicoes_svrl = modelo_svrl.predict(x_teste)
qualidade_svrl0 = mean_squared_error(y_teste, predicoes_svrl)
del modelo_svrl, predicoes_svrl
print(f'Tempo gasto: {time.process_time()- a} s')

print('\nModelo - SVR')                     # SVR
a = time.process_time()
modelo_svr = SVR()
modelo_svr = modelo_svr.fit(x_treino, y_treino)
predicoes_svr = modelo_svr.predict(x_teste)
qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
del modelo_svr, predicoes_svr
print(f'Tempo gasto: {time.process_time()- a} s')

print('Modelo - Ávore de Decisão')          # Ávore de Decisão
a = time.process_time()
modelo_dt = DecisionTreeRegressor()
modelo_dt = modelo_dt.fit(x_treino, y_treino)
predicoes_dt = modelo_dt.predict(x_teste)
qualidade_dt0 = mean_squared_error(y_teste, predicoes_dt)
del modelo_dt, predicoes_dt
print(f'Tempo gasto: {time.process_time()- a} s')

print('Modelo - Falso (média)')             # Média
a = time.process_time()
modelo_falso = DummyRegressor()
modelo_falso = modelo_falso.fit(x_treino, y_treino)
predicoes_falsas = modelo_falso.predict(x_teste)
qualidade_media0 = mean_squared_error(y_teste, predicoes_falsas)
del modelo_falso, predicoes_falsas
print(f'Tempo gasto: {time.process_time()- a} s')

print('Modelo - Falso (mediana)')           # Mediana
a = time.process_time()
modelo_falso = DummyRegressor(strategy="median")
modelo_falso = modelo_falso.fit(x_treino, y_treino)
predicoes_falsas = modelo_falso.predict(x_teste)
qualidade_mediana0 = mean_squared_error(y_teste, predicoes_falsas)
del modelo_falso, predicoes_falsas
print(f'Tempo gasto: {time.process_time()- a} s')







# ##################################       Remover as notas abaxido de 100      ######################################
# del x, y, x_treino, y_treino, x_teste, y_teste
# f = 100 # filto do valor mínimo
# notas_uteis = notas[ (notas.linguagem_codigo > f) &         # Fazer em Apenas um comando
#                      (notas.cienc_humanas > f) &
#                      (notas.cienc_naturais > f) &
#                      (notas.matematica > f) &
#                      (notas.redacao > f)      ]
#
# x = notas_uteis[['cienc_naturais', 'cienc_humanas', 'matematica', 'redacao']]
# y = notas_uteis['linguagem_codigo']
# x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=326784)      # Separa a amostra em elementos de treino e de teste
#
# print('Dataframe sem as notas abaixo de 100')
# print(f'Dados para treino (x e y): x = {x_treino.shape} e y = {y_treino.shape}')
# print(f'Dados para teste (x e y): x = {x_teste.shape} e y = {y_teste.shape}')
#
#
#
#
#
# print('\nModelo - Linear SVR')              # Linear SVR
# modelo_svrl = LinearSVR(max_iter=1000)
# modelo_svrl = modelo_svrl.fit(x_treino, y_treino)
# predicoes_svrl = modelo_svrl.predict(x_teste)
# qualidade_svrl0 = mean_squared_error(y_teste, predicoes_svrl)
# del modelo_svrl, predicoes_svrl
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('\nModelo - SVR')                     # SVR
# a = time.process_time()
# modelo_svr = SVR()
# modelo_svr = modelo_svr.fit(x_treino, y_treino)
# predicoes_svr = modelo_svr.predict(x_teste)
# qualidade_svr0 = mean_squared_error(y_teste, predicoes_svr)
# del modelo_svr, predicoes_svr
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('Modelo - Ávore de Decisão')          # Ávore de Decisão
# a = time.process_time()
# modelo_dt = DecisionTreeRegressor()
# modelo_dt = modelo_dt.fit(x_treino, y_treino)
# predicoes_dt = modelo_dt.predict(x_teste)
# qualidade_dt0 = mean_squared_error(y_teste, predicoes_dt)
# del modelo_dt, predicoes_dt
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('Modelo - Falso (média)')             # Média
# a = time.process_time()
# modelo_falso = DummyRegressor()
# modelo_falso = modelo_falso.fit(x_treino, y_treino)
# predicoes_falsas = modelo_falso.predict(x_teste)
# qualidade_media0 = mean_squared_error(y_teste, predicoes_falsas)
# del modelo_falso, predicoes_falsas
# print(f'Tempo gasto: {time.process_time()- a} s')
#
# print('Modelo - Falso (mediana)')           # Mediana
# a = time.process_time()
# modelo_falso = DummyRegressor(strategy="median")
# modelo_falso = modelo_falso.fit(x_treino, y_treino)
# predicoes_falsas = modelo_falso.predict(x_teste)
# qualidade_mediana0 = mean_squared_error(y_teste, predicoes_falsas)
# del modelo_falso, predicoes_falsas
# print(f'Tempo gasto: {time.process_time()- a} s')
#
#
#
#
#
#
# Qualidade do teste        # Seria o "erro quadrático"
print('Avaliação de desempenho dos métodos:')

print(f'lin. SRV0: \tPontuação = {qualidade_svrl0:.2f}, Raiz = {math.sqrt(qualidade_svrl0):.2f}')
print(f'SRV0: \t\tPontuação = {qualidade_svr0:.2f}, Raiz = {math.sqrt(qualidade_svr0):.2f}')
print(f'DT0: \t\tPontuação = {qualidade_dt0:.2f}, Raiz = {math.sqrt(qualidade_dt0):.2f}')
print(f'Media0: \tPontuação = {qualidade_media0:.2f}, Raiz = {math.sqrt(qualidade_media0):.2f}')
print(f'Mediana0: \tPontuação = {qualidade_mediana0:.2f}, Raiz = {math.sqrt(qualidade_mediana0):.2f}')




# print(f'SRV1: \t\tPontuação = {qualidade_svr1:.2f}, Raiz = {math.sqrt(qualidade_svr1):.2f}')
#
# print(f'lin. SRV1: \tPontuação = {qualidade_svrl1:.2f}, Raiz = {math.sqrt(qualidade_svrl1):.2f}')
#
# print(f'DT1: \t\tPontuação = {qualidade_dt1:.2f}, Raiz = {math.sqrt(qualidade_dt1):.2f}')
#
# print(f'Falso1: \tPontuação = {qualidade_falso1:.2f}, Raiz = {math.sqrt(qualidade_falso1):.2f}')
#
# print(f'Media0: \tPontuação = {qualidade_media0:.2f}, Raiz = {math.sqrt(qualidade_media0):.2f}')
#
# print(f'Mediana0: \tPontuação = {qualidade_media0:.2f}, Raiz = {math.sqrt(qualidade_media0):.2f}')