# Aula 5 - Regressão e Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split

# Forma de tentar minimizar a aleatoriedade no processo
import numpy as np
np.random.seed(43267)       # Fixa o inicio da geração dos termos aleatórios

enem = pd.read_csv('B:\Programação\Quarentena_Dados\dados\enem_sample_2018_43278.csv')

colunas_de_notas = enem[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']]    # Pegar as colunas de notas
notas = colunas_de_notas.dropna()                                                                       # Remover quem não possui nota em alguma prova
notas.columns = ['cienc_naturais', 'cienc_humanas', 'linguagem_codigo', 'matematica','redacao']         # Renomear as colunas

    # Tentar adivinhar a nota de Linguagem_codigo a partir das outras notas
    # Funciona de forma similar a uma regressão (x, Y)
x = notas[['cienc_naturais', 'cienc_humanas', 'matematica', 'redacao']]
y = notas['linguagem_codigo']

#  Treino da inteligencia
# Ele seleciona alguns elementos para "ensinar" e outros para "testar" a qualidade do teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=326784)      # random_state é outra forma de fixar a escolha de termos aleatorios
print('Grupos de treino (x e y)')                                                       # não é muito eficiente, pois um método pode chamar outro que utilizam random, e este não seguira este padrão ...
print(x_treino.shape)
print(y_treino.shape)
print('Grupos de teste (x e y)')
print(x_teste.shape)
print(y_teste.shape)

