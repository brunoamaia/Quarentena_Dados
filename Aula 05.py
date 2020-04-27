# Aula 5 - Regressão e Machine Learning

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import math

# Formatação geral para apresentar os dados com 2 casas decimais
pd.options.display.float_format = '{:,.2f}'.format
sns.set_style("whitegrid")

#   Bibliotecas para machine learning       # procurar SQLearning
from sklearn.metrics import mean_squared_error          # Analise de "qualidade" da Inteligencia Artificial (IA)
from sklearn.dummy import DummyRegressor                # Modelo de regressão que faz a média simples (pior modelo possivel), serve para comparação de eficiencia
from sklearn.model_selection import train_test_split    # Separar a amostra em treino e teste           # SVR - regressão   #SVC - classificação
from sklearn.svm import LinearSVR                       # Uma forma de inteligencia artificial para regressão Lienar
from sklearn.svm import SVR                             # Forma mais robusta e pesada de Inteligencia Artificial
from sklearn.tree import DecisionTreeRegressor          # Arvore de Decisão, forma mais rápida de treinar uma IA


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
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=326784)      # Separa a amostra em elementos de treino e de teste
print(f'Dados para treino (x e y): x = {x_treino.shape} e y = {y_treino.shape}')                                                       # random_state é outra forma de fixar a escolha de termos aleatorios
#print(x_treino.shape)                                                                   # não é muito eficiente, pois um método pode chamar outro que utilizam random, e este não seguira este padrão ...
print(f'Dados para teste (x e y): x = {x_teste.shape} e y = {y_teste.shape}')


# Criar o modelo de inteligencia artificial
print('Criação e treino da inteligencia artificial (IA)')
# modelo = SVR()        # Cria um modelo Não Linear     (é muito "pesado")
print('Modelo - Linear SVR')
modelo_svr = LinearSVR()    #   Máquina de Vetores de suporte (SVM, do inglês: support vector machine)
modelo_svr = modelo_svr.fit(x_treino, y_treino)               # .fit - Realiza o treino (forma de aprender as regras, ou tentar)
predicoes_svr = modelo_svr.predict(x_teste) # .predict - Saida dos valores estimados pela IA
# plot.figure(figsize=(10,10))
# sns.scatterplot(x=y_teste, y=(predicoes_svr - y_teste))  # plotar diferença entre o projetado e o real
# plot.show()
#print(modelo_svr)

print('Modelo - Ávore de Decisão')
modelo_dt = DecisionTreeRegressor()        # Avore de decisão (é bem rápido)
modelo_dt = modelo_dt.fit(x_treino, y_treino)
predicoes_dt = modelo_dt.predict(x_teste)  # Saida dos valores estimados pela IA
# plot.figure(figsize=(10,10))
# sns.scatterplot(x=y_teste, y=(predicoes_dt - y_teste))  # plotar diferença entre o projetado e o real
# plot.show()
#print(modelo_dt)

print('Modelo - Falso (média)')
modelo_falso = DummyRegressor()     # Teste utilizando média (falsa IA)
modelo_falso = modelo_falso.fit(x_treino, y_treino)
predicoes_falsas = modelo_falso.predict(x_teste)
# plot.figure(figsize=(10,10))
# sns.scatterplot(x = y_teste, y = (predicoes_falsas - y_teste))       # Erro da previsão utilizando apenas a média dos dados do "treino"
# plot.show()
#print(modelo_falso)

# Qualidade do teste        # Seria o "erro quadrático"
print('Qualidade dos modelos utilizando o Erro Quadrático Médio.')
qualidade_svr = mean_squared_error(y_teste, predicoes_svr)
qualidade_dt = mean_squared_error(y_teste, predicoes_dt)
qualidade_falsas = mean_squared_error(y_teste, predicoes_falsas)

### Avaliação dos métodos
print('Avaliação de desempenho dos métodos:')
# print(f"Método 1: Pontuação = {avaliacao_metodo:.2f}; Raiz = {math.sqrt(avaliacao_metodo):.2f}")
print(f'SRV: \tPontuação = {qualidade_svr:.2f}, Raiz = {math.sqrt(qualidade_svr):.2f}')
print(f'DT: \tPontuação = {qualidade_dt:.2f}, Raiz = {math.sqrt(qualidade_dt):.2f}')
print(f'Falso: \tPontuação = {qualidade_falsas:.2f}, Raiz = {math.sqrt(qualidade_falsas):.2f}')


### Gráficos Adicionais
# plotar/confrontar os resulados de um eixo com a previsão \o/
# sns.scatterplot(x=x_teste['matematica'].values, y=predicoe_notas_limguagem)     # Previsões     (fundo)
# sns.scatterplot(x=x_teste['matematica'].values, y=y_teste)                      # Valores Reais (parte de cima)
# plot.show()


# Vamos utilizar uma métrica para nos dizer como nosso modelo está indo
# Utilizaremos o Erro Quadrático Médio.
# Existem centenas de métricas de avaliação, tudo vai depender do que você precisa e o que você está prevendo.







## Desafio 1 da Tais Spadini
# Explore os parâmetros C e o max_iter do modelo LinesSVR. Não há garantias que o resultado será melhor.

## Desafio 2 do Thiago Gonçalves
# No gráfico em que plotamos a média com o valor previsto, plote a média das 4 notas ao invés de uma.

## Desafio 3 do Paulo Silveira
# Remover as notas zero e testar os mesmos modelos, comparando o resultado

## Desafio 4 do Guilherme Silveira
# Interpretar tudo que foi feito e compartilhar suas conclusões

## Desafio 5 do Thiago Gonçalves
# Calcule as métricas de erro que utilizamos (mean square root error) também no conjunto de treino, e veja o que acontece comparado com o conjunto de teste.
