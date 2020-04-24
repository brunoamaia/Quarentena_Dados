# Aula 4: Estatísticas no ENEM 2018
import pandas as pd

# importações para plotar gráficos (no caso, utilizou para o gráficode correlações)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot

# Formatação geral para apresentar os dados com 2 casas decimais
pd.options.display.float_format = '{:,.2f}'.format

enem = pd.read_csv('B:\Programação\Quarentena_Dados\dados\enem_sample_2018_43278.csv')
#print(enem.head(5))

colunas_de_notas = enem[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']]    # Pegar as colunas de notas
notas = colunas_de_notas.dropna()                                                                       # Remove quem não possui nota em alguma prova
notas.columns = ['cienc_naturais', 'cienc_humanas', 'linguagem_codigo', 'matematica','redacao']
a = len(notas)
notas = notas.drop_duplicates()
print(f'Foram removidas {len(colunas_de_notas)-a} linhas de dados "em branco"')
print(f'Foram removidas {a-len(notas)} linhas de dados duplicados')
corr = notas.corr()
print('Analised e Correlação')
print(corr)

# Gráfico de correlações  (motivo de importar tantas bibliotecas)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plot.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(240, 10, sep=20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.heatmap(corr)
plot.show()

notas1 = notas[['cienc_humanas', 'linguagem_codigo','redacao']]     # Reduzir colunas para rodar mais rápido
# notas_teste = notas1.replace(0,'')
# notas_teste = notas_teste.dropna()
notas_teste = notas1
sns.pairplot(notas_teste)
# sns.pairplot(notas_teste, kind='reg')       #Poltar com linha de regressão linear
# sns.set(style="ticks", color_codes=True)
# sns.pairplot(notas_teste, hue='species')     #plotar "pareamento" junto com uma reta de regressão
plot.show()


## Desafio 1
# Se a pessoa não teve presença, preencher a nota da pessoa com algum valor. Seria a média, mediana, zero?

## Desafio 2
# Corrigir a Matriz de correlação (a que precisou importar várias bibliotecas).

## Desafio 3
# pairplot dos acertos de cada categoria e a redação    (comparar gabarito com as questões marcadas)
# pois a nota das questões é calculada com base em quantas pessoas acertaram cada questão.
# Esse é um teste para ver o impacto causado na nota final por esse modelo

# Desafio 4
## Remover as notas "zero"

## Desafio 5
# Quais questões tiveram mais acertos (gabarito x acertos x erros)

## Desafio 6
#  Estudar oq ue as pessoas que estudam o assunto estão discutindo e conclusões que já chegaram sobre a utilização de informações (principalmente sensensiveis) para machine learning e data science. Podcast do datahackers também sobre o assunto.