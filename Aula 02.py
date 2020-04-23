import matplotlib.pyplot as plt
import pandas as pd     # Imortar a biblioteca "pandas"
import seaborn as sns   # biblioteca para trabalhar com os gráficos


### Importar o nome dos filmes ###
filmes = pd.read_csv("B:\Programação\Quarentena_Dados\dados\movies.csv")    # Existem outros read: read_pdf, read_xls, ...
filmes.columns = ['filmeId', 'titulo', 'genero']

### Importar as avaliações dos filmes ###
avaliacoes = pd.read_csv('B:\Programação\Quarentena_Dados\dados\Ratings.csv')
avaliacoes.columns = ['usuarioId', 'filmeId', 'nota', 'momento']

generos_de_filmes = filmes['genero'].str.get_dummies('|').sum().sort_values(ascending=False)   # Tabela com os generos e quantas vezes aparece cada genero. Em ordem Decrescente
n = len(generos_de_filmes)
# print(generos_de_filmes)

###         SEABORN
# *** O matplotlib é quem gerencia os "plot". O seaborn tem parametros predefinidos e torna mais simples algumas operações.
#       Portanto, caso tenha dominio do Matplotlib, talvez não seja necessário utilizar o seaborn
# procurar as opções de customização:
#   palettes,   (cores)
#   aesthetics,       (estilo do gráfico, grid, ...

    # Ajuste das configurações do gráfico
sns.set_style("whitegrid")          # Colocar grid em todos os gráficos
plt.figure(figsize=(18,8))          # Tamanho do gráfico
sns.barplot(x = generos_de_filmes.index,        # ".index" Pega os indices
            y = generos_de_filmes.values,       # ".values" Pega os valores
            palette = sns.color_palette('BuGn_r', n_colors=n+5 ) )     # Palete de cores (Degrade de verde) , e a quantidade de variações para o degrade (+x para não terminar em branco)
plt.show()

# Hitograma das Notas do filme
    # Filme 1
id = 1
notas_filme_1 = avaliacoes.query(f'filmeId == {id}') ['nota']
nome = filmes.query(f'filmeId == {id}')['titulo'].to_string()
print(f'Média do filme {nome}: {notas_filme_1.mean():.2f}')   # 2 Duas casas Decimais
print(notas_filme_1.describe())
notas_filme_1.plot(kind = 'hist', title=f'{nome}')          # Gráfico Hisotgrama
plt.show()


id = 2
notas_filme_2 = avaliacoes.query(f'filmeId == {id}') ['nota']
nome = filmes.query(f'filmeId == {id}')['titulo'].to_string()
print(f'\n\nMédia do filme {nome}: {notas_filme_2.mean():.2f}')   # 2 Duas casas Decimais
print(notas_filme_2.describe())
notas_filme_2.plot(kind = 'hist', title=f'{nome}')
plt.show()
notas_filme_2.plot.box()                        # Boxplot
plt.show()

sns.boxplot(data=avaliacoes.query('filmeId in [1, 2, 919, 46578]'), x='filmeId', y='nota')   # plotar vários boxplot juntos para comparar
plt.show()
### Desafios

###         Desafio 1           ###
# Rotacionar os thicks no gráfico (nome dos generos)

###         Desafio 2           ###
#  Compar filmes com notas parecidas e achar distribuições bem diferentes

###         Desafio 3           ###
#  Fazer o boxplot dos 10 filmes com mais votos

###         Desafio 4           ###
#  No boxplot de vários filmes (linha 54), deixar com tamnho adequado e com o nome dos filmes (thicks)

###         Desafio 5           ###
#  Calcular média moda e mediana dos filmes. Encontrar filmes com a moda próxima de 1, 3 e 5

###         Desafio 6           ###
#  Plotar o boxplot e o histograma (um do lado do outro) de um filme

###         Desafio 7           ###
#  Gráfico de notas médias por ano
