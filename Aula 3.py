import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns   # biblioteca para trabalhar com os gráficos

sns.set_style("whitegrid")

imdb = pd.read_csv('B:\Programação\Quarentena_Dados\dados\movie_metadata.csv')
#print(imdb.head())
a = 1

print('\n\nListar termos unicos e quantidade que apareceram (Filme colorido ou preto e branco):')
print(imdb['color'].value_counts())                 # Contar quantas vezes aparece cada valor unico
print('\n\nNormalizar o valor dos  termos unicos (Filme colorido ou preto e branco): ')
print(imdb['color'].value_counts(normalize=True))   # Normalizar os valores
# print(imdb['director_name'].value_counts().tail(20))    # Lista os Diretores e quantos filmes fizeram e depois Pega as ultimas linhas
a = 1

# Pegar a coluna das cores      #### REfazer esta parte
color_or_bw = imdb.query("color in ['Color', ' Black and White']")  # Criar nova tabela para estudar a importância/relação de filmes coloridos/PeB (remove os demais)
color_or_bw = color_or_bw.dropna().query('budget > 0  |  gross > 0')     # Remove as linhas sem dados (dropna()) e as linhas com dados
color_or_bw['color_0_ou_1'] = (color_or_bw['color']=='Color')*1   # Tentativa de Criar uma coluna que torna a variável Preto/Branco em binária
sns.scatterplot(data=color_or_bw, x="color_0_ou_1", y="gross")
plot.show()
print('\n\nDataframe organizado por filmes coloridos Ou preto e branco')
print(color_or_bw["color_0_ou_1"].value_counts())
# color_or_bw['color_0_ou_1'] = color_or_bw['color'] == 'Color'     # Outra forma de fazer, mas não etá aceitando
# df["b"] = df["value"] == 3                                    # Exemplo
# print(f'\n\n Tamanho da variável: {len(color_or_bw)}')


### Verificar gasto e ganho apenas com os filmes dos USA, devido a imprecissão de conversão monetária
imdb2 = imdb.drop_duplicates()
print(f'\nHaviam {len(imdb)} colunas e após remover os duplicado, passamos a ter {len(imdb2)} colunas.\nOu seja, haviam {len(imdb)-len(imdb2)} filmes duplicados')
imdb_usa = imdb2.query('country == "USA"')
budget_gross = imdb_usa[['budget', 'gross']]    # Criar um novo DataFrame apenas com budget e gross, nesse caso, apenas de filmes dos eua
budget_gross = budget_gross.dropna().query('budget > 0  |  gross > 0')    # Remover linhas sem dados, e as que possuem valor = 0 (pois é uma informação que não faz sentido, e provavelmente é um outlouer
#sns.scatterplot(x='budget', y='gross', data=budget_gross)
#plot.show()

## # Verificar lucro/prejuizo
imdb_usa = imdb_usa.dropna().query('budget > 0  |  gross > 0')  # Remover dados que não são relevantes
imdb_usa['lucro'] = imdb_usa['gross'] - imdb_usa['budget']
gasto_lucro = imdb_usa[['budget', 'lucro']]
#sns.scatterplot(x='budget', y='lucro', data=gasto_lucro)
#plot.show()
print(gasto_lucro.sort_values('lucro'))
print('filme com o maior prejuizo')
print(imdb2.query('budget == 263700000'))


### Perguntas gerais
# - Aventura tem nota melhor que comédia?
# - Diretor com mais filmes te nota melhor?
# - As repostas são sespcíficas para a amortar ou para o mundo?
# - Quais correlaçoes (simples) exsitem entre os dados?
#       * budget x gross (orçamento x faturamento
#       * title_year x algo?
#       *
sns.pairplot(data=imdb_usa[['gross', 'budget', 'lucro', 'title_year']])
plot.show()         ## Plotar vários gráficos comparando as variáveis selecionadas (duas a duas)
print('\n\nCorrelação entre as variáves: gross (ganho), budget (gasto), lucro, e ano')
print(imdb_usa[['gross', 'budget', 'lucro', 'title_year']].corr())      # Correlação entre as variáveis selecionadas



## Desafio 1
# Fazer o boxplot da nota dos filmes coloridos e PeB

## Desafio 2
# filmes mais novos tiveram mais prejuizo?
lucro_ano = imdb_usa[['title_year', 'lucro']]
#sns.scatterplot(x='title_year', y='lucro', data=lucro_ano)
#plot.show()

## Desafio 3
# Quais são os filmes próximos de 1940 que tiveram muito lucro

## Desafio 4
# Confirmar que o diretor que faz muitos filmes e não da lucro é o Woody Allen
filmes_por_diretor = imdb_usa['director_name'].value_counts()               # conta filmes por diretor
gross_director = imdb_usa[['director_name', 'gross']].set_index('director_name').join(filmes_por_diretor, on='director_name')   # Insere a coluna com os dados de quantos filmes foram feitos por aquele diretor. O set_index adiciona um "sub titulo/rotulo" pois não podem ter 2 colunas com o mesmo nome
gross_director.columns=['dindim','filmes_irmaos']   #renomear
gross_director = gross_director.reset_index()
sns.scatterplot(x='filmes_irmaos', y='dindim', data=gross_director)
plot.show()
print('\n\n*#*#*#*#  Desafio 4  #*#*#*#*')
print('Confirmar que o diretor que faz muitos filmes e não da lucro é o Woody Allen')
print(filmes_por_diretor)


## Desafio 5
# Calcular a correlação entre os dados, dos filmes a partir de 2000
# realizar a interpretação

## Desafio 6
# Fazer uma reta nos gráficos de pareamento e verificar se serve para realizar alguma previsão
# por exemplo gross/lucro

## Desafio 7
# Utilizar a nota nas corerlações
# verificar também a quantidade de votos que se obteve























