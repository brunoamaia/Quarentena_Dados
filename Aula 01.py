import matplotlib.pyplot as plt
import pandas as pd     # Imortar a biblioteca "pandas"


### Importar / Ler os arquivos brutos ###
filmes = pd.read_csv("B:\Programação\Quarentena_Dados\dados\movies.csv")    # Existem outros read: read_pdf, read_xls, ...
avaliacoes = pd.read_csv('B:\Programação\Quarentena_Dados\dados\Ratings.csv')
            # este metodo cra a variável como um DataFrame (ver + no pandas)

### Operações básicas que podem ser realizadas com as tabelas ###
filmes.columns = ['filmeId', 'titulo', 'genero']                    # renomeia as colunas (no caso,     traduzimos elas)
avaliacoes.columns = ['usuarioId', 'filmeId', 'nota', 'momento']

# print('Tabelas Importadas: \nFilmes:')
# print(filmes)
# print('\n\nAvaliações:')
# print(avaliacoes)

#filmes.head()   # Mostra as primeiras  colunas dos dados importados (só funciona no Notebook)
#print('\n\n Lista com o nome dos Filmes')
#print(filmes)

#?filmes        # Chama o help (documentação) da função

# nome = filmes.query('filmeId == 1')['title']
# print('\n\nNome do filme selecionado, no caso, com Id = 1')
# print(nome)


# resumo = avaliacoes.describe()  # Mostra um resumo das informaçoes Importadas
# print(resumo)

# print(avaliacoes)
# avaliacoes.shape    #Mostra o tamanho do DataFrame
# len(avaliacoes)     # Pega a quantidade de linhas

# avaliacoes.columns #mostra o nome das colunas (no Notebook)
#
# nota = avaliacoes.query('filmeId==1').describe()    # Buscar apenas os filmes com
# print('\n\nResumo das informações do Fimlme com Id = 1')
# print(nota)
#
# nota1 = avaliacoes.query('filmeId==1')      # Forma de buscar os filmes com Id=1
# nota1 = nota1.query('filmeId==1')['nota']   # Forma de buscar apenas a coluna de "Notas" dos filmes com Id=1
# print('\n\nColuna das notas do filme com Id = 1')
# print(nota1)
#
# mediafilme1 = avaliacoes.query('filmeId==1')['nota'].mean() #forma direta de pegar a média, da nota, dos filmes com Id=1 (não é muito recomendado)
# print(mediafilme1)

### Começando a trabalhar com as tabelas ###
#     Vamos primeiro pegar a média de todos os filmes

nota_media_dos_filmes = avaliacoes.groupby('filmeId')['nota'].mean()    # Pega a média das notas de cada filme
    # este método retorna uma "série"
print('\n\nNota média dos filmes.')
print(nota_media_dos_filmes)

filmes_com_media = filmes.join(nota_media_dos_filmes, on='filmeId')     # Insere a coluna de filmes (nome dos filmes), na tabela de filmes_com_media
print(filmes_com_media.describe())


###         Desafio 1           ###
# Identificar os filmes que não receberam avaliação
filmes_sem_nota = filmes_com_media.query('nota.isna()')    # Forma de separar os filmes sem nota
# filmes_sem_nota = filmes_com_media.query('nota.isnull()', engine='python')  # Forma  que funcionou no colab.research.google
filmes_s_n = filmes_sem_nota['titulo'].values.tolist()        # Transforma a coluna de Titulos em uma lista
n = len(filmes_s_n)
print('\n\n*#*#*#*#  Desafio 1  #*#*#*#*')
print('Encotrar os filmes que não receberam nenhuma nota.')
print(f'Foram encontrados {n} filmes sem avaliação, sendo eles:')
print(*filmes_s_n, sep = "\n")              # Imprimir cada valor em uma linha


###         Desafio 2         ###
# Renomear a coluna "nota", para "media"
# filmes_com_media.columns = ['filmeId', 'titulo', 'generos', 'media']    # Quando sabe a posição exata e/ou quer mudar mais que um nome
filmes_com_media.rename(columns={'nota':'media'}, inplace=True)         # Modifcar do nome "A" para o "B"
print('\n\n*#*#*#*#  Desafio 2  #*#*#*#*')
print('Renomear a coluna "nota", para "media".')
print('Segue a Tabela com o nome da coluna modificada:')
print(filmes_com_media)
filmes_com_media.head(5)
# filmes_com_media.columns = ['filmeId', 'media']



###         Desafio 3         ###
# Colocar a quantidade de avaliações que cada filme recebeu
contar_votos = pd.DataFrame({'filmeId':avaliacoes['filmeId']})     # Criar DataFrame com os filmeId
contar_votos['votos'] = contar_votos                              # Duplicar apra manter a "chave" (filmeId). Pois erá utilizada para linkar com a tabela de nomes
contar_votos = contar_votos.groupby('filmeId')['votos'].count()   # Contar quantas vezes apareceu cada Id (como ficou com 3 colunas, a 1ª mostra o Id do filme)
contar_votos = filmes.join(contar_votos, on='filmeId')     # Insere a coluna de filmes (nome dos filmes), na tabela de cotar votos
contar_votos = contar_votos.drop('filmeId', 1)                             # Remover uma coluna ('nome', eixo: 0 para linhas (x) ou 1 para colunas (y))
print('\n\n*#*#*#*#  Desafio 3  #*#*#*#*')
print('Colocar a quantidade de avaliações que cada filme recebeu.')
print('Tabela com a quantidade de voto dos filmes: ')
print(contar_votos)


###         Desafio 4         ###
# Arredondar as casas decimais das médias
filmes_com_media['media'] = filmes_com_media['media'].round(decimals=2)
print('\n\n*#*#*#*#  Desafio 4  #*#*#*#*')
print('Arredondar para duas casas decimais as médias.')
print('Tabela com os valores arredondados: ')
print(filmes_com_media)


###         Desafio 5         ###
# Verificar quantos são os gêneros únicos e quais são (esse aqui o bixo pega kkk)
#generos_unicos = pd.DataFrame(filmes.genero.str.split('|').tolist()).stack().unique()   # Forma direta de chamar o comando
        # O comando foi "quebrado" para ser mais facil entender
#generos_contados = filmes['genero'].str.get_dummies('|').sum()              # Forma direta para separar e contar os generos
    # .str habilita as ferramentas de String. .get_dumies('*'), separa no caracter '*', cada item separado se torna uma nova coluna. .sum faz o somatório, neste caso somou os valores das colunas
    # Fazer "na unha"
generos = filmes['genero']  # Pega a coluna de Generos
generos = generos.str.split('|')          # Separar os generos da mesma linha (que estão perarados por "|") o .str Serve para trabalhar com string
generos = generos.tolist()                # Tornar uma lista
generos = pd.DataFrame(generos).stack()   # Remodela o Dataframe. Coloca cada estilo em uma linha nova (essas linhas novas são criadas dentro de cada filme)
generos_unicos = generos.unique()                # Verifica os generos unicos
generos_unicos.sort()                                   # Ordenar de forma alfabetica os generos

#generos_unicos = pd.DataFrame(generos_unicos, columns=['genero'])   # Coloca o nome de "genero" na coluna e reeestrutura como um Dataframe (pois tinha tornado uma lista)
n = len(generos_unicos)     # Quantidade de generos
print('\n\n*#*#*#*#  Desafio 5  #*#*#*#*')
print('Verificar quantos são os gêneros únicos e quais são (esse aqui o bixo pega kkk)')
print(f'Foram encontrados {n} generos, eles são os seguintes: ')
#print(generos_unicos)                           # Imprimir como Dataframe
print(*generos_unicos, sep = "\n")              # Imprimir cada valor em uma linha


###         Desafio 6         ###
# Quantas vezes aparece cada gênero
# generos_quantidade, dum = filmes(data, 'genres', genre_labels)
contar_generos = filmes['genero'].str.get_dummies('|').sum()              # Usando o método direto
                # .str habilita as ferramentas de String. .get_dumies('*'), separa no caracter '*', cada item separado se torna uma nova coluna. .sum faz o somatório, neste caso somou os valores das colunas
                #  O comando cria uma série. Existe apenas uma coluna (quantidade), mas cada valor possui um indice (genero)
contar_generos =  contar_generos.sort_values(ascending=False)     # ordenar pelo valor (so tem na coluna de quantidade) de forma decrescente
    # método mais trabalhoso
# contar_generos = pd.DataFrame(generos, columns=['genero'])      # Importar o Dataframe remodeado com os generos
# contar_generos['contador'] = contar_generos['genero']           # Duplicar coluna para fazer a contagem
# contar_generos = contar_generos.groupby('genero')['contador'].count()   # Contador

print('\n\n*#*#*#*#  Desafio 6  #*#*#*#*')
print('Arredondar para duas casas decimais as médias')
print('Tabela com os valores arredondados: ')
print(contar_generos)


###         Desafio 7         ###
# Criar um gráfico para o "Desafio 6". Pode ser de barra


print('\n\n*#*#*#*#  Desafio 7  #*#*#*#*')
print('Criar um gráfico para o "Desafio 6". Pode ser de barra')
# print('Tabela com os valores arredondados: ')
# print()
contar_generos.describe()
plt.interactive(False)

# contar_generos.plot(kind='pie', title='Porcentagem dos generos nos filmes analisados') # Gráfico de pizza
contar_generos.plot(kind='bar', title='Filmes por genero') # Gráfico de barras
plt.show()



