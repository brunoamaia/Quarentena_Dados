import pandas as pd     # Imortar a biblioteca "pandas"
import math
import matplotlib.pyplot as plt

### Importar / Ler os arquivos brutos ###
filmes = pd.read_csv("B:\Programação\Quarentena_Dados\dados\movies.csv")    # Existem outros read: read_pdf, read_xls, ...
avaliacoes = pd.read_csv('B:\Programação\Quarentena_Dados\dados\Ratings.csv')
            # este metodo cra a variável como um DataFrame (ver + no pandas)

### Operações básicas que podem ser realizadas com as tabelas ###
filmes.columns = ['filmeId', 'titulo', 'genero']                    # renomeia as colunas (no caso,     traduzimos elas)
avaliacoes.columns = ['usuarioId', 'filmeId', 'nota', 'momento']

print('Tabelas Importadas: \nFilmes:')
print(filmes)
print('\n\nAvaliações:')
print(avaliacoes)
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
print('\n\n*#*#*#*#  Desafio 1  #*#*#*#*')
print('Encotrar os filmes que não receberam nenhuma nota')
print('Filmes sem avaliação: \n')
print(*filmes_s_n, sep = "\n")              # Imprimir cada valor em uma linha


###         Desafio 2         ###
# Renomear a coluna "nota", para "media"
# filmes_com_media.columns = ['filmeId', 'titulo', 'generos', 'media']    # Quando sabe a posição exata e/ou quer mudar mais que um nome
filmes_com_media.rename(columns={'nota':'media'}, inplace=True)         # Modifcar do nome "A" para o "B"
print('\n\n*#*#*#*#  Desafio 2  #*#*#*#*')
print('Renomear a coluna "nota", para "media"')
print('Tabela com o nome da coluna modificada: \n')
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
print('Colocar a quantidade de avaliações que cada filme recebeu')
print('Quantidade de voto dos filmes: ')
print(contar_votos)


###         Desafio 4         ###
# Arredondar as casas decimais das médias
filmes_com_media['media'] = filmes_com_media['media'].round(decimals=2)
print('\n\n*#*#*#*#  Desafio 4  #*#*#*#*')
print('Arredondar para duas casas decimais as médias')
print('Tabela com os valores arredondados: ')
print(filmes_com_media)


###         Desafio 5         ###
# Verificar quantos são os gêneros únicos e quais são (esse aqui o bixo pega kkk)
generos = filmes['genero']  # Pega a coluna de Generos
# .unique()         # Verifica os generos unicos
#print(generos_unicos)
# print('\n\n*#*#*#*#  Desafio 5  #*#*#*#*')
# print('Arredondar para duas casas decimais as médias')
# print('Tabela com os valores arredondados: ')
# print()

###         Desafio 6         ###
# Quantas vezes aparece cada gênero

# print('\n\n*#*#*#*#  Desafio 6  #*#*#*#*')
# print('Arredondar para duas casas decimais as médias')
# print('Tabela com os valores arredondados: ')
# print()


###         Desafio 7         ###
# Criar um gráfico para o "Desafio 6". Pode ser de barra

# print('\n\n*#*#*#*#  Desafio 7  #*#*#*#*')
# print('Arredondar para duas casas decimais as médias')
# print('Tabela com os valores arredondados: ')
# print()



