import pandas as pd

imdb = pd.read_csv('B:\Programação\Quarentena_Dados\dados\movie_metadata.csv')
#print(imdb.head())
a = 1

print('\n\nListar termos unicos e quantidade que apareceram (Filme colorido ou preto e branco):')
print(imdb['color'].value_counts())                 # Contar quantas vezes aparece cada valor unico
print('\n\nNormalizar o valor dos  termos unicos (Filme colorido ou preto e branco): ')
print(imdb['color'].value_counts(normalize=True))   # Normalizar os valores
# print(imdb['director_name'].value_counts().tail(20))    # Lista os Diretores e quantos filmes fizeram e depois Pega as ultimas linhas
a = 1

# Pegar a coluna das cores
color_or_bw = imdb.query("color in ['Color', ' Black and White']")  # Criar nova tabela para estudar a importância/relação de filmes coloridos/PeB (remove os demais)
# color_or_bw['color_0_ou_1'] = (color_or_bw['color']=='Color')*1   # Tentativa de Criar uma coluna que torna a variável Preto/Branco em binária
# color_or_bw['color_0_ou_1'] = color_or_bw['color'] == 'Color'     # Outra forma de fazer, mas não etá aceitando
# df["b"] = df["value"] == 3                                    # Exemplo
# print(f'\n\n Tamanho da variável: {len(color_or_bw)}')




### Perguntas gerias
# - Aventura tem nota melhor que comédia?
# - Diretor com mais filmes te nota melhor?
# - As repostas são sespcíficas para a amortar ou para o mundo?
# - Quais correlaçoes (simples) exsitem entre os dados?
#       * budget x gross (orçamento x faturamento
#       * title_year x algo?
#       *