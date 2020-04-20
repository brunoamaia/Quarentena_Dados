import pandas as pd     # Imortar a biblioteca "pandas"

filmes = pd.read_csv("B:\Programação\Quarentena_Dados\dados\movies.csv")     # Importar/Ler os dados do arquivo "movies.csv"
filmes.head()   # Mostra as primeiras  colunas dos dados importados (só funciona no Notebook)

print(filmes)