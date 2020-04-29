import pandas as pd

a = [ ('jack', 34, 'Sydeny' , 'Australia'),
      ('Riti', 30, 'Delhi' , 'India' ),
      ('Vikas', 31, 'Mumbai' , 'India' )]

b = [('Neelu', 32, 'Bangalore' , 'India' ),
      ('John', 16, 'New York' , 'US') ,
      ('Mike', 17, 'las vegas' , 'US')]

Dataa = pd.DataFrame(a, columns = ['Name' , 'Age', 'City' , 'Country'])
Datab = pd.DataFrame(b, columns = ['Name' , 'Age', 'City' , 'Country'])

print('Dataframe A')
print(Dataa)
print('\nDataframe B')
print(Datab)

# dataa = Dataa.append({Datab}, ignore_index=True)
# modDfObj = dfObj.append(pd.Series(['Raju', 21, 'Bangalore', 'India'], index=dfObj.columns ), ignore_index=True)
datac = Dataa.append(pd.DataFrame(Datab), ignore_index=True)

print('\nDataframe C')
print(datac)


dados_treino = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DADOS_TREINO.csv')
dados_teste = pd.read_csv('B:\Programação\Quarentena_Dados\dados\desafio\DADOS_TESTE.csv')

print('Informações dos dados:')
print(f'Dados do Treino: \t{dados_treino.shape}')
print(f'Dados do Teste: \t{dados_teste.shape}')

dados_todos = dados_treino.append(pd.DataFrame(dados_teste), ignore_index=True)
print(f'Dados do Dataframe final: \t{dados_todos.shape}')

print('\nDados dos conjuntos:')
print('Final dos Dados de Treino:')
print(dados_treino.tail(3))
print('Inicio dos Dados de Teste')
print(dados_teste.head(3))

print('\nRegião de Interseção dos DataFrames')
print(dados_todos.loc[149997:150002])
