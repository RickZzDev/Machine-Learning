#pelo longo?
#perna curta?
#faz auau?

porco1 = [0,1,0]
porco2 = [0,1,1]
porco3 = [1,1,0]

cachorro1 = [0,1,1]
cachorro2 = [1,0,1]
cachorro3 = [1,1,1]
#O treino são os dados para treinar o algoritmo
treino_x = [porco1,porco2,porco3,cachorro1,cachorro2,cachorro3]
#usando classificacao binaria para dizer qual é porco e qual é cachorro, também é treino
treino_y = [1,1,1,0,0,0] #1 é porco, 0 é cachorro

from sklearn.svm import LinearSVC
#Criando um cerebro vazio
modelo = LinearSVC()
#Passando os treinos para ele
modelo.fit(treino_x,treino_y)
#Criando um animal para passa lo a IA
animal_misterioso = [1,1,1]
#Pedindo uma predict do animal misterioso
modelo.predict([animal_misterioso])

#CRIANDO NOVOS ANIMAIS PARA TESTE
misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]
#Colocando esses animais em um array
teste_x = [misterio1,misterio2,misterio3]
#Definindo o que cada animal é (porco ou cachorro)
teste_y = [0,1,1]
#Enviando ao modelo e recebendo a resposta
previsoes = modelo.predict(teste_x)
previsoes

from sklearn.metrics import accuracy_score
#Saber a taxa de acerto, o primeiro parametro é as certezas e o segundo o predict
accuracy_score(teste_y,previsoes)