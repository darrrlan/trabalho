import numpy as np

class MLP(object):
    def __init__(self, entrada, oculta, saida, taxaDeAprendizado=0.1, pesos=None):
        self.entrada = entrada
        self.oculta = oculta
        self.saida = saida
        self.taxaDeAprendizado = taxaDeAprendizado

        if pesos is None:
            # Inicialize os pesos e bias aleatoriamente se não forem fornecidos
            min_valor = -0.5
            max_valor = 0.5

            self.pesos_input_oculta = np.random.uniform(min_valor, max_valor, (self.entrada, self.oculta))
            self.bias_oculta = np.random.uniform(min_valor, max_valor, self.oculta)
            self.pesos_oculta_saida = np.random.uniform(min_valor, max_valor, (self.oculta, self.saida))
            self.bias_saida = np.random.uniform(min_valor, max_valor, self.saida)
        else:
            # Use os pesos fornecidos pelo AG
            self.pesos_input_oculta, self.bias_oculta, self.pesos_oculta_saida, self.bias_saida = pesos

    def getTaxaDeAprendizado(self):
        return self.taxaDeAprendizado

    def setTaxaDeAprendizado(self, taxa):
        self.taxaDeAprendizado = taxa

    def ativacaoSigmoidal(self, valor):
        return 1 / (1 + np.exp(-valor))

    def derivadaAtivacaoSigmoidal(self, valor):
        return valor * (1 - valor)

    def erroQuadraticoMedio(self, esperado, valor):
        return np.mean(np.square(esperado - valor))

    def feedForward(self, entrada):
        self.saida_oculta = self.ativacaoSigmoidal(np.dot(entrada, self.pesos_input_oculta) + self.bias_oculta)
        self.saida_rede = self.ativacaoSigmoidal(np.dot(self.saida_oculta, self.pesos_oculta_saida) + self.bias_saida)
        return self.saida_rede

    def backpropagation(self, entrada, target):
        self.feedForward(entrada)
        erro_saida = target - self.saida_rede
        gradiente_saida = erro_saida * (self.saida_rede * (1 - self.saida_rede))
        self.pesos_oculta_saida += self.taxaDeAprendizado * np.outer(self.saida_oculta, gradiente_saida)
        self.bias_saida += self.taxaDeAprendizado * gradiente_saida
        erro_oculta = np.dot(gradiente_saida, self.pesos_oculta_saida.T)
        gradiente_oculta = erro_oculta * (self.saida_oculta * (1 - self.saida_oculta))
        self.pesos_input_oculta += self.taxaDeAprendizado * np.outer(entrada, gradiente_oculta)
        self.bias_oculta += self.taxaDeAprendizado * gradiente_oculta
        return self.saida_rede
    

    def set_weights(self, weights):
        # Defina as formas dos pesos
        pesos_input_oculta_shape = (5, 8)  # A forma dos pesos de entrada para oculta
        pesos_oculta_saida_shape = (8, 1)  # A forma dos pesos de oculta para saída

        # Extraia os pesos do array fornecido e reorganize-os
        pesos_input_oculta = np.array(weights[:pesos_input_oculta_shape[0]*pesos_input_oculta_shape[1]]).reshape(pesos_input_oculta_shape)
        pesos_oculta_saida = np.array(weights[pesos_input_oculta_shape[0]*pesos_input_oculta_shape[1]:pesos_input_oculta_shape[0]*pesos_input_oculta_shape[1] + pesos_oculta_saida_shape[0]*pesos_oculta_saida_shape[1]]).reshape(pesos_oculta_saida_shape)
        bias_oculta = np.array(weights[-(pesos_input_oculta_shape[1] + pesos_oculta_saida_shape[0]):-(pesos_oculta_saida_shape[0])])
        bias_saida = np.array([weights[-1]])  # Pegue o último elemento como bias de saída

        # Configure os pesos internos com os valores extraídos
        self.pesos_input_oculta = pesos_input_oculta
        self.bias_oculta = bias_oculta
        self.pesos_oculta_saida = pesos_oculta_saida
        self.bias_saida = bias_saida


    def treinamento(self, entrada, esperado, epocas):
        for _ in range(epocas):
            for i in range(len(entrada)):
                entrada_atual = entrada[i]
                esperado_atual = esperado[i]

                saida = self.feedForward(entrada_atual)

                self.backpropagation(entrada_atual, esperado_atual)

                erro = self.erroQuadraticoMedio(esperado_atual, saida)
                print(f'Época {_}, Amostra {i}, Erro: {erro}')
