import pygame
import numpy as np

from random import randint
from scipy.spatial.distance import euclidean
from pygame.locals import *

import sys
import threading

from Formiga import Formiga
from Dados import Data


class Colonia():
    def __init__(self, tamanhoGrid = 100, raioVisao = 5, numeroFormigas = 100, interacoes = 5*10**6):

        self.tamanhoGrid        = tamanhoGrid               # Tamanho do grid
        self.raioVisao          = raioVisao                 # Raio de visão das formigas
        self.numeroFormigas     = numeroFormigas            # Número de formigas
        self.interacoes         = interacoes                # Interações para aumentar a velocidade do algoritmo
        self.trabalhadoras      = list()                      # Lista de formigas trabalhadoras

        ''' Lê o dataset '''
        dataset           = np.loadtxt('dataset_1_4.txt')
        listaInicializada = list()

        for elemento in dataset:
            dado = Data(elemento[0:-1], elemento[-1]*130%255) # 230%255 para coloração
            listaInicializada.append(dado)
        self.datasetCarregado = listaInicializada

        ''' Calculando alpha '''
        distanciaEuclidiana = 0
        for dadoX in self.datasetCarregado:
            for dadoY in self.datasetCarregado:
                distanciaEuclidiana += euclidean(dadoX.get_atributo(), dadoY.get_atributo())
        self.alpha  = distanciaEuclidiana / (len(self.datasetCarregado) ** 2)

        ''' Gera o grid '''
        self.grid = np.empty((self.tamanhoGrid, self.tamanhoGrid), dtype = object)
        for dado in self.datasetCarregado:
            i = np.random.randint(0, self.tamanhoGrid)
            j = np.random.randint(0, self.tamanhoGrid)
            while self.grid[i,j] is not None:
                i = np.random.randint(0, self.tamanhoGrid)
                j = np.random.randint(0, self.tamanhoGrid)
            self.grid[i,j] = dado

        ''' Inicializa as formigas, as cria e adiciona a lista de trabalhadoras'''
        for i in range(self.numeroFormigas):
            x = np.random.randint(0, self.tamanhoGrid-1)
            y = np.random.randint(0, self.tamanhoGrid-1)
            formiga = Formiga(x, y, self.raioVisao, self.grid, self.interacoes, self.alpha)
            self.trabalhadoras.append(formiga)


    ''' Inicia execução sequencial '''
    def _iniciacaoSequencial(self):
        for i in range(self.interacoes):
            for formiga in self.trabalhadoras:
                formiga.run()
        listaInicializada = list()
        for formiga in self.trabalhadoras:
            listaInicializada.append(formiga._get_carrying())

    ''' Converte uma matriz de dados em matriz de ints para mostrar na tela via pygame '''
    def _get_matrizInteiros(self):
        matrizInicializada = np.zeros((self.tamanhoGrid, self.tamanhoGrid))
        
        for i in range(self.tamanhoGrid):
            for j in range(self.tamanhoGrid):
                if self.grid[i,j] != None and type(self.grid[i,j]) == Data:
                    data = self.grid[i,j]
                    matrizInicializada[i,j] = data.get_grupo()
                else:
                    matrizInicializada[i,j] = self.grid[i,j]
        return matrizInicializada

    ''' Método principal da colônia '''
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        screen.set_alpha(None)

        t        = threading.Thread(target=self._iniciacaoSequencial)
        t.daemon = True
        t.start()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            int_arr    = self._get_matrizInteiros()
            surface    = pygame.surfarray.make_surface(int_arr)
            newsurface = pygame.transform.scale(surface, (800, 600))
            screen.blit(newsurface, (0,0))
            pygame.display.flip()

if __name__ == "__main__":
    colonia = Colonia()
    colonia.run()
