import numpy as np
from scipy.spatial.distance import euclidean

from threading import Thread

from Dados import Data

class Formiga():
    
    ''' Função de inicialização dos parâmetros (formiga)'''
    def __init__(self, x, y, raioVisao, grid, its, alpha):

        ######################################################################
        self.grid           = grid
        self.raioVisao      = raioVisao
        self.x              = x
        self.y              = y
        self.interacoes     = its

        ######################################################################

        # Calcula quantos dados uma formiga pode ver ao seu redor
        self.raioDados = 1
        for i in range(self.raioVisao):
            self.raioDados = self.raioDados + 2

        ######################################################################
        
        self.carregando       = False  # auxiliar booleano para indicar se a formiga está carregando um dado
        self.data             = None   
        self.c                = self.raioVisao * 10
        self.alpha            = alpha

        ######################################################################
        
    ''' Gera uma posição randomica com um numero de passos aleatórios.
        Tamanho do passo = aceleração do algoritmo = self.grid.shape[0]'''
    def _posicaoRandomica(self):
        shapeGrid    = self.grid.shape[0]
        tamanhoPasso = np.random.randint(1, shapeGrid)

        x = self.x + np.random.randint(-1 * tamanhoPasso, 1 * tamanhoPasso + 1)
        y = self.y + np.random.randint(-1 * tamanhoPasso, 1 * tamanhoPasso + 1)

        if x < 0: 
            x = shapeGrid + x
        if x >= shapeGrid: 
            x = x - shapeGrid
        if y < 0:
            y = shapeGrid + y
        if y >= shapeGrid: 
            y = y - shapeGrid

        return x, y

    ''' Dada uma array de duas dimensões, retorna uma array cujo elemento central é array[x,y] -> vizinhos '''
    def _vizinhos(self, array, x, y, n=3):
        array = np.roll( np.roll(array, shift=- x + 1, axis=0), shift = -y + 1, axis = 1)
        return array[:n, :n]

    ''' Apenas pega um dado se a média de similaridade aplicada a sigmoid possui um valor maior que um número aleatório entre 0 e 1 '''
    def _pegar(self):

        ######################################################################

        # vizinhosVistos = quantidade de dados na vizinhança
        vizinhosVistos = self._vizinhos(self.grid, self.x, self.y, n = self.raioDados)
        # dados para fórmula
        fi             = self._mediaSimilaridade(vizinhosVistos)
        sig            = ((1 - np.exp(-(self.c * fi))) / (1 + np.exp(-(self.c * fi))))
        # f deve variar no intervalo [0,1]
        f              = 1 - sig
        rd             = np.random.uniform(0.0, 1.0)

        ######################################################################
        
        # caso f seja maior que rd, o dado é carregado
        if (f >= rd):
            self.carregando           = True
            self.data                 = self.grid[self.x, self.y]
            self.grid[self.x, self.y] = None
            return True
        return False

    ''' Larga o dado se a média de similaridade aplicada a sigmoid possui um valor maior que um número aleatório entre 0 e 1 '''
    def _largar(self):

        ######################################################################
        
        # vizinhosVistos = quantidade de dados na vizinhança
        vizinhosVistos = self._vizinhos(self.grid, self.x, self.y, n = self.raioDados)
        fi             = self._mediaSimilaridade(vizinhosVistos)
        f              = ((1 - np.exp(-(self.c * fi))) / (1 + np.exp(-(self.c * fi))))
        rd             = np.random.uniform(0.0, 1.0)

        ######################################################################

        if (f >= rd):
            self.carregando = False
            self.grid[self.x, self.y] = self.data
            self.data = None
            return True
        return False

    ''' Move a formiga pelo grid. Enquanto existirem formigas carregando dados, as mesmas são movidas '''
    def run(self):

        grid = self.grid
        x    = self.x
        y    = self.y

        if grid[x,y] == None:
            if self.carregando:
                self._largar()
        elif grid[x,y] != None:
            if not self.carregando:
                self._pegar()

        self.x, self.y = self._posicaoRandomica()

        self.interacoes-= 1
        if self.interacoes<= 0 and self.carregando:
            while self.carregando:
                grid = self.grid
                x, y = self.x, self.y

                if grid[x,y] == None:
                    if self.carregando:
                        self._largar()
                elif grid[x,y] != None:
                    if not self.carregando:
                        self._pegar()

                self.x, self.y = self._posicaoRandomica()
                self.interacoes-= 1

    ''' Calcula a média de similaridade (vizinhança) entre um dado e os outros ao redor da formiga'''
    def _mediaSimilaridade(self, vizinhosVistos):
        somatorioRetorno = 0
        shape            = vizinhosVistos.shape[0]

        if self.carregando:
            data = self.data.get_atributo()
        else:
            data = self.grid[self.x, self.y].get_atributo()

        for i in range(shape):
            for j in range(shape):
                retorno = 0
                if vizinhosVistos[i,j] != None:
                    retorno = 1 - (euclidean(data, vizinhosVistos[i,j].get_atributo())) / ((self.alpha))
                    somatorioRetorno += retorno

        fi = somatorioRetorno / (self.raioDados**2)
        
        if fi > 0: 
            return fi
        else: 
            return 0
