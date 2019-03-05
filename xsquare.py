
# -*- coding: utf-8 -*-
import numpy as np
from math import pi, pow

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
class Lata:
    def __init__(self, n=10, maxsol=32, gensize=5, term=10, prob=0.3):
        self.n = n
        self.maxsol = maxsol
        self.pob = []
        self.gen = 0
        self.gensize = gensize
        self.term = term
        self.prob = prob
        self.best = [100000000000000,0]
        np.random.seed()

    # Genera la poblacion inicial, segun el tamaño de maz sol. 
    def gen_population(self):
        # Genera pares aleatorios
        dv = np.random.uniform(low=0, high=self.maxsol, size=self.n)
        hv = np.random.uniform(low=0, high=self.maxsol, size=self.n)

        # Posteriormente los converte en binarios y llena con ceros
        print("Población generada: ")
        for i in range(len(dv)):
            d = "{0:b}".format(int(dv[i]))
            h = "{0:b}".format(int(hv[i]))
            d = d.zfill(self.gensize)
            h = h.zfill(self.gensize)
            print(str(i),":",str(d),str(h))
            self.pob.append([d,h,i])

    #Ciclo principal
    def calc(self):
        self.data = []
        while True:
            print(" * Probar fitess y condiciones")
            for pair in self.pob:
                res = self.g(pair)
                cond = self.f(pair)


                # ¿Si es el mejor elemento, guardar?
                if res > 0:
                    if self.best[0] > cond:
                        self.best = [cond, pair]
                        
                    
                # Si no se ha alcanzado la condición de término
                if not self.term:
                    if res > 0:
                        print("Solución encontrada 1: ", str(self.best),end=" ")
                        print(res, cond)
                        return pair
                else:
                    if self.gen > self.term:
                        print("Solución encontrada 2: ", str(self.best), end=" ")
                        return self.best
            # Una revisada la condición de término se pasa a 
            # crear la nueva generación
            print(" * Crear nueva generación") 
            new_gen = []
            i = 0
            count = 0 #Contador de la nueva generación

            # Genera nueva población por cruza
            while i < ((len(self.pob))/4):
                
                # Selecciona aleatoria oponentes
                r = np.random.uniform(0, len(self.pob), size=2)
                j = int(r[0])
                k = int(r[1])
                #Primer torneo!
                if self.f(self.pob[j]) < self.f(self.pob[k]):
                    winner1 = j
                else:
                    winner1 = k

                # Selecciona aleatoriamente  los contrincantes
                r = np.random.uniform(0, len(self.pob), size=2)
                j = int(r[0])
                k = int(r[1])
                #Segundo torneo
                if self.f(self.pob[j]) < self.f(self.pob[k]):
                    winner2 = j
                else:
                    winner2 = k

                # Ahora hacemos la cruza de los mejores individuos
                elem = self.cross(self.pob[winner1], self.pob[winner2])
                elem[0].append(count)
                count += 1
                elem[1].append(count)
                count += 1
                print(elem[0])
                print(elem[1])
                # Agrego a la nueva generación
                new_gen.append(elem[0])
                new_gen.append(elem[1])
                i += 1
            i = 0

            # Ahora vamos a mutar!
            while count < len(self.pob)-1:

                # Seleccionamos la primer víctima
                r = np.random.randint(0,len(self.pob))
                e = self.mutt(self.pob[r][:2])
                e.append(count)
                print(e)
                new_gen.append(e)
                count += 1

                # Seleccionamos la primer víctima
                r = np.random.randint(0,len(self.pob))
                e = self.mutt(self.pob[r][:2])
                e.append(count)
                new_gen.append(e)
                print(e)
                count += 1
                i += 1
            print(new_gen) 
            self.pob = new_gen 
            # Transformamos a binarios :)
            if self.best[1] == 0:
                x1 = np.random.uniform(low=0, high=self.maxsol)
                x2 = np.random.uniform(low=0, high=self.maxsol)
                d = "{0:b}".format(int(x1))
                d = d.zfill(self.gensize)
                
                h = "{0:b}".format(int(x2))
                h = h.zfill(self.gensize)
                
                self.best[0] = self.f([d,h])
                self.best[1] = [d,h]

            # Agregamos el mejor de la generación a la próxima
            best = self.best[1]
            best.append(count)
            self.pob.append(best)
            self.gen += 1

            self.data.append([self.gen, self.best]) 
            print("Mejor solución de la generación: ", str(self.best))
            print("Volumen:", str(300+self.g(self.best[1])))
            print("d:", str(int(self.best[1][0],2)))
            print("h:", str(int(self.best[1][1],2)))
            input();
            print("-------------------- Nueva Generación -------------------")


    # Cálculo del la función de fitness
    def f(self, pair):
        d = 0.65*(float(int(pair[0],2)))
        h = float(int(pair[1],2))
        res = ((pi*pow(d,2))/2) + pi*d*h
        return res

    # Cálculo de la restrucción
    def g(self,pair):
        print(pair)
        d = float(int(pair[0],2))
        h = float(int(pair[1],2))
        res = ((pi*pow(d,2)*h)/4)-300
        return res

    
    # Se realiza la cruza tomando el punto de inicio self.crossf
    def cross(self, pair1, pair2):
        a1 = pair1[0]
        b1 = pair1[1]

        a2 = pair2[0]
        b2 = pair2[1]

        # Generamos nuevos números
        fin = np.random.randint(2,7) # Punto de cruza

        na1 = a1[:fin] + b2[fin:]
        nb1 = b1[:fin] + a2[fin:]

        fin = np.random.randint(2,7) # Punto de cruza 2

        na2 = a2[:fin] + b1[fin:]
        nb2 = b2[:fin] + a1[fin:]
        return [[na1, nb1], [na2,nb2]]
    
    # Función que genera la multación de los individuos
    def mutt(self, pair):
        new = []
        print(pair[0])
        p1 = np.array(list(pair[0]))
        p2 = np.array(list(pair[1]))

        np.random.shuffle(p1)
        np.random.shuffle(p2)

        print(p1)

        pn1 = "".join(p1)
        pn2 = "".join(p2)
        new.append(pn1)
        new.append(pn2)
        return new


    #Método que grafica los valores encontrados en forma de cilindros
    def drawCyl(self): 

        d  = int(self.best[1][0],2)
        h  = int(self.best[1][1],2)
        center = 0

        fig = plt.figure()
        ax = Axes3D(fig, azim=30, elev=30)
        x=np.linspace(center-d, center+d, 100)
        z=np.linspace(0, h, 100)
        Xc, Zc=np.meshgrid(x, z)
        Yc = np.sqrt(d**2 -(Xc - center) ** 2) + center 
        # Draw parameters
        rstride = 20
        cstride = 10
        ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
        ax.plot_surface(Xc, (2*center-Yc), Zc, alpha=0.2, rstride=rstride, cstride=cstride)


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_title("Mejor cilindro encontrado por el algoritmo evolutivo")

        plt.show()

    def draw(self):
        x = [i[0] for i in self.data]
        y = [int(i[1][0]) for i in self.data]
        print(x)
        print(y)
        
        plt.plot(x,y)
        plt.title("Mejor elemento global: desarrollo de volúmen del cilindo por cada generació por cada generación")
        plt.xlabel("Generación")
        plt.ylabel("Volúmen de la lata")
        
        plt.show()


if __name__ == "__main__":
    lata = Lata()
    lata.gen_population()
    lata.calc()
    #lata.drawCyl()
    lata.draw()

# License: GPL v3+
# Copyright: Jesss Mager 2017