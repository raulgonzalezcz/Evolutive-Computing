from __future__ import division
import random
import math
import matplotlib.pyplot as plt

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func_obj(individual):
    # Ackley function
    #result = -20 * math.exp(-0.2 * math.sqrt(0.5 * (individual[0]**2 + individual[1]**2))) - math.exp(0.5 * (math.cos(2 * math.pi * individual[0]) + math.cos(2 * math.pi * individual[1]))) + 20 + math.e

    # Rosenbrock function
    """
    x = individual[0]
    y = individual[1]
    a = 1 - x
    b = y - x*x
    result = (a*a) + (b*b*100)
    """

    #Eggholder
    result = (- (individual[1]+47)*math.sin(math.sqrt(abs( individual[1] + (individual[0]/2) + 47)) ) ) - individual[0]*math.sin(math.sqrt(abs( individual[0] - (individual[1]+47))))
    
    return result

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,R,c1,c2,pos_best_g):
        for i in range(0,num_dimensions):

            vel_cognitive=c1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=R*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,R,c1,c2,costFunc,x0,bounds,count,num_particles,maxiter):
        global num_dimensions
        bestPSO = []
        meanPSO = []

        for cont in range(count):
            num_dimensions=len(x0)
            err_best_g=-1                   # best error for group
            pos_best_g=[]                   # best position for group

            best = []
            mean = []

            # establish the swarm
            swarm=[]
            for i in range(0,num_particles):
                swarm.append(Particle(x0))

            # begin optimization loop
            i=0
            while i < maxiter:
                #print i,err_best_g
                # cycle through particles in swarm and evaluate fitness
                err_mean = 0
                for j in range(0,num_particles):
                    swarm[j].evaluate(costFunc)
                    err_mean += float(swarm[j].err_i)

                    # determine if current particle is the best (globally)
                    if swarm[j].err_i < err_best_g or err_best_g == -1:
                        pos_best_g=list(swarm[j].position_i)
                        err_best_g=float(swarm[j].err_i)

                # cycle through swarm and update velocities and position
                for j in range(0,num_particles):
                    swarm[j].update_velocity(R,c1,c2,pos_best_g)
                    swarm[j].update_position(bounds)

                #Get best and mean results
                if(i!=0):
                    best.append(err_best_g)
                    mean.append(err_mean/num_particles)

                i+=1

            # print final results
            print('FINAL:')
            print(pos_best_g)
            print(err_best_g)
            print(len(best))
            print(len(mean))
            self.plotResultsMethod(maxiter-1, best, mean, "Resultados usando PSO")
            bestPSO.append(best)
            meanPSO.append(mean)
        sumas = []
        sumas1 = []
        #print(len(total_values[0]))
        for i in range(maxiter-1):
            sumaPBest=0
            sumaPMean=0
            for j in range(count):
                 sumaPBest += bestPSO[j][i]
                 sumaPMean += meanPSO[j][i]
            sumas.append(sumaPBest/count)
            sumas1.append(sumaPMean/count)
        print("Len final:", len(sumas))
        self.plotResultsMethod(maxiter-1,sumas,sumas1,"Comparación de representaciones en 10 intentos (PSO)")



    def plotFinalResults(self, data, title,color):
    # data = [ [ key,[data_values] ] , ]
        x = [i for i in range(len(data))]
        y = data
        #a = data[2]
        #b = data[3]
        #print(x)
        #print(y)
            
        l1, = plt.plot(x,y,color)
        #l3, = plt.plot(x,a[1],'g')
        #l4, = plt.plot(x,b[1],'k')

        plt.title(title)
        plt.xlabel("No. Generación")
        plt.ylabel("Valor obtenido")
           
        # plt.legend([l1, l2, l3],['('+ schema[y[0]]+ ')', '('+ schema[z[0]]+ ')', '('+ schema[a[0]]+ ')'])
        plt.legend([l1],['(PSO)'])
        plt.show()

    def plotResultsMethod(self,maxiter, dataB, dataM, title):
        x = [i for i in range(maxiter)]
        #print(x)
        #print(y)
            
        l1, = plt.plot(x,dataB,'r')
        l2, = plt.plot(x,dataM,'b')

        plt.title(title)
        plt.xlabel("No. Generación")
        plt.ylabel("Valor obtenido")
           
        plt.legend([l1, l2],['(Mejor individuo)', '(Promedio de población)'])
        plt.show()

#--- RUN ----------------------------------------------------------------------+

R= 1.1      # constant inertia weight (how much to weigh the previous velocity)
c1= 0.9        # cognative constant
c2= 1 - c1                      # social constant
initial_ack=[2.5,2.5]               # initial starting location [x1,x2...]
initial_egg = [125,125]
bounds_ack=[(-5,5),(-5,5)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
bounds_egg=[(-512,512),(-512,512)]
count = 5
PSO(R,c1,c2,func_obj,initial_egg,bounds_egg,count,num_particles=50,maxiter=101)