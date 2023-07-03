    # -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 10:19:36 2020

@author: liwenjie
"""

import numpy as np
from  Assign import TransTime#, Optimize
import matplotlib.pyplot as plt
import time

Spoint  = 7+1
Dpoint  = 14+1
TCpoint = 4+1
MaNum   = 4+1
Requests= 16+1
JibRadius = 70
Dis_ST, DeliverTime = TransTime()
# print(Dis_ST)
# AssignResult = Optimize()
ST = time.perf_counter()
RequestDetail = np.array([[8,2], [14,1], [12,4], [5,3], [3,1],
        [3,2], [3,3], [7,4], [9,3], [4,4], [2,4], [1,2],
        [13,2],[11,3], [6,1], [10,4], [0,0]])

List = [[4,6,12], [1,7,8,11], [5,9,10,15,16], [2,3,13,14]]
#List = [[4,6,12], [1,7,8,11], [5,9,10,15,16], [2,3,13,14]]
MaxLen = max(len(x) for x in List)
NewList = list(map(lambda l:l + [Requests]*(MaxLen - len(l)), List))
AssignResult = RequestDetail[np.array(NewList, dtype = np.int16) - 1]
# print(AssignResult)     
# print(AssignResult.shape)     


class GA(object):
################################# Initialize the Sequence Model #################################
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, cross_coefficent, tower_num):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.cross_coefficent = cross_coefficent
        self.tower_num = TowerNum
        self.pop = np.array([[np.random.permutation(self.DNA_size) for i in np.arange(self.tower_num)]for _ in np.arange(self.pop_size)], dtype = np.int16)
        # self.pop[self.pop_size -1] = N2N()
################################# Choose a Supply Point for Each Request #################################
    def assign_supply(self, deliver_time, dis_st, material_store, requests):
        supply_chain = np.empty_like(self.pop, dtype = np.int16)
        for k, request in enumerate(requests):
            for j, detail in enumerate(request):   
                temp_supply = []
                temp_time = []
                if detail[1] == 0: #如果是虚拟运输任务，则分配虚拟物料供应点
                    temp_supply.append(Spoint)
                else:
                    for i in material_store:                 
                        if detail[1] == i[1] and dis_st[k, i[0]-1] <= JibRadius:  #是否存储相应材料
                            temp_supply.append(i[0]) #挑出存储有所需材料的供应点
                if len(temp_supply) > 1:
                    for i in temp_supply:
                        temp_time.append(deliver_time[k, i-1, detail[0]-1]) #计算材料供应所需时间
                    posibility = 100 / np.array(temp_time) 
                    supply = np.random.choice(temp_supply, size=self.pop_size, replace=True, p=posibility / posibility.sum()) #距离最近的供应方案具有较大的概率被选为供应点
                    supply_chain[:, k, j] = supply
                else :
                    supply_chain[:, k, j] = temp_supply
        return supply_chain
################################# Ignore the virtual transportation requests #################################
    def exchange(self, request, DNA):
        De = request[:, :, 0]
        for i in range(self.DNA_size):
            individual = DNA[i]
            for k in range(self.tower_num):
                sequence_sin = individual[k]
                location = np.argwhere(De[k] == 0).flatten() #虚拟任务的位置
                if location.shape[0] > 0:#是否存在虚拟任务
                    real = np.delete(sequence_sin, location) #删除虚拟任务
                    real_se = np.argsort(real) #重新排序
                    se_new = np.arange(real.shape[0])
                    for j, d in enumerate(real_se):
                        real[d] = se_new[j]
                    DNA[i, k] = np.append(real, np.arange(real_se.shape[0], De.shape[0]+1))   #补全任务排序 
        return DNA
################################# Translate DNA #################################
    def translateDNA(self, DNA, supply_chain, requests):
        demand = requests[:, :, 0] #获取需求位置坐标
        supply_point = np.empty_like(DNA)
        demand_point = np.empty_like(DNA)
        for i,d in enumerate(DNA):
            for k,s in enumerate(d):
                dese = demand[k]
                demand_point[i, k, :] = dese[s] #按照DNA的顺序对request进行重新排序  
                supply = supply_chain[i, k, :]
                supply_point[i, k, :] = supply[s] #按照DNA顺序对Supply排序
        return supply_point, demand_point
################################# Get the Fitness of each Individual #################################
    def get_fitness(self, supply_point, demand_point, deliver_time, startpoint):
        Start_time = np.empty((self.pop_size, self.tower_num, 2 * self.DNA_size), dtype=np.float32)
        Single_time = np.empty((self.pop_size, self.tower_num), dtype=np.float32)
        for a in range(self.pop_size): #个体编号
            for k in range(self.tower_num): #塔吊编号
                ST, FT = 0, 0
                Time_unloading, Time_loading = 1, 1
                HookLocation = startpoint[k] # 吊钩初始位置
                for x in range(self.DNA_size): #基因编号
                    if supply_point[a, k, x] != Spoint: 
                        s, d1, d2 =  supply_point[a, k, x], HookLocation, demand_point[a, k, x] 
                        StartTime1 = ST
                        StartTime2 = StartTime1 + deliver_time[k, s-1, d1-1] + Time_loading #Demand to Supply
                        ST = StartTime2 + deliver_time[k, s-1, d2-1] + Time_unloading  #Supply to Demand
                        Start_time[a, k, 2 * x] =  StartTime1
                        Start_time[a, k, 2 * x + 1 ] =  StartTime2
                        FT += deliver_time[k, s-1, d2-1] + Time_unloading + + deliver_time[k, s-1, d1-1] + Time_loading
                        HookLocation = d2
                    else:
                        Start_time[a, k, 2 * x] =  ST
                        Start_time[a, k, 2 * x + 1 ] =  ST
                Single_time[a, k] = FT
            ##
        total_time = Single_time.sum(axis=1) #总时间
        # total_time = Single_time.max(axis=1)  #运行时间最大值
        # print(Start_time)
        fitness = np.exp(1000 / total_time)
        return fitness, total_time
################################# Select the Good Invidual to Next Population #################################
    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return idx
################################# Crossover Operator (Partially Mapped Crossover) for Sequence Part #################################
    def PMX(self, parent_A, parent_B, start, end):    
        keep =parent_A
        for x_ in range(start, end+1):
            childA1 = parent_A.copy()
            element = parent_B[x_]
            locationA = np.where(childA1 == element)
            location_A= np.ravel(locationA)
            swap_A, swap_B = childA1[x_], childA1[location_A[0]]
            childA1[x_], childA1[location_A[0]] = swap_B, swap_A
        for y_ in range(start, end+1):
            childB1=parent_B.copy()
            element = keep[y_]
            locationB = np.where(childB1 == element)
            location_B= np.ravel(locationB)
            swap_A, swap_B = childB1[y_], childB1[location_B[0]]
            childB1[y_], childB1[location_B[0]] = swap_B, swap_A
        return childA1, childB1
################################# Crossover Operator (Single Mapped Crossover) for Supply Location Part #################################
    def SPC(self, parent_A, parent_B, start, end): 
        child_A = np.hstack((parent_A[:start], parent_B[start:end], parent_A[end:]))
        child_B = np.hstack((parent_B[:start], parent_A[start:end], parent_B[end:]))
        return child_A, child_B
################################# Crossover Operatoration in Improved Genetic Algorithm #################################
    def crossover(self, parent_a, parent_b, pop, fitness, supply_chain, mean_fitness, max_fitness):
        pop_childA = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        pop_childB = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        supply_childA = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        supply_childB = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        parentASe = pop[parent_a]
        parentBSe = pop[parent_b]
        parentASu = supply_chain[parent_a]
        parentBSu = supply_chain[parent_b]
        if fitness[parent_a] >= mean_fitness : 
            for k in range(self.tower_num):
                if np.random.rand() <= self.cross_rate - self.cross_coefficent * ((max_fitness - fitness[parent_a]) / (max_fitness - mean_fitness)) :
                    Start_point = np.random.randint(0, self.DNA_size/2+1)
                    End_point = np.random.randint(self.DNA_size/2, self.DNA_size)
                    #Partially Mapped Crossover(Sequence)
                    pop_childA[k, :], pop_childB[k, :] = self.PMX(parentASe[k], parentBSe[k], Start_point, End_point)
                    #Single point Crossover(Supply)
                    supply_childA[k, :], supply_childB[k, :] = self.SPC(parentASu[k], parentBSu[k], Start_point, End_point)
                else:
                    pop_childA = pop[parent_a]
                    pop_childB = pop[parent_b]
                    supply_childA = supply_chain[parent_a]
                    supply_childB = supply_chain[parent_b]
        else:
            for k in range(self.tower_num):
                if np.random.rand() <= self.cross_rate :
                    Start_point = np.random.randint(0, (self.DNA_size)/2+1)
                    End_point = np.random.randint((self.DNA_size)/2, self.DNA_size)
                    #Partially Mapped Crossover(Sequence)
                    pop_childA[k, :], pop_childB[k, :] = self.PMX(parentASe[k], parentBSe[k], Start_point, End_point)
                    #Single point Crossover(Supply)
                    supply_childA[k, :], supply_childB[k, :] = self.SPC(parentASu[k], parentBSu[k], Start_point, End_point)
                else:
                    pop_childA = pop[parent_a]
                    pop_childB = pop[parent_b]
                    supply_childA = supply_chain[parent_a]
                    supply_childB = supply_chain[parent_b]
        return pop_childA, pop_childB, supply_childA, supply_childB
################################# Mutate Operator for Sequence Part #################################
    def mutate(self, child):
        # print(child)
        for k in range(self.tower_num):
            for point in range(self.DNA_size):
                Prob = np.random.rand()
                #Swap
                if  Prob < self.mutate_rate/3:
                    swap_point = np.random.randint(0, self.DNA_size)
                    swapA, swapB = child[k, point], child[k, swap_point]
                    child[k, point], child[k, swap_point] = swapB, swapA  #exchange the sequence
                #Inversion
                elif self.mutate_rate/3 <= Prob < 2*self.mutate_rate/3:
                    inversion_point = np.random.randint(0, self.DNA_size)
                    if inversion_point > point:
                        inversionA = child[k, point : inversion_point]
                        inversionB = inversionA[::-1]
                        child[k, point:inversion_point] = inversionB
                    elif inversion_point == point :
                        pass
                    else:
                        inversionA = child[k, inversion_point:point]
                        inversionB = inversionA[::-1]
                        child[k, inversion_point:point] = inversionB
                #Insertion             
                elif 2* self.mutate_rate/3 <= Prob < self.mutate_rate:
                    insertion_point = np.random.randint(0, self.DNA_size)
                    insertA = child[k, point]
                    a = np.delete(child[k], point)
                    child[k] = np.insert(a,insertion_point,insertA)
        return child
################################# Mutate Operator for Supply Part #################################
    def supply_mutate(self, child, request, material_store, dis_st): #mutate to Supply
        for k in range(self.tower_num):
            for j in range(self.DNA_size):
                if np.random.rand() < self.mutate_rate:
                    temp_supply = []
                    if request[k, j, 1] == 0: #如果是虚拟运输任务，则分配虚拟物料供应点
                            temp_supply.append(Spoint)
                    else:
                        for i in material_store:                 
                            if request[k, j, 1] == i[1] and dis_st[k, i[0]-1] <= JibRadius:  #是否存储相应材料
                                temp_supply.append(i[0]) #挑出存储有所需材料的供应点
                    child[k, j] = np.random.choice(temp_supply, size = 1) #生成新的供应链
        return child
################################# Evolve Operator for Population #################################    
    def evolve(self, fitness, supply_chain, request, material_store, dis_st):
        mean_ = np.mean(fitness)
        max_ = np.max(fitness)
        idx = self.select(fitness) #得到适应度高个体的编号
        pop = self.pop[idx] #挑选出优秀个体
        supply_chain = supply_chain[idx] #挑选出优秀个体（Supply）
        pair = np.random.choice(np.arange(self.pop_size), size = ((self.pop_size//2), 2), replace=False)
        for z in pair: #挑选的父代A
            #Crossover 
            child_popA, child_popB, child_supA, child_supB = self.crossover(z[0], z[1], pop, fitness, supply_chain, mean_, max_)
            #sequence mutate
            pop[z[0]] = self.mutate(child_popA)
            pop[z[1]] = self.mutate(child_popB)
            #supply mutate
            supply_chain[z[0]] = self.supply_mutate(child_supA, request, material_store, dis_st)
            supply_chain[z[1]] = self.supply_mutate(child_supB, request, material_store, dis_st)
        self.pop = pop
################################# Improved Genetic Algorithm with Elitist Preservation #################################
if __name__ == '__main__':
################################# Algorithm Parameters #################################
    Material_store = np.array([[1,3],[1,4],[2,1],[2,2],[3,1],[3,3],[3,4],[4,2],[5,1],[5,4],[6,2],[6,3],[7,1],[7,2]]) #Material存储信息
    InitLocation = np.array([5, 2, 6, 11])
    N_CITIES = AssignResult.shape[1]  # DNA size
    CROSS_RATE = 0.85
    CROSS_COEFFICENT = 0.5
    MUTATE_RATE = 0.015
    POP_SIZE = 20
    N_GENERATIONS = 50
    TowerNum = AssignResult.shape[0]
################################# Get The Elite Individual #################################
    Best_invidual = np.empty((AssignResult.shape[0], AssignResult.shape[1]), dtype = np.int16)
    Best_supply = np.empty((AssignResult.shape[0], AssignResult.shape[1]), dtype = np.int16)
    Best_fitness = 0
    BF = []
    MF = []
################################# Starts #################################
    print(30 * '-', ' Start ', 30 * '-')
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, cross_coefficent=CROSS_COEFFICENT, tower_num = TowerNum)
    ga.pop = ga.exchange(AssignResult, ga.pop)
    SupplyPoint = ga.assign_supply(DeliverTime, Dis_ST,Material_store, AssignResult)
    for generation in range(N_GENERATIONS):
        SupplySequence, DemandSequence = ga.translateDNA(ga.pop, SupplyPoint, AssignResult)
        Fitness, OperationTime = ga.get_fitness(SupplySequence, DemandSequence, DeliverTime, InitLocation)
        best_idx = np.argmax(Fitness) #最优的适应度值
        BF.append(OperationTime[best_idx])
        mean_fitness = np.mean(OperationTime)#平均的适应度值
        MF.append(mean_fitness)
        if np.max(Fitness) > Best_fitness: #获取目前为止最优的结果
            Best_fitness = np.max(Fitness)
            Best_invidual = ga.pop[best_idx]
            Best_supply = SupplyPoint[best_idx]
        # print('Gen:', generation, '| best fit: %.2f' % OperationTime[best_idx],)
        # print('Best sequence', ga.pop[best_idx])
        # print('Best Supply', SupplyPoint[best_idx])
        # print('-----------------------------')
        ga.evolve(Fitness, SupplyPoint, AssignResult, Material_store, Dis_ST)
        ga.pop = ga.exchange(AssignResult, ga.pop)
        SupplySequence, DemandSequence = ga.translateDNA(ga.pop, SupplyPoint, AssignResult)
        Fitness, OperationTime = ga.get_fitness(SupplySequence, DemandSequence, DeliverTime, InitLocation)
        temporary = np.argmin(Fitness)
        ga.pop[temporary] = Best_invidual #用目前最好的个体替换掉交叉后最差的个体
        SupplyPoint[temporary] = Best_supply
################################# Ends #################################
################################# Print Result #################################
    FT=time.perf_counter()
    print('Calculation Time is','%f' %(FT - ST),'s')
    print('The Best Solution is ', '%.2f'% BF[generation], 'min')
    print(29 * '-', ' Solution ', 29 * '-')
    for k in range(TowerNum):
        Result = []
        Sequence = Best_invidual[k]
        supply   = Best_supply[k]
        demand   = AssignResult[k, :, 0]
        supply_re= supply[Sequence]
        demand_re= demand[Sequence]
        Ini = InitLocation[k]
        Result.append('D%d'%Ini)
        for i in range(N_CITIES):
            if supply_re[i] != Spoint:
                Result.append('S%d'%supply_re[i])
                Ini = demand_re[i]
                Result.append('D%d'%Ini)
        print('Scheduling of Tower Crane', k+1, 'is')
        print( Result)
        print(70 * '-')
################################# Plot Result #################################    
    plt.figure()
    plt.plot(np.arange(N_GENERATIONS)+1, BF, label = 'max', color = 'black',linewidth = 1)
    plt.plot(np.arange(N_GENERATIONS)+1, MF, label = 'mean', linestyle='--',color = 'gray',linewidth = 1)
    plt.xlim(0, N_GENERATIONS)
    plt.xlabel('Iterations')
    plt.ylabel('Total transport time')
    plt.title('Genetic algorithm process')
    plt.legend(loc = 0, ncol = 2)
    # plt.savefig('result.png', dpi=500)