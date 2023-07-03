    # -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 10:19:36 2020

@author: liwenjie
"""

import numpy as np
from  Assign import TransTime, Judgement#, Optimize
import matplotlib.pyplot as plt
import time
from pylab import *

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

Spoint  = 7+1
Dpoint  = 14+1
TCpoint = 4+1
MaNum   = 4+1
Requests= 20+1
JibRadius = 70
Dis_ST, DeliverTime = TransTime()
# print(Dis_ST)
# AssignResult = Optimize()
ST = time.perf_counter()
RequestDetail = np.array([[1,2], [3,4], [12,3], [5,3], [2,4],
                          [2,1], [11,1], [10,4], [3,3], [8,2], 
                          [10,1], [10,3],[14,4],[11,3], [1,2], 
                          [1,1], [6,4], [5,2], [10,2], [7,4], 
                          [0,0]])
List = [[1,4,15,16,18], [5,6,10,20], [2,8,9,11,12,17,19], [3,7,13,14]]
MaxLen = max(len(x) for x in List)
NewList = np.array(list(map(lambda l:l + [Requests]*(MaxLen - len(l)), List)), dtype = np.int16)
AssignResult = RequestDetail[NewList - 1]
Index = []
Set = [[3,7], [1,4],[6,10]] #任务优先级集合
for priority in Set:
    tempory_set = []
    for d in priority:
        location = np.where(NewList == d)
        tempory_set.append(location)
    Index.append(tempory_set)
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

#''''''''''''''''''''''''''优化前''''''''''''''''''''''''''#
    # def assign_supply(self, deliver_time, dis_st, material_store, requests):
    #     supply_chain = np.empty_like(self.pop, dtype = np.int16)
    #     for k, request in enumerate(requests):
    #         for j, detail in enumerate(request):   
    #             temp_supply = []
    #             temp_time = []
    #             if detail[1] == 0: #如果是虚拟运输任务，则分配虚拟物料供应点
    #                 temp_supply.append(Spoint)
    #             else:
    #                 for i in material_store:                 
    #                     if detail[1] == i[1] and dis_st[k, i[0]-1] <= JibRadius:  #是否存储相应材料
    #                         temp_supply.append(i[0]) #挑出存储有所需材料的供应点
    #             if len(temp_supply) > 1:
    #                 for i in temp_supply:
    #                     temp_time.append(deliver_time[k, i-1, detail[0]-1]) #计算材料供应所需时间
    #                 posibility = 100 / np.array(temp_time) 
    #                 supply = np.random.choice(temp_supply, size=self.pop_size, replace=True, p=posibility / posibility.sum()) #距离最近的供应方案具有较大的概率被选为供应点
    #                 supply_chain[:, k, j] = supply
    #             else :
    #                 supply_chain[:, k, j] = temp_supply
    #     return supply_chain
    
#''''''''''''''''''''''''''优化后''''''''''''''''''''''''''#
    def assign_supply(self, deliver_time, dis_st, material_store, requests):
        supply_chain = np.empty_like(self.pop, dtype=np.int16)
        is_virtual = (requests[:, :, 1] == 0)
        temp_supply = [[] for _ in range(requests.shape[0])]
        temp_time = [[] for _ in range(requests.shape[0])]
        for i, j, k in np.argwhere(~is_virtual):
            detail = requests[i, j, :]
            for l, (m, n) in enumerate(material_store):
                if detail[1] == n and dis_st[i, m-1] <= JibRadius:
                    temp_supply[i].append(m)
                    temp_time[i].append(deliver_time[i, m-1, detail[0]-1])
            if len(temp_supply[i]) > 1:
                posibility = 100 / np.array(temp_time[i])
                supply = np.random.choice(temp_supply[i], size=self.pop_size, replace=True, p=posibility / posibility.sum())
                supply_chain[:, i, j] = supply
            else:
                supply_chain[:, i, j] = temp_supply[i][0]
        for i, j in np.argwhere(is_virtual):
            supply_chain[:, i, j] = Spoint
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
                    DNA[i, k] = np.append(real, np.arange(real_se.shape[0], De.shape[1]))   #补全任务排序 
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
################################# Calculate the waiting time of each Individual #################################
    def waiting(self, supply_point, demand_point, deliver_time, starttime):
        Time_in = np.zeros((Overlapscenario.shape[0], self.tower_num, 2 * self.DNA_size)) + 300 #Start_time(OVerlapping area, Tower crane, Time in and out, path)
        Time_out = np.zeros((Overlapscenario.shape[0], self.tower_num, 2 * self.DNA_size)) + 300 #Start_time(OVerlapping area, Tower crane, Time in and out, path)
        waitingtime = np.zeros_like(starttime)
        for k in np.arange(self.tower_num): #塔吊编号
            path, p = [], []
            demand = InitLocation[k]
            path.append('D%d'%demand)
            p.append(demand)
            for x in np.arange(self.DNA_size): #基因编号
                if supply_point[k, x] != Spoint:
                    supply = supply_point[k, x]
                    path.append('S%d'% supply)
                    p.append(supply)
                    demand = demand_point[k, x]
                    path.append('D%d'%demand)
                    p.append(demand)
            for o in np.arange(1, Overlapscenario.shape[0]+1):
                if Time_AB[k+1,o] != 0:
                    outpoint = []
                    for i, d in enumerate(path):
                        if d not in Overlap[o-1]:
                            outpoint.append(i)
                    start = 0
                    while(start != len(path) - 1): 
                        inside, time_in, time_out=  0, 0, 0
                        if start in outpoint: #起始点不在干涉区域内
                            end = start + 1
                            while(end not in outpoint): #寻找下一个在干涉区外的点
                                if end != len(path) - 1:
                                    end += 1
                                else:
                                    break
                            # print(start, end)
                            if (end - start) > 1: #吊臂进入干涉区域并停留
                                if start % 2 == 1: #start是supply
                                    time_in = starttime[k, start] + sum(Path_A[ p[start], p[start +1], k+1, o,point] * Time_SC[point, p[start],k+1] for point in range(2*o-1, 2*o+1))
                                    inside += (sum(Path_A[ p[start], p[start +1],k+1, o,point] * Time_DC[point, p[start+1],k+1] for point in range(2*o-1, 2*o+1)) + 1)
                                else:#start是demand
                                    time_in = starttime[k, start] + sum(Path_B[ p[start], p[start +1], k+1, o,point] * Time_DC[point, p[start],k+1] for point in range(2*o-1, 2*o+1))
                                    inside += (sum(Path_B[ p[start], p[start+1], k+1, o, point] * Time_SC[point, p[start+1],k+1] for point in range(2*o-1, 2*o+1)) +1)
                                for i in range(1, end-start-1):
                                    if (start + i) % 2 == 1:
                                        inside += (deliver_time[k, p[start+i] -1, p[start+i+1]-1] + 1)
                                    else:
                                        inside += (deliver_time[k, p[start+i+1] -1, p[start+i]-1] + 1)
                                if end == len(path) - 1 and end not in outpoint:
                                    inside += 100
                                else:
                                    if end % 2 == 0: #end是demand, (end-1)则为supply
                                        inside += sum(Path_A[ p[end-1], p[end],k+1,o,point] * Time_SC[point, p[end-1],k+1] for point in range(2*o-1, 2*o+1))
                                    else: #end是supply, (end-1)则为demand
                                        inside += sum(Path_B[ p[end-1], p[end],k+1,o,point] * Time_DC[point, p[end-1],k+1] for point in range(2*o-1, 2*o+1))
                                time_out = time_in + inside 
                            else: #判断吊臂是否经过干涉区域
                                if start % 2 == 0: #start是demand点
                                    if Path[p[end], p[start],k+1,o] == 1:
                                        time_in = starttime[k, start] + sum(Path_B[ p[start], p[end], k+1, o, point] * Time_DC[point,p[start], k+1] for point in range(2*o-1, 2*o+1))
                                        if end == len(path) - 1 and end not in outpoint:
                                            time_out = time_in + 100
                                        else:
                                            time_out = time_in + Time_AB[k+1, o]
                                    else:
                                        time_in = 300.0
                                        time_out = 300.0
                                else:#start是supply点
                                    if Path[ p[start], p[end], k+1,o] == 1:
                                        time_in = starttime[k, start] + sum(Path_A[ p[start], p[end], k+1, o, point] * Time_SC[point, p[start], k+1] for point in range(2*o-1, 2*o+1))
                                        if end == len(path) - 1 and end not in outpoint:
                                            time_out = time_in + 100
                                        else:
                                            time_out = time_in + Time_AB[k+1, o]
                                    else:
                                        time_in = 300.0
                                        time_out = 300.0
                        else: #起始点在干涉区域内
                            end = start + 1
                            while(end not in outpoint):
                                if end != len(path) - 1:
                                    end += 1
                                else:
                                    break
                            for i in range(1, end-start):
                                if i % 2 == 1:
                                    inside += (deliver_time[k, p[start+i] -1, p[start+i-1]-1] + 1)
                                else:
                                    inside += (deliver_time[k, p[start+i-1] -1, p[start+i]-1] + 1)
                            if end == len(path) - 1 and end not in outpoint:
                                time_out = time_in + inside + 100
                            else:
                                if end % 2 == 0: #end是demand点
                                    time_out = time_in + inside + sum(Path_A[p[end-1], p[end], k+1, o, point] * Time_SC[point, p[end-1], k+1] for point in range(2*o-1, 2*o+1))
                                else: #end是supply点
                                    time_out = time_in + inside + sum(Path_B[p[end-1], p[end], k+1, o, point] * Time_DC[point, p[end-1], k+1] for point in range(2*o-1, 2*o+1))
                        Time_in[o-1, k, start] = time_in
                        Time_out[o-1, k, start] = time_out
                        start = end
        a = np.min(Time_in) #按照进入干涉区域时间先后判断
        while(a < 300):
            b = np.where(Time_in == a)
            area, crane, point = b
            for i in range(len(area)):
                in1, out1 = a, Time_out[area[i], crane[i], point[i]]
                for x in Overlapscenario[area[i]]:
                    if x != crane[i]:
                        k2 = x #寻找同一干涉区域的相邻塔吊
                in2, out2 = Time_in[area[i], k2, :], Time_out[area[i], k2, :]
                dif2, dif1 = out2 - in1 + Buffer, out1 - in2 + Buffer
                coll1, coll2 = np.ravel(np.where(dif1 > 0)), np.ravel(np.where(dif2 > 0))
                colloc = list(set(coll1) & set(coll2)) #判断差值都大于0的位置
                if 0 < len(colloc) <= 1:#判断是否有碰撞发生
                    waiting = min(dif2[colloc[0]], dif1[colloc[0]])
                    if in1 > in2[colloc]: #后进让先进
                        waitingtime[crane[i], point[i]]  += waiting
                        starttime[crane[i], point[i]:]  += waiting
                        Time_in[:, crane[i], point[i]:]  += waiting
                        Time_out[:, crane[i], point[i]:] += waiting
                    else:
                        waitingtime[k2, colloc[0]]  += waiting
                        starttime[k2, colloc[0]:]  += waiting
                        Time_in[:, k2, colloc[0]:]  += waiting
                        Time_out[:, k2, colloc[0]:] += waiting
                elif 1 < len(colloc):
                    j = min(j  for j in colloc)
                    waiting = min(dif2[j], dif1[j])
                    if in1 < in2[j]: #后进让先进
                        waitingtime[crane[i], point[i]]  += waiting
                        starttime[crane[i], point[i]:]  += waiting
                        Time_in[:, crane[i], point[i]:]  += waiting
                        Time_out[:, crane[i], point[i]:] += waiting
                    else:
                        waitingtime[k2, j]  += waiting
                        starttime[k2, j:]  += waiting
                        Time_in[:, k2, j:]  += waiting
                        Time_out[:, k2, j:] += waiting
            Time_in[b] = 500
            a = np.min(Time_in)
        return waitingtime, starttime
################################# Check the priority of each Individual #################################
    def check_priority(self, individual, index, Start_Time, End_Time):
        PunishNumber = 0
        for pair in index:
            sequence_r1, sequence_r2 = individual[pair[0]], individual[pair[1]]
            crane_r1, crane_r2 = pair[0][0], pair[1][0]
            finishtime = End_Time[crane_r1, 2* sequence_r1]
            starttime = Start_Time[crane_r2, 2 *sequence_r2]
            if finishtime > starttime:
                PunishNumber += 1
        return PunishNumber
################################# Get the Fitness of each Individual #################################
    def get_fitness(self, supply_point, demand_point, deliver_time, startpoint, pop, index):
        Start_time = np.empty((self.pop_size, self.tower_num, 2 * self.DNA_size), dtype=np.float32)
        End_time = np.empty_like(Start_time)
        Waiting_time = np.zeros_like(Start_time)
        Single_time = np.empty((self.pop_size, self.tower_num), dtype=np.float32)
        Violation_Number = np.zeros(self.pop_size, dtype=np.int16)
        Time_unloading, Time_loading = 1, 1
        for a in range(self.pop_size): #个体编号
            for k in range(self.tower_num): #塔吊编号
                ST, FT = 0, 0
                HookLocation = startpoint[k] # 吊钩初始位置
                for x in range(self.DNA_size): #基因编号
                    if supply_point[a, k, x] != Spoint: 
                        s, d1, d2 =  supply_point[a, k, x], HookLocation, demand_point[a, k, x] 
                        StartTime1 = ST
                        StartTime2 = StartTime1 + deliver_time[k, s-1, d1-1] + Time_loading #Demand to Supply
                        ST = StartTime2 + deliver_time[k, s-1, d2-1] + Time_unloading  #Supply to Demand
                        Start_time[a, k, 2 * x] =  StartTime1
                        Start_time[a, k, 2 * x + 1 ] =  StartTime2
                        FT += (deliver_time[k, s-1, d2-1] + Time_unloading)*6 +  (deliver_time[k, s-1, d1-1] + Time_loading)*3
                        HookLocation = d2
                    else:
                        Start_time[a, k, 2 * x] =  ST
                        Start_time[a, k, 2 * x + 1 ] =  ST
                Single_time[a, k] = FT
            Waiting_time[a], Start_time[a] = self.waiting(supply_point[a], demand_point[a], deliver_time, Start_time[a])
            End_time[a, :, : -1] = Start_time[a, :, 1: ]
            End_time[a, :, 2 * self.DNA_size - 1] = Single_time[a, :]
            Violation_Number[a] = self.check_priority(pop[a], index, Start_time[a], End_time[a])
        total_wait = Waiting_time.sum(axis=2)
        total_time = np.sum(Single_time + 10*total_wait , axis=1) + 100*Violation_Number#总时间
        # total_time = Single_time.max(axis=1)  #运行时间最大值
        fitness = 1000 / total_time
        return fitness, total_time, total_wait.sum(axis=1)
################################# Tournament Selection #################################
    def select(self, fitness):
        idx = np.empty(self.pop_size, dtype = np.int16)
        for i in range(self.pop_size):
            tournament = np.random.randint(0, self.pop_size, 5) # 5为锦标赛规模
            fitness_tournament = fitness[tournament]
            best = np.argmax(fitness_tournament)
            idx[i] = tournament[best]
        # idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
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
    def crossover(self, parentA_Se, parentB_Se, parentA_Su, parentB_Su, mean_fitness, max_fitness, invidual_fitness):
        pop_childA = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        pop_childB = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        supply_childA = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        supply_childB = np.empty((self.tower_num, self.DNA_size), dtype= np.int16)
        if invidual_fitness >= mean_fitness : 
            for k in range(self.tower_num):
                if np.random.rand() <= self.cross_rate - self.cross_coefficent * ((max_fitness - invidual_fitness) / (max_fitness - mean_fitness)) :
                    Start_point = np.random.randint(0, self.DNA_size/2+1)
                    End_point = np.random.randint(self.DNA_size/2, self.DNA_size)
                    #Partially Mapped Crossover(Sequence)
                    pop_childA[k, :], pop_childB[k, :] = self.PMX(parentA_Se[k], parentB_Se[k], Start_point, End_point)
                    #Single point Crossover(Supply)
                    supply_childA[k, :], supply_childB[k, :] = self.SPC(parentA_Su[k], parentB_Su[k], Start_point, End_point)
                else:
                    pop_childA = parentA_Se
                    pop_childB = parentB_Se
                    supply_childA = parentA_Su
                    supply_childB = parentB_Su
        else:
            for k in range(self.tower_num):
                if np.random.rand() <= self.cross_rate :
                    Start_point = np.random.randint(0, (self.DNA_size)/2+1)
                    End_point = np.random.randint((self.DNA_size)/2, self.DNA_size)
                    #Partially Mapped Crossover(Sequence)
                    pop_childA[k, :], pop_childB[k, :] = self.PMX(parentA_Se[k], parentB_Se[k], Start_point, End_point)
                    #Single point Crossover(Supply)
                    supply_childA[k, :], supply_childB[k, :] = self.SPC(parentA_Su[k], parentB_Su[k], Start_point, End_point)
                else:
                    pop_childA = parentA_Se
                    pop_childB = parentB_Se
                    supply_childA = parentA_Su
                    supply_childB = parentB_Su
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
        mean_, max_ = np.mean(fitness), np.max(fitness)
        idx = self.select(fitness)
        pop = self.pop[idx] #挑选出优秀个体
        supply_chain = supply_chain[idx] #挑选出优秀个体（Supply）
        best_individual = np.argmax(fitness) #得到适应度最高个体的编号
        parentASe, parentASu = pop[best_individual], supply_chain[best_individual]
        for z in range(self.pop_size): #挑选的父代A
            #Crossover 
            parentBSe = pop[z]
            parentBSu = supply_chain[z]
            invidual_ = fitness[z]
            child_popA, child_popB, child_supA, child_supB = self.crossover(parentASe, parentBSe, parentASu, parentBSu, mean_, max_, invidual_)
            #sequence mutate
            pop[best_individual] = self.mutate(child_popA)
            pop[z] = self.mutate(child_popB)
            #supply mutate
            supply_chain[best_individual] = self.supply_mutate(child_supA, request, material_store, dis_st)
            supply_chain[z] = self.supply_mutate(child_supB, request, material_store, dis_st)
        self.pop = pop

################################# Improved Genetic Algorithm with Elitist Preservation #################################
if __name__ == '__main__':
################################# Algorithm Parameters #################################
    N_CITIES = AssignResult.shape[1]  # DNA size
    CROSS_RATE = 0.85
    CROSS_COEFFICENT = 0.4
    MUTATE_RATE = 0.018
    POP_SIZE = 30
    N_GENERATIONS = 50
    TowerNum = AssignResult.shape[0]
################################# Construction Site Parameters #################################
    Material_store = np.array([[1,3],[1,4],[2,1],[2,2],[3,1],[3,3],[3,4],[4,2],[5,1],[5,4],[6,2],[6,3],[7,1],[7,2]]) #Material存储信息
    InitLocation = np.array([5, 2, 6, 11])
    Overlapscenario = np.array([[0,1], [0,2], [1,2], [1,3]])
    Overlap = [['D2', 'D3'], ['D3', 'D4'], ['D3', 'D6'], ['S3', 'S4']]
    Time_SC, Time_DC, Time_AB, Path_A, Path_B, Path = Judgement()
    Buffer = 0.25
################################# Get The Elite Individual #################################
    Best_invidual = np.empty((AssignResult.shape[0], AssignResult.shape[1]), dtype = np.int16)
    Best_supply = np.empty((AssignResult.shape[0], AssignResult.shape[1]), dtype = np.int16)
    Best_fitness = 0
    BF = []
    MF = []
    WT = []
################################ Starts #################################
    print(30 * '-', ' Start ', 30 * '-')
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE, cross_coefficent=CROSS_COEFFICENT, tower_num = TowerNum)
    ga.pop = ga.exchange(AssignResult, ga.pop)
    SupplyPoint = ga.assign_supply(DeliverTime, Dis_ST,Material_store, AssignResult)
    for generation in range(N_GENERATIONS):
        SupplySequence, DemandSequence = ga.translateDNA(ga.pop, SupplyPoint, AssignResult)
        Fitness, OperationTime, WaitingTime = ga.get_fitness(SupplySequence, DemandSequence, DeliverTime, InitLocation, ga.pop, Index)
        best_idx = np.argmax(Fitness) #最优的适应度值
        BF.append(OperationTime[best_idx])
        WT.append(WaitingTime[best_idx])
        mean_fitness = np.mean(OperationTime)#平均的适应度值
        MF.append(mean_fitness)
        if np.max(Fitness) > Best_fitness: #获取目前为止最优的结果
            Best_fitness = np.max(Fitness)
            Best_invidual = ga.pop[best_idx]
            Best_supply = SupplyPoint[best_idx]
        print('Gen:', generation, '| best fit: %.2f' % OperationTime[best_idx],)
        print('Best sequence', ga.pop[best_idx])
        print('Best Supply', SupplyPoint[best_idx])
        print('-----------------------------')
        ga.evolve(Fitness, SupplyPoint, AssignResult, Material_store, Dis_ST)
        ga.pop = ga.exchange(AssignResult, ga.pop)
        SupplySequence, DemandSequence = ga.translateDNA(ga.pop, SupplyPoint, AssignResult)
        Fitness, OperationTime, WaitingTime = ga.get_fitness(SupplySequence, DemandSequence, DeliverTime, InitLocation, ga.pop, Index)
        temporary = np.argmin(Fitness)
        ga.pop[temporary] = Best_invidual #用目前最好的个体替换掉交叉后最差的个体
        SupplyPoint[temporary] = Best_supply
################################# Ends #################################
################################# Print Result #################################
    FT=time.perf_counter()
    print('Calculation Time is','%f' %(FT - ST),'s')
    print('The Best Solution is ', '%.2f'% BF[generation], 'min')
    print('The waiting time is', '%.2f'% WT[generation], 'min')
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
    plt.plot(np.arange(N_GENERATIONS)+1, BF, label = '各代最优',linewidth = 1)
    plt.plot(np.arange(N_GENERATIONS)+1, MF, label = '各代平均', linestyle='--',linewidth = 1)
    # plt.plot(np.arange(N_GENERATIONS)+1, WT, label = '等待时间', linestyle='-.',linewidth = 1)
    plt.xlim(0, N_GENERATIONS)
    plt.xlabel('迭代次数')
    plt.ylabel('运输费用/CNY')
    plt.title('改进遗传算法计算过程')
    plt.legend(loc = 0, ncol = 2)
    plt.show()
    # plt.savefig('result.png', dpi=500)