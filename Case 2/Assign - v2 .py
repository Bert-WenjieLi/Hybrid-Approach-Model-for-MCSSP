# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:15:14 2020
@author: liwenjie
"""

# from gurobipy import *
import numpy as np
import xlrd
import time 

ST = time.time()
################################# Construction Parameters #################################
Spoint  = 7+1
Dpoint  = 14+1
TCpoint = 4+1
MaNum   = 4+1
Requests= 20+1
Cpoint=8+1
Sequence=4+1
OverZone=4+1
################################# Tower Crane Parameters #################################
Vr = 60
Vw = 0.5
Vh = 136
Alpah = 0
Beta = 1
SafetyHight = 2 * 1.5
JibRadius = 70

################################# Operation Time Calculation #################################
#Read Construction Site Data
data=xlrd.open_workbook(r'4 tower crane locations_new.xls')
supply=data.sheet_by_name('supply')
demand=data.sheet_by_name('demand')
crane=data.sheet_by_name('crane')

def Distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

#supply positions data
sx={}
sy={}
sz={}
for i in np.arange(1,Spoint):
    sx[i], sy[i], sz[i] = supply.row(i)[1].value, supply.row(i)[2].value, supply.row(i)[3].value

#demand positions data
dx={}
dy={}
dz={}
Cor_d = np.empty((Dpoint-1, 3))
for j in np.arange(1,Dpoint):
    dx[j], dy[j], dz[j] = demand.row(j)[1].value, demand.row(j)[2].value, demand.row(j)[3].value
    Cor_d[j-1, 0], Cor_d[j-1, 1],  Cor_d[j-1, 2] = dx[j], dy[j], dx[j]

def Coordinate_Demand():
    return Cor_d
    
#crane positions data
cx={}
cy={}
cz={}
for k in np.arange(1,TCpoint):
    cx[k], cy[k], cz[k] = crane.row(k)[1].value, crane.row(k)[2].value, crane.row(k)[3].value

Dis_STC={}
Dis_DTC={}
Dis_SD={}
Time_sd_r={}
Time_sd_w={}
Time_sd_co={}
Time_sd_v={}
Time_sd_run={}
DeliverTime = np.empty((TCpoint-1, Spoint, Dpoint), dtype=np.float32)      #(TowerCrane, SupplyPoint, DemandPoint)
Dis_sd = np.empty((TCpoint-1, Spoint-1), dtype=np.float64)   #(TowerCrane, SupplyPoint)
for i in range(1,Spoint):
    for j in range(1,Dpoint):
        for k in range(1,TCpoint): 
            Dis_STC[i,k] = Distance(sx[i],sy[i],cx[k],cy[k])
            Dis_sd[k-1, i-1] = Dis_STC[i,k]
            Dis_DTC[j,k] = Distance(dx[j],dy[j],cx[k],cy[k])
            Dis_SD[i,j] = Distance(dx[j],dy[j],sx[i],sy[i]) 
            Time_sd_r[i,j,k] = np.abs(Dis_STC[i,k]-Dis_DTC[j,k]) / Vr
            Time_sd_w[i,j,k] = np.arccos(-(Dis_SD[i,j]**2-Dis_DTC[j,k]**2-Dis_STC[i,k]**2)/(2*Dis_STC[i,k]*Dis_DTC[j,k])) / Vw
            Time_sd_co[i,j,k] = np.maximum(Time_sd_r[i,j,k],Time_sd_w[i,j,k]) + Alpah * np.minimum(Time_sd_r[i,j,k],Time_sd_w[i,j,k])
            Time_sd_v[i,j,k] = np.abs(sz[i]-dz[j] + SafetyHight) / Vh
            Time_sd_run[i,j,k] = np.maximum(Time_sd_co[i,j,k],Time_sd_v[i,j,k]) + Beta * np.minimum(Time_sd_co[i,j,k],Time_sd_v[i,j,k])
            DeliverTime[k-1, i-1, j-1] = Time_sd_run[i,j,k]
            # if i==3 and j==14 and k==4:
            #     print(i,j,k,Time_sd_run[i,j,k])
            #DeliverTime[k-1, Spoint, Dpoint] = 100
#print(DeliverTime)            
def TransTime():
    return Dis_sd, DeliverTime

################################# Requests Assignment Model #################################
#Material Store
MaStore={}
for i in range(1,Spoint):
    for m in range(1,MaNum):
        MaStore[i,m]=0
MaStore[1,3]=1
MaStore[1,4]=1
MaStore[2,1]=1
MaStore[2,2]=1
MaStore[3,1]=1
MaStore[3,3]=1
MaStore[3,4]=1
MaStore[4,2]=1
MaStore[5,1]=1
MaStore[5,4]=1
MaStore[6,2]=1
MaStore[6,3]=1
MaStore[7,2]=1
MaStore[7,1]=1

#Material Demand(random)
# np.random.seed(5)
ReNumber = np.random.permutation(np.arange(1, Requests))
ReDemand = np.random.randint(1, Dpoint, Requests-1)
ReMaType = np.random.randint(1, MaNum , Requests-1)
MaDe={}
for r in range(1,Requests):
    for j in range(1,Dpoint):
        for m in range(1,MaNum):
            MaDe[r,j,m]=0
for r in range(1,Requests):
    MaDe[ReNumber[r-1], ReDemand[r-1], ReMaType[r-1]] = 1
# for r in range(1,Requests):
#     for j in range(1,Dpoint):
#         for m in range(1,MaNum):
#             if MaDe[r,j,m]!=0:
#                 print(r,j,m)
RequestDetail = np.empty((Requests, 2), dtype = np.int16) #物料运输任务
for j in range(1,Dpoint):
    for m in range(1,MaNum):
        for r in range(1,Requests):
            if MaDe[r,j,m] == 1:
                RequestDetail[r-1, 0],  RequestDetail[r-1, 1]= j, m
RequestDetail[Requests-1, 0],  RequestDetail[Requests-1, 1] = 0, 0

#Creat the requests assignment model
ma = Model('MaterialRequestsAssigement')

MaSu={} #选择供应位置i运输物料m至需求位置j，完成任务r
for i in range(1,Spoint):
    for j in range(1,Dpoint):
        for m in range(1,MaNum):
            for r in range(1,Requests):
                MaSu[r,i,j,m] = ma.addVar(vtype = GRB.BINARY, name = "x1")
ReSe={} #任务r被分配至塔吊k
for r in range(1,Requests):
    for k in range(1,TCpoint):
        ReSe[r,k] = ma.addVar(vtype = GRB.BINARY, name = "x2")
SuDe={} #塔吊k将物料m由供应位置i运输至需求位置j完成任务r
for i in range(1,Spoint):
    for j in range(1,Dpoint):
        for m in range(1,MaNum):
              for k in range(1,TCpoint):
                for r in range(1,Requests):
                    SuDe[r,i,j,m,k] = ma.addVar(vtype = GRB.BINARY, name = "x3")
Time = ma.addVar(vtype=GRB.CONTINUOUS, name = "y")
ma.update

for r in range(1,Requests):
    ma.addConstr(quicksum(ReSe[r,k] for k in np.arange(1,TCpoint)) == 1)
    for j in range(1,Dpoint):
        for m in range(1,MaNum):
            for k in np.arange(1,TCpoint):
                ma.addConstr(quicksum(SuDe[r,i,j,k,m] for i in np.arange(1,Spoint)) <= 1)
                for i in range(1,Spoint):
                    ma.addConstr( 3 - ReSe[r,k] - MaDe[r,j,m] - MaStore[i,m] >= 1 - SuDe[r,i,j,k,m])
                    ma.addConstr(SuDe[r,i,j,m,k] * (Dis_STC[i,k] - JibRadius) <= 0)
                    ma.addConstr(SuDe[r,i,j,m,k] * (Dis_DTC[j,k] - JibRadius) <= 0)

# for k in range(1,TCpoint):
#     ma.addConstr(quicksum(ReSe[r,k] for r in np.arange(1,Requests)) == 6)

for k in range(1,TCpoint):
    ma.addConstr(Time >= quicksum(SuDe[r,i,j,m,k] * Time_sd_run[i,j,k] for r in np.arange(1,Requests) for i in np.arange(1,Spoint) for j in np.arange(1,Dpoint) for m in np.arange(1,MaNum)))

ma.setObjective(Time, GRB.MINIMIZE)

def Optimize():
    ma.Params.OutputFlag = 0 #是否输出求解过程,1为输出，0为不输出
    ma.optimize()
    if ma.status == GRB.status.OPTIMAL:
        List = []
        for k in range(1,TCpoint):
            Listk = []
            for r in range(1,Requests):
                if ReSe[r,k].x == 1:
                    Listk.append(r)
            List.append(Listk)  #将任务编号分配给相应塔吊
    MaxLen = max(len(x) for x in List)
    NewList = np.array(list(map(lambda l:l + [Requests]*(MaxLen - len(l)), List)), dtype = np.int16)#补全任务数量，转化为array，虚拟任务用物料种类m=0代替
    AssignResult = RequestDetail[NewList - 1] #将总物料运输任务分配给各塔吊
    #挑选出含有优先级限制的任务的位置
    Set = [[3,7], [2,4],[8,15]] #任务优先级集合
    index = []
    for priority in Set:
        tempory_set = []
        for d in priority:
            location = np.where(NewList == d)
            tempory_set.append(location)
        index.append(tempory_set)
    return AssignResult, index

################################# Transportation Path judgement Model #################################
ax={}
ay={}
for p in range(1,Cpoint):
    ax[p], ay[p]=demand.row(p)[6].value, demand.row(p)[7].value
##if the operate path will entire the overlap space
def Judgement():
    Dis_CTC, Dis_CS, Dis_CD={}, {}, {}
    Time_SC, Time_DC = {}, {}
    for i in range(1,Spoint):
        for j in range(1,Dpoint):
            for k in range(1,TCpoint):
                for  p in range(1,Cpoint):
                    Dis_CTC[p,k] = Distance(ax[p],ay[p],cx[k],cy[k]) #distance of tower crane and point A or B
                    Dis_CS[p,i] = Distance(ax[p],ay[p],sx[i],sy[i])  #distance of point A or B and supply point
                    Dis_CD[p,j] = Distance(ax[p],ay[p],dx[j],dy[j])  #distance of point A or B and demand point
                    #operate time from supply point to point A or B
                    Time_SC[p,i,k] = np.arccos(-(Dis_CS[p,i]**2-Dis_CTC[p,k]**2-Dis_STC[i,k]**2)/(2*Dis_STC[i,k]*Dis_CTC[p,k]))/0.5
                    Time_DC[p,j,k] = np.arccos(-(Dis_CD[p,j]**2-Dis_CTC[p,k]**2-Dis_DTC[j,k]**2)/(2*Dis_DTC[j,k]*Dis_CTC[p,k]))/0.5 
    Time_AB={}
    for k in range(1,TCpoint):
        for o in range(1,OverZone):
            Time_AB[k,o]=0
    Time_AB[1,1]=2.72
    Time_AB[2,1]=2.72
    Time_AB[1,2]=2.85
    Time_AB[3,2]=2.85
    Time_AB[2,3]=2.93
    Time_AB[3,3]=2.93
    Time_AB[2,4]=3.60
    Time_AB[4,4]=3.60
    #D在干涉区域
    Path_C={}
    for j in range(1,Dpoint):
        for o in range(1,OverZone):
            Path_C[j,o]=0
    Path_C[2,1]=1
    Path_C[3,1]=1
    Path_C[3,2]=1
    Path_C[3,3]=1
    Path_C[4,2]=1
    Path_C[6,3]=1
    #S在干涉区域
    Path_D={}
    for i in range(1,Spoint):
        for o in range(1,OverZone):
            Path_D[i,o]=0
    Path_D[3,4]=1
    Path_D[4,4]=1
            
    Time_P1={}
    Time_P2={}
    for i in range(1,Spoint):
        for j in range(1,Dpoint):
            for k in range(1,TCpoint):
                for o in range(1,OverZone):
                    if Time_AB[k,o]!=0 :#and Dis_STC[i,k]<=70 and Dis_DTC[j,k]<=70:
                        p = 2 * o - 1
                        if Path_C[j,o]==1 or Path_D[i,o]==1:
                            Time_P1[i,j,k,o] = Time_SC[p,i,k]+Time_DC[p,j,k]
                            Time_P2[i,j,k,o] = Time_SC[p+1,i,k]+Time_DC[p+1,j,k]
                        else:
                            Time_P1[i,j,k,o] = Time_SC[p,i,k]+Time_DC[p+1,j,k]+Time_AB[k,o]
                            Time_P2[i,j,k,o] = Time_SC[p+1,i,k]+Time_DC[p,j,k]+Time_AB[k,o]
    Path_A, Path_B, Path={}, {}, {}
    for i in range(1,Spoint):
        for j in range(1,Dpoint):
            for k in range(1,TCpoint):
                for o in range(1,OverZone):
                    if Time_AB[k,o] != 0:#and Dis_STC[i,k]<=70 and Dis_DTC[j,k]<=70:
                        p = 2 * o - 1
                        #正向运动 S2D
                        if Time_P1[i,j,k,o]<=Time_sd_run[i,j,k]:
                            Path_A[i,j,k,o,p] = 1
                        else:
                            Path_A[i,j,k,o,p] = 0
                        if Time_P2[i,j,k,o]<=Time_sd_run[i,j,k]:
                            Path_A[i,j,k,o,p+1] = 1
                        else:
                            Path_A[i,j,k,o,p+1] = 0
                        #反向运动 D2S
                        if Path_C[j,o]==1 or Path_D[i,o]==1:
                            Path_B[j,i,k,o,p],  Path_B[j,i,k,o,p+1] = Path_A[i,j,k,o,p],  Path_A[i,j,k,o,p+1]
                        else:
                            Path_B[j,i,k,o,p],  Path_B[j,i,k,o,p+1] = Path_A[i,j,k,o,p+1],  Path_A[i,j,k,o,p]
    for i in range(1,Spoint):
        for j in range(1,Dpoint):
            for k in range(1,TCpoint):
                for o in range(1,OverZone):
                    p = 2*o-1
                    if Time_AB[k,o]!=0 :#and Dis_STC[i,k]<=70 and Dis_DTC[j,k]<=70:
                        if Path_A[i,j,k,o,p]==1 or Path_A[i,j,k,o,p+1]==1:
                            Path[i,j,k,o]=1
                        elif Path_C[j,o]==1 and Path_D[i,o]==1:
                            Path[i,j,k,o]=1
                        else:
                            Path[i,j,k,o]=0   
    return Time_SC, Time_DC, Time_AB, Path_A, Path_B, Path     

