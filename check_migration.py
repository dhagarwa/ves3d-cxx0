#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 04:47:39 2018

@author: dhwanit
"""
sh_order = 32
nodes = 2*sh_order*(sh_order +1)
path = "/Users/dhwanit/Google Drive/Biros Research/ves3d-cxx/results/set_exp2/"
for i in range(100):
    filename = path + "exp2_ves_32_" + str(i) + ".txt"
    k = open(filename,'r')
    lines = k.readlines()     #i am reading lines here
    #print(len(lines))
    mean = 0
    for line in lines[nodes+1:2*nodes +1]:            #taking each line
        conv_int = float(line)         #converting string to int
        mean += conv_int
        
    mean = mean/nodes
    print(mean)