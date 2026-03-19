
import os 
import timeit

import time

def time_block(name, fn):
    t0 = time.perf_counter()
    for i in range (0, 1000):
        out = fn()   
    t1 = time.perf_counter()
    print(f"{name}: {(t1 - t0)/1000:.6f} sec")
    return out




os.add_dll_directory(r"C:\msys64\mingw64\bin")

import pyqmlcore as pyq
li2 = [i for i in range(0,1000)]
li1 = [10,10,100]
li3 = [j*2 for j in range(0, 1000)]

tens = pyq.Tensor(data = li2,dim = li1, type = pyq.float64)
tens_1 = pyq.Tensor(data = li3,dim = li1, type = pyq.float32)

def add():
    tens_2 = tens+tens_1
    #print(tens_2)

time_block("add", add)

print(pyq.float32)

