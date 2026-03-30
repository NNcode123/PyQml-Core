
import os 
import numpy as np
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
print(pyq.float32)
li2 = [i*0.2 for i in range(0,100)]
li3 = [i for i in range(0,1000000)]
li1 = [1,10,10]
li4 = [100,100,100]
#li3 = [j*2/10 for j in range(0, 100000)]


#tens_list = pyq.Tensor(li3, li4)
#print(tens_list)

#pyqml tensor and astype funcs 
tens = pyq.arange(0,40,.2).astype(int)
tens = tens 
print(tens)
print(tens.dtype)

tens_1 = pyq.Tensor(li3,dim = [100,100,100], type = pyq.int64)
tens_1 = tens_1.astype(float)
tens_3 = tens+tens_1
print(tens_1.dtype)
print(tens_3.dtype)
print(tens_3)

tens_4 =pyq.arange(0,50,1)
print(tens_4.dtype)
print(tens_4)

#tens_3 = tens+tens_1
#print(tens.shape)
#print(tens.dtype)



def add():
    tens_2 = pyq.arange(0,1000000,1) + pyq.arange(0,1000000,1)
  
    #tens_4 = tens*tens_1
    #tens_5 = tens/tens_1
    
    
 


def add_numpy():
    res_tens = np.arange(0,1000000) + np.arange(0,1000000)


    


time_block("add", add)
time_block("numm",add_numpy)



