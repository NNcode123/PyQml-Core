
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


tens_1 = pyq.Tensor(li3,dim = [100,100,100], type = pyq.int64)
tens = pyq.Tensor(li2, li1)
numpy_tensor = pyq.to_numpy(tens_1)


print(numpy_tensor)

tens_3 = tens+tens_1
tens_3 = tens*tens_1

tensqy = tens+tens
print(tens)
print(tensqy)



print(tens.shape)
print(tens.dtype)





 
def add():
    tens_2 = pyq.arange(0,1000000,1) + pyq.arange(0,1000000,1)
  
    #tens_4 = tens*tens_1
    #tens_5 = tens/tens_1


def add_numpy():
    res_tens = np.arange(0,1000000) + np.arange(0,1000000)


    


time_block("add", add)
time_block("numm",add_numpy)



