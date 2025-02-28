#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import pandas as pd
import gc


# In[2]:


t0=time.perf_counter();
with open("dac_var.txt", "r") as file:
    lines = file.readlines()
dac_var = [list(map(float, line.split())) for line in lines if line.strip()]   
with open("msd_var.txt", "r") as file:
    lines = file.readlines()
msd_var = [list(map(float, line.split())) for line in lines if line.strip()] 
with open("len_var.txt", "r") as file:
    lines = file.readlines()
len_var = [list(map(float, line.split())) for line in lines if line.strip()] 
with open("parameters.txt", "r") as file:
    lines = file.readlines()
parameters = [list(map(float, line.split())) for line in lines if line.strip()] 
t1=time.perf_counter();
print("loading took ",t1-t0," seconds");
del lines
gc.collect();


# In[3]:


sim_var = pd.DataFrame({
    "dac_var": dac_var,
    "msd_var": msd_var,
    "len_var": len_var,
    "parameters": parameters
})
sim_var.to_pickle("sim_var.pkl");


# In[4]:


del msd_var,dac_var,len_var,sim_var
gc.collect()


# In[5]:


t0=time.perf_counter();
with open("dac.txt", "r") as file:
    lines = file.readlines()
dac = [list(map(float, line.split())) for line in lines if line.strip()]   
with open("msd.txt", "r") as file:
    lines = file.readlines()
msd = [list(map(float, line.split())) for line in lines if line.strip()] 
with open("len.txt", "r") as file:
    lines = file.readlines()
len = [list(map(float, line.split())) for line in lines if line.strip()] 
with open("parameters.txt", "r") as file:
    lines = file.readlines()
parameters = [list(map(float, line.split())) for line in lines if line.strip()] 
t1=time.perf_counter();
print("loading took ",t1-t0," seconds");
del lines
gc.collect();


# In[6]:


sim = pd.DataFrame({
    "dac": dac,
    "msd": msd,
    "len": len,
    "parameters": parameters
})
sim.to_pickle("sim.pkl");


# In[7]:


del msd,dac,len,sim
gc.collect()


# In[8]:


gc.collect()


# In[ ]:




