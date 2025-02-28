import csv
import numpy as np
import pandas as pd
import tidynamics
import matplotlib.pyplot as plt
import random
import subprocess
import os
import sys

# mean+std of track length from selected cell tracks
lmean=29.00925925;
lstd=11.65117651853;
# minimal length
n0=15; 

def CalcMSD(xx):
	mm=tidynamics.msd(xx);
	return mm;
	
def CalcDAC(xx):
	vel=xx[1:]-xx[:-1]
	norm = np.sqrt(np.sum(vel**2, axis=1))
	norm_vec = np.column_stack([norm, norm])
	if np.nan not in norm:
		vel = vel / norm_vec
	else:
		for value in range(len(norm_vec)):
			if value != 0:
				vel[i] = vel[i] / norm_vec[i]
			else:
				vel[i] = 0
	val2vec = tidynamics.acf(vel)
	return val2vec;
	
def WriteThree(parameters,mm,dac,l,fparameters,fmm,fdac,fl):
	zeile = ' '.join(map(str, mm)) + '\n'
	with open(fmm, 'a') as f: f.write(zeile)
	zeile = ' '.join(map(str, dac)) + '\n'
	with open(fdac, 'a') as f: f.write(zeile)	
	zeile = ' '.join(map(str, parameters)) + '\n'
	with open(fparameters, 'a') as f: f.write(zeile)	
	zeile = str(l) + '\n'
	with open(fl, 'a') as f: f.write(zeile)
	
	
N=sys.argv[1];

for i in range(int(N)):
	
	#motion_strength=np.exp(random.uniform(np.log(0.01),np.log(10))); # 0.5 +- 
	#advection_velocity=np.exp(random.uniform(np.log(0.01),np.log(10))); # 1.4 +-
	#RaT_time_step =np.exp(random.uniform(np.log(0.01),np.log(10)));# 0.1
	#RaT_shape=np.exp(random.uniform(np.log(0.01),np.log(10))); # 0.5 +-
	#RaT_scale=np.exp(random.uniform(np.log(0.1),np.log(100))); # 4 

	motion_strength=np.random.lognormal(mean=np.log(0.5), sigma=0.6);
	advection_velocity=np.random.lognormal(mean=np.log(1.4), sigma=0.7);
	RaT_scale=np.random.lognormal(mean=np.log(3.8), sigma=0.8);
	RaT_shape=0.5
	RaT_time_step = 0.1;

	cmd="/home/rmuelle/bin/morpheus-2.3.7";
	args=["--set", "motion_strength={}".format(motion_strength), \
		  "--set", " advection_velocity={}".format(advection_velocity), \
		  "--set", " RaT_time_step={}".format(RaT_time_step), \
		  "--set", " RaT_shape={}".format(RaT_shape), \
		  "--set", " RaT_scale={}".format(RaT_scale), \
		  "../Run-and-tumble_rm.xml"]
		  		  
	process = subprocess.run([cmd]+args,stdout=subprocess.DEVNULL)

	if (process.returncode!=0):
		if os.path.exists("msd.txt"):
			os.remove("msd.txt");
		if os.path.exists("dac.txt"):
			os.remove("dac.txt");
		if os.path.exists("parameters.txt"):
			os.remove("parameters.txt");
		sys.exit(process.returncode);

	parameters=np.array([motion_strength,  advection_velocity, RaT_scale]);
	zeile = ' '.join(map(str, parameters)) + '\n'
	with open("parameters.txt", 'a') as f: f.write(zeile)		

	data = pd.read_csv("logger.csv", sep='\s+')
	xx=np.column_stack((data["advection_cell_center.x"].values,data["advection_cell_center.y"].values));	
	zeile = ' '.join(map(str, xx[:,0])) + '\n'
	with open("xx.txt", 'a') as f: f.write(zeile)			
	zeile = ' '.join(map(str, xx[:,1])) + '\n'
	with open("yy.txt", 'a') as f: f.write(zeile)						
		
	mm=CalcMSD(xx);
	dac=CalcDAC(xx);	
	zeile = ' '.join(map(str, mm)) + '\n'
	with open("msd.txt", 'a') as f: f.write(zeile)
	zeile = ' '.join(map(str, dac)) + '\n'
	with open("dac.txt", 'a') as f: f.write(zeile)
	zeile = str(xx.shape[0]) + '\n';
	with open("len.txt", 'a') as f: f.write(zeile)				
				
	n=-1;
	while not (n>=n0 and n<=xx.shape[0]):
		n=int(round(np.random.normal(loc=lmean, scale=lstd)));
	
	xx=xx[0:n,:];	
	mm=CalcMSD(xx);
	dac=CalcDAC(xx);	
	zeile = ' '.join(map(str, mm)) + '\n'
	with open("msd_var.txt", 'a') as f: f.write(zeile)
	zeile = ' '.join(map(str, dac)) + '\n'
	with open("dac_var.txt", 'a') as f: f.write(zeile)
	zeile = str(xx.shape[0]) + '\n'
	with open("len_var.txt", 'a') as f: f.write(zeile)
	
