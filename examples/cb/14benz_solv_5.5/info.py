import numpy as np
import os

info={}
info['name']='14benz'
info['nsubs']=[5,6]
info['nblocks']=np.sum(info['nsubs'])
info['ncentral']=0
info['nreps']=1
info['nnodes']=1
info['enginepath']=os.environ['CHARMMEXEC']
info['temp']=298.15