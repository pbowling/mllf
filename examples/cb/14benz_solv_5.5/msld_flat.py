##
## pyCHARMM script drafted from many examples
##
##   ** MSLD with BLADE (no OMM support) **
##

import os
import sys
import numpy as np
import pandas

##############################################
# Load pyCHARMM libraries

import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.scalar as scalar
from pycharmm.lib import charmm as libcharmm

##############################################

info={}
info['name']='14benz'
info['nsubs']=[5,6]
info['nblocks']=np.sum(info['nsubs'])
info['ncentral']=0
info['nreps']=1
info['nnodes']=1
info['enginepath']=os.environ['CHARMMEXEC']
info['temp']=298.15


##############################################
# Set up global parameters

# variables
box = 32.964000
pmegrid = 32

# msld variables
fnex = 5.5

nsites=len(info['nsubs'])

# nonbonded conditions
nb_fswitch = False          # normal fswitching functions
nb_pme = True               # normal PME

# dynamics conditions                
blade = True

# dynamics variables
cpt_on   = True             # run with CPT for NPT?
timestep = 0.002            # ps
# ns1      = 500000           # number of MD steps per 1 ns
# total_ns = 1                # total number of production ns sampling
# nequil = int(ns1*(1/10))    # equil for 100 ps
# nprod  = ns1*total_ns       # prod sampling for 5 ns
nsavc  = 1000               # dcd save frequency
esteps = 125000             # number of time steps to discard for equilibration
nsteps = 375000             # number of time steps for MD simulation to use for sampling

##############################################
# Stream variables and system setup files
# Flexible loader: prefer --vars-file CLI arg, then MSLD_VARS_FILE env var, then step-based filename, then default
def _load_variables_module():
    """Load variables from a python file (e.g., variablesflat.py) into this script's globals.

    Search order:
      1. CLI arg --vars-file / -v
      2. Environment variable MSLD_VARS_FILE
      3. CLI arg --step N (maps to variables{step}.py)
      4. Fallback to 'variablesflat.py'

    The variables file is executed in an isolated dict; keys commonly expected
    (info, info['temp'], box, nsites, nsubs, bias, etc.) are merged into this module's
    globals when present.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--vars-file', '-v', type=str, help='Path to variables python file')
    parser.add_argument('--step', '-s', type=int, help='Step number to select variables{step}.py')
    parser.add_argument('--out-dir', '-o', type=str, help='Output directory for simulation outputs (res/, dcd/, logs/)')
    # parse only known args so this file can be embedded in other runs
    args, _ = parser.parse_known_args()

    vars_path = None
    # 1) explicit file
    if args.vars_file:
        vars_path = args.vars_file
    # 2) env var
    elif os.environ.get('MSLD_VARS_FILE'):
        vars_path = os.environ.get('MSLD_VARS_FILE')
    # 3) step
    elif args.step is not None:
        cand = f'variables{args.step}.py'
        if os.path.exists(cand):
            vars_path = cand
    # 4) default
    if vars_path is None:
        if os.path.exists('variablesflat.py'):
            vars_path = 'variablesflat.py'
        else:
            # last resort: try variables.py
            vars_path = 'variables.py' if os.path.exists('variables.py') else None

    if vars_path is None:
        raise FileNotFoundError('No variables file found (check --vars-file, MSLD_VARS_FILE, or current directory)')

    # execute the variables file in an isolated namespace and merge results
    vns = {}
    with open(vars_path, 'r') as vf:
        code = vf.read()
    exec(compile(code, vars_path, 'exec'), vns)

    # merge commonly expected names into this module's globals
    merges = ['info', 'temp', 'box', 'pmegrid', 'fnex', 'nsubs', 'nsites', 'nblocks', 'ncentral',
              'nreps', 'nnodes', 'bias', 'name', 'minimizeflag', 'nsavc', 'nsteps', 'esteps']
    for name in merges:
        if name in vns:
            globals()[name] = vns[name]

    # also merge any dict named 'info' entries into local info (if present)
    if 'info' in vns and isinstance(vns['info'], dict):
        info.update(vns['info'])

    # expose chosen out-dir to module globals so the rest of the script can
    # place res/, dcd/, and other output folders under that location
    globals()['out_dir'] = args.out_dir if hasattr(args, 'out_dir') else None

    return vars_path


# perform the load (this may inspect CLI args or env vars)
_vars_file_used = _load_variables_module()
print(f"Loaded variables from {_vars_file_used}")

##############################################
# Read in toppar files, coordinate files, etc.

# toppar files
read.rtf('prep/top_all36_msld.rtf')
read.rtf('prep/full_ligand.rtf',append=True)
read.prm('prep/par_all36_msld.prm',flex=True)
read.prm('prep/full_ligand.prm',flex=True,append=True)

ligseg = 'LIG' 
resnum = '1'
read.sequence_pdb('prep/full_ligand.pdb')
gen.new_segment(ligseg,setup_ic=True)
read.pdb('prep/full_ligand.pdb',resid=True)

# bomblev -1  ! JZV

#  Hybrid Ligand Setup
#  (1) read in patch files
for i in range(len(info['nsubs'])):
  for j in range(info['nsubs'][i]):
    read.rtf(f'prep/site{i+1}_sub{j+1}_pres.rtf',append=True)

# (3) add atoms for each substituent
pycharmm.lingo.charmm_script('ic generate')

# pycharmm.lingo.charmm_script('autogen nopatch') # Don't autogen every step
for i in range(len(info['nsubs'])):
  for j in range(info['nsubs'][i]):
    pycharmm.lingo.charmm_script(f'patch p{i+1}_{j+1} {ligseg} {resnum} setup')
    read.pdb(f'prep/site{i+1}_sub{j+1}_frag.pdb',resid=True)
    ic.prm_fill(replace_all=False) # ic param
    ic.build() # ic build
select.store_selection('togenerate',pycharmm.SelectAtoms(seg_id=ligseg,res_id=resnum))
pycharmm.lingo.charmm_script('auto angle dihe sele .bonded. .bonded. .bonded. togenerate end')

# (2) delete atoms in common core ligand 
#    atoms taken from site1_sub1.txt site2_sub1.txt
select.store_selection('todelete',pycharmm.SelectAtoms().by_res_and_type(ligseg,resnum,'C4 C5 H4 H5'))
pycharmm.lingo.charmm_script('delete atom select todelete end')

# Hybrid Ligand Block
# Substituent definitions
for site in range(len(info['nsubs'])):
  for sub in range(info['nsubs'][site]):
    selname='site'+str(site+1)+'sub'+str(sub+1)
    # extract alchem patch atoms from patch file
    sub_atoms=[]
    rtffile = 'prep/site'+str(site+1)+'_sub'+str(sub+1)+'_pres.rtf'
    for line in open(rtffile,'r'):
      if line[0:4] == 'ATOM': sub_atoms.append(line.split()[1].upper())
    atoms_in_sub = pycharmm.SelectAtoms().by_res_and_type(ligseg,resnum,' '.join(sub_atoms))
    # saves the identified atoms in the charmm variable
    select.store_selection(selname,atoms_in_sub)

# delete angles and dihedrals between alchem groups
# pycharmm.lingo.charmm_script('auto angle dihe')
settings.set_bomb_level(-1)
for site in range(nsites):
    for sub1 in range(info['nsubs'][site]):
        for sub2 in range(sub1+1,info['nsubs'][site]):
            pycharmm.lingo.charmm_script('dele connectivity sele {} show end sele {} show end'
            .format(f'site{site+1}sub{sub1+1}',f'site{site+1}sub{sub2+1}'))

solvated=True
if solvated==True:
  read.sequence_pdb('prep/solvent.pdb')
  gen.new_segment('WT00',setup_ic=True,angle=False, dihedral=False)
  read.pdb('prep/solvent.pdb',resid=True)


# # write out psf, crd, pdb files
# write.psf_card('patch.psf')
# write.coor_card('patch.crd')
# write.coor_pdb('patch.pdb')


##############################################
# Create water box & periodic images

# MODIFY to set up periodic images or SBC
coor.stat()
crystal.define_cubic(box)
# crystal.build(14.0)
pycharmm.lingo.charmm_script('open read card unit 14 name prep/cubic.xtl\ncrystal read card unit 14\nclose unit 14')
image.setup_segment(0.0, 0.0, 0.0, ligseg )
if solvated==True:
  image.setup_residue(0.0, 0.0, 0.0, 'WT00')

##############################################
# Set up BLOCK module for MSLD

# check that the system's net charge is 0
pycharmm.lingo.charmm_script('set charge = ?cgtot')
netQ = pycharmm.lingo.get_charmm_variable('CHARGE')
tol=1e-8
if (netQ > tol) or (netQ < (-1*tol)):
    print("ERROR: system net charge not equal to zero!! Exiting...")
    #pycharmm.lingo.charmm_script('stop')


# MSLD BLOCK module
# ** the block module passed by pycharmm.lingo CANNOT be divided into parts
#    it must be passed as one complete unit
# ** Therefore, multiple strings are created and passed at once to lingo
blockplusone = info['nblocks'] + 1
knoe = 118.4  # for newest version of CATS, use 118.4 for everything
# initialize block
block_init='''
!! BLOCK setup
BLOCK {}
   clear
END
BLOCK {}
'''.format(blockplusone,blockplusone)
# load blocks
block_call=''
ii=2
sub0=np.sum(info['nsubs'])-info['nsubs']
for site in range(nsites):
    for sub in range(info['nsubs'][site]):
        block_call+='Call {} sele {} show end\n'.format(ii,f'site{site+1}sub{sub+1}')
        ii+=1
# for softcore atoms (not sure how to loop-create CATS atoms...?)
block_parm='''
! scat on
! scat k {}
! cats sele atom ?segid ?resid ?atomname .or. [list of atom names to cat]

qldm theta
lang info['temp'] {}
soft on
pmel ex

ldin 1 1.0  0.0  5.0  0.0  5.0'''.format(knoe,info['temp'])
# ldin lines
block_ldin=''
sitestr=''
sub0=np.cumsum(info['nsubs'])-info['nsubs']
for site in range(nsites):
    for sub in range(info['nsubs'][site]):
        if sub == 0:
            tmplmb=1.0-(0.01*(info['nsubs'][site]-1))
        else:
            tmplmb=0.01
        iii=sub0[site]+sub
        bii=iii+2
        block_ldin+='ldin {} {:.4f} 0.0 5.0 {} 5.0\n'.format(bii,tmplmb,bias['b'][0,iii])
        sitestr+=str(site+1)+'  '
# add in exclusions with adex
block_adex=''
for site in range(nsites):
    for sub1 in range(info['nsubs'][site]):
        for sub2 in range(sub1+1,info['nsubs'][site]):
            block_adex+='adex {} {}\n'.format(sub0[site]+sub1+2,sub0[site]+sub2+2)

# msld parameters
block_msld='''
!!rmla bond thet dihe impr
rmla bond thet impr
msld 0  {} fnex {}
msma
'''.format(sitestr,fnex)

# msld variable biases
block_varb='ldbi {}\n'.format((5*info['nblocks']*(info['nblocks']-1))//2)
sub0=np.cumsum(info['nsubs'])-info['nsubs']
ibias=0
for si in range(nsites):
    for sj in range(si,nsites):
        for ii in range(info['nsubs'][si]):
            for jj in range(info['nsubs'][sj]):
                if (si != sj) or (jj > ii):
                    # bii should be iii+2
                    iii=sub0[si]+ii
                    jjj=sub0[sj]+jj
                    bii=iii+2
                    bjj=jjj+2
                    # !vbrex! these lines needed in vb.inp, not here, add to biases below
                    # !vbrex! if si==sj:
                    # !vbrex!     c_shift=2.0*(myrep-ncentral)
                    # !vbrex!     s_shift=0.5*(myrep-ncentral)
                    block_varb+='ldbv {} {} {}  6  0.00 {} 0\n'.format(ibias+1,bii,bjj,-bias['c'][iii,jjj])
                    block_varb+='ldbv {} {} {} 10 -5.56 {} 0\n'.format(ibias+2,bii,bjj,-bias['x'][iii,jjj])
                    block_varb+='ldbv {} {} {}  8 0.017 {} 0\n'.format(ibias+3,bii,bjj,-bias['s'][iii,jjj])
                    block_varb+='ldbv {} {} {} 10 -5.56 {} 0\n'.format(ibias+4,bjj,bii,-bias['x'][jjj,iii])
                    block_varb+='ldbv {} {} {}  8 0.017 {} 0\n'.format(ibias+5,bjj,bii,-bias['s'][jjj,iii])
                    ibias+=5
block_varb+='END\n'

#print('''** TEST **
pycharmm.lingo.charmm_script('''
{}
{}
{}
{}
{}
{}
{}'''.format(block_init,block_call,block_parm,block_ldin,block_adex,block_msld,block_varb))
#pycharmm.lingo.charmm_script('stop')



##############################################
# Set NonBonded settings & SP energy calc
cutnb = 12.0
cutim = cutnb
ctofnb = 10.0
ctonnb = 9.0

## nbond switching
## use a dictionary so that it becomes easy to switch between w/ vs w/o PME
nbonds_dict = {'cutnb':cutnb,'cutim':cutim,
           'ctonnb':ctonnb,'ctofnb':ctofnb,
           'atom':True,'vatom':True,
           'cdie':True,'eps':1.0,
           'inbfrq':-1, 'imgfrq':-1}

if nb_pme:
    nbonds_dict['switch']=True
    nbonds_dict['vfswitch']=True
    nbonds_dict['ewald']=True
    nbonds_dict['pmewald']=True
    nbonds_dict['kappa']=0.32
    nbonds_dict['fftx']=pmegrid
    nbonds_dict['ffty']=pmegrid
    nbonds_dict['fftz']=pmegrid
    nbonds_dict['order']=6

elif nb_fswitch:
    nbonds_dict['fswitch']=True
    nbonds_dict['vfswitch']=True
    nbonds_dict['ewald']=False
    nbonds_dict['pmewald']=False

else: 
    print("NonBonded Parameter Error - both pme and switch are false")
    pycharmm.lingo.charmm_script('stop')

nbonds=pycharmm.NonBondedScript(**nbonds_dict)
nbonds.run()
energy.show()

##############################################
# Minimize the system

if minimizeflag==True:
  minimize.run_sd(nstep=250,nprint=50,step=0.005,tolenr=1e-3,tolgrd=1e-3)
  energy.show()
  #minimize.run_abnr(nstep=250,nprint=50,tolenr=1e-3,tolgrd=1e-3)
  #energy.show()

  # write out psf, crd, pdb files
  write.psf_card('prep/minimized.psf')
  write.coor_card('prep/minimized.crd')
  write.coor_pdb('prep/minimized.pdb')

## Read in minimized coordinates
read.coor_card('prep/minimized.crd',resid=True)


##############################################
# Set up and run Dynamics

# dynamics conditions
if blade:
    useblade = 'prmc pref 1 iprs 100 prdv 100'
    gscal = 0.1
    ntrfrq=0
    leap = True
    openmm = False
else: 
    print("MSLD can only be run with BLADE - exiting...")
    pycharmm.lingo.charmm_script('stop')

# set shake
shake.on(bonh=True,fast=True,tol=1e-7)
dyn.set_fbetas(np.full((psf.get_natom()),gscal,dtype=float))

# initialize blade
pycharmm.lingo.charmm_script('energy blade')

# set up output directories (allow overriding via --out-dir)
_out_dir = globals().get('out_dir')
if _out_dir:
    out_base = _out_dir
else:
    out_base = '.'

res_dir = os.path.join(out_base, 'res')
dcd_dir = os.path.join(out_base, 'dcd')

os.makedirs(res_dir, exist_ok=True)
os.makedirs(dcd_dir, exist_ok=True)

# set up dynamics dictionary of parameters
dynamics_dict = {'cpt':cpt_on,'leap':True,'langevin':True,
    'timestep':timestep,
    'nsavc':nsavc,
    'nsavl':10,  # frequency for saving lambda values in lamda-dynamics
    'nprint': 1000, # Frequency to write to output
    'iprfrq': 1000, # Frequency to calculate averages
    'ntrfrq':ntrfrq,
    'firstt':info['temp'],'finalt':info['temp'],'tstruct':info['temp'],'tbath':info['temp'],
    'iasors': 1,'iasvel':1,'iscvel': 0,'iscale': 0,
    'ihtfrq':0,'ieqfrq':0,'ichecw': 0,
    'inbfrq':-1,'imgfrq':-1,'ihbfrq':0,'ilbfrq':0,
    'echeck': -1}


if cpt_on:
    dynamics_dict['pconstant'] = True
    dynamics_dict['pmass'] = psf.get_natom()*0.12
    dynamics_dict['pref'] = 1.0
    dynamics_dict['pgamma'] = 20.0
    dynamics_dict['hoover'] = True
    dynamics_dict['reft'] = info['temp']
    dynamics_dict['tmass'] = 1000


if blade:
    dynamics_dict['omm'] = False
    dynamics_dict['blade'] = useblade


# MD equilibration
heat_dcd_path = os.path.join(dcd_dir, '{}_{}.dcd'.format(info['name'],'heat'))
dcd_file = pycharmm.CharmmFile(file_name=heat_dcd_path, 
               file_unit=1,formatted=False,read_only=False)
heat_res_path = os.path.join(res_dir, '{}_{}.res'.format(info['name'],'heat'))
res_file = pycharmm.CharmmFile(file_name=heat_res_path, 
               file_unit=2,formatted=True,read_only=False)
lam_path = os.path.join(res_dir, '{}_{}.lmd'.format(info['name'],'heat'))
lam_file = pycharmm.CharmmFile(file_name=lam_path, 
               file_unit=3,formatted=False,read_only=False)

if not 'restartfile' in locals():
    dynamics_dict['start']  = True
    dynamics_dict['restart']= False
    dynamics_dict['iunrea'] = -1
else:
    # restartfile may or may not be set by variables; guard access
    prv_rest = pycharmm.CharmmFile(file_name=locals().get('restartfile'), 
                   file_unit=4,formatted=True,read_only=False)
    dynamics_dict['start']  = False
    dynamics_dict['restart']= True
    dynamics_dict['iunrea'] = prv_rest.file_unit
dynamics_dict['nstep']  = esteps
dynamics_dict['isvfrq'] = esteps # Frequency to save restart file
dynamics_dict['iunwri'] = res_file.file_unit
dynamics_dict['iuncrd'] = dcd_file.file_unit
dynamics_dict['iunldm'] = lam_file.file_unit

equil_dyn = pycharmm.DynamicsScript(**dynamics_dict)
equil_dyn.run()

if 'restartfile' in locals():
    prv_rest.close()
dcd_file.close()
res_file.close()
lam_file.close()

write.coor_pdb(os.path.join(dcd_dir, '{}_fframe.{}.pdb'.format(info['name'],'equil'))) # write out final frame


# MD production
flat_dcd_path = os.path.join(dcd_dir, '{}_{}.dcd'.format(info['name'],'flat'))
dcd_file = pycharmm.CharmmFile(file_name=flat_dcd_path, 
               file_unit=1,formatted=False,read_only=False)
flat_res_path = os.path.join(res_dir, '{}_{}.res'.format(info['name'],'flat'))
res_file = pycharmm.CharmmFile(file_name=flat_res_path, 
               file_unit=2,formatted=True,read_only=False)
prv_rest = pycharmm.CharmmFile(file_name=heat_res_path, 
               file_unit=4,formatted=True,read_only=False)
lam_file = pycharmm.CharmmFile(file_name=os.path.join(res_dir, '{}_{}.lmd'.format(info['name'],'flat')), 
               file_unit=3,formatted=False,read_only=False)

dynamics_dict['start']  = False
dynamics_dict['restart']= True
dynamics_dict['nstep']  = nsteps
dynamics_dict['isvfrq'] = nsteps # Frequency to save restart file
dynamics_dict['iunrea'] = prv_rest.file_unit
dynamics_dict['iunwri'] = res_file.file_unit
dynamics_dict['iuncrd'] = dcd_file.file_unit
dynamics_dict['iunldm'] = lam_file.file_unit

prod_dyn = pycharmm.DynamicsScript(**dynamics_dict)
prod_dyn.run()

dcd_file.close()
res_file.close()
lam_file.close()

# write.coor_pdb('dcd/{}_fframe.{}.pdb'.format(info['name'],'flat')) # write out final frame

#if openmm: pycharmm.lingo.charmm_script('omm clear')
#if blade: pycharmm.lingo.charmm_script('blade off')

# # collect lambda statistics
proc_lam = pycharmm.CharmmFile(file_name=os.path.join(res_dir, '{}_{}.lmd'.format(info['name'],'flat')), 
            file_unit=33,formatted=False,read_only=False)
pycharmm.lingo.charmm_script('traj lamb print ctlo 0.95 cthi 0.99 first {} nunit {}'.format(proc_lam.file_unit,1))


##############################################
# FINISHED

pycharmm.lingo.charmm_script('stop')


