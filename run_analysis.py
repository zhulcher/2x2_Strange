# SKELETON

# 1. Read in regular analysis file using SPINE (FILE1)
# 2. dump Hip/Mip prediction/truth to h5 and store in a file to be read in (FILE2)


#min_hip_range=something
#min_forwardness=something
#hip_to_mip_dist=something
#mip_range=[something,something]
#michel_dist=something

#lambda_decay_len=something
#lambda_kinematic_bounds=[something,something]

#potential_K=[]
#predicted_K=[]
#predicted_K_michel=[]

#predicted_L=[]


#loop over number of events in (FILE1)
    #read in the event of FILE1
    #read in the (same) event in FILE2

    #FIND PRIMARY KAONS LOOP---------------------
        # loop over particles:
            #if not hip.is_primary: continue
            #if HIP_range<min_hip_len : continue
            #if forwardness<min_forwardness: continue
            ###
            #some cut on extra daughters I need to think more about
            ###
            #potential_K+=[particle.trackid]

        # loop over particles:
            #if MIP_range<mip_range[0]: continue
            #if MIP_range>mip_range[1]: continue
            # z=dist_hipend_mipstart
            #if z[0]>min_hip_to_mip_dist:continue
            ###
            #some cut on extra daughters I need to think more about
            ###
            #predicted_K+=[z[1],particle.trackid]

        # loop over particles (again for Michels):
            #z=MIP_to_michel
            #if z[0]>michel_dist:continue
            #predicted_K_michel+=[z[1]+[particle.trackid]]
    #END FIND PRIMARY KAONS LOOP---------------------

    #FIND LAMBDAS LOOP---------------------------
        # loop over particles (p1):
            #if not potential_lambda_hip: continue
            #loop over particles (p2):
                #if not potential_lambda_mip: continue
                #if lambda_decay_len<lambda_decay_len: continue
                #if lambda_kinematic not contained within lambda_kinematic_bounds: continue
                #predicted_L+=[particle.trackid]
    #END FIND LAMBDAS LOOP---------------------------

    #compute efficiency and purity
    
#/sdf/data/neutrino/sindhuk/Minirun5/MiniRun5_1E19_RHC.flow.0000001.larcv.root

###########ignore the draft code below this line##############################

# import os
import sys

from analysis import *

# filepath=sys.argv[1]
# file=os.path.basename(filepath)
# filenum=file.split('_')[1]+'_'+file.split('_')[2].split('-')[0]

# outfile='outloc/processed_'+filenum+'.npy'

# if os.path.isfile(outfile):exit()
# print("files:",filepath,file,filenum)

SOFTWARE_DIR = '/Users/zhulcher/Documents/GitHub/spine' #or wherever on sdf
DATA_DIR = '/home/' # Change this path if you are not on SDF (see main README)

# Set software directory
sys.path.append(SOFTWARE_DIR)

newsemseg="utils/output_HM.h5"

#######read in the analysis file I generate from HIP/MIP prediction##################
from spine.io.read import HDF5Reader
reader = HDF5Reader(newsemseg) #set the file name
#######read in the analysis file I generate from HIP/MIP prediction##################

######read in the analysis file that everyone looks at##################
import yaml
from spine.driver import Driver
DATA_PATH = DATA_DIR + 'dummy.h5' # This is the analysis file generated from the sample
anaconfig = 'anaconfig.cfg'
anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', DATA_PATH))
print(yaml.dump(anaconfig))
driver = Driver(anaconfig)
######read in the analysis file that everyone looks at##################





print("starting")

for ENTRY_NUM in range(driver.max_iteration):
    print(ENTRY_NUM)
    data = driver.process(entry=ENTRY_NUM)
    hipmip= reader[0]
    print(reader['seg_label'])

    pot_kaons=[]
    pred_kaons=[]
    pred_kaons_michel=[]
    true_kaons=[]

    HIP=1
    MIP=2
    for i in reco_particles:
        if HIP_candidate(i):
            pot_kaons+=[particle/cluster_id]
    for i in CLUSTERS:
        if MIP_candidate(i,pot_kaons):
            pred_kaons+=[i.parent_id]
        if MIP_michel(i,pot_kaons):
            pred_kaons_michel+=[i.parent_id]
        if true_k_with_mu(i):
            true_kaons+=[i.parent_id]

    efficiency=len(set(pred_kaons)&set(true_kaons))/len(true_kaons) #total efficiency
    purity=len(set(pred_kaons)&set(true_kaons))/len(pred_kaons) #total purity







    #I need to do some association in here which I think should be easy enough, 



# np.save(outfile,outlist)