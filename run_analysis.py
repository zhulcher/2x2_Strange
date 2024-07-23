# SKELETON

# 1. Read in regular analysis file using SPINE
# 2. dump Hip/Mip prediction/truth to h5 and store in a file to be read in


#loop over number of events in one of the files
    #read in the event of the coresponding to the analysis file
    #read in the (same) event coresponding to the (Hip/Mip) file
        # loop over particles:
        #     if HIP_candidate:
        #         store the K candidate mother in place1
        # loop over particles (again):
        #     if MIP_candidate:
        #        store the K candidate mother in place2
        # loop over particles (again again):
        #     if MIP_michel:   
        #         store the K candidate mother in place4

        #true_lambdas=len(true_lambda(particle_list))
        #true_K=len(true_k_with_mu(particle_list))

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