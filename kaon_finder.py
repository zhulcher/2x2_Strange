
import os
import sys

filepath=sys.argv[1]
file=os.path.basename(filepath)
filenum=file.split('_')[1]+'_'+file.split('_')[2].split('-')[0]

outfile='outloc/processed_'+filenum+'.npy'

if os.path.isfile(outfile):exit()
print("files:",filepath,file,filenum)

SOFTWARE_DIR = '/Users/zhulcher/Documents/GitHub/spine' #or wherever on sdf
DATA_DIR = '/home/' # Change this path if you are not on SDF (see main README)

# Set software directory
sys.path.append(SOFTWARE_DIR)

import yaml
from spine.driver import Driver

#######read in the analysis file that everyone looks at##################
DATA_PATH = DATA_DIR + 'dummy.h5' # This is the analysis file generated from the sample
anaconfig = 'anaconfig.cfg'
anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', DATA_PATH))
print(yaml.dump(anaconfig))
driver = Driver(anaconfig)
#######read in the analysis file that everyone looks at##################

#######read in the analysis file I generate from HIP/MIP prediction##################
from spine.io.read import HDF5Reader
reader = HDF5Reader(whatever.h5) #set the file name
#######read in the analysis file I generate from HIP/MIP prediction##################



def HIPMIP_pred(voxels,sparse3d_pcluster_semantics_zach_or_something_like_that):
    return
    #returns a semantic segmentation prediction for a cluster bc not guaranteed unique
    #majority vote
    #i need to have the files in hand to think clearer about this




def HIP_candidate(particle): 
    return 
    #returns true if HIP candidate satisfying containment cuts, nothing fancy
    #cut on forward or backwardness of kaon in detector? dunno yet

def MIP_candidate(particle,hip_candidates,range=[400,600]): 
    return
    #returns true if MIP candidate if a HIP candidate is their mother
        #mother determined as start point of MIP epsilon away from end point of MIP
    #cuts MIPS with range outside of set bounds
    #cuts candidates with any extra daughters associated to their parent
    #also cuts on containment of daughter particle

def extra_nice_MIP(particle,hip_candidates,range=[400,600]):
    return 
    #returns true if MIP candidates above has a single electron daughter of this MIP


def true_k_with_mu(particle):#returns true if cluster is a contained muon with true kaon as parent
    return
def true_lambda(particle): #returns identifiers for true lambdas
    return

def potential_lambda(particle):
    return
    # returns back to back hip mip with
        #the connection point separated from the vertex of the event
        #the hip and mip full range
        #some mass or kinematic cut I need to workshop


    


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
        if HIP_candidate(i):#I don't have any confidence this is exactly what I want
            pot_kaons+=[particle/cluster_id]
    for i in CLUSTERS:
        if MIP_candidate(i,pot_kaons):
            pred_kaons+=[i.parent_id]
        if extra_nice_MIP(i,pot_kaons):
            pred_kaons_michel+=[i.parent_id]
        if true_k_with_mu(i):
            true_kaons+=[i.parent_id]

    efficiency=len(set(pred_kaons)&set(true_kaons))/len(true_kaons) #total efficiency
    purity=len(set(pred_kaons)&set(true_kaons))/len(pred_kaons) #total purity







    #I need to do some association in here which I think should be easy enough, 


    
        
    # sumE=pair[0]._truth_depositions

# np.save(outfile,outlist)