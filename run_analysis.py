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
                #predicted_L+=[[p1.trackid,p2.trackid]]
    #END FIND LAMBDAS LOOP---------------------------

    #compute efficiency and purity
    
#/sdf/data/neutrino/sindhuk/Minirun5/MiniRun5_1E19_RHC.flow.0000001.larcv.root

###########ignore the draft code below this line##############################

# import os
import sys



from analysis.analysis_cuts import direction_acos
from spine.io.read import HDF5Reader

import yaml
from spine.driver import Driver

from analysis.analysis_cuts import *
import numpy as np

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

def main(min_hip_range=0,max_acos=0,max_hip_to_mip_dist_K=np.inf,mip_range=[0,np.inf],
         max_michel_dist=np.inf,min_lambda_decay_len=0,lambda_kinematic_bounds=[-np.inf,np.inf],max_hip_to_mip_dist_lam=np.inf):

    newsemseg="utils/output_HM.h5"

    #######read in the analysis file I generate from HIP/MIP prediction##################
    
    reader = HDF5Reader(newsemseg) #set the file name
    #######read in the analysis file I generate from HIP/MIP prediction##################

    ######read in the analysis file that everyone looks at##################
    
    DATA_PATH = DATA_DIR + 'dummy.h5' # This is the analysis file generated from the sample
    anaconfig = 'anaconfig.cfg'
    anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', DATA_PATH))
    print(yaml.dump(anaconfig))
    driver = Driver(anaconfig)
    ######read in the analysis file that everyone looks at##################


    potential_K=[]
    predicted_K=[]
    predicted_K_michel=[]

    predicted_L=[]

    # TRUTH=False #Orwell would be proud
    TRUTH=True #Orwell is displeased

    print("starting")

    for ENTRY_NUM in range(len(driver)):
        print(ENTRY_NUM)
        data = driver.process(entry=ENTRY_NUM)
        hipmip= reader[ENTRY_NUM]
        print(reader['seg_label'])

        # true_kaons=[]
        # true_lambdas=[]

        # HIP=1
        # MIP=2
        if TRUTH:
            particles=data['truth_particles']
            interactions=data['truth_interactions']
        else:
            particles=data['reco_particles']
            interactions=data['reco_interactions']



        
        sparse3d_pcluster_semantics_HM=something

        #STANDARD [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        #HM [SHOWR_SHP, HIP_SHP, MIP_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        #TODO add in something about the collision distance

        # FIND PRIMARY KAONS LOOP---------------------
        for hip_candidate in particles:
            if not hip_candidate.is_primary: continue #PRIMARY
            if HIPMIP_pred(hip_candidate,sparse3d_pcluster_semantics_HM)!=HIP: continue #HIP
            if not is_contained(hip_candidate): continue #CONTAINED
            if hip_candidate<min_hip_range: continue #RANGE
            if direction_acos(hip_candidate)>max_acos: continue #FORWARD
            if daughters(hip_candidate,particles)!=[0,0,1,0,0,0,0]: continue #EXTRA DAUGHTERS
            potential_K+=[[hip_candidate]]

        for mip_candidate in particles:
            if HIPMIP_pred(mip_candidate,sparse3d_pcluster_semantics_HM)!=MIP: continue #MIP
            if not is_contained(mip_candidate): continue #CONTAINED
            if mip_candidate.length<mip_range[0] or mip_candidate.length>mip_range[1]:continue #RANGE
            z=dist_hipend_mipstart(mip_candidate,potential_K) 
            if z[0]>max_hip_to_mip_dist_K:continue #START NEAR END OF ONE OF PREV KAONS
            daughters=daughters(mip_candidate,particles)
            if (daughters!=[0,0,1,0,0,0,0] and np.linalg.norm(daughters)!=0):continue #EXTRA DAUGHTERS
                
            predicted_K+=[z[1]+[mip_candidate]]

        for michel_candidate in particles:
            if HIPMIP_pred(michel_candidate,sparse3d_pcluster_semantics_HM)!=MICHL_SHP: continue #MICHL
            z=MIP_to_michel(michel_candidate,predicted_K)
            if z[0]>max_michel_dist:continue #START NEAR END OF ONE OF PREV MUONS
            predicted_K_michel+=[z[1]+[michel_candidate]]
        # END FIND PRIMARY KAONS LOOP---------------------

        # FIND LAMBDAS LOOP---------------------------
        for lam_hip_candidate in particles:
            if not is_contained(lam_hip_candidate): continue #CONTAINED
            if HIPMIP_pred(lam_hip_candidate,sparse3d_pcluster_semantics_HM)!=HIP: continue #HIP
            for lam_mip_candidate in particles:
                if not is_contained(lam_mip_candidate): continue #CONTAINED
                if HIPMIP_pred(lam_mip_candidate,sparse3d_pcluster_semantics_HM)!=MIP: continue #MIP

                if np.linalg.norm(lam_hip_candidate.position-lam_mip_candidate.position)>max_hip_to_mip_dist_lam: continue  #DIST FROM MIP TO HIP START
                if lambda_decay_len(lam_hip_candidate,lam_mip_candidate,interactions)<min_lambda_decay_len: continue #EFFECTIVE DECAY LENGTH
                if lambda_kinematic(lam_hip_candidate,lam_mip_candidate)<lambda_kinematic_bounds[0] or lambda_kinematic(lam_hip_candidate,lam_mip_candidate)>lambda_kinematic_bounds[1]: continue #KINEMATIC
                #TODO there's maybe some daughter cut I could use here?
                predicted_L+=[[lam_hip_candidate.trackid,lam_mip_candidate.trackid]]

            #TODO do something
        # END FIND LAMBDAS LOOP---------------------------

        # if is_sim:
        #     pk=[tuple(j) for i in predicted_K.values() for j in i]
            
        #     efficiency_K=len(set(predicted_K)&set(true_kaons))/len(true_kaons) #total efficiency
        #     purity_K=len(set(predicted_K)&set(true_kaons))/len(predicted_K) #total purity

        #     pkm=[tuple(j) for i in predicted_K_michel.values() for j in i]

        #     efficiency_K_mich=len(set(predicted_K_michel)&set(true_kaons_michel))/len(true_kaons_michel) #total efficiency
        #     purity_K_mich=len(set(predicted_K_michel)&set(true_kaons_michel))/len(predicted_K_michel) #total purity

        #     pL=[tuple(j) for i in predicted_L.values() for j in i]

        #     efficiency_L=len(set(predicted_L)&set(true_lambdas))/len(true_lambdas) #total efficiency
        #     purity_L=len(set(predicted_L)&set(true_lambdas))/len(predicted_L) #total purity

        

    return 
if __name__ == "__main__":
    main()