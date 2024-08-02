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
            #some cut on extra children I need to think more about
            ###
            #potential_K+=[particle.trackid]

        # loop over particles:
            #if MIP_range<mip_range[0]: continue
            #if MIP_range>mip_range[1]: continue
            # z=dist_hipend_mipstart
            #if z[0]>min_hip_to_mip_dist:continue
            ###
            #some cut on extra children I need to think more about
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

import random
from collections import Counter
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

# Set software directory
sys.path.append(SOFTWARE_DIR)

from spine.data.meta import *

def main(mode:bool=True,HMh5="",analysish5="",min_hip_range=0,max_acos=0,mip_range=[0,np.inf],
         min_lambda_decay_len=0,lambda_mass_bounds=[-np.inf,np.inf],
         max_hip_to_mip_dist_lam=np.inf,max_vertex_error=np.inf,
         max_child_angle=np.pi,max_child_dist=np.inf)->list[dict[int,list[list[Particle]]]]:

    # newsemseg="utils/output_HM.h5"

    #######read in the analysis file I generate from HIP/MIP prediction##################
    
    reader = HDF5Reader(HMh5) #set the file name
    #######read in the analysis file I generate from HIP/MIP prediction##################

    ######read in the analysis file that everyone looks at##################
    
     # This is the analysis file generated from the sample
    anaconfig = 'anaconfig.cfg'
    anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', analysish5))
    print(yaml.dump(anaconfig))
    driver = Driver(anaconfig)
    ######read in the analysis file that everyone looks at##################


    potential_K:dict[int,list[list[Particle]]]={}
    predicted_K:dict[int,list[list[Particle]]]={}
    predicted_K_michel:dict[int,list[list[Particle]]]={}

    predicted_L:dict[int,list[list[Particle]]]={}

    print("starting")

    for ENTRY_NUM in range(len(driver)):
        print(ENTRY_NUM)
        data = driver.process(entry=ENTRY_NUM)
        sparse3d_pcluster_semantics_HM=reader[ENTRY_NUM]['seg_label']
        print(reader['seg_label'])

        # true_kaons=[]
        # true_lambdas=[]
        # HIP=1
        # MIP=2
        if mode==True:
            particles:list[Particle] =data['truth_particles']
            interactions:list[Interaction]=data['truth_interactions']
        else:
            particles:list[Particle] =data['reco_particles']
            interactions:list[Interaction] =data['reco_interactions']

        #check that voxels line up, sometimes
        if random.randint(0, 100)==0:
            meta:Met= data['meta']
            p_to_check=random.randint(0, len(particles))
            idx=particles[p_to_check].id
            voxels1= sparse3d_pcluster_semantics_HM[idx,1:4]
            voxels2=particles[p_to_check].points
            # assert sum(voxels1)==sum(voxels2)
            assert set(meta.index(voxels1))==set(meta.index(voxels2))

        #STANDARD [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        #HM [SHOWR_SHP, HIP_SHP, MIP_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        #TODO add in something about the collision distance
        #TODO check for near module boundary
        #TODO angular resolution check
        #TODO direction estimate check 

        # FIND PRIMARY KAONS LOOP---------------------
        for hip_candidate in particles:
            if not hip_candidate.is_primary: continue #PRIMARY
            if HIPMIP_pred(hip_candidate,sparse3d_pcluster_semantics_HM)!=HIP: continue #HIP
            if not is_contained(hip_candidate): continue #CONTAINED
            if hip_candidate.length<min_hip_range: continue #RANGE
            if direction_acos(hip_candidate)>max_acos: continue #FORWARD
            if ENTRY_NUM not in potential_K: potential_K[ENTRY_NUM]=[]
            potential_K[ENTRY_NUM]+=[[hip_candidate]]

        if ENTRY_NUM in potential_K:
            for mip_candidate in particles:
                if HIPMIP_pred(mip_candidate,sparse3d_pcluster_semantics_HM)!=MIP: continue #MIP
                if not is_contained(mip_candidate): continue #CONTAINED
                if mip_candidate.length<mip_range[0] or mip_candidate.length>mip_range[1]:continue #RANGE
                z=dist_end_start(mip_candidate,[k[0] for k in potential_K[ENTRY_NUM]])
                if z[0]>max_child_dist:continue #START NEAR END OF ONE OF PREV KAONS
                parent_K=potential_K[ENTRY_NUM][int(z[1])][0]
                kids=children(parent_K.end_point,particles,max_child_dist,max_child_angle,0)
                kid_sem_seg=Counter([HIPMIP_pred(p,sparse3d_pcluster_semantics_HM) for p in kids])
                if kid_sem_seg[HIP]>0: continue #EXTRA CHILDREN
                if ENTRY_NUM not in predicted_K: predicted_K[ENTRY_NUM]=[]
                predicted_K[ENTRY_NUM]+=[[parent_K,mip_candidate]]
        
        if ENTRY_NUM in predicted_K:
            for michel_candidate in particles:
                if HIPMIP_pred(michel_candidate,sparse3d_pcluster_semantics_HM)!=MICHL_SHP: continue #MICHL
                z=dist_end_start(michel_candidate,[k[1] for k in potential_K[ENTRY_NUM]])
                if z[0]>max_child_dist:continue #START NEAR END OF ONE OF PREV MUONS

                parent_K=predicted_K[ENTRY_NUM][int(z[1])][0]
                kids=children(parent_K.end_point,particles,max_child_dist,max_child_angle,0)
                kid_sem_seg=Counter([HIPMIP_pred(p,sparse3d_pcluster_semantics_HM) for p in kids])
                if kid_sem_seg[HIP]>0: continue #EXTRA CHILDREN
                if ENTRY_NUM not in predicted_K_michel: predicted_K_michel[ENTRY_NUM]=[]
                predicted_K_michel[ENTRY_NUM]+=[predicted_K[ENTRY_NUM][int(z[1])]+[michel_candidate]]
        # END FIND PRIMARY KAONS LOOP---------------------

        # FIND LAMBDAS LOOP---------------------------
        for lam_hip_candidate in particles:
            if not is_contained(lam_hip_candidate): continue #CONTAINED
            if HIPMIP_pred(lam_hip_candidate,sparse3d_pcluster_semantics_HM)!=HIP: continue #HIP
            for lam_mip_candidate in particles:
                if not is_contained(lam_mip_candidate): continue #CONTAINED
                if HIPMIP_pred(lam_mip_candidate,sparse3d_pcluster_semantics_HM)!=MIP: continue #MIP
                if vertex_angle_error(lam_mip_candidate,lam_hip_candidate,interactions)>max_vertex_error: continue
                if np.linalg.norm(lam_hip_candidate.start_point-lam_mip_candidate.start_point)>max_hip_to_mip_dist_lam: continue  #DIST FROM MIP TO HIP START
                if lambda_decay_len(lam_hip_candidate,lam_mip_candidate,interactions)<min_lambda_decay_len: continue #EFFECTIVE DECAY LENGTH
                L_mass=lambda_mass(lam_hip_candidate,lam_mip_candidate)
                if L_mass<lambda_mass_bounds[0] or L_mass>lambda_mass_bounds[1]: continue #KINEMATIC

                kids=lambda_children(lam_mip_candidate,lam_hip_candidate,particles,max_child_dist,max_child_angle,0)
                kid_sem_seg=Counter([HIPMIP_pred(p,sparse3d_pcluster_semantics_HM) for p in kids])
                if Counter(kid_sem_seg)!=Counter([HIP,MIP]): continue
                #TODO make some cut on the AM information (lambda_AM)
                predicted_L[ENTRY_NUM]+=[[lam_hip_candidate,lam_mip_candidate]]

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


    return [predicted_K,predicted_K_michel,predicted_L]
if __name__ == "__main__":
    main(mode=True,HMh5="utils/output_HM.h5",analysish5= '/home/dummy.h5')