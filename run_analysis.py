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

import yaml
from spine.driver import Driver
from spine.io.read import HDF5Reader
from spine.utils.globals import MICHL_SHP
from analysis.analysis_cuts import *
# from spine.data.meta import *

# filepath=sys.argv[1]
# file=os.path.basename(filepath)
# filenum=file.split('_')[1]+'_'+file.split('_')[2].split('-')[0]

# outfile='outloc/processed_'+filenum+'.npy'

# if os.path.isfile(outfile):exit()
# print("files:",filepath,file,filenum)

SOFTWARE_DIR = '/Users/zhulcher/Documents/GitHub/spine' #or wherever on sdf

# Set software directory
sys.path.append(SOFTWARE_DIR)


def main(HMh5,analysish5,mode:bool=True,outfile='',use_only_truth=False
        #,min_hip_range=0,max_acos=0,mip_range=[0,np.inf],
        # min_lambda_decay_len=0,lambda_mass_bounds=[-np.inf,np.inf],
         #max_hip_to_mip_dist_lam=np.inf,max_vertex_error=np.inf,
        # max_child_angle=np.pi,max_child_dist=np.inf
        ):

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



    potential_K:dict[int,list[PotK]]={}
    predicted_K:dict[int,list[PredK]]={}
    predicted_K_michel:dict[int,list[PredK_Mich]]={}
    predicted_L:dict[int,list[Pred_L]]={}

    print("starting")

    for ENTRY_NUM in range(len(driver)):
        print(ENTRY_NUM)
        data = driver.process(entry=ENTRY_NUM)
        sparse3d_pcluster_semantics_HM=reader[ENTRY_NUM]['seg_label']

        # true_kaons=[]
        # true_lambdas=[]
        # HIP=1
        # MIP=2
        if mode:
            particles:list[Particle] =data['truth_particles']
            interactions:list[Interaction]=data['truth_interactions']
        else:
            particles:list[Particle] =data['reco_particles']
            interactions:list[Interaction] =data['reco_interactions']

        if use_only_truth and not mode:
            raise ValueError("How exactly do you get truth out of reco?")

        #check that voxels line up, sometimes
        if random.randint(0, 100)==0:
            meta:Met= data['meta']
            p_to_check=random.randint(0, len(particles)-1)
            idx=particles[p_to_check].index
            voxels1= sparse3d_pcluster_semantics_HM[idx,1:4]
            voxels2=meta.to_px(particles[p_to_check].points,floor=True)
            assert set(meta.index(voxels1))==set(meta.index(voxels2))

        #STANDARD [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        #HM [SHOWR_SHP, HIP_SHP, MIP_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        #TODO check for near module boundary
        #TODO angular resolution check
        #TODO direction estimate check

        #TODO move function computing into the analysis cuts classes
        #TODO use truth attribute to keep particles which are truth or not
        #TODO base around muon with michel and kaon secondary

        HM_pred=[HIPMIP_pred(p,sparse3d_pcluster_semantics_HM) for p in particles]
        HM_sc=[HM_score(p,sparse3d_pcluster_semantics_HM) for p in particles]
        # FIND PRIMARY KAONS LOOP---------------------
        for hip_candidate in particles:
            if not hip_candidate.is_primary:
                continue #PRIMARY
            if HM_pred[hip_candidate.id]!=HIP_HM:
                continue #HIP
            if hip_candidate.reco_length==0:
                continue
            # print("we have a HIP")
            if not is_contained(hip_candidate.points,mode='detector'):
                continue #CONTAINED
            if not is_contained(np.array([hip_candidate.start_point]),mode='tpc',margin=1):
                continue #CONTAINED VERTEX
            if use_only_truth:
                # print(hip_candidate.pdg_code)
                if hip_candidate.pdg_code!=321:
                    continue
            # if hip_candidate.reco_length<min_hip_range: continue #RANGE
            # if direction_acos(hip_candidate.start_dir)>max_acos: continue #FORWARD
            if ENTRY_NUM not in potential_K:
                potential_K[ENTRY_NUM]=[]
            # print("hip len",hip_candidate.reco_length)
            if direction_acos(hip_candidate.start_dir)<0 or direction_acos(hip_candidate.start_dir)>np.pi:
                raise ValueError("bad acos",direction_acos(hip_candidate.start_dir),ENTRY_NUM)
            potential_K[ENTRY_NUM]+=[PotK(hip_candidate.id,hip_candidate.reco_length,direction_acos(hip_candidate.start_dir),HM_sc[hip_candidate.id])]


        if ENTRY_NUM in potential_K:
            for mip_candidate in particles:
                if HM_pred[mip_candidate.id]!=MIP_HM:
                    continue #MIP
                if not is_contained(mip_candidate.points,mode='detector'):
                    continue #CONTAINED
                if mip_candidate.reco_length==0:
                    continue
                # if mip_candidate.reco_length<mip_range[0] or mip_candidate.reco_length>mip_range[1]:continue #RANGE
                pairs=dist_end_start(mip_candidate,[particles[k.hip_id] for k in potential_K[ENTRY_NUM]])

                if use_only_truth:
                    # print(mip_candidate.pdg_code,mip_candidate.parent_pdg_code,mip_candidate.creation_process)
                    if mip_candidate.pdg_code!=-13:
                        continue
                    if mip_candidate.parent_pdg_code!=321:
                        continue
                # if z[0]>max_child_dist:continue #START NEAR END OF ONE OF PREV KAONS
                for z in pairs:
                    parent_K=particles[potential_K[ENTRY_NUM][int(z[1])].hip_id]

                    assert HM_pred[parent_K.id]==HIP_HM

                    closest_kids=children(parent_K,[p for p in particles if HM_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]],ignore=[mip_candidate.id,parent_K.id])

                    #TODO if meet up point is in the wall, backtrack and add it to HIP and MIP
                    # mip_candidate.parent
                    if ENTRY_NUM not in predicted_K:
                        predicted_K[ENTRY_NUM]=[]
                    # print(mip_candidate.reco_length)
                    predicted_K[ENTRY_NUM]+=[PredK(potential_K[ENTRY_NUM][int(z[1])],mip_candidate.id,mip_candidate.reco_length,z[0],closest_kids,HM_sc[mip_candidate.id])]

        if ENTRY_NUM in predicted_K:
            for michel_candidate in particles:
                if HM_pred[michel_candidate.id]!=MICHL_SHP:
                    continue #MICHL
                pairs=dist_end_start(michel_candidate,[particles[k.mip_id] for k in predicted_K[ENTRY_NUM]])

                if use_only_truth:
                    if michel_candidate.pdg_code!=-11:
                        continue
                    if michel_candidate.parent_pdg_code!=-13:
                        continue
                    if michel_candidate.ancestor_pdg_code!=321:
                        continue
                for z in pairs:
                    # if z[0]>max_child_dist:continue #START NEAR END OF ONE OF PREV MUONS

                    parent_mu=particles[predicted_K[ENTRY_NUM][int(z[1])].mip_id]

                    # print(parent_mu.reco_length,parent_mu.reco_length,michel_candidate.end_t,parent_mu.last_step,michel_candidate.first_step)
                    # print()
                    assert HM_pred[parent_mu.id]==MIP_HM

                    K_id:int=predicted_K[ENTRY_NUM][int(z[1])].hip_id

                    #TODO if meet up point is in the wall, backtrack and add it to HIP and MIP

                    closest_kids=children(parent_mu,[p for p in particles if HM_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]],ignore=[parent_mu.id,K_id,michel_candidate.id])
                    # if kid_sem_seg[HIP]>0: continue #EXTRA CHILDREN
                    if ENTRY_NUM not in predicted_K_michel:
                        predicted_K_michel[ENTRY_NUM]=[]

                    # dt=np.nan
                    # ddist=np.nan
                    # if mode and is_contained(michel_candidate.start_point,mode="tpc",margin=1) and is_contained(parent_mu.end_point,mode="tpc",margin=2):
                    #     # t_to_dist=(parent_mu.t-michel_candidate.t)/np.linalg.norm(parent_mu.last_step-michel_candidate.first_step)*1.5/1000/10
                    #     if michel_candidate.parent_id==parent_mu.id:
                    #         dt=(michel_candidate.t-parent_mu.t)
                    #         ddist=np.linalg.norm(parent_mu.last_step-michel_candidate.first_step)
                    #         # print(np.abs(parent_mu.last_step-michel_candidate.first_step)-np.array([1.648/10/1000*dt,0,0]))
                    #         # print(np.linalg.norm(parent_mu.last_step-michel_candidate.first_step),parent_mu.last_step,michel_candidate.first_step)
                    predicted_K_michel[ENTRY_NUM]+=[PredK_Mich(predicted_K[ENTRY_NUM][int(z[1])],michel_candidate.id,z[0],closest_kids,HM_sc[michel_candidate.id])]
        # END FIND PRIMARY KAONS LOOP---------------------
        #TODO reorganize code around muons
        # FIND LAMBDAS LOOP---------------------------
        for lam_hip_candidate in particles:
            if not is_contained(lam_hip_candidate.points,mode='detector'):
                continue #CONTAINED
            if HM_pred[lam_hip_candidate.id]!=HIP_HM:
                continue #HIP
            if lam_hip_candidate.reco_length==0:
                continue
            if use_only_truth:
                if abs(lam_hip_candidate.pdg_code)!=2212:
                    continue
                # if not np.isclose(np.linalg.norm(lam_hip_candidate.end_momentum),0):continue
                skip=False
                for p in particles:
                    if p.creation_process=='4::121' and p.parent_id==lam_hip_candidate.id:
                        skip=True
                        break
                if skip: continue

            for lam_mip_candidate in particles:
                if not is_contained(lam_mip_candidate.points,mode='detector'):
                    continue #CONTAINED
                if HM_pred[lam_mip_candidate.id]!=MIP_HM:
                    continue #MIP
                if lam_mip_candidate.reco_length==0:
                    continue
                if use_only_truth:
                    if abs(lam_mip_candidate.pdg_code)!=211:
                        continue
                    # if not np.isclose(np.linalg.norm(lam_mip_candidate.end_momentum),0):continue
                    # if HM_pred[lam_mip_candidate.id]==MIP_HM and abs(lam_mip_candidate.pdg_code)==2212: raise Exception("HOW?",sparse3d_pcluster_semantics_HM[lam_mip_candidate.index,-1])
                    if lam_mip_candidate.parent_id!=lam_hip_candidate.parent_id:
                        continue
                    if abs(lam_mip_candidate.parent_pdg_code)!=3122:
                        continue
                    skip=False
                    for p in particles:
                        if p.creation_process=='4::121' and p.parent_id==lam_hip_candidate.parent_id:
                            skip=True
                            break
                        if p.creation_process=='4::121' and p.parent_id==lam_hip_candidate.id:
                            skip=True
                            break
                        if p.parent_id==lam_hip_candidate.parent_id and p.id not in [lam_hip_candidate.id,lam_mip_candidate.id]: raise Exception("WHY???",p.id,[lam_hip_candidate.id,lam_mip_candidate.id])
                    if skip: continue
                    if lam_hip_candidate.creation_process!='6::201' or lam_mip_candidate.creation_process!='6::201': continue
                    print(lam_mip_candidate.end_momentum,lam_hip_candidate.end_momentum,lam_hip_candidate.reco_ke,lam_mip_candidate.reco_ke)

                VAE=vertex_angle_error(lam_mip_candidate,lam_hip_candidate,interactions)
                # if VAE>max_vertex_error: continue
                coll_dist=collision_distance(lam_hip_candidate,lam_mip_candidate)
                # if coll_dist[2]>max_hip_to_mip_dist_lam: continue  #DIST FROM MIP TO HIP START

                lam_decay_len=lambda_decay_len(lam_hip_candidate,lam_mip_candidate,interactions)
                # if lam_decay_len<min_lambda_decay_len: continue #EFFECTIVE DECAY LENGTH
                lam_mass2=lambda_mass_2(lam_hip_candidate,lam_mip_candidate)
                # if lam_mass<lambda_mass_bounds[0] or lam_mass>lambda_mass_bounds[1]: continue #KINEMATIC
                # AM=lambda_AM(lam_hip_candidate,lam_mip_candidate)

                lam_closest_kids=lambda_children(lam_mip_candidate,lam_hip_candidate,[p for p in particles if HM_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]])
                prot_closest_kids=children(lam_hip_candidate,[p for p in particles if HM_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]],ignore=[lam_hip_candidate.id,lam_mip_candidate.id])
                pi_closest_kids=children(lam_mip_candidate,[p for p in particles if HM_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]],ignore=[lam_hip_candidate.id,lam_mip_candidate.id])

                #TODO if meet up point is in the wall, backtrack and add it to HIP and MIP

                momenta=momenta_projections(lam_hip_candidate,lam_mip_candidate,interactions)

                lam_acos=direction_acos((lam_hip_candidate.momentum+lam_mip_candidate.momentum)/np.linalg.norm(lam_hip_candidate.momentum+lam_mip_candidate.momentum))


                if ENTRY_NUM not in predicted_L:
                    predicted_L[ENTRY_NUM]=[]
                predicted_L[ENTRY_NUM]+=[Pred_L(lam_hip_candidate.id,lam_mip_candidate.id,lam_hip_candidate.reco_length,lam_mip_candidate.reco_length,VAE,lam_mass2,lam_decay_len,momenta,coll_dist,
                                                lam_closest_kids,prot_closest_kids,pi_closest_kids,lam_acos,
                                                HM_sc[lam_hip_candidate.id],HM_sc[lam_mip_candidate.id])]

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

    print(predicted_K)
    print(predicted_K_michel)
    print(predicted_L)
    if outfile!='':
        np.save(outfile,np.array([predicted_K,predicted_K_michel,predicted_L]))
    # raise Exception(potential_K.keys(),predicted_K.keys())
    return [predicted_K,predicted_K_michel,predicted_L]
if __name__ == "__main__":
    # main(mode=True,HMh5="lambdas_10/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_10/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_10.npy')
    # main(mode=True,HMh5="kaons_10/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_10/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_10.npy')
    # main(mode=True,HMh5="kaons/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons/output_0_0000-analysis_truth.h5')




    # main(mode=True,HMh5="lambdas_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_250_only_truth.npy',use_only_truth=True)
    # main(mode=True,HMh5="lambdas_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_250.npy')
    # main(mode=True,HMh5="kaons_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_250_only_truth.npy',use_only_truth=True)
    # main(mode=True,HMh5="kaons_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_250.npy')


    main(mode=True,HMh5="kaons_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_100_only_truth.npy',use_only_truth=True)

    main(mode=True,HMh5="kaons_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_100.npy')




    main(mode=True,HMh5="lambdas_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_100_only_truth.npy',use_only_truth=True)
    main(mode=True,HMh5="lambdas_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_100.npy')
