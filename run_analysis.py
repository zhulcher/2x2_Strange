# from scipy.spatial.distance import cdist

# import os
import sys
import os
import random
import numpy as np
import yaml
SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf

# min_len=2.5

# Set software directory
sys.path.append(SOFTWARE_DIR)
from spine.driver import Driver
from spine.io.read import HDF5Reader
from spine.data import Meta as Met
from spine.data.out import TruthParticle,RecoParticle,RecoInteraction,TruthInteraction
Interaction = RecoInteraction | TruthInteraction

Particle = TruthParticle|RecoParticle
from analysis.analysis_cuts import *

analysis_type='icarus'
if analysis_type=='2x2':
    full_containment='detector'
else:
    full_containment='module'


K_MIN_KE=40
LAM_MIN_KE=50

margin0=[[15,15],[15,15],[10,60]]

from collections import Counter


def main(HMh5,analysish5,mode:bool=True,outfile=''):

    # newsemseg="utils/output_HM.h5"

    #######read in the analysis file I generate from HIP/MIP prediction##################

    reader = HDF5Reader(HMh5) #set the file name
    #######read in the analysis file I generate from HIP/MIP prediction##################

    ######read in the analysis file that everyone looks at##################

    # This is the analysis file generated from the sample
    anaconfig = 'configs/anaconfig.cfg'
    anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', analysish5))
    print(yaml.dump(anaconfig))
    driver = Driver(anaconfig)
    ######read in the analysis file that everyone looks at##################

    # potential_K:dict[int,list[PotK]]={}
    # predicted_K:dict[int,list[PredK]]={}
    predicted_K_mu_mich:dict[int,list[PredKaonMuMich]]={}
    predicted_L:dict[int,list[Pred_L]]={}
    num_nu=0
    nu_type_K=[]
    nu_type_L=[]

    print("starting")
    # process_codes=[]
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

        if random.randint(0, 100)==0:
            meta:Met= data['meta']
            if len(particles)>0:
                p_to_check=random.randint(0, len(particles)-1)
                idx=particles[p_to_check].index
                voxels1= sparse3d_pcluster_semantics_HM[idx,1:4]
                voxels2=meta.to_px(particles[p_to_check].points,floor=True)
                assert set(meta.index(voxels1))==set(meta.index(voxels2))

        # STANDARD [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        # HM [SHOWR_SHP, HIP_SHP, MIP_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]
        

        HM_pred=[HIPMIP_pred(p,sparse3d_pcluster_semantics_HM) for p in particles]
        HM_acc=[HIPMIP_acc(p,sparse3d_pcluster_semantics_HM) for p in particles]
    
        if mode:
            for inter in interactions:
                assert type(inter)==TruthInteraction
                if inter.nu_id!=-1 and inter.is_contained:
                    num_nu+=1

        # print("starting kaons")
        for hip_candidate in particles:
            # if interactions[mip_candidate.interaction_id].nu_id==-1:continue

            # Truth=False
            Truth_K=False
            truth_list=[]
            reason=""
            if mode:
                assert type(hip_candidate)==TruthParticle
                truth_list=np.array([hip_candidate.is_primary , # or np.isclose(np.linalg.norm(hip_candidate.position-hip_candidate.ancestor_position),0) #0 or (abs(hip_candidate.parent_pdg_code)==321 and process_map[hip_candidate.ancestor_creation_process]=='primary')
                        hip_candidate.ke>K_MIN_KE, #1
                        # is_contained(hip_candidate.points,mode=full_containment), #2
                        is_contained(interactions[hip_candidate.interaction_id].vertex,mode=full_containment,margin=margin0),
                        hip_candidate.pdg_code==321, #4), #3 
                        # all_daughters_contained(hip_candidate.ancestor_track_id,particles) #5
                        ])
            # pass_prelims=True
                Truth_K=np.all(truth_list)
                if Truth_K:
                    myint=interactions[hip_candidate.interaction_id]
                    nu_type_K+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]
                if not Truth_K:
                    reason=str((np.argwhere(truth_list==False))[0][0])

            pass_prelims=is_contained(interactions[hip_candidate.interaction_id].reco_vertex,mode=full_containment,margin=margin0)
            # pass_prelims=is_contained(interactions[hip_candidate.interaction_id].vertex,mode=full_containment)*is_contained(hip_candidate.points,mode=full_containment)
            pass_prelims*=HM_pred[hip_candidate.id]==HIP_HM
            
            if not (Truth_K or pass_prelims):continue
            # print("kaon")
            # if type(mip_candidate)==TruthParticle:
                # if mip_candidate.creation_process not in ["Decay","primary","muIoni","conv","compt","eBrem","annihil","neutronInelastic","nCapture","muPairProd","muMinusCaptureAtRest"]:  process_codes+=[mip_candidate.creation_process]
                # if mip_candidate.pdg_code==-13 and mip_candidate.parent_pdg_code==321 and mip_candidate.creation_process=="6::201":
                #     print("found a muon from kaon but may not be contained")
            # if mip_candidate.reco_length>70 or mip_candidate.reco_length<15: continue
            
            
            # if not is_contained(hip_candidate.points,mode=full_containment): continue
            
            
            # if mip_candidate.reco_length<min_len: continue
            
            if ENTRY_NUM not in predicted_K_mu_mich:
                predicted_K_mu_mich[ENTRY_NUM] = []
            # print(mip_candidate.reco_length)
            # if 
            predicted_K_mu_mich[ENTRY_NUM]+=[PredKaonMuMich(hip_candidate,particles,interactions,HM_acc,HM_pred,truth=Truth_K,reason=reason,truth_list=truth_list)]
        # print("starting lambda")
        for lam_hip_candidate in particles:
            for lam_mip_candidate in particles:
                if lam_mip_candidate.interaction_id!=lam_hip_candidate.interaction_id: continue
            
            # pass_prelims=True

            # if not is_contained(interactions[lam_hip_candidate.interaction_id].vertex,mode=full_containment): continue

            # if not is_contained(lam_hip_candidate.points,mode=full_containment):
            #     continue #CONTAINED
            # if mode:
            #     if HM_pred[lam_hip_candidate.id]!=HIP_HM: continue #HIP
            # if lam_hip_candidate.reco_length<min_len:
                # continue
            # print("lambda")
            
                # if not is_contained(interactions[lam_mip_candidate.interaction_id].vertex,mode=full_containment): continue

                # if not is_contained(lam_mip_candidate.points,mode=full_containment):
                    # continue #CONTAINED
                # if mode:
                    # if HM_pred[lam_mip_candidate.id]!=MIP_HM: continue #MIP
                # if lam_mip_candidate.reco_length<min_len:
                #     continue

                # if np.min(cdist(lam_hip_candidate.points,lam_mip_candidate.reco_length))>200:
                #     continue
                Truth_lam=False
                reason=""
                if mode:
                    assert type(lam_mip_candidate)==TruthParticle
                    assert type(lam_hip_candidate)==TruthParticle
                    # print("help",lam_mip_candidate.parent_creation_process,lam_mip_candidate.parent_pdg_code)
                    truth_list=np.array([lam_mip_candidate.parent_pdg_code==3122,#0
                                lam_hip_candidate.parent_pdg_code==3122,#1
                                lam_hip_candidate.parent_track_id==lam_mip_candidate.parent_track_id,#2
                                abs(lam_mip_candidate.pdg_code)==211,#3
                                abs(lam_hip_candidate.pdg_code)==2212,#4
                                is_contained(interactions[lam_hip_candidate.interaction_id].vertex,mode=full_containment,margin=margin0),
                                # is_contained(lam_mip_candidate.points,mode=full_containment),#5
                                # is_contained(interactions[lam_mip_candidate.interaction_id].vertex,mode=full_containment),#6   TODO#interaction vertex messed up?????????
                                # is_contained(lam_hip_candidate.points,mode=full_containment),#7
                                # is_contained(interactions[lam_hip_candidate.interaction_id].vertex,mode=full_containment),#8
                                process_map[lam_mip_candidate.creation_process]=='6::201',#9
                                process_map[lam_mip_candidate.parent_creation_process]=='primary' or (lam_mip_candidate.ancestor_pdg_code==3122 and process_map[lam_mip_candidate.ancestor_creation_process]=='primary'),#10
                                # all_daughters_contained(lam_mip_candidate.ancestor_track_id,particles),#11
                                process_map[lam_hip_candidate.creation_process]=='6::201',#12
                                process_map[lam_hip_candidate.parent_creation_process]=='primary'  or (lam_hip_candidate.ancestor_pdg_code==3122 and process_map[lam_hip_candidate.ancestor_creation_process]=='primary'),#13
                                # all_daughters_contained(lam_hip_candidate.ancestor_track_id,particles)#14
                    ])

                    Truth_lam=np.all(truth_list)
                    if Truth_lam:
                        myint=interactions[lam_hip_candidate.interaction_id]
                        nu_type_L+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]
                    if not Truth_lam:
                        reason=str((np.argwhere(truth_list==False))[0][0])
                # pass_prelims=True
                pass_prelims=is_contained(interactions[lam_hip_candidate.interaction_id].reco_vertex,mode=full_containment,margin=margin0)

                # pass_prelims=is_contained(interactions[lam_hip_candidate.interaction_id].vertex,mode=full_containment)*is_contained(lam_hip_candidate.points,mode=full_containment)
                # pass_prelims*=is_contained(interactions[lam_mip_candidate.interaction_id].vertex,mode=full_containment)*is_contained(lam_mip_candidate.points,mode=full_containment)
                pass_prelims*=HM_pred[lam_hip_candidate.id]==HIP_HM
                pass_prelims*=HM_pred[lam_mip_candidate.id]==MIP_HM

                if not (Truth_lam or pass_prelims):
                    continue

                

                if ENTRY_NUM not in predicted_L:
                    predicted_L[ENTRY_NUM]=[]
                predicted_L[ENTRY_NUM]+=[Pred_L(lam_hip_candidate,lam_mip_candidate,particles,interactions,HM_acc,HM_pred,truth=Truth_lam,reason=reason)]


    print(predicted_K_mu_mich)
    # raise Exception(process_codes)
    # print(predicted_K_michel)
    print(predicted_L)
    if outfile!='':
        np.save(outfile,np.array([predicted_K_mu_mich,predicted_L,num_nu,Counter(nu_type_K),Counter(nu_type_L)]))
    # raise Exception(potential_K.keys(),predicted_K.keys())
    return [predicted_K_mu_mich, predicted_L,num_nu]
if __name__ == "__main__":
    # main(mode=True,HMh5="lambdas_10/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_10/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_10.npy')
    # main(mode=True,HMh5="kaons_10/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_10/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_10.npy')
    # main(mode=True,HMh5="kaons/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons/output_0_0000-analysis_truth.h5')

    # main(mode=True,HMh5="lambdas_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_250_only_truth.npy',use_only_truth=True)
    # main(mode=True,HMh5="lambdas_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_250.npy')
    # main(mode=True,HMh5="kaons_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_250_only_truth.npy',use_only_truth=True)
    # main(mode=True,HMh5="kaons_250/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_250/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_250.npy')

    # for path in ["2024-08-14-lambdas","2024-08-14-kaons"]:
    #     main(
    #         mode=True,
    #         HMh5=path+"/output_0_0000-analysis_HM_truth.h5",
    #         analysish5=path+"/output_0_0000-analysis_truth.h5",
    #         outfile="npyfiles/"+path+".npy",
    #         assign_truth=True,
    #     )


    # for path in ["2024-08-16-lambdas","2024-08-16-kaons"]:
    #     main(
    #         mode=True,
    #         HMh5=path+"/output_0_0000-analysis_HM_truth.h5",
    #         analysish5=path+"/output_0_0000-analysis_truth.h5",
    #         outfile="npyfiles/"+path+".npy",
    #         assign_truth=True,
    #     )

    # for path in ["2024-10-16-kaons","2024-08-16-lambdas"]:
    #     main(
    #         mode=True,
    #         HMh5=path+"/output_0_0000-analysis_HM_truth.h5",
    #         analysish5=path+"/output_0_0000-analysis_truth.h5",
    #         outfile="npyfiles/"+path+".npy",
    #         assign_truth=True,
    #     )
    # FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_files/"
    # SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_npy/"

    


    if len(sys.argv[1:])==0:
        FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_files/"
        SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_npy/"
        for path in os.listdir(FILEDIR):
            main(
                mode=True,
                HMh5=FILEDIR+path+"/analysis_HM_both.h5",
                analysish5=FILEDIR+path+"/analysis_both.h5",
                outfile=SAVEDIR+"npyfiles/"+path+".npy"
            )
    if len(sys.argv[1:])==1:
        SAVEDIR=os.path.dirname(sys.argv[1]).replace("_files","_npy")
        # raise Exception(SAVEDIR)
        main(
            mode=True,
            HMh5=sys.argv[1]+"/analysis_HM_both.h5",
            analysish5=sys.argv[1]+"/analysis_both.h5",
            outfile=os.path.join(SAVEDIR,"npyfiles/"+os.path.basename(os.path.normpath(sys.argv[1]))+".npy")
            )

    
    # import os
    # path="beam/processed"
    # directories = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    # for entry in sorted(directories):
    #     full_path = os.path.join(path, entry)
    #     if os.path.isdir(full_path):
    #         print(entry)
    #         # print(path)
    #         main(
    #             mode=True,
    #             HMh5=path+"/"+entry+"/output_0_0000-analysis_HM_truth.h5",
    #             analysish5=path+"/"+entry+"/output_0_0000-analysis_truth.h5",
    #             outfile="npyfiles/beam/"+entry+".npy",
    #             assign_truth=True,
    #         )

    # main(mode=True,HMh5="kaons_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_100_truth.npy',assign_truth=True)

    # # main(mode=True,HMh5="kaons_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'kaons_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_K_100.npy')

    # main(mode=True,HMh5="lambdas_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_100_truth.npy',assign_truth=True)
    # main(mode=True,HMh5="lambdas_100/output_0_0000-analysis_HM_truth.h5",analysish5= 'lambdas_100/output_0_0000-analysis_truth.h5',outfile='npyfiles/test_L_100.npy')
