# from scipy.spatial.distance import cdist

# import os
import sys
import os
# import random
import numpy as np
# from numpy._core.numerictypes import bool_
import yaml
SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf
from collections import Counter

# min_len=2.5

# Set software directory
sys.path.append(SOFTWARE_DIR)
from spine.driver import Driver
from spine.io.read import HDF5Reader
from spine.data import Meta as Met
from spine.data.out import TruthParticle,RecoParticle,RecoInteraction,TruthInteraction
Interaction = RecoInteraction | TruthInteraction
from scipy.spatial.distance import cdist

Particle = TruthParticle|RecoParticle
from analysis.analysis_cuts import *

analysis_type='icarus'
if analysis_type=='2x2':
    full_containment='detector'
else:
    full_containment='module'


# K_MIN_KE=40
# LAM_MIN_KE=50



# from collections import Counter


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
    predicted_L:dict[int,list[Pred_Neut]]={}
    predicted_K0s:dict[int,list[Pred_Neut]]={}


    primarymip: dict[int,PrimaryMIP]={}

    truth_interaction_map = {}

    num_nu=0
    nu_type_K=[]
    nu_type_L=[]
    nu_type_K0s=[]

    

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


        perm = np.lexsort(data['points_label'].T)
        perm_inverse=np.argsort(perm)

        # perm = np.arange(len(data['points_label']))
        # perm_inverse = np.argsort(perm)



        # if random.randint(0, 100)==0:
        meta:Met= data['meta']
        # if len(particles)>0:
        if mode:
            for p_to_check in range(len(particles)):
            # p_to_check=random.randint(0, len(particles)-1)
                idx=perm_inverse[particles[p_to_check].index]
                voxels1= sparse3d_pcluster_semantics_HM[idx,1:4]
                voxels2=meta.to_px(particles[p_to_check].points,floor=True)
                assert set(meta.index(voxels1))==set(meta.index(voxels2)),(set(meta.index(voxels1)),set(meta.index(voxels2)),meta.to_px(data['points_label'][perm][:20],floor=True),reader[ENTRY_NUM]['seg_label'][:20])#(meta.to_px(data['points_label'][:10]),sparse3d_pcluster_semantics_HM[:10,1:4])#(len(set(meta.index(voxels1))),len(set(meta.index(voxels2))),np.shape(data['points_label']),np.shape(reader[ENTRY_NUM]['seg_label']),data['points_label'][:10],reader[ENTRY_NUM]['seg_label'][:10],meta.index(voxels1)[:10],meta.index(voxels2)[:10],voxels1[:10],voxels2[:10],mode,type(particles[0]),)

        # STANDARD [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        # HM [SHOWR_SHP, HIP_SHP, MIP_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

        HM_pred=[HIPMIP_pred(p,sparse3d_pcluster_semantics_HM,perm_inverse) for p in particles]
        HM_acc=[HIPMIP_acc(p,sparse3d_pcluster_semantics_HM,perm_inverse) for p in particles]

        #TODO fix this 
        # for p in particles:
            
        #     if p.num_voxels>0 and abs(p.pdg_code) in [2212,321,3112,3222]:
        #         if type(p)==TruthParticle:
        #             assert HM_pred[p.id]==HIP_HM,(p.pdg_code,p.shape,set(HM_pred),HM_pred[p.id],HIP_HM,len(p.points),p.num_voxels,HM_acc[p.id], Counter(sparse3d_pcluster_semantics_HM[perm_inverse[p.index], -1]))
                
        #             assert HM_acc[p.id]==1,(p.pdg_code,p.shape,set(HM_pred),HM_pred[p.id],HIP_HM,len(p.points),p.num_voxels,HM_acc[p.id], Counter(sparse3d_pcluster_semantics_HM[perm_inverse[p.index], -1]))

        

        primarymip[ENTRY_NUM]=PrimaryMIP(particles,HM_pred)

        truth_interaction_map[ENTRY_NUM]={}
    
        if mode:
            for inter in interactions:
                assert type(inter)==TruthInteraction
                if inter.nu_id!=-1 and is_contained(inter.vertex,margin=0):
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
                        # hip_candidate.ke>K_MIN_KE, #1
                        # is_contained(hip_candidate.points), #2
                        is_contained(interactions[hip_candidate.interaction_id].vertex,margin=margin0),
                        hip_candidate.pdg_code==321, #4), #3 
                        # all_daughters_contained(hip_candidate.ancestor_track_id,particles) #5
                        ])
            # pass_prelims=True
                Truth_K=np.all(truth_list)
                if Truth_K:
                    myint=interactions[hip_candidate.interaction_id]
                    nu_type_K+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]

                    if hip_candidate.interaction_id not in truth_interaction_map[ENTRY_NUM]:
                        truth_interaction_map[ENTRY_NUM][hip_candidate.interaction_id]=[]

                    truth_interaction_map[ENTRY_NUM][hip_candidate.interaction_id]+=[hip_candidate]
                if not Truth_K:
                    reason=str((np.argwhere(truth_list==False))[0][0])

            pass_prelims=is_contained(interactions[hip_candidate.interaction_id].reco_vertex,margin=0)
            # pass_prelims=is_contained(interactions[hip_candidate.interaction_id].vertex)*is_contained(hip_candidate.points)
            pass_prelims*=HM_pred_hotfix(hip_candidate,HM_pred)==HIP_HM

            if pass_prelims:

                dist = np.min(cdist(hip_candidate.points, [interactions[hip_candidate.interaction_id].vertex,interactions[hip_candidate.interaction_id].reco_vertex]))
                
                pass_prelims*=(dist<10)

            # points_i = np.vstack([hip_candidate.start_point, hip_candidate.end_point])
            # points_j = np.vstack([mip_candidate.start_point, mip_candidate.end_point])

            
            if not (Truth_K or pass_prelims):continue
            # print("kaon")
            # if type(mip_candidate)==TruthParticle:
                # if mip_candidate.creation_process not in ["Decay","primary","muIoni","conv","compt","eBrem","annihil","neutronInelastic","nCapture","muPairProd","muMinusCaptureAtRest"]:  process_codes+=[mip_candidate.creation_process]
                # if mip_candidate.pdg_code==-13 and mip_candidate.parent_pdg_code==321 and mip_candidate.creation_process=="6::201":
                #     print("found a muon from kaon but may not be contained")
            # if mip_candidate.reco_length>70 or mip_candidate.reco_length<15: continue
            
            
            # if not is_contained(hip_candidate.points): continue
            
            
            # if mip_candidate.reco_length<min_len: continue
            
            if ENTRY_NUM not in predicted_K_mu_mich:
                predicted_K_mu_mich[ENTRY_NUM] = []
            # print(mip_candidate.reco_length)
            # if 
            predicted_K_mu_mich[ENTRY_NUM]+=[PredKaonMuMich(hip_candidate,particles,interactions,HM_acc,HM_pred,truth=Truth_K,reason=reason,truth_list=truth_list)]
        # print("starting lambda")

        if type(particles[0])==TruthParticle: 
            existing_parent_track_ids=Counter([p.parent_track_id for p in particles])
            existing_track_ids=Counter([p.track_id for p in particles])

        for lam_hip_candidate in particles:
            for lam_mip_candidate in particles:
                if lam_mip_candidate.interaction_id!=lam_hip_candidate.interaction_id: continue
                Truth_lam=False
                reason=""
                if mode:
                    assert type(lam_mip_candidate)==TruthParticle
                    assert type(lam_hip_candidate)==TruthParticle
                    # print("help",lam_mip_candidate.parent_creation_process,lam_mip_candidate.parent_pdg_code)
                    truth_list=np.array([lam_mip_candidate.parent_pdg_code==3122,#0
                                lam_hip_candidate.ancestor_track_id==lam_mip_candidate.ancestor_track_id,
                                (lam_hip_candidate.parent_pdg_code==3122 and lam_hip_candidate.parent_track_id==lam_mip_candidate.parent_track_id and set([process_map[lam_mip_candidate.creation_process],process_map[lam_hip_candidate.creation_process]])==set(['6::201'])) or 
                                    (lam_hip_candidate.parent_pdg_code==2212 and process_map[lam_mip_candidate.creation_process]=='6::201' and lam_hip_candidate.creation_process=='protonInelastic' and
                                    lam_hip_candidate.parent_creation_process=='Decay' and existing_parent_track_ids[lam_hip_candidate.parent_track_id]==1 and existing_track_ids[lam_hip_candidate.parent_track_id]==0),#2
                                lam_mip_candidate.pdg_code==-211,#3
                                lam_hip_candidate.pdg_code==2212,#4
                                is_contained(interactions[lam_hip_candidate.interaction_id].vertex,margin=margin0),#5
                                # is_contained(lam_mip_candidate.points),
                                # is_contained(interactions[lam_mip_candidate.interaction_id].vertex),
                                # is_contained(lam_hip_candidate.points),
                                # is_contained(interactions[lam_hip_candidate.interaction_id].vertex),
                                # process_map[lam_mip_candidate.creation_process] in ['6::201',"4::151"],# or lam_mip_candidate.creation_process=='lambdaInelastic',#6
                                # process_map[lam_mip_candidate.parent_creation_process]=='primary' or (lam_mip_candidate.ancestor_pdg_code==3122 and process_map[lam_mip_candidate.ancestor_creation_process]=='primary'),#7#TODO this is weakened due to a bug 
                                lam_mip_candidate.ancestor_pdg_code==3122,#7
                                # all_daughters_contained(lam_mip_candidate.ancestor_track_id,particles),#8
                                # process_map[lam_hip_candidate.creation_process] in ['6::201',"4::151"],# or lam_hip_candidate.creation_process=='lambdaInelastic',#9
                                # process_map[lam_hip_candidate.parent_creation_process]=='primary'  or (lam_hip_candidate.ancestor_pdg_code==3122 and process_map[lam_hip_candidate.ancestor_creation_process]=='primary'),#10#TODO this is weakened due to a bug 
                                lam_hip_candidate.ancestor_pdg_code==3122,#10,
                                # all_daughters_contained(lam_hip_candidate.ancestor_track_id,particles)#11
                    ])

                    Truth_lam=np.all(truth_list)
                    if Truth_lam:
                        myint=interactions[lam_hip_candidate.interaction_id]
                        nu_type_L+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]

                        if lam_hip_candidate.interaction_id not in truth_interaction_map[ENTRY_NUM]:
                            truth_interaction_map[ENTRY_NUM][lam_hip_candidate.interaction_id]=[]

                        truth_interaction_map[ENTRY_NUM][lam_hip_candidate.interaction_id]+=[lam_hip_candidate,lam_mip_candidate]
                    if not Truth_lam:
                        reason=str((np.argwhere(truth_list==False))[0][0])
                # pass_prelims=True
                pass_prelims=is_contained(interactions[lam_hip_candidate.interaction_id].reco_vertex,margin=0)

                # pass_prelims=is_contained(interactions[lam_hip_candidate.interaction_id].vertex)*is_contained(lam_hip_candidate.points)
                # pass_prelims*=is_contained(interactions[lam_mip_candidate.interaction_id].vertex)*is_contained(lam_mip_candidate.points)
                pass_prelims*=HM_pred_hotfix(lam_hip_candidate,HM_pred)==HIP_HM
                pass_prelims*=HM_pred_hotfix(lam_mip_candidate,HM_pred)==MIP_HM

                if pass_prelims:

                    dist = np.min(cdist(lam_hip_candidate.points, lam_mip_candidate.points))
                
                    pass_prelims*=(dist<10)

                if not (Truth_lam or pass_prelims):
                    continue

                

                if ENTRY_NUM not in predicted_L:
                    predicted_L[ENTRY_NUM]=[]
                predicted_L[ENTRY_NUM]+=[Pred_Neut(lam_hip_candidate,lam_mip_candidate,particles,interactions,HM_acc,HM_pred,truth=Truth_lam,reason=reason,mass1=PROT_MASS,mass2=PION_MASS)]


        for K0s_mip1_candidate in particles:
            for K0s_mip2_candidate in particles:
                if K0s_mip1_candidate.interaction_id!=K0s_mip2_candidate.interaction_id: continue
                if K0s_mip1_candidate.id>=K0s_mip2_candidate.id:continue
                Truth_K0s=False
                reason=""
                if mode:
                    # if K0s_mip1_candidate.parent_pdg_code==310 and K0s_mip2_candidate.parent_pdg_code==310: raise Exception("k0s anc",K0s_mip1_candidate.ancestor_pdg_code,K0s_mip1_candidate.creation_process,K0s_mip2_candidate.parent_creation_process,K0s_mip2_candidate.ancestor_creation_process,K0s_mip1_candidate.pdg_code,K0s_mip2_candidate.pdg_code)
                    assert type(K0s_mip1_candidate)==TruthParticle
                    assert type(K0s_mip2_candidate)==TruthParticle
                    truth_list=np.array([K0s_mip1_candidate.parent_pdg_code==310,#0
                                K0s_mip2_candidate.parent_pdg_code==310,#1
                                K0s_mip1_candidate.parent_track_id==K0s_mip2_candidate.parent_track_id,#2
                                abs(K0s_mip1_candidate.pdg_code)==211,#3
                                abs(K0s_mip2_candidate.pdg_code)==211,#4
                                is_contained(interactions[K0s_mip1_candidate.interaction_id].vertex,margin=margin0),#5
                                process_map[K0s_mip1_candidate.creation_process]=='6::201',#6
                                process_map[K0s_mip1_candidate.parent_creation_process]=='primary' or (K0s_mip1_candidate.ancestor_pdg_code in [310,311] and process_map[K0s_mip1_candidate.ancestor_creation_process] in ['primary','6::201']),#7
                                process_map[K0s_mip2_candidate.creation_process]=='6::201',#8
                                process_map[K0s_mip2_candidate.parent_creation_process]=='primary'  or (K0s_mip2_candidate.ancestor_pdg_code in [310,311] and process_map[K0s_mip2_candidate.ancestor_creation_process] in ['primary','6::201']),#9
                    ])

                    Truth_K0s=np.all(truth_list)
                    if Truth_K0s:
                        myint=interactions[K0s_mip1_candidate.interaction_id]
                        nu_type_K0s+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]
                    if not Truth_K0s:
                        reason=str((np.argwhere(truth_list==False))[0][0])
                # pass_prelims=True
                pass_prelims=is_contained(interactions[K0s_mip1_candidate.interaction_id].reco_vertex,margin=0)

                

                
                pass_prelims*=HM_pred_hotfix(K0s_mip1_candidate,HM_pred)==MIP_HM
                pass_prelims*=HM_pred_hotfix(K0s_mip2_candidate,HM_pred)==MIP_HM

                if pass_prelims:

                    dist = np.min(cdist(K0s_mip1_candidate.points, K0s_mip2_candidate.points))
                
                    pass_prelims*=(dist<10)

                if not (Truth_K0s or pass_prelims):
                    continue

                

                if ENTRY_NUM not in predicted_K0s:
                    predicted_K0s[ENTRY_NUM]=[]
                predicted_K0s[ENTRY_NUM]+=[Pred_Neut(K0s_mip1_candidate,K0s_mip2_candidate,particles,interactions,HM_acc,HM_pred,truth=Truth_K0s,reason=reason,mass1=PION_MASS,mass2=PION_MASS)]


    print(predicted_K_mu_mich)
    # raise Exception(process_codes)
    # print(predicted_K_michel)
    print(predicted_L)
    print(predicted_K0s)
    print(primarymip)
    print(truth_interaction_map)
    if outfile!='':
        np.savez(outfile,PREDKAON=predicted_K_mu_mich,PREDLAMBDA=predicted_L,PREDK0S=predicted_K0s,NUMNU=num_nu)#,Counter(nu_type_K),Counter(nu_type_L),Counter(nu_type_K0s),primarymip,truth_interaction_map]))
    # raise Exception(potential_K.keys(),predicted_K.keys())
    return [predicted_K_mu_mich, predicted_L,predicted_K0s,num_nu]
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

    import argparse

    parser = argparse.ArgumentParser(description='Script to Run Analysis')
    parser.add_argument('--mode', type=str, choices=["Truth", "Reco"], help='Reco or Truth running mode')
    parser.add_argument('--dir', type=str, help='Directory of h5 files, npyfile will go in same level directory with _files replaced with _npy')
    # parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    assert args.mode
    mode=True
    if args.mode=="Reco":
        mode=False
    if args.mode=="Truth":
        mode=True
    print(f"Mode: {args.mode}")
    if args.dir:
        print(f"Dir: {args.dir}")
        SAVEDIR=os.path.dirname(sys.argv[1]).replace("_files","_npy")
        os.makedirs(SAVEDIR+"/npyfiles/", exist_ok=True)
        print(f"SAVEDIR: {SAVEDIR}")
        # raise Exception(SAVEDIR)
        main(
            mode=mode,
            HMh5=sys.argv[1]+"/analysis_HM_both.h5",
            analysish5=sys.argv[1]+"/analysis_both.h5",
            outfile=os.path.join(SAVEDIR,"npyfiles/"+os.path.basename(os.path.normpath(sys.argv[1])))
            )
    else:

        
        # FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_files/"
        # SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_npy/"
        FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files_reco/"
        SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files_reco/"


        print(f"Running Default Dir: {FILEDIR}")
        print(f"SAVEDIR: {SAVEDIR}")
        os.makedirs(SAVEDIR+"npyfiles/", exist_ok=True)
        for path in os.listdir(FILEDIR):
            main(
                mode=mode,
                HMh5=FILEDIR+path+"/analysis_HM_both.h5",
                analysish5=FILEDIR+path+"/analysis_both.h5",
                outfile=SAVEDIR+"npyfiles/"+path
            )
        
        # main(
        #     mode=False,
        #     HMh5=sys.argv[1]+"_reco/analysis_HM_reco.h5",
        #     analysish5=sys.argv[1]+"_reco/analysis_reco.h5",
        #     outfile=os.path.join(SAVEDIR,"npyfiles_reco/"+os.path.basename(os.path.normpath(sys.argv[1])))
        #     )

    
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
