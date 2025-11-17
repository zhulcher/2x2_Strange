import sys
import os
import yaml

from analysis.analysis_cuts import *
from collections import Counter

USE_HM=False

sys.path.append(SOFTWARE_DIR)
from spine.driver import Driver
from spine.io.core.read import HDF5Reader
from spine.data import Meta as Met


def closest_reco_particle_to_truth_start(p:ParticleType,particles:list[ParticleType],truth_particles:list[TruthParticle],skip:Optional[RecoParticle]=None)->Optional[ParticleType]:
    if type(p)==TruthParticle:
        return p
    assert type(p)==RecoParticle

    if not p.is_matched:
        return p
    best_match:TruthParticle=truth_particles[p.match_ids[0]]
    assert p.id in best_match.match_ids
    if len(best_match.match_ids)==1: 
        assert best_match.match_ids[0]==p.id
        return p


    closest_to_start=None
    best_dist=np.inf

    for i in best_match.match_ids:
        if type(skip)==RecoParticle:
            if skip.id==i: continue
        match_back=particles[i].match_overlaps[particles[i].match_ids==best_match.id]
        assert len(match_back)==1
        if p.reco_length<min_len/2: continue
        if match_back[0]<.1: continue
        # if best_match.match_overlaps[j]<.25: continue
        dist = np.min(cdist(particles[i].points, [best_match.start_point]))
        if dist<best_dist:
            closest_to_start=particles[i]
            best_dist=dist
    # if closest_to_start is None: return p
    # assert closest_to_start is not None
    return closest_to_start



def main(HMh5,analysish5,mode:bool=True,outfile='',compare_truth="True"):

    # This is the analysis file generated from the sample
    anaconfig = 'configs/anaconfig.cfg'
    anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', analysish5))
    if mode:
        anaconfig['build']['mode']='truth'
    else:
        anaconfig['build']['mode']='both'
    print(yaml.dump(anaconfig))
    driver = Driver(anaconfig)
    ######read in the analysis file that everyone looks at##################

    # potential_K:dict[int,list[PotK]]={}
    # predicted_K:dict[int,list[PredK]]={}
    predicted_K_mu_mich:list[PredKaonMuMich]=[]
    predicted_L:list[Pred_Neut]=[]
    predicted_K0s:list[Pred_Neut]=[]
    # inters:dict[tuple[int,int],Interaction]={}
        


    

    print("starting")
    # process_codes=[]
    reader=None
    if USE_HM: reader = HDF5Reader(HMh5) #set the file name
    for ENTRY_NUM in range(len(driver)):
        print(ENTRY_NUM)
        data = driver.process(entry=ENTRY_NUM)
                
                # perm = np.lexsort(data['points'].T)

        

        # true_kaons=[]
        # true_lambdas=[]
        # HIP=1
        # MIP=2
        if mode:
            particles:list[ParticleType] =data['truth_particles']
            interactions:list[InteractionType]=data['truth_interactions']
            
        else:
            particles:list[ParticleType] =data['reco_particles']
            interactions:list[InteractionType] =data['reco_interactions']

        truth_particles:list[TruthParticle] =data['truth_particles']
        truth_interactions:list[TruthInteraction]=data['truth_interactions']


        meta:Met= data['meta']

        perm_inverse=None
        HM_pred=None

        if USE_HM and not mode:
            assert reader is not None
            sparse3d_pcluster_semantics_HM=reader[ENTRY_NUM]['segmentation']


        if USE_HM and mode:
            assert reader is not None
            sparse3d_pcluster_semantics_HM=reader[ENTRY_NUM]['seg_label']
            index_set=set({})
            for p in particles:
                for i in p.index:
                    index_set.add(i)

            perm = np.lexsort(data['points_label'].T) 
            perm_inverse=np.argsort(perm)

            assert len(perm)==len(reader[ENTRY_NUM]['seg_label']),(len(perm),len(reader[ENTRY_NUM]['seg_label']))

            # assert len(perm)==len(sparse3d_pcluster_semantics_HM),(len(perm),len(sparse3d_pcluster_semantics_HM),len(reader[ENTRY_NUM]['seg_label']),len(data['points_label']),len(reader[ENTRY_NUM]['segmentation']),len(data['points']),len(index_set),max(index_set))
            
            for p_to_check in range(len(particles)):
            # p_to_check=random.randint(0, len(particles)-1)
                idx=perm_inverse[particles[p_to_check].index]
                voxels1= sparse3d_pcluster_semantics_HM[idx,1:4]
                voxels2=meta.to_px(particles[p_to_check].points,floor=True)
                assert set(meta.index(voxels1))==set(meta.index(voxels2)),(set(meta.index(voxels1)),set(meta.index(voxels2)),meta.to_px(data['points_label'][perm][:20],floor=True),reader[ENTRY_NUM]['seg_label'][:,1:4][:20])#(meta.to_px(data['points_label'][:10]),sparse3d_pcluster_semantics_HM[:10,1:4])#(len(set(meta.index(voxels1))),len(set(meta.index(voxels2))),np.shape(data['points_label']),np.shape(reader[ENTRY_NUM]['seg_label']),data['points_label'][:10],reader[ENTRY_NUM]['seg_label'][:10],meta.index(voxels1)[:10],meta.index(voxels2)[:10],voxels1[:10],voxels2[:10],mode,type(particles[0]),)
            # assert max(perm_inverse)<len(particles),(max(perm_inverse),len(particles))
            # STANDARD [SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]

            # HM [SHOWR_SHP, HIP_SHP, MIP_SHP, MICHL_SHP, DELTA_SHP, LOWES_SHP,UNKWN_SHP]
        # assert len(particles)-1==max([p.index for p in particles]),(len(particles)-1,max([p.index for p in particles]))


        # missing_indx=set(tuple(i) for i in meta.to_px(data['points_label'][perm],floor=True))-set(reader[ENTRY_NUM]['seg_label'][:,1:4])
        # raise Exception(len(missing_indx))

            set1 = set(map(tuple, meta.to_px(data['points_label'],floor=True)))
            set2 = set(map(tuple, reader[ENTRY_NUM]['seg_label'][:,1:4]))



            # Elements in a1 but not in a2
            missing_in_a2 = np.array(list(set1 - set2))
            # Elements in a2 but not in a1
            missing_in_a1 = np.array(list(set2 - set1))

        

            if len(missing_in_a2): raise Exception("In a1 but not in a2:\n", missing_in_a2)
            if len(missing_in_a1): raise Exception("In a2 but not in a1:\n", missing_in_a1)

        # if len(set1)!=len(set2):
        # raise Exception(len(set1),len(set2))
        
            HM_pred:list[np.ndarray]=[HIPMIP_pred(p,sparse3d_pcluster_semantics_HM,perm_inverse,mode=mode) for p in particles]
        # HM_acc:list[np.ndarray]=[HIPMIP_acc(p,sparse3d_pcluster_semantics_HM,perm_inverse,mode=mode) for p in particles]

        for i in range(len(interactions)):
            assert interactions[i].id==i,(interactions[i].id,i)
        for i in range(len(truth_interactions)):
            assert truth_interactions[i].id==i,(truth_interactions[i].id,i)

        #TODO fix this 
        # for p in particles:
            
        #     if p.num_voxels>0 and abs(p.pdg_code) in [2212,321,3112,3222]:
        #         if type(p)==TruthParticle:
        #             assert HM_pred[p.id]==HIP_HM,(p.pdg_code,p.shape,set(HM_pred),HM_pred[p.id],HIP_HM,len(p.points),p.num_voxels,HM_acc[p.id], Counter(sparse3d_pcluster_semantics_HM[perm_inverse[p.index], -1]))
                
        #             assert HM_acc[p.id]==1,(p.pdg_code,p.shape,set(HM_pred),HM_pred[p.id],HIP_HM,len(p.points),p.num_voxels,HM_acc[p.id], Counter(sparse3d_pcluster_semantics_HM[perm_inverse[p.index], -1]))

        

        # primarymip[ENTRY_NUM]=PrimaryMIP(particles,HM_pred)

        # truth_interaction_map[ENTRY_NUM]={}
    
        # if mode:
            # for inter in interactions:
                # assert type(inter)==TruthInteraction
                # if inter.nu_id!=-1 and is_contained(inter.vertex,margin=0):
                    # num_nu+=1

        # print("starting kaons")
        for hip_candidate in particles:
            if hip_candidate.interaction_id==-1: continue
            if hip_candidate.reco_length<=min_len/4: continue

            assert hip_candidate.id in [i.id for i in interactions[hip_candidate.interaction_id].particles],(hip_candidate.id,[i.id for i in interactions[hip_candidate.interaction_id].particles],type(hip_candidate),type(interactions[hip_candidate.interaction_id]),hip_candidate.interaction_id,hip_candidate.pdg_code)
            assert len(hip_candidate.points)
                # continue
            # if interactions[mip_candidate.interaction_id].nu_id==-1:continue



            # Truth=False
            Truth_K=False
            truth_list=[]
            reason=""
            if compare_truth:
                if mode:
                    assert type(hip_candidate)==TruthParticle
                    truth_list=np.array([hip_candidate.is_primary or (hip_candidate.parent_id==hip_candidate.id and hip_candidate.ancestor_pdg_code==321), # or np.isclose(np.linalg.norm(hip_candidate.position-hip_candidate.ancestor_position),0) #0 or (abs(hip_candidate.parent_pdg_code)==321 and process_map[hip_candidate.ancestor_creation_process]=='primary')
                            # hip_candidate.ke>K_MIN_KE, #1
                            # is_contained(hip_candidate.points), #2
                            # is_contained(interactions[hip_candidate.interaction_id].vertex,margin=margin0),
                            hip_candidate.pdg_code==321, #4), #3 
                            # all_daughters_contained(hip_candidate.ancestor_track_id,particles) #5
                            ])
                else:
                    # raise Exception(dir(hip_candidate))
                    assert type(hip_candidate)==RecoParticle


                    match=hip_candidate.match_ids
                    # assert len(match)>0

                    if len(match)>0:
                        m0=match[0]

                        # if 

                        match_hip=truth_particles[m0]
                        assert type(match_hip)==TruthParticle

                        truth_list=np.array([len(match)>0,
                            match_hip.is_primary or (match_hip.parent_id==match_hip.id and match_hip.ancestor_pdg_code==321), # or np.isclose(np.linalg.norm(hip_candidate.position-hip_candidate.ancestor_position),0) #0 or (abs(hip_candidate.parent_pdg_code)==321 and process_map[hip_candidate.ancestor_creation_process]=='primary')
                                # hip_candidate.ke>K_MIN_KE, #1
                                # is_contained(hip_candidate.points), #2
                                # is_contained(truth_interactions[match_hip.interaction_id].vertex,margin=margin0),
                                match_hip.pdg_code==321, #4), #3 
                                # all_daughters_contained(hip_candidate.ancestor_track_id,particles) #5
                                closest_reco_particle_to_truth_start(hip_candidate,particles,truth_particles)==hip_candidate,
                                ])
                    else:
                        truth_list=np.array([False,
                                False, # or np.isclose(np.linalg.norm(hip_candidate.position-hip_candidate.ancestor_position),0) #0 or (abs(hip_candidate.parent_pdg_code)==321 and process_map[hip_candidate.ancestor_creation_process]=='primary')
                                # hip_candidate.ke>K_MIN_KE, #1
                                # is_contained(hip_candidate.points), #2
                                False,
                                False, #4), #3 
                                # all_daughters_contained(hip_candidate.ancestor_track_id,particles) #5
                                ])



                    # raise Exception(particles[0].match_ids,particles[0].match_overlaps)
                    # match=truth_particles[particles[0].match_ids
                
            # pass_prelims=True
                Truth_K=np.all(truth_list)


                # if Truth_K:
                    # myint=interactions[hip_candidate.interaction_id]
                    # nu_type_K+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]

                    # if hip_candidate.interaction_id not in truth_interaction_map[ENTRY_NUM]:
                        # truth_interaction_map[ENTRY_NUM][hip_candidate.interaction_id]=[]

                    # truth_interaction_map[ENTRY_NUM][hip_candidate.interaction_id]+=[hip_candidate]
                if not Truth_K:
                    reason=str((np.argwhere(truth_list==False))[0][0])
            
            pass_prelims=is_contained(reco_vert_hotfix(interactions[hip_candidate.interaction_id]),margin=0)
            # pass_prelims=is_contained(interactions[hip_candidate.interaction_id].vertex)*is_contained(hip_candidate.points)
            pass_prelims*=(HM_pred_hotfix(hip_candidate,HM_pred)==HIP_HM or hip_candidate.pid in [PION_PID,PROT_PID])#hip_candidate.shape==TRACK_SHP#

            pass_prelims*=hip_candidate.is_primary

            pass_prelims*=len(interactions[hip_candidate.interaction_id].particles)>=3

            # if pass_prelims:

            #     if mode:
            #         dist = np.min(cdist(hip_candidate.points, [interactions[hip_candidate.interaction_id].vertex,interactions[hip_candidate.interaction_id].reco_vertex]))
            #     else:
                    
            #         if len([interactions[hip_candidate.interaction_id].vertex])==0:
            #             dist=np.inf
            #         else:
            #             assert len(hip_candidate.points)>0,(len(hip_candidate.points),len(hip_candidate.depositions))
            #             assert len(interactions[hip_candidate.interaction_id].vertex)>0,len(interactions[hip_candidate.interaction_id].vertex)
            #             dist = np.min(cdist(hip_candidate.points, [interactions[hip_candidate.interaction_id].vertex]))
                
            #     # pass_prelims*=(dist<10)
            fm=interactions[hip_candidate.interaction_id].is_flash_matched
            pass_prelims*=fm

            
            if not (Truth_K or pass_prelims):continue

            
            # print("kaon")
            # if type(mip_candidate)==TruthParticle:
                # if mip_candidate.creation_process not in ["Decay","primary","muIoni","conv","compt","eBrem","annihil","neutronInelastic","nCapture","muPairProd","muMinusCaptureAtRest"]:  process_codes+=[mip_candidate.creation_process]
                # if mip_candidate.pdg_code==-13 and mip_candidate.parent_pdg_code==321 and mip_candidate.creation_process=="6::201":
                #     print("found a muon from kaon but may not be contained")
            # if mip_candidate.reco_length>70 or mip_candidate.reco_length<15: continue
            
            
            # if not is_contained(hip_candidate.points): continue
            
            
            # if mip_candidate.reco_length<min_len: continue
            
            # if ENTRY_NUM not in predicted_K_mu_mich:
                # predicted_K_mu_mich[ENTRY_NUM] = []
            # print(mip_candidate.reco_length)
            # if 

            # delta_shp
            predicted_K_mu_mich+=[PredKaonMuMich(ENTRY_NUM,hip_candidate,particles,interactions,HM_pred,truth=Truth_K,reason=reason,truth_list=truth_list,truth_particles=truth_particles,truth_interactions=truth_interactions)]
        # print("starting lambda")

        if mode:
            existing_parent_track_ids=Counter([p.parent_track_id for p in particles])
            existing_track_ids=Counter([p.track_id for p in particles])
        else:
            existing_parent_track_ids=Counter([truth_particles[p.match_ids[0]].parent_track_id for p in particles if len(p.match_ids)>0])
            existing_track_ids=Counter([truth_particles[p.match_ids[0]].track_id for p in particles if len(p.match_ids)>0])

        for lam_hip_candidate in particles:
            #TODO fix this 
            if lam_hip_candidate.interaction_id==-1: continue
            assert len(lam_hip_candidate.points)
            if lam_hip_candidate.reco_length<=min_len/4: continue
            assert lam_hip_candidate.id in [i.id for i in interactions[lam_hip_candidate.interaction_id].particles],(lam_hip_candidate.id,[i.id for i in interactions[lam_hip_candidate.interaction_id].particles],type(lam_hip_candidate),type(interactions[lam_hip_candidate.interaction_id]),lam_hip_candidate.interaction_id,lam_hip_candidate.pdg_code)
                # continue
            for lam_mip_candidate in particles:

                if lam_mip_candidate.id==lam_hip_candidate.id:
                    continue
                if lam_mip_candidate.reco_length<=min_len/4: continue
                if lam_mip_candidate.interaction_id==-1: continue
                assert lam_mip_candidate.id in [i.id for i in interactions[lam_mip_candidate.interaction_id].particles],(lam_mip_candidate.id,[i.id for i in interactions[lam_mip_candidate.interaction_id].particles],type(lam_mip_candidate),type(interactions[lam_mip_candidate.interaction_id]),lam_mip_candidate.interaction_id,lam_mip_candidate.pdg_code)
                assert len(lam_mip_candidate.points)
                    # continue
                if lam_mip_candidate.interaction_id!=lam_hip_candidate.interaction_id: continue
                Truth_lam=False
                reason=""
                if compare_truth:
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
                                    # is_contained(interactions[lam_hip_candidate.interaction_id].vertex,margin=margin0),#5
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
                    else:
                        # raise Exception(dir(lam_mip_candidate))
                        assert type(lam_mip_candidate)==RecoParticle
                        assert type(lam_hip_candidate)==RecoParticle

                        hip_match=lam_hip_candidate.match_ids
                        mip_match=lam_mip_candidate.match_ids



                        if len(hip_match)>0 and len(mip_match)>0:
                            # assert 
                            mh=hip_match[0]

                            match_hip=truth_particles[mh]

                            
                            # assert 
                            mm=mip_match[0]

                            match_mip=truth_particles[mm]



                            truth_list=np.array([len(hip_match)>0,
                                                len(mip_match)>0,
                                                match_mip.parent_pdg_code==3122,#0
                                                match_hip.ancestor_track_id==match_mip.ancestor_track_id,
                                                (match_hip.parent_pdg_code==3122 and match_hip.parent_track_id==match_mip.parent_track_id and set([process_map[match_mip.creation_process],process_map[match_hip.creation_process]])==set(['6::201'])) or 
                                                    (match_hip.parent_pdg_code==2212 and process_map[match_mip.creation_process]=='6::201' and match_hip.creation_process=='protonInelastic' and
                                                    match_hip.parent_creation_process=='Decay' and existing_parent_track_ids[match_hip.parent_track_id]==1 and existing_track_ids[match_hip.parent_track_id]==0),#2
                                        match_mip.pdg_code==-211,#3
                                        match_hip.pdg_code==2212,#4
                                        # is_contained(truth_interactions[match_hip.interaction_id].vertex,margin=margin0),#5
                                        # is_contained(match_mip.points),
                                        # is_contained(truthinteractions[match_mip.interaction_id].vertex),
                                        # is_contained(match_hip.points),
                                        # is_contained(truthinteractions[match_hip.interaction_id].vertex),
                                        # process_map[match_mip.creation_process] in ['6::201',"4::151"],# or match_mip.creation_process=='lambdaInelastic',#6
                                        # process_map[match_mip.parent_creation_process]=='primary' or (match_mip.ancestor_pdg_code==3122 and process_map[match_mip.ancestor_creation_process]=='primary'),#7#TODO this is weakened due to a bug 
                                        match_mip.ancestor_pdg_code==3122,#7
                                        # all_daughters_contained(match_mip.ancestor_track_id,particles),#8
                                        # process_map[match_hip.creation_process] in ['6::201',"4::151"],# or match_hip.creation_process=='lambdaInelastic',#9
                                        # process_map[match_hip.parent_creation_process]=='primary'  or (match_hip.ancestor_pdg_code==3122 and process_map[match_hip.ancestor_creation_process]=='primary'),#10#TODO this is weakened due to a bug 
                                        match_hip.ancestor_pdg_code==3122,#10,
                                        closest_reco_particle_to_truth_start(lam_mip_candidate,particles,truth_particles,skip=lam_hip_candidate)==lam_mip_candidate,
                                        closest_reco_particle_to_truth_start(lam_hip_candidate,particles,truth_particles,skip=lam_mip_candidate)==lam_hip_candidate,
                                        # unique_hip_match,
                                        # unique_mip_match
                                        # all_daughters_contained(match_hip.ancestor_track_id,particles)#11
                                        ])
                        else:truth_list=np.array([len(hip_match)>0,len(mip_match)>0,
                                False,#0
                                        False,
                                        False,
                                        False,#3
                                        False,#4
                                        False,#5
                                        # is_contained(match_mip.points),
                                        # is_contained(truthinteractions[match_mip.interaction_id].vertex),
                                        # is_contained(match_hip.points),
                                        # is_contained(truthinteractions[match_hip.interaction_id].vertex),
                                        # process_map[match_mip.creation_process] in ['6::201',"4::151"],# or match_mip.creation_process=='lambdaInelastic',#6
                                        # process_map[match_mip.parent_creation_process]=='primary' or (match_mip.ancestor_pdg_code==3122 and process_map[match_mip.ancestor_creation_process]=='primary'),#7#TODO this is weakened due to a bug 
                                        False,#7
                                        # all_daughters_contained(match_mip.ancestor_track_id,particles),#8
                                        # process_map[match_hip.creation_process] in ['6::201',"4::151"],# or match_hip.creation_process=='lambdaInelastic',#9
                                        # process_map[match_hip.parent_creation_process]=='primary'  or (match_hip.ancestor_pdg_code==3122 and process_map[match_hip.ancestor_creation_process]=='primary'),#10#TODO this is weakened due to a bug 
                                        False,#10,
                                        # all_daughters_contained(match_hip.ancestor_track_id,particles)#11
                                        ])


                    Truth_lam=np.all(truth_list)
                    # if Truth_lam:
                        # myint=interactions[lam_hip_candidate.interaction_id]
                        # nu_type_L+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]

                        # if lam_hip_candidate.interaction_id not in truth_interaction_map[ENTRY_NUM]:
                            # truth_interaction_map[ENTRY_NUM][lam_hip_candidate.interaction_id]=[]

                        # truth_interaction_map[ENTRY_NUM][lam_hip_candidate.interaction_id]+=[lam_hip_candidate,lam_mip_candidate]
                    if not Truth_lam:
                        reason=str((np.argwhere(truth_list==False))[0][0])
                # pass_prelims=True
                if mode:
                    pass_prelims=is_contained(reco_vert_hotfix(interactions[lam_hip_candidate.interaction_id]),margin=0)
                else:
                    pass_prelims=is_contained(reco_vert_hotfix(interactions[lam_hip_candidate.interaction_id]),margin=0)

                # pass_prelims=is_contained(interactions[lam_hip_candidate.interaction_id].vertex)*is_contained(lam_hip_candidate.points)
                # pass_prelims*=is_contained(interactions[lam_mip_candidate.interaction_id].vertex)*is_contained(lam_mip_candidate.points)
                pass_prelims*=(HM_pred_hotfix(lam_hip_candidate,HM_pred)==HIP_HM or lam_hip_candidate.pid in [3,4])#lam_hip_candidate.shape==TRACK_SHP#
                pass_prelims*=(HM_pred_hotfix(lam_mip_candidate,HM_pred)==MIP_HM or lam_mip_candidate.pid in [3,4])#lam_mip_candidate.shape==TRACK_SHP

                if pass_prelims:

                    dist = np.min(cdist(lam_hip_candidate.points, lam_mip_candidate.points))
                
                    pass_prelims*=(dist<10)

                fm=interactions[lam_hip_candidate.interaction_id].is_flash_matched
                # if pass_prelims and not mode:
                   
                #     if not fm:
                #     # if not interactions[hip_candidate.interaction_id].is_flash_matched:
                #         for v in interactions:
                #             if v.is_flash_matched:
                #                 if np.linalg.norm(reco_vert_hotfix(v)-reco_vert_hotfix(interactions[lam_hip_candidate.interaction_id]))<500:
                #                     fm=True
                #                     break
                #                 # continue
                #             # if i.id==self.interaction.id:
                #                 # continue
                #             # if i.is_flash_matched:
                #             # print(v[0],inter)

                #             # if v[2]==self.interaction.id: continue
                #                 # if 
                #                 # d=np.linalg.norm(v[0]-inter)

                            
                #             # if d<best_dist:
                #             #     best_dist=d
                #             #     inter=v[0]
                pass_prelims*=fm

                if not (Truth_lam or pass_prelims):
                    continue

                

                # if ENTRY_NUM not in predicted_L:
                    # predicted_L[ENTRY_NUM]=[]
                predicted_L+=[Pred_Neut(ENTRY_NUM,lam_hip_candidate,lam_mip_candidate,particles,interactions,HM_pred,truth=Truth_lam,reason=reason,mass1=PROT_MASS,mass2=PION_MASS,truth_particles=truth_particles,truth_interactions=truth_interactions)]


        # for K0s_mip1_candidate in particles:
        #     for K0s_mip2_candidate in particles:
        #         if K0s_mip1_candidate.interaction_id!=K0s_mip2_candidate.interaction_id: continue
        #         if K0s_mip1_candidate.id>=K0s_mip2_candidate.id:continue
        #         Truth_K0s=False
        #         reason=""
        #         if compare_truth:
        #             if mode:
        #                 # if K0s_mip1_candidate.parent_pdg_code==310 and K0s_mip2_candidate.parent_pdg_code==310: raise Exception("k0s anc",K0s_mip1_candidate.ancestor_pdg_code,K0s_mip1_candidate.creation_process,K0s_mip2_candidate.parent_creation_process,K0s_mip2_candidate.ancestor_creation_process,K0s_mip1_candidate.pdg_code,K0s_mip2_candidate.pdg_code)
        #                 assert type(K0s_mip1_candidate)==TruthParticle
        #                 assert type(K0s_mip2_candidate)==TruthParticle
        #                 truth_list=np.array([K0s_mip1_candidate.parent_pdg_code==310,#0
        #                             K0s_mip2_candidate.parent_pdg_code==310,#1
        #                             K0s_mip1_candidate.parent_track_id==K0s_mip2_candidate.parent_track_id,#2
        #                             abs(K0s_mip1_candidate.pdg_code)==211,#3
        #                             abs(K0s_mip2_candidate.pdg_code)==211,#4
        #                             is_contained(interactions[K0s_mip1_candidate.interaction_id].vertex,margin=margin0),#5
        #                             process_map[K0s_mip1_candidate.creation_process]=='6::201',#6
        #                             process_map[K0s_mip1_candidate.parent_creation_process]=='primary' or (K0s_mip1_candidate.ancestor_pdg_code in [310,311] and process_map[K0s_mip1_candidate.ancestor_creation_process] in ['primary','6::201']),#7
        #                             process_map[K0s_mip2_candidate.creation_process]=='6::201',#8
        #                             process_map[K0s_mip2_candidate.parent_creation_process]=='primary'  or (K0s_mip2_candidate.ancestor_pdg_code in [310,311] and process_map[K0s_mip2_candidate.ancestor_creation_process] in ['primary','6::201']),#9
        #                 ])
        #             else:
        #                 raise Exception(dir(K0s_mip1_candidate))

        #             Truth_K0s=np.all(truth_list)
        #             # if Truth_K0s:
        #                 # myint=interactions[K0s_mip1_candidate.interaction_id]
        #                 # nu_type_K0s+=[(myint.current_type,myint.lepton_pdg_code,myint.interaction_type,myint.interaction_mode)]
        #             if not Truth_K0s:
        #                 reason=str((np.argwhere(truth_list==False))[0][0])
        #         # pass_prelims=True
        #         pass_prelims=is_contained(reco_vert_hotfix(interactions[K0s_mip1_candidate.interaction_id]),margin=0)

                

                
        #         pass_prelims*=HM_pred_hotfix(K0s_mip1_candidate,HM_pred)==MIP_HM
        #         pass_prelims*=HM_pred_hotfix(K0s_mip2_candidate,HM_pred)==MIP_HM

        #         if pass_prelims:

        #             dist = np.min(cdist(K0s_mip1_candidate.points, K0s_mip2_candidate.points))
                
        #             pass_prelims*=(dist<10)

        #         if not (Truth_K0s or pass_prelims):
        #             continue

                

        #         if ENTRY_NUM not in predicted_K0s:
        #             predicted_K0s[ENTRY_NUM]=[]
        #         predicted_K0s[ENTRY_NUM]+=[Pred_Neut(K0s_mip1_candidate,K0s_mip2_candidate,particles,interactions,HM_acc,HM_pred,truth=Truth_K0s,reason=reason,mass1=PION_MASS,mass2=PION_MASS)]


    print(predicted_K_mu_mich)
    # raise Exception(process_codes)
    # print(predicted_K_michel)
    print(predicted_L)
    print(predicted_K0s)
    # print(primarymip)
    # print(truth_interaction_map)
    print("saving to ", outfile)
    # raise Exception("hold on",predicted_K_mu_mich,predicted_L)
    if outfile!='':
        
    
        np.savez_compressed(outfile,PREDKAON=predicted_K_mu_mich,PREDLAMBDA=predicted_L)#,PREDK0S=predicted_K0s,NUMNU=num_nu)#,Counter(nu_type_K),Counter(nu_type_L),Counter(nu_type_K0s),primarymip,truth_interaction_map]))
        import subprocess
        from pathlib import Path
        for s in ["statistics_plot_kp.py","statistics_plot_lam.py"]:#,"statistics_plot_lam.py","statistics_plot_assoc_CCNC.py"]:
            for m in ["reco"]:#,"truth"]:
                script = Path(__file__).parent / s
                cmd = [sys.executable, str(script), "--mode", m,"--single_file",os.path.basename(outfile+".npz")]
                subprocess.run(cmd, check=True)

    # raise Exception(potential_K.keys(),predicted_K.keys())
    return [predicted_K_mu_mich, predicted_L,predicted_K0s]
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Script to Run Analysis')
    parser.add_argument('--mode', type=str, choices=["truth", "reco"], help='Reco or Truth running mode')
    parser.add_argument('--dir', type=str,default="", help='Directory of h5 files, npyfile will go in same level directory with _files replaced with _npy')
    parser.add_argument('--compare_truth', type=str, choices=["True", "False"],default="True", help='Should truth info be recorded')
    # parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    assert args.mode
    # assert args.dir
    # mode=True
    if args.mode=="reco":
        mode=False
    elif args.mode=="truth":
        mode=True
    else:
        raise Exception("bad mode",args.mode)
    print(f"Mode: {args.mode}")
    if args.dir:
        print(f"Dir: {args.dir}")
        SAVEDIR=os.path.dirname(args.dir).replace("_files","_analysis")
        os.makedirs(SAVEDIR+"/npyfiles/", exist_ok=True)
        print(f"SAVEDIR: {SAVEDIR}")
        # raise Exception(SAVEDIR)

        analysis_both = os.path.join(args.dir, "analysis_both.h5")
        analysis_reco = os.path.join(args.dir, "analysis_reco.h5")

        analysis_HM_both = os.path.join(args.dir, "analysis_HM_both.h5")
        analysis_HM_reco = os.path.join(args.dir, "analysis_HM_reco.h5")

        # Prefer analysis_both.h5, but fall back to analysis_reco.h5 if it exists
        if os.path.exists(analysis_both):
            analysish5 = analysis_both
            analysisHMh5=analysis_HM_both
        elif os.path.exists(analysis_reco):
            analysish5 = analysis_reco
            analysisHMh5=analysis_HM_reco
        else:
            raise FileNotFoundError("Neither analysis_both.h5 nor analysis_reco.h5 found in {}".format(args.dir))
        main(
            mode=mode,
            HMh5=analysisHMh5,
            analysish5=analysish5,
            outfile=os.path.join(SAVEDIR,"npyfiles/"+os.path.basename(os.path.normpath(args.dir))),
            compare_truth=args.compare_truth
            )
    else:

        
        # FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_files/"
        # SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_npy/"
        # FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files_reco/"
        # SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files_reco/"

        FILEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files_truth/"
        SAVEDIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_analysis/"




        print(f"Running Default Dir: {FILEDIR}")
        print(f"SAVEDIR: {SAVEDIR}")
        os.makedirs(SAVEDIR+"npyfiles/", exist_ok=True)
        for path in os.listdir(FILEDIR):
            analysis_both = os.path.join(FILEDIR+path, "analysis_both.h5")
            analysis_reco = os.path.join(FILEDIR+path, "analysis_reco.h5")

            # Prefer analysis_both.h5, but fall back to analysis_reco.h5 if it exists
            if os.path.exists(analysis_both):
                analysish5 = analysis_both
            elif os.path.exists(analysis_reco):
                analysish5 = analysis_reco
            else:
                raise FileNotFoundError("Neither analysis_both.h5 nor analysis_reco.h5 found in {}".format(FILEDIR+path))
            main(
                mode=mode,
                HMh5=FILEDIR+path+"/analysis_HM_both.h5",
                analysish5=analysish5 ,
                # outfile=SAVEDIR+"npyfiles/"+path,
                compare_truth=args.compare_truth
            )

