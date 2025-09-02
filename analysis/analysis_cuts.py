""" 
This file contains output classes and cut functions useful for reconstructing kaons and 
lambdas in a liquid argon TPC using the reconstruction package SPINE https://github.com/DeepLearnPhysics/spine
"""
# from ast import Module
# from collections import Counter

# from numba import bool_



import copy
# from _pytest.monkeypatch import V
from scipy.spatial.distance import cdist
# from sympy import false                   
# import string
SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf

import sys
# Set software directory
sys.path.append(SOFTWARE_DIR)
from spine.data.out import TruthParticle,RecoParticle,RecoInteraction,TruthInteraction
from spine.utils.globals import DELTA_SHP, MUON_PID,PHOT_PID, PION_PID, PROT_PID,KAON_PID,KAON_MASS,PROT_MASS, PION_MASS,MICHL_SHP,TRACK_SHP,SHOWR_SHP,LOWES_SHP
from spine.utils.geo.manager import Geometry
from spine.utils.energy_loss import csda_ke_lar
# from spine.utils.vertex import get_pseudovertex
import numpy as np
# from scipy import stats as st

# from copy import deepcopy
from numba import njit

@njit
def norm3d(arr):
    # out = np.empty(len(arr),dtype=np.float64)
    # for i in range(len(arr)):
        # x, y, z = arr[i]
        # out[i] = 
    return (arr[0]**2 + arr[1]**2 + arr[2]**2) ** 0.5



margin0=[[15,15],[15,15],[10,60]]

PI0_MASS=135.0 #[MeV/c^2]


E_PI0_Kp_Decay=(KAON_MASS**2+PI0_MASS**2-PION_MASS**2)/2/KAON_MASS
P_PI0_Kp_Decay=np.sqrt(E_PI0_Kp_Decay**2-PI0_MASS**2)

# from scipy.spatial.distance import cdist


# TODO spine.TruthParticle.mass propagated to truth particles in larcv
# TODO spine.TruthParticle.parent_end_momentum
# TODO id vs parentid vs orig id vs the actual parentid that I want

# TODO some sort of particle flow predictor
# TODO decay at rest predictor?

# TODO deal with particles with small scatters wih len cut

# TODO Kaon/ Michel flash timing?
# TODO add k0s info

#TODO add sigma photon as a primary 

drift_dir_map={}

analysis_type='icarus'
if analysis_type=='2x2':
    full_containment='detector'
else:
    full_containment='module'
    drift_dir_map={0:np.array([-1,0,0]),
                   1:np.array([1,0,0]),
                   2:np.array([-1,0,0]),
                   3:np.array([1,0,0]),}

min_len=2.5


HIP_HM = 7
MIP_HM = TRACK_SHP
SHOWR_HM = SHOWR_SHP
MICHL_HM=MICHL_SHP

LAM_MASS = 1115.683   # [MeV/c^2]
SIG0_MASS = 1192.642  # [MeV/c^2]


# Particle = RecoParticle | TruthParticle
Interaction = RecoInteraction | TruthInteraction

Particle = TruthParticle|RecoParticle


Model="icarus"

process_map={}

if Model=="2x2":
    process_map["4::121"]="4::121"
    process_map["6::201"]="6::201"
    process_map["4::151"]="4::151"

if Model=="icarus":
    for i in ["primary"]:
        process_map[i]=i
    for i in ["muIoni","conv","compt","eBrem","annihil","nCapture","muPairProd","muMinusCaptureAtRest","hIoni","muBrems","photonNuclear","eIoni","electronNuclear","phot",'',"hadElastic","hBrems","hPairProd"]:
        process_map[i]="other"
    process_map["muonNuclear"]="maybe_something"
    process_map["Decay"]="6::201"
    for i in ['pi+Inelastic', 'protonInelastic',"pi-Inelastic","kaon0LInelastic","kaon-Inelastic","kaon+Inelastic","lambdaInelastic","dInelastic","anti-lambdaInelastic","sigma-Inelastic","kaon0SInelastic","sigma+Inelastic","anti_neutronInelastic","neutronInelastic","anti_protonInelastic","tInelastic",'He3Inelastic','anti_sigma-Inelastic','alphaInelastic']:
        process_map[i]="4::121"
    process_map["hBertiniCaptureAtRest"]="4::151"


def reco_vert_hotfix(inter:Interaction):
    if type(inter)==TruthInteraction:
        return inter.reco_vertex
    elif type(inter)==RecoInteraction:
        return inter.vertex
    else:
        raise Exception("type not allowed",type(inter))
    

def truth_interaction_id_hotfix(inter:Interaction):
    if type(inter)==TruthInteraction:
        return inter.id
    elif type(inter)==RecoInteraction:
        if inter.is_matched:
            return inter.match_ids[0]
        return None
    else:
        raise Exception("type not allowed",type(inter))


def is_primary_hotfix(p:Particle)->bool:

    if type(p)==TruthParticle:
        if abs(p.pdg_code) in [3222,3112] and process_map[p.creation_process]=="primary":
            return True
    return p.is_primary

def HM_pred_hotfix(p:Particle,hm_pred:dict[int,np.ndarray],old=False)->int:


    if old and hm_pred is not None:
        if type(p)==TruthParticle:
            if p.shape==TRACK_SHP:
                if p.pdg_code in [2212,321,-321]:
                    assert np.argmax(hm_pred[p.id])==HIP_HM and p.shape==TRACK_SHP,(p.shape,hm_pred[p.id],p.pdg_code)
            if abs(p.pdg_code) in [3222,3112,2212,321]:
                if np.argmax(hm_pred[p.id])!=HIP_HM:print("THIS IS AN ERROR TO FIX???????? Particle Grouping",p.pdg_code)
                return HIP_HM
        if p.shape!=TRACK_SHP:
            return p.shape
        assert p.shape==TRACK_SHP
        # assert (np.sum(hm_pred[p.id][MIP_HM])+np.sum(hm_pred[p.id][HIP_HM]))>0,hm_pred[p.id]
        if hm_pred[p.id][MIP_HM]>hm_pred[p.id][HIP_HM]:
            return MIP_HM
        else:
            assert hm_pred[p.id][HIP_HM]>=hm_pred[p.id][MIP_HM]
            return HIP_HM
    if p.shape!=TRACK_SHP:
        return p.shape
    if p.pid in [2,3]:
        return MIP_HM
    if p.pid in [4,5]:
        return HIP_HM
    if type(p)==TruthParticle:
        if p.pdg_code==3222:
            return HIP_HM
    else:
        raise Exception(p.shape,p.pid,p.pdg_code)
    return 1000
    # if p.shape!=TRACK_SHP: return p.shape
    # return np.argmax(hm_pred[p.id])


def angle_between(a, b):
    anorm= norm3d(a)
    bnorm= norm3d(b)
    if anorm==0 or bnorm==0:
        cos_theta=1
    else:
        assert anorm>0 and bnorm>0
        a = np.asarray(a)
        b = np.asarray(b)
        cos_theta = np.dot(a, b) / (anorm * bnorm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  



Geo = Geometry(detector=Model)

# class ExtraChild:
#     # child_id:int
#     dist_to_parent:float
#     # proj_dist_to_parent:float
#     angle_to_parent:float
#     # truth:bool
#     child:Particle
#     child_hm_pred: int

#     """
#     Storage class for extra children of any particle
#     Attributes
#     ----------
#     dist_to_parent: float
#         parent end to child start distance
#     proj_dist_to_parent: float
#         parent to child projected distance of closest approach
#     angle_to_parent: float
#         angle between child start direction and line from parent end to child start
#     truth: bool
#         is this a true child of the parent?
#     child: Particle
#         Particle object for the child
#     child_hm_pred: int
#         HM prediction for the child
#     parent: None|Particle
#         If the parent particle is specified, ignore the start point and use the point of closest approach from the parent (line) to the child (line)
#     """
#     def __init__(self, child:Particle,parent_end_pt:np.ndarray,hm_pred:list[np.ndarray],parent:None|Particle=None):
#         self.child=child
        
#         par_end=parent_end_pt
#         self.dist_to_parent=float(norm3d(child.start_point-par_end))
#         # self.proj_dist_to_parent=np.nan
#         # if parent!=None and not is_contained(parent.points,mode=full_containment):
#             # print(parent.reco_end_dir,parent.reco_end_dir)
#             # par_end = get_pseudovertex(
#             #     start_points=np.array([parent.end_point, child.start_point], dtype=float),
#             #     directions=[parent.reco_end_dir, child.reco_start_dir],
#             # )
#             # self.proj_dist_to_parent=collision_distance(parent,child,orientation=["end","start"])[-1]

#         self.child_hm_pred=HM_pred_hotfix(child,hm_pred)
#         if self.dist_to_parent==0:
#             self.angle_to_parent=0
#         # else:
#             # self.angle_to_parent=direction_acos((child.start_point-par_end)/self.dist_to_parent,child.reco_start_dir)
#         # self.truth=False
#         """
#         Initialize with all of the necessary particle attributes

#         Parameters
#         ----------
#         child : Particle
#             child particle
#         parent_end_pt: list[float]
#             end point of parent particle
#         hm_pred: list[np.ndarray]
#             HM predictions for the particles
#         """


# class PotK:
#     """
#     Storage class for primary Kaons and their cut parameters

#     Attributes
#     ----------
#     hip_len: float
#         len attribute of the particle object
#     dir_acos: float
#         arccos of the particle direction with the beam direction
#     k_hm_acc:float
#         percent of the voxels of this particle
#         whose Hip/Mip semantic segmentation matches the overall prediction
#     """

#     hip_len: float
#     dir_acos: float
#     k_hm_acc: float
#     dist_from_hip: float
#     proj_dist_from_hip:float
#     k_extra_children: list[ExtraChild]
#     # dist_Hausdorff_from_hip:float
#     pred_end: list[float]
#     truth: bool
#     hip: Particle
#     truth_list:list[bool]
    

#     def __init__(self, kaon:Particle,hm_acc:float,mip:Particle,particles:list[Particle],interactions:list[Interaction],set_truth:bool):
#         """
#         Initialize with all of the necessary particle attributes

#         Parameters
#         ----------
#         kaon : Particle
#             Kaon candidate particle object
#         hm_acc:float
#             percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
#         """
#         self.hip_len = kaon.reco_length
#         self.dir_acos = direction_acos(kaon.reco_start_dir)
#         self.k_hm_acc = hm_acc
#         self.k_extra_children=[]
#         self.dist_from_hip=0
#         self.proj_dist_from_hip=0
#         # self.dist_Hausdorff_from_hip=0
#         self.hip=kaon
#         self.truth_list=[False]

        
        # if set_truth:
        #     assert isinstance(mip,TruthParticle) and isinstance(self.hip,TruthParticle) and isinstance(kaon,TruthParticle)
        #     vis_part=[part.orig_id for part in particles if part.reco_length>3]
        #     # true_par_id=mip.parent_id
        #     fake_hip=kaon
        #     id_to_particle={}
        #     for par in particles:
        #         id_to_particle[par.orig_id]=par

        #     done=False
        #     while not done:
        #         done=True
        #         for id in fake_hip.children_id:
        #             if id not in id_to_particle:
        #                 break
        #             if abs(id_to_particle[id].pdg_code)==321 and norm3d(fake_hip.end_point-id_to_particle[id].start_point)<3 and id not in vis_part:
        #                 done=False
        #                 fake_hip=id_to_particle[id]
        #                 break
        #     self.truth_list=[abs(fake_hip.pdg_code)==321,(mip.parent_id==fake_hip.orig_id or (mip.parent_id in fake_hip.children_id and mip.parent_id not in vis_part)),abs(mip.parent_pdg_code)==321,(self.hip.is_primary or bool(norm3d(self.hip.start_point-interactions[mip.interaction_id].vertex)<min_len)),bool(norm3d(fake_hip.end_point-mip.start_point)<3)]
        # self.truth=bool(np.all(self.truth_list))

# class PotMich:
#     """
#     Storage class for primary Kaons with muon child and michel and their cut parameters

#     Attributes
#     ----------
#     mich_id : int
#         id for hip associated to this class
#     dist_to_mich: float
#         distance from mich start to mip end
#     mu_extra_children: list[ExtraChild]
#         extra children for the mip
#     HM_acc_mich:float
#         percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
#     """

#     mich_id: int
#     dist_to_mich: float
#     proj_dist_to_mich:float
#     mu_extra_children: list[ExtraChild]
#     # decay_t_to_dist: float
#     mich_hm_acc: float
#     truth: bool
#     mich:Particle

#     def __init__(
#         self, mich:Particle, mich_hm_acc
#     ):
#         """
#         Initialize with all of the necessary particle attributes

#         Parameters
#         ----------
#         pred_K:PredK
#             PredK object with associated hip and mip information
#         mich_id : int
#             id for hip associated to this class
#         dist_to_mich: float
#             distance from mich start to mip end
#         mu_extra_children: list[ExtraChild]
#             extra children parameters as defined in 'children' function for the mip
#         HM_acc_mich:float
#             percent of the voxels of this particle whose Hip/Mip semantic segmentation
#             matches the overall prediction
#         """

#         self.mich_id = mich.id
#         self.dist_to_mich = 0
#         self.proj_dist_to_mich=0
#         self.mu_extra_children = []
#         self.mich_hm_acc = mich_hm_acc
#         self.truth=False
#         self.mich=mich

#         # self.decay_t=decay_t
#         # self.decay_sep=decay_sep




class PredKaonMuMich:
    __slots__ = ('event_number', 'truth','reason','pass_failure','error',
                'truth_list','hip','hm_pred','fm_interactions','particles',
                'potential_kaons','truth_interaction_vertex','truth_Kp','truth_interaction_nu_id',
                'truth_hip','truth_michel','truth_pi0_gamma','real_K_momentum',
                'truth_interaction_overlap','decay_mip_dict','truth_interaction_id','is_flash_matched','match_overlaps','reco_vertex','primary_particle_counts')
    """
    Storage class for primary Kaons with muon child and their cut parameters

    Attributes
    ----------
    mip_len_base: float
        len attribute of the particle object
    potential_michels: list[Particle]
        list of michel candidates corresponding to this mip candidate
    truth: bool
        is this a truth muon coming from a kaon
    """

    # mip_len_base: float
    # mu_hm_acc: float
    # potential_kaons: list[PotK]
    # potential_michels: list[PotMich]
    # mu_extra_children:list[ExtraChild]
    truth:bool
    # mip:Particle

    # true_signal:bool
    potential_kaons:list[tuple[Particle,list[tuple[list[Particle],list[str]]]]]

    def __init__(
        self,
        # pot_k: PotK,
        ENTRY_NUM:int,
        K_hip: Particle,
        particles: list[Particle],
        interactions: list[Interaction],
        # hm_acc:list[float],
        hm_pred:dict[int,np.ndarray],
        
        truth:bool,
        reason:bool,
        truth_list,
        truth_particles,
        truth_interactions
    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        pot_k:PotK
            PotK object with associated hip information
        dist_to_hip: float
            distance from mip start to hip end
        """

        # self.mip_len_base=mip.reco_length
        # self.mu_hm_acc = hm_acc[mip.id]
        # self.potential_michels=[]
        self.event_number=ENTRY_NUM
        self.truth=truth
        self.reason=reason
        self.pass_failure=""
        # self.mip=mip
        # self.true_signal=False
        self.error=""
        self.truth_list=truth_list
        self.hip=K_hip
        # self.accepted_mu=None
        # self.hm_pred=hm_pred
        


        

        

        self.fm_interactions=[(reco_vert_hotfix(i),i.is_flash_matched,i.id) for i in interactions]
        if self.truth and type(Particle)==TruthParticle:
            assert abs(self.hip.pdg_code)==321,self.hip.pdg_code
        interaction=interactions[self.hip.interaction_id]
        self.particles=[p for p in interaction.particles if len(p.points)>=3 and is_contained(p.start_point,margin=-5) and (p.shape in [MICHL_SHP,SHOWR_SHP,LOWES_SHP,DELTA_SHP] or is_contained(p.end_point,margin=-5))]
        self.potential_kaons=[]

        self.is_flash_matched=interaction.is_flash_matched
        # interaction.


        self.hm_pred={}
        if hm_pred is not None:
            for i in range(len(hm_pred)):
                if particles[i].interaction_id==interaction.id:
                    self.hm_pred[particles[i].id]=hm_pred[particles[i].id]


        self.truth_interaction_vertex=[np.nan,np.nan,np.nan]
        self.truth_interaction_nu_id=None
        self.truth_Kp={}

        self.truth_pi0_gamma={}

        self.truth_michel={}

        self.truth_interaction_id=truth_interaction_id_hotfix(interaction)

        # self.truth_mips={}

        self.truth_interaction_overlap=[[],[]]

        self.match_overlaps=interaction.match_overlaps

        self.reco_vertex=reco_vert_hotfix(interaction)

        self.primary_particle_counts=interaction.primary_particle_counts


        # self.primaries=[]

        self.decay_mip_dict={}

        if type(self.hip)==TruthParticle:
            assert type(interactions[0])==TruthInteraction,(type(interactions[0]))
            assert type(interaction)==TruthInteraction
            self.truth_interaction_vertex=interaction.vertex
            self.truth_interaction_nu_id=interaction.nu_id
            self.truth_hip=self.hip
                

        else:
            assert type(interactions[0])==RecoInteraction,type(interactions[0])
            self.truth_hip=None
            if self.hip.is_matched:
                self.truth_hip=truth_particles[self.hip.match_ids[0]]
            if interaction.is_matched:
                self.truth_interaction_overlap=[truth_interactions[interaction.match_ids[0]].match_ids,truth_interactions[interaction.match_ids[0]].match_overlaps]
                self.truth_interaction_vertex=truth_interactions[interaction.match_ids[0]].vertex
                self.truth_interaction_nu_id=truth_interactions[interaction.match_ids[0]].nu_id

        for n,k in enumerate(truth_particles):
            if k.is_primary and k.pdg_code==321:
                self.truth_Kp[n]=k
            if k.pdg_code==22 and k.parent_pdg_code==111 and k.ancestor_pdg_code==321 and process_map[k.parent_creation_process]=='6::201':
                self.truth_pi0_gamma[n]=k
            # if k.pdg_code in [-13,211] and k.parent_pdg_code==321 and process_map[k.creation_process]=='6::201':
            #     self.truth_mips[n]=k
            if abs(k.pdg_code) in [211,13] and process_map[k.creation_process]=='6::201' and k.parent_pdg_code==321 and k.ancestor_pdg_code==321:
                self.decay_mip_dict[n]=k
            if k.pdg_code==-11 and k.parent_pdg_code in [-13,211] and process_map[k.creation_process]=='6::201' and k.ancestor_pdg_code==321:
                self.truth_michel[n]=k
            # if k.is_primary:
                # self.primaries+=[n]

        
        # for n,i in enumerate(truth_particles):
            # i
        assert self.hip in interaction.particles,(self.hip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles],type(interaction))


        

        self.real_K_momentum=K_hip.reco_momentum
        if self.truth and type(Particle)==TruthParticle:
            assert type(K_hip)==TruthParticle
            self.real_K_momentum=momentum_from_children_ke(K_hip,particles,KAON_MASS)

        if type(K_hip)==TruthParticle and self.truth:
            assert HM_pred_hotfix(K_hip,hm_pred)==HIP_HM,(HM_pred_hotfix(K_hip,hm_pred),HIP_HM,K_hip.id,K_hip.pdg_code)
        # done=False
        # while not done: #this loop goes over all of the hips connected to the end of the kaon, and constructs a hadronic group which hopefully contains the kaon end. 
        #     done=True
        #     # print("looking")
        #     for p in particles:
        #         # print([r[0] for r in self.potential_kaons])
        #         if p not in [r[0] for r in self.potential_kaons] and HM_pred_hotfix(p,hm_pred)==HIP_HM:
        #             # print("getting here")
        #             for k in list(self.potential_kaons).copy():
        #                 # print(k[0])
        #                 if norm3d(p.start_point-k[0].end_point)<min_len:
                            
        #                     self.potential_kaons+=[[p,[],0]]
        #                     done=False

        # for k in self.potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group 
        #     for p in particles:
        #         if p in [r[0] for r in self.potential_kaons]:continue
        #         if HM_pred_hotfix(p,hm_pred)==HIP_HM and norm3d(p.start_point-k[0].end_point)<min_len:
        #             raise Exception("how",HM_pred_hotfix(p,hm_pred),[r[0].id for r in self.potential_kaons])
        #         if HM_pred_hotfix(p,hm_pred)==MICHL_HM:
        #             continue
        #         if is_primary_hotfix(p):
        #             continue
                
        #         if HM_pred_hotfix(p,hm_pred)==MIP_HM and norm3d(p.start_point-k[0].end_point)<min_len:
                    
        #             add_it=True
        #             for c in particles: #this loop looks for mips or hips at the end of this mip and rejects it if so
        #                 #TODO allow mips at the end of the mip, and add the lengths
        #                 #TODO add in counts for particles connecting at each end
        #                 if HM_pred_hotfix(c,hm_pred) in [HIP_HM] and norm3d(c.start_point-p.end_point)<min_len:
        #                     add_it=False
        #                     break
        #                 # if HM_pred_hotfix(c,hm_pred) in [HIP_HM,MIP_HM] and norm3d(c.start_point-p.start_point)<min_len and norm3d(c.end_point-p.start_point)>norm3d(c.start_point-p.start_point):
        #                 #     add_it=False
        #                 #     break
        #             if add_it:
        #                 k[1]+=[p]
        #             # else:
        #             #     k[1]+=[-1] ????
        #         if HM_pred_hotfix(p,hm_pred)==SHOWR_HM and norm3d(p.start_point-k[0].end_point)<14*4:
        #             k[2]+=1
        # # print(self.potential_kaons)

        


        # else: print("good end")
    # @profile
    def pass_cuts(self,cuts:dict[str,dict[str,bool|list]|bool])->bool:

        self.pass_failure=[]
        # self.potential_kaons=[[self.hip,[],[]]]
        # passing_len=False
        # klens=[]
        # michlens=[]
        # passed=True

        # closest_flash_matched_interaction=None
        # best_dist=np.inf
        # particles:list[Particle]=interaction.particles
        # vertex=reco_vert_hotfix(i)

        # inter=reco_vert_hotfix(interaction)
        inter=self.reco_vertex

        if True:
            if True:
                if type(self.hip)==TruthParticle:
                    NON_PRIMARY_HIPS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred)==HIP_HM and (not is_primary_hotfix(p))]
                    NON_PRIMARY_MIPS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred)==MIP_HM and (not is_primary_hotfix(p))]
                    NON_PRIMARY_MICHLS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred) in [MICHL_SHP,LOWES_SHP,SHOWR_SHP] and (not is_primary_hotfix(p))]# and p.reco_ke<60] #
                    NON_PRIMARY_SHWRS=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred) in [SHOWR_SHP,LOWES_SHP]  and (not is_primary_hotfix(p))] #
                else:
                    NON_PRIMARY_HIPS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred)==HIP_HM]
                    NON_PRIMARY_MIPS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred)==MIP_HM]
                    NON_PRIMARY_MICHLS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred) in [MICHL_SHP,LOWES_SHP,SHOWR_SHP]]# and p.reco_ke<60] #(not is_primary_hotfix(p))
                    NON_PRIMARY_SHWRS:list[Particle]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred) in [SHOWR_SHP,LOWES_SHP]] #and (not is_primary_hotfix(p))

                # assert self.hip in interaction.particles,(self.hip.id,[i.id for i in particles],type(interaction))#[i.id for i in interaction.primary_particles]
                
                MIP_CHAINS:list[tuple[list[Particle],list[str]]]=[([i],[]) for i in copy.copy(NON_PRIMARY_MIPS)]

                #TODO bring this back but fixed?
                for i in MIP_CHAINS:
                    done=False
                    while done==False:
                        done=True
                        for j in NON_PRIMARY_MIPS:
                            n1=norm3d(i[0][-1].end_point-j.start_point)
                            if n1<min_len and n1<norm3d(i[0][-1].start_point-j.start_point) and j not in i:# and angle_between(i[-1].end_dir,j.start_dir)<np.pi/12:
                                i[0].append(j)
                                done=False
                        # for j in NON_PRIMARY_HIPS:
                        #     n1=norm3d(i[-1].end_point-j.start_point)
                        #     if n1<min_len and n1<norm3d(i[-1].start_point-j.start_point) and j not in i:
                        #         # i+=[j]
                        #         MIP_CHAINS.remove(i)
                        #         done=True
                        #         break
                            
                
                self.potential_kaons=[(self.hip,copy.deepcopy(MIP_CHAINS))]
                done=False

                # if self.hip.momentum[2]<0 and np.abs(self.hip.momentum[2])>norm3d(self.hip.momentum[:2]):
                #     # self.pass_failure+=[c]
                #     # passed=False
                #     ship_copy=copy.deepcopy(self.hip)
                # # if self.hip.momentum[2]<0:# and np.abs(self.hip.momentum[2])>norm3d(self.hip.momentum[:2]):
                #     # ship_copy=copy.deepcopy(self.hip)
                #     ship_copy.start_point,ship_copy.end_point=self.hip.end_point,self.hip.start_point
                #     self.potential_kaons+=[[ship_copy,copy.copy(MIP_CHAINS),[]]]
                #     assert np.isclose(ship_copy.start_point@self.hip.end_point,self.hip.end_point@self.hip.end_point),(ship_copy.start_point@self.hip.end_point,self.hip.end_point@self.hip.end_point)

                
                

                while not done: #this loop goes over all of the hips connected to the end of the kaon, and constructs a hadronic group which hopefully contains the kaon end. 
                    done=True
                    # print("looking")
                    
                    for p in NON_PRIMARY_HIPS:
                        # print([r[0] for r in self.potential_kaons])
                        if p not in [r[0] for r in self.potential_kaons]:
                            # print("getting here")
                            for k in self.potential_kaons:
                                # print(k[0])
                                n1=norm3d(p.start_point-k[0].end_point)
                                n2=norm3d(p.start_point-k[0].start_point)
                                if n1<min_len and n1<n2:

                                    self.potential_kaons+=[(p,copy.deepcopy(MIP_CHAINS))]
                                    done=False
                                    break
                                # elif n2<min_len and n2<n1:
                                #     self.potential_kaons+=[[p,copy.copy(MIP_CHAINS),[]]]
                                #     done=False
                                #     break

                # ship_copy=copy.deepcopy(self.hip)
                # if self.hip.momentum[2]<0:# and np.abs(self.hip.momentum[2])>norm3d(self.hip.momentum[:2]):
                #     # ship_copy=copy.deepcopy(self.hip)
                #     ship_copy.start_point,ship_copy.end_point=self.hip.end_point,self.hip.start_point
                #     self.potential_kaons+=[[ship_copy,copy.copy(MIP_CHAINS),[]]]

                # print()
                
                
                # Geo = Geometry(detector="icarus")
                # cath_pos=Geo.tpc[Geo.get_closest_module([[-.00001,0,0]])[0]].cathode_pos


        # if type(self.hip)==RecoParticle:
            

            # if not self.is_flash_matched:
            #     for v in self.fm_interactions:
            #         # if np.isclose(norm3d(v[0]-inter),0): continue
            #         if not v[1]: continue

            #         # print("FM_interactions",v[0],self.reco_vertex)
            #         # if i.id==interaction.id:
            #             # continue
            #         # if i.is_flash_matched:
            #         d=norm3d(v[0]-inter)
            #         if d<best_dist:
            #             best_dist=d
            #                 # closest_flash_matched_interaction=i
            # # particles+=closest_flash_matched_interaction.particles
            # # if reco_vert_hotfix(i)[2]<inter:
            #     # vertex=reco_vert_hotfix(i)
            # # else:

            

            
            # assert best_dist!=0
        # if self.is_flash_matched:
            # print("its flashmatched")
        # if not self.is_flash_matched:
            # print("best_dist",best_dist,type(self.hip)==RecoParticle,inter)

        
        # hm_pred=self.hm_pred



        
        
        for c in cuts:
            # if len(self.pass_failure)>=3: break

            checked=False

            if c=="Contained HIP":
                checked=True
                if not is_contained(self.hip.end_point,margin=1):
                    self.pass_failure+=[c]
                    #passed=False

            elif c=="Connected Non-Primary MIP":
                checked=True
                for k in self.potential_kaons:
                    for p in copy.copy(k[1]):
                            # assert type(p)==list[Particle],(type(p),type[p[0]])
                            # plen=np.sum([i.reco_length for i in p])

                            # if plen>=40:
                            #     mini_checked=True
                            #     k[2]+=[[p,r]]
                            #     k[1].remove(p)
                            #     continue

                            mip_start=p[0][0].start_point
                            mip_end=p[0][-1].end_point



                            n1=norm3d(mip_start-k[0].end_point)
                            if (n1>min_len or n1>=norm3d(mip_start-k[0].start_point)):
                                if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                    k[1].remove(p)
                                else:
                                    p[1].append(c)
                if np.sum([len(k[1]) for k in self.potential_kaons])==0:
                    self.pass_failure+=[c]
                    #passed=False


            elif c=="Contained MIP":
                checked=True
                for k in self.potential_kaons:
                    for p in copy.copy(k[1]):
                            # assert type(p)==list[Particle],(type(p),type[p[0]])
                            # plen=np.sum([i.reco_length for i in p])

                            # if plen>=40:
                            #     mini_checked=True
                            #     k[2]+=[[p,r]]
                            #     k[1].remove(p)
                            #     continue

                            # mip_start=p[0].start_point
                            mip_end=p[0][-1].end_point

                            if not is_contained(mip_end,margin=1):
                                # k[2]+=[[p,c]]
                                if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                    k[1].remove(p)
                                else:
                                    p[1].append(c)
                # print(np.sum([len(k[1]) for k in self.potential_kaons]))
                if np.sum([len(k[1]) for k in self.potential_kaons])==0:
                    self.pass_failure+=[c]
                    #passed=False




            

            elif c=="Initial HIP":
                checked=True
                if HM_pred_hotfix(self.hip,self.hm_pred)!=HIP_HM and self.hip.pid not in  [3,4]:
                    self.pass_failure+=[c]
                    #passed=False

            
            elif c==rf"Primary $K^+$":
                checked=True
                # if type(interaction)==TruthInteraction:
                if (not is_primary_hotfix(self.hip)) or norm3d(inter-self.hip.start_point)>5*min_len:# or HM_pred_hotfix(self.hip,hm_pred)!=HIP_HM:
                    self.pass_failure+=[c]
                    #passed=False
                # else:
                #     if not is_primary_hotfix(self.hip):#norm3d(self.hip.start_point-inter)>cuts[c] and 
                #         self.pass_failure+=[c]
                #         #passed=False

            elif c=="Correct HIP TPC Assoc.":
                checked=True
                # if get_tpc_id(self.hip.start_point) not in set([i[1] for i in self.hip.sources]) and get_tpc_id(self.hip.end_point) not in set([i[1] for i in self.hip.sources]):#self.hip.module_ids:
                if not self.hip.is_contained:
                    
                    self.pass_failure+=[c]
                    #passed=False

                    # if self.truth:
                        # print("Correct HIP Module Assoc.",get_tpc_id(self.hip.start_point),get_tpc_id(self.hip.end_point),self.hip.sources)


            elif c=="Kaon Len":
                checked=True
                if self.hip.reco_length<cuts[c]:
                    self.pass_failure+=[c]
                    #passed=False

            elif c=="Valid Interaction":
                checked=True
                if type(self.hip)==TruthParticle:
                    if self.truth_interaction_nu_id==-1:
                        self.pass_failure+=[c]
                        #passed=False
                elif type(self.hip)==RecoParticle:
                    # print()
                    if not self.is_flash_matched:# and best_dist>cuts[c]:
                        if self.truth: print("FAILED AT VALID INTERACTION")#,best_dist)
                        self.pass_failure+=[c]
                        #passed=False
                else:
                    raise Exception(type(self.hip))
                

            elif c=="Forward HIP":
                checked=True
                # if self.hip.momentum[2]<0 and np.abs(self.hip.momentum[2])>cuts[c]*norm3d(self.hip.momentum[:2]):
                if angle_between(self.hip.momentum,np.array([0,0,1]))>cuts[c]:
                    found_another=False
                    for p in self.particles:
                        if p.id!=self.hip.id and norm3d(p.start_point-self.hip.start_point)<min_len:
                            found_another=True
                            break
                    if not found_another:
                        self.pass_failure+=[c]
                        #passed=False
            #         ship_copy=copy.deepcopy(self.hip)
            #     # if self.hip.momentum[2]<0:# and np.abs(self.hip.momentum[2])>norm3d(self.hip.momentum[:2]):
            #         # ship_copy=copy.deepcopy(self.hip)
            #         ship_copy.start_point,ship_copy.end_point=self.hip.end_point,self.hip.start_point
            #         self.potential_kaons+=[[ship_copy,copy.copy(MIP_CHAINS),[]]]
            #         assert np.isclose(ship_copy.start_point@self.hip.end_point,self.hip.end_point@self.hip.end_point),(ship_copy.start_point@self.hip.end_point,self.hip.end_point@self.hip.end_point)
            elif c=="More Than Downgoing":
                checked=True
                ending=[]
                starting=[]
                for p in self.particles:
                    if norm3d(p.end_point-inter)<min_len/2:
                        ending+=[p]
                    if norm3d(p.start_point-inter)<min_len/2:
                        starting+=[p]
                    if len(ending)>1 or len(starting)>1:
                        break
                if len(starting)==1 and len(ending)==1:
                    # if norm3d(starting[0].start_point-ending[0].end_point)<min_len:
                    if np.abs(angle_between(starting[0].momentum,ending[0].end_dir)-np.pi/2)>np.pi/2-cuts[c][1] and np.abs(angle_between(starting[0].momentum,np.array([0,-1,0]))-np.pi/2)>np.pi/2-cuts[c][0]:
                        self.pass_failure+=[c]
                        #passed=False

                elif len(starting)==1 and angle_between(starting[0].momentum,np.array([0,-1,0]))<cuts[c][0]:
                    self.pass_failure+=[c]
                    #passed=False
                
            elif c=="":
                checked=True
                # #passed=False

                self.pass_failure+=[""]
            
            elif c=="MIP_CUTS":
                checked=True

                # Geo.define_containment_volumes(-5, mode=full_containment)
                # conp=[p for p in particles if len(p.points)>=3 and is_contained(p.start_point,margin=-5,define_con=False) and (p.shape in [MICHL_SHP,SHOWR_SHP,LOWES_SHP,DELTA_SHP] or is_contained(p.end_point,margin=-5,define_con=False))]
                # conp=[p for p in particles if len(p.points)>=3 and p.is_contained]
                
        
                

                for r in cuts[c]:
                    if len(self.pass_failure)>=3: break

                    mini_checked=False
                    # skip_the_rest=False
                    # for rr in cuts[c]:
                        # if skip_the_rest and cuts[c]!="Single MIP Decay": continue
                        # potk=deepcopy(self.potential_kaons)
                    # found=False
                    # if np.sum([len(k[1]) for k in self.potential_kaons])==0:
                    #     mini_checked=True

                    #TODO does this mip len thing actually do what I want????????
                    for k in self.potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                        for p in copy.copy(k[1]):
                            # assert type(p)==list[Particle],(type(p),type[p[0]])
                            plen=np.sum([i.reco_length for i in p[0]])

                            # if plen>=40:
                            #     mini_checked=True
                            #     k[2]+=[[p,r]]
                            #     k[1].remove(p)
                            #     continue

                            mip_start=p[0][0].start_point
                            mip_end=p[0][-1].end_point
                        # for p in NON_PRIMARY_MIPS:

                            #NICE CHECK TODO assert np.all([HM_pred_hotfix(i,self.hm_pred)==MIP_HM for i in p])
                            # if is_primary_hotfix(p):
                            #     continue
                            # if p in PK: continue
                            # if HM_pred_hotfix(p,hm_pred)==HIP_HM and norm3d(p.start_point-k[0].end_point)<min_len:
                            #     raise Exception("how",HM_pred_hotfix(p,hm_pred),[r[0].id for r in self.potential_kaons])
                            # if HM_pred_hotfix(p,hm_pred)!=MIP_HM:
                            #     continue
                            # add_it=True


                            # if r=="Connected Non-Primary MIP":
                            #     mini_checked=True
                            # #(norm3d(kk.start_point-self.hip.end_point)<=norm3d(kk.start_point-self.hip.start_point))
                            #     n1=norm3d(mip_start-k[0].end_point)
                            #     if (n1>min_len or n1>=norm3d(mip_start-k[0].start_point)):
                            #         k[2]+=[[p,r]]
                            #         k[1].remove(p)
                                    
                                    # add_it=False

                            if r==r"HIP $K^+ or >0 Scatter$":
                                mini_checked=True
                                if k[0]==self.hip and self.hip.pid in [2,3]:
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                                
                            

                            #TODO allow mips at the end of the mip, and add the lengths
                            #TODO add in counts for particles connecting at each end

                            if r=="Michel Child":
                                mini_checked=True
                                mich_child=False 
                                for other in NON_PRIMARY_MICHLS:
                                    if other.reco_ke>80: continue
                                    # if other==p[-1]:
                                    #     continue
                                        # raise Exception("How?",p[-1].id,p[-1].shape,other.id,other.shape)
                                    # if norm3d(other.start_point-p.end_point)<min_len*5: #TODO this is due to a bug
                                    if np.min(cdist(other.points, [mip_end]))<cuts[c][r][0] and np.min(cdist(other.points[:,1:], [mip_end[1:]]))<cuts[c][r][1] and norm3d(other.start_point-mip_end)<norm3d(other.start_point-mip_start):# and np.dot(other.start_point-mip_end,drift_dir_map[get_tpc_id(mip_end)])>-min_len:
                                        mich_child=True
                                        break
                                if not mich_child:
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                                    


                            if r==r"Low MIP len $\pi^0$ Tag":
                                mini_checked=True
                                # if p.reco_length!=-1:print(p.reco_length)

                                
                                has_pi0=(plen>=40)
                                if not has_pi0:
                                    for other in NON_PRIMARY_SHWRS:
                                        if other.reco_ke<10: continue# i think the minimal energy of a photon from the pi0 is around 20MeV
                                        if other.reco_ke>300: continue# i think the maximum energy is around 225 MeV
                                        if (norm3d(other.start_point-mip_start)<cuts[c][r][0] 
                                            # and norm3d(other.start_point-mip_end)>min_len 
                                            and np.min(cdist(other.points, [mip_end]))>min_len
                                            and impact_parameter(mip_start,other.start_point,other.momentum)<impact_parameter(self.hip.start_point,other.start_point,other.momentum)
                                            and (impact_parameter(mip_start,other.start_point,other.momentum)<cuts[c][r][1])):#or angle_between(other.start_point-mip_start,other.momentum)<cuts[c][r][2])
                                            # and cos_gamma_to_pip_bounds(other.reco_ke)[0]<np.cos(angle_between(other.start_point-mip_start,p[0].momentum)) and np.cos(angle_between(other.start_point-mip_start,p[0].momentum))<cos_gamma_to_pip_bounds(other.reco_ke)[1]):
                                            # and abs((cos_gamma_to_E(mip_start,other.start_point,p[0].momentum)-other.reco_ke)/other.reco_ke)<cuts[c][r][2]):
                                            has_pi0+=1 #TODO apparently a 3112 doesnt get a reco length?
                                if not has_pi0:
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                                    

                            if r=="MIP Child At Most 1 Michel":
                                mini_checked=True
                                # for other in NON_PRIMARY_HIPS:
                                check1=np.any([(norm3d(other.start_point-mip_end)<min_len)*(other.reco_length>min_len) for other in NON_PRIMARY_HIPS if other!=p[-1]])
                                check2=np.any([(norm3d(other.start_point-mip_end)<min_len)*(other.reco_length>min_len) for other in NON_PRIMARY_MIPS if other!=p[-1]])
                                if check1 or check2:

                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                                    
                                    # add_it=False
                                    
                                
                            # if r=="MIP Child At Most 1 Michel":
                            #     mini_checked=True
                            #     # for other in NON_PRIMARY_MIPS:
                            #     if :
                            #         k[2]+=[[p,r]]
                            #         k[1].remove(p)
                            #         # break
                            #         # add_it=False
                            #         # break
                            #     # break
                            if r=="Single MIP Decay":
                                mini_checked=True
                                if np.sum([(norm3d(other.start_point-mip_start)<min_len)*(other.reco_length>min_len)*(norm3d(other.start_point-mip_start)<norm3d(other.start_point-self.hip.start_point)) for other in NON_PRIMARY_MIPS+NON_PRIMARY_HIPS if other!=self.hip])>1:
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                                # if other.shape==TRACK_SHP and norm3d(other.start_point-p.start_point)<min_len and other.reco_length>min_len:
                                    # add_it=False
                                    # ???????
                                    
                            if r=="Bragg Peak":
                                mini_checked=True
                                # close_voxels=[]
                                found=False
                                
                                for i in self.particles:
                                    if found: break
                                    # assert type(i)==Particle
                                    # print(i.num_voxels,len(i.points),len(i.depositions))
                                    for v in range(len(i.points)):
                                        if norm3d(i.points[v]-k[0].end_point)<min_len:
                                            if i.depositions[v]>=cuts[c][r]:
                                                found=True
                                                break
                                if not found:
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                            if r=="Come to Rest":
                                mini_checked=True
                                ctr=come_to_rest(k[0])
                                if ctr<cuts[c][r][0] or cuts[c][r][1]<ctr:
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)





                            if r=="Valid MIP Len":
                                mini_checked=True
                                if not ((cuts[c][r]["mu_len"][0]<plen and plen<cuts[c][r]["mu_len"][1]) or 
                                        (cuts[c][r]["pi_len"][0]<plen and plen<cuts[c][r]["pi_len"][1])):
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)


                            if r=="Correct MIP TPC Assoc.":
                                mini_checked=True
                                if not p[0][0].is_contained or not p[0][-1].is_contained:
                                    
                                    # k[2]+=[[p,r]]
                                    if len(self.pass_failure) or len(p[1]):
                                # k[2]+=[[p,c]]
                                        k[1].remove(p)
                                    else:
                                        p[1].append(c)

                                    if self.truth:
                                        print("Correct MIP Module Assoc.",get_tpc_id(p[0][0].start_point),set([i[1] for i in p[0][0].sources]),get_tpc_id(p[0][-1].start_point),set([i[1] for i in p[0][-1].sources]))


                                    
                                    # ?????????????

                            # for other in particles: #this loop looks for mips or hips at the end of this mip and rejects it if so
                                # if c in PK: continue
                                

                                # if other.shape in [MICHL_SHP,LOWES_SHP] and 
                                    
                                # elif c.orig_parent_id==p.orig_id and abs(p.pdg_code)==13 and abs(other.pdg_code)==11 and norm3d(other.start_point-p.end_point)<min_len:
                                #         raise Exception(HM_pred_hotfix(other,hm_pred),other.shape,p.shape,norm3d(other.start_point-p.end_point))

                                # if other.shape in [SHOWR_SHP,LOWES_SHP] and :
                                    

                                
                                    
                                
                                    
                                
                            # if r=="Michel Child":
                            #     checked=True
                            #     if not mich_child:
                            #         # add_it=False

                            
                                    # add_it=False

                            
                                    # add_it=False
                                    
                                        
                            # # if rounds[r]>=rounds["Contained MIP"] and not is_contained(p.points):
                            # #     add_it=False
                            #     # continue
                            # found=found or add_it

                            # if r==list(cuts[c].keys())[-2] and c==list(cuts[c].keys())[-2]:
                            #     k[1]+=[(p,passed*add_it)]
                        # valid=0
                        # for k in potk:
                        #     # if len(k[1])!=1: validlist+=[False]
                        #     for kk in k[1]: valid+=1
                                # if kk==-1:
                                #     continue
                                    # if self.truth: 
                                    #     print("contained mips")
                                    # if passed:
                                    #     self.pass_failure="contained mips"
                                    #     #passed=False

                                # else:
                                

                        # if self.hip.reco_length<min_len:
                        #     if self.truth: 
                        #         print("kaon_too_short")
                        #     if passed:
                        #         self.pass_failure="kaon_too_short"
                        #         #passed=False
                        # if not found:?????????????

                    if np.sum([len(k[1]) for k in self.potential_kaons])==0:
                        mini_checked=True
                        self.pass_failure+=[r]
                        #passed=False
                            # skip_the_rest=True       

                    if not mini_checked:
                        raise Exception(c,r, "not found in K+ MIP cuts")
                    
            if not checked:
                raise Exception(c,"not found in K+ cuts")
            

        # print("TRYING A KAON",self.truth)

        # # potential_kaons=[]
        # done=False
        # while not done:
        #     done=True
        #     for r in self.particles:



        # for k in self.potential_kaons:
        #     if k.proj_dist_from_hip<cuts["par_child_dist max"][0] and len([(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in k.k_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and (p.child_hm_pred in [SHOWR_HM,MICHL_HM] or p.child.reco_length>min_len)])==0 and (Model=='2x2' or min(norm3d(k.hip.end_point-self.mip.start_point),norm3d(k.hip.start_point-self.mip.start_point))<cuts["par_child_dist max"][0]) and norm3d(k.hip.end_point-self.mip.start_point)<norm3d(k.hip.start_point-self.mip.start_point):
        #         klens+=[k.dist_from_hip]
        # for mich in self.potential_michels:
        #     if mich.proj_dist_to_mich<cuts["par_child_dist max"][0]:
        #         michlens+=[mich.dist_to_mich]
        
        # childcut=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in self.mu_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and p.child_hm_pred in [HIP_HM,MIP_HM] and ((direction_acos(p.child.reco_start_dir,self.mip.reco_end_dir)>np.pi/12 and p.child.reco_length>min_len) or (not is_contained(p.child.end_point,mode=full_containment)))]
        # if len(childcut)!=0:
        #     if self.truth: 
        #         print("muon child",self.truth, childcut)
        #     if passed:
        #         self.pass_failure="muon child"
                
        #     passed=False
        


        # if np.any([not is_contained(k[0].points,mode=full_containment) for k in self.potential_kaons]):
        #     if self.truth: 
        #         print("contained kaon")
        #     if passed:
        #         self.pass_failure="contained kaon"
        #         passed=False

        # if np.any([np.any(np.array(k[1])==-1) for k in self.potential_kaons]):
        #     if self.truth: 
        #         print("contained mips")
        #     if passed:
        #         self.pass_failure="contained mips"
        #         passed=False

        # if interaction.id!==-1:
        #     if self.truth: 
        #         print("neutrino interaction")
        #     if passed:
        #         self.pass_failure="neutrino interaction"
        #         passed=False


        

        

        # if not np.any(validlist):
            # self.good_mu=self.potential_kaons[np.argwhere(validlist)[0][0]][0]

            # assert type(self.good_mu)==Particle
            # if self.truth: 
            #     print("valid muon or pion decay",len(self.potential_kaons))#,[k[1:] for k in self.potential_kaons],validlist)
            # if passed:
            

        # if self.hip.reco_ke<40:
        #     # if self.truth: 
        #     #     print("min KE",len(self.potential_kaons))#,[k[1:] for k in self.potential_kaons],validlist)
        #     # if passed:
        #     self.pass_failure+=["min KE"]
        #     passed=False
            

        # if not is_contained(self.mip.points,mode=full_containment):
        #     if len(klens)==0: 
        #         if self.truth:
        #             print("missing required kaon")
        #         if passed:
        #             self.pass_failure="missing required kaon"
        #         passed=False
        # else:
        #     klens+=[0]

        # if cuts["require_michel"] or self.require_michel:
        #     if len(self.potential_michels)==0: 
        #         if self.truth:
        #             print("missing required michel")
        #         if passed:
        #             self.pass_failure="missing required michel"
        #         passed=False
        # else:
        #     michlens+=[0]

        # for k in klens:
        #     # for mich in michlens:
        #     #     estlen=self.mip_len_base+k+mich
        #     #     if cuts["mu_len"][0]<estlen and estlen<cuts["mu_len"][1]:
        #     #         passing_len=True
        #     #         break
        #     if passing_len: break
        # if not passing_len:
        #     if self.truth:
        #         print("muon passing len",klens,michlens,self.mip_len_base)
        #     if passed:
        #         self.pass_failure="muon passing len"
        #     passed=False
        # return passed #TODO more cuts

        # if self.pass_failure==["Valid MIP Len","Forward HIP"]:

            # self.hip.start_point,self.hip.end_point=self.hip.end_point,self.hip.start_point
            # print("start")
            # print(self.hip.momentum)
            # self.hip.momentum*=-1
            # print(self.hip.momentum)
            # print("end")
            # passed=self.pass_cuts(cuts)
            # self.hip.start_point,self.hip.end_point=self.hip.end_point,self.hip.start_point
            # self.hip.momentum*=-1
            # assert len(self.potential_kaons)
            # return 

        
        return len(self.pass_failure)==1


class Pred_Neut:
    # __slots__ = ('a', 'b')
    __slots__ = ('event_number','mip','hip','hm_pred','truth','reason','pass_failure','error','mass1','mass2',
                'truth_hip','truth_mip','particles','fm_interactions','truth_interaction_vertex','truth_interaction_overlap',
                'truth_interaction_nu_id','real_hip_momentum','real_mip_momentum','real_hip_momentum_reco','real_mip_momentum_reco',
                'vae','truth_interaction_id','is_flash_matched','reco_vertex','primary_particle_counts')
    """
    Storage class for neutrals with two track decays after some distance

    Attributes
    ----------
    vae: float
        angle between the line constructed from the momenta of the hip and mip and
        the line constructed from the interaction vertex and the decay point
    mass2:float
        reconstructed mass squared
    momenta: list[float]
        shape(4) [hip transv. momentum, mip transv. momentum,hip long momentum, mip long momentum]
    # coll_dist: float
    #     shape(3) [t1,t2, dist]: the distance from the start point to the
    #     point along the vector start direction which is the point of
    #     closest approach to the other particle's corresponding line for the hip (t1) and mip (t2)
    #     along with the distance of closest approach of these lines (dist)
    dir_acos:float
        arccos of the direction with the beam direction
    # hip_extra_children: list[ExtraChild]
    #     extra children for the hip
    # mip_extra_children: list[ExtraChild]
    #     extra children for the mip
    # extra_children: list[ExtraChild]
    #     extra children for the neutral
    mip: Particle
        candidate pion
    hip: Particle
        candidate proton
    """

    vae: float
    mass1: float
    mass2: float
    momenta: list[float]
    # coll_dist: list[float]
    dir_acos: float
    # prot_extra_children: list[ExtraChild]
    # pi_extra_children: list[ExtraChild]
    # extra_children: list[ExtraChild]
    truth:bool
    mip: Particle
    hip: Particle
    mass1:float
    mass2:float

    #TODO add in flag that the particle was added in special and check for this after
    

    def __init__(
        self,
        ENTRY_NUM:int,
        hip:Particle,
        mip:Particle,
        particles:list[Particle],
        interactions: list[Interaction],
        hm_pred:dict[int,np.ndarray],
        truth:bool,
        reason:bool,
        mass1:float,
        mass2:float,
        truth_particles:list[Particle],
        truth_interactions:list[Interaction]

    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        hip: Particle
            hip particle object for convenience, this will be identified with particle/mass1
        mip: Particle
            mip particle object
        particles: list[Particle]
            list of particle objects for the event
        interactions: list[Interaction]
            list of interactions for the event
        hm_pred: list[np.ndarray]
            hip mip semantic segmentation prediction for each particle
        truth:bool
            is this a signal neutral or not
        """

        self.event_number=ENTRY_NUM
        interaction=interactions[hip.interaction_id]
        self.particles=[p for p in interaction.particles if len(p.points)>=3 and is_contained(p.start_point,margin=-5) and (p.shape in [MICHL_SHP,SHOWR_SHP,LOWES_SHP,DELTA_SHP] or is_contained(p.end_point,margin=-5))]
        self.mip=mip
        self.hip=hip

        # self.hm_pred=hm_pred

        self.hm_pred={}
        if hm_pred is not None:
            for i in range(len(hm_pred)):
                if particles[i].interaction_id==interaction.id:
                    self.hm_pred[particles[i].id]=hm_pred[particles[i].id]
        
        self.truth=truth
        self.reason=reason
        self.pass_failure=""
        self.error=""

        self.mass1=mass1
        self.mass2=mass2

        self.truth_interaction_vertex=[np.nan,np.nan,np.nan]
        self.truth_interaction_nu_id=None

        self.fm_interactions=[(reco_vert_hotfix(i),i.is_flash_matched,i.id) for i in interactions]


        self.truth_interaction_overlap=[[],[]]

        if type(self.hip)==TruthParticle:
            assert type(interactions[0])==TruthInteraction,type(interactions[0])
            assert type(interaction)==TruthInteraction
            self.truth_hip=self.hip
            self.truth_mip=self.mip
            self.truth_interaction_vertex=interaction.vertex
            self.truth_interaction_nu_id=interaction.nu_id
        else:
            assert type(interactions[0])==RecoInteraction,type(interactions[0])
            self.truth_hip=None
            self.truth_mip=None
            # self.truth_interaction=None
            if self.hip.is_matched:
                self.truth_hip=truth_particles[self.hip.match_ids[0]]
            if self.mip.is_matched:
                self.truth_mip=truth_particles[self.mip.match_ids[0]]
            if interaction.is_matched:
                self.truth_interaction_vertex=truth_interactions[interaction.match_ids[0]].vertex
                self.truth_interaction_nu_id=truth_interactions[interaction.match_ids[0]].nu_id
                self.truth_interaction_overlap=[truth_interactions[interaction.match_ids[0]].match_ids,truth_interactions[interaction.match_ids[0]].match_overlaps]
                # self.truth_interaction=truth_interactions[interaction.match_ids[0]]
            

        assert self.hip in interaction.particles,(self.hip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles])
        assert self.mip in interaction.particles,(self.mip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles])

        self.real_hip_momentum=hip.reco_momentum
        self.real_hip_momentum_reco=momentum_from_children_ke_reco(hip,particles,mass1,ignore=[mip])
        self.real_mip_momentum_reco=momentum_from_children_ke_reco(mip,particles,mass2,ignore=[hip])
        # self.hip_children_contained_reco=all_children_contained_reco(hip,particles,ignore=[mip])
        # self.mip_children_contained_reco=all_children_contained_reco(mip,particles,ignore=[hip])
        if self.truth and type(hip)==TruthParticle:
            assert type(hip)==TruthParticle
            self.real_hip_momentum=momentum_from_children_ke(hip,particles,mass1)


        self.real_mip_momentum=mip.reco_momentum
        if self.truth and type(mip)==TruthParticle:
            assert type(mip)==TruthParticle
            self.real_mip_momentum=momentum_from_children_ke(mip,particles,mass2)

        # self.vae = self.vertex_angle_error()
        # self.mass2 = self.mass_2()
        # self.decaylen = self.decay_len()
        # self.momenta: list[float] = self.momenta_projections()
        # self.coll_dist = collision_distance(hip,mip)
        
        if self.truth and type(mip)==TruthParticle:
            assert type(self.mip)==TruthParticle and type(self.hip)==TruthParticle
            # self.true_signal=abs(self.mip.parent_pdg_code)==3122 and abs(self.hip.pdg_code)==2212 and abs(self.mip.pdg_code)==211 and self.mip.parent_id==self.hip.parent_id and process_map[self.hip.creation_process]=='6::201' and process_map[self.mip.creation_process]=='6::201'
            # truth_parsed,self.reason=self.is_truth(particles)
            # assert truth_parsed==self.truth, (truth_parsed,self.reason)
        # if self.truth: print("We got a true lambda")

        # guess_start = get_pseudovertex(
        #     start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        #     directions=[hip.reco_start_dir, mip.reco_start_dir],
        # )

        

        # self.pi_extra_children=[]
        # self.prot_extra_children=[]
        
        # self.pot_parent:list[tuple[Particle,bool]]=[]

        self.truth_interaction_id=truth_interaction_id_hotfix(interaction)

        self.is_flash_matched=interaction.is_flash_matched

        self.reco_vertex=reco_vert_hotfix(interaction)

        self.primary_particle_counts=interaction.primary_particle_counts
        
    
    def pass_cuts(self,cuts:dict)->bool:
        # passed=True
        extra_children=[]
        self.pass_failure=[]

        pot_parent:list[tuple[Particle,bool]]=[]

        guess_start=(self.hip.start_point+self.mip.start_point)/2

        # particles:list[Particle]=interaction.particles

        for p in self.particles:
            # if p.interaction_id!=hip.interaction_id: continue
            if p.id not in [self.mip.id,self.hip.id] and HM_pred_hotfix(p,self.hm_pred) in [SHOWR_HM,MIP_HM,HIP_HM,MICHL_HM]:
                # self.prot_extra_children +=[ExtraChild(p,hip.end_point,hm_pred,hip)]
                # self.pi_extra_children +=[ExtraChild(p,mip.end_point,hm_pred,mip)]
                extra_children += [p]
                if HM_pred_hotfix(p,self.hm_pred) in [MIP_HM,HIP_HM]: pot_parent+=[(p,False)]
        # print("TRYING a lambda",self.truth)
        # if not (is_contained(self.hip.points,mode=full_containment,margin=2) and is_contained(self.mip.points,mode=full_containment,margin=2)):
        #     if self.truth: print("lam child containment issue")
        #     if passed:
        #         self.pass_failure="lam child containment issue"
        #     passed=False

        # if norm3d(self.mip.start_point-self.hip.start_point)>cuts["lam_cont_dist max"]:
        #     if self.truth: print("lam child contained start distance max", norm3d(self.mip.start_point-self.hip.start_point))
        #     if passed:
        #         self.pass_failure="lam child contained start distance max"
        #     passed=False

        # if self.coll_dist[-1]>cuts["lam_proj_dist max"]:
        #     if self.truth: print("lam_proj_dist max", self.coll_dist[-1])
        #     if passed:
        #         self.pass_failure="lam_proj_dist max"
        #     passed=False

        
        

        inter = self.reco_vertex

        # best_dist=np.inf
        # new_vertex=None

        # if type(self.hip)==RecoParticle:
        #     if not self.is_flash_matched:
        #         for v in self.fm_interactions:
        #             if not v[1]:
        #                 continue
        #             # if i.id==interaction.id:
        #                 # continue
        #             # if i.is_flash_matched:
        #             # print(v[0],inter)

        #             # if np.isclose(norm3d(v[0]-inter),0): continue
        #             d=norm3d(v[0]-inter)

                    
        #             if d<best_dist:
        #                 best_dist=d
        #                 inter=v[0]
        #                 # new_vertex

        # assert self.hip in particles,(self.hip.id,[i.id for i in particles])#,[i.id for i in interaction.primary_particles],type(interaction))
        # assert self.mip in particles,(self.mip.id,[i.id for i in particles])#,[i.id for i in interaction.primary_particles],type(interaction))
        michels:list[Particle]=[p for p in self.particles if p.shape in [MICHL_SHP,LOWES_SHP]]
        assert norm3d(inter)>0,inter

        

        
        ldl=float(norm3d(inter - guess_start))


        assert ldl<10000, (inter,guess_start)
        # self.decaylen=ldl
        if inter[2]<self.reco_vertex[2]:
            self.reco_vertex=inter
            # self.decaylen=float(norm3d(inter - guess_start))

        
        # guess_start = get_pseudovertex(
        #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
        #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
        # )
        # assert norm3d(guess_start-guess_start)==0
        # dir1 = guess_start - inter
        # vae=0
        # if not np.isclose(ldl,0):
        #     dir2 = self.real_hip_momentum_reco + self.real_mip_momentum_reco

        #     ret = angle_between(dir1,dir2)
        #     if passed: assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,dir1,dir2)
            
        #     vae=self.decaylen*np.sin(min(ret,np.pi/2))

            

        # # vae1=0
        # # if (not np.isclose(norm3d(dir1),0)) and (not np.isclose(norm3d(self.real_mip_momentum_reco),0)) and (not np.isclose(norm3d(self.real_hip_momentum_reco),0)):
        # #     vae1=point_to_plane_distance(inter,guess_start,self.real_hip_momentum_reco,self.real_mip_momentum_reco)
        # self.vae=vae

        self.vae=impact_parameter(inter,guess_start,self.real_hip_momentum_reco + self.real_mip_momentum_reco)

        #TODO make a cut of vae with other pid choices 


        primary_shapes=np.bincount([p.shape for p in self.particles if is_primary_hotfix(p)],minlength=10)


        MIP_prot_KE_rat=abs(come_to_rest(self.mip,PROT_MASS))
        MIP_pi_KE_rat=abs(come_to_rest(self.mip,PION_MASS))
        # HIP_prot_KE_rat=abs(come_to_rest(self.hip,PROT_MASS))
        # HIP_pi_KE_rat=abs(come_to_rest(self.hip,PION_MASS))


        for c in cuts:

            if len(self.pass_failure)>=3: break
            checked=False


            if c=="MIP Child":
                checked=True
                if HM_pred_hotfix(self.mip,self.hm_pred)!=MIP_HM and self.mip.pid not in [PION_PID,PROT_PID]:#:# and 
                    self.pass_failure+=[c]
                    #passed=False
                if MIP_prot_KE_rat<min(.5,MIP_pi_KE_rat):
                    self.pass_failure+=[c]
                    #passed=False

            if c=="HIP Child":
                checked=True
                if HM_pred_hotfix(self.hip,self.hm_pred)!=HIP_HM and self.hip.pid not in [PION_PID,PROT_PID]:#:# and s
                    self.pass_failure+=[c]
                    #passed=False
                # if HIP_pi_KE_rat<min(.5,HIP_prot_KE_rat):
                #     self.pass_failure+=[c]
                #     #passed=False

                    # PROT_PID

            if c=="Primary HIP-MIP":
                checked=True
                if type(self.hip)==TruthParticle:
                    if is_primary_hotfix(self.hip) or is_primary_hotfix(self.mip):# or HM_pred_hotfix(self.hip,self.hm_pred)!=HIP_HM or HM_pred_hotfix(self.mip,self.hm_pred)!=MIP_HM:
                        # if self.truth: print("nonprimary hip/mip", norm3d(self.mip.start_point-self.hip.start_point))
                        # if passed:
                        self.pass_failure+=[c]
                        #passed=False
                
                elif type(self.hip)==RecoParticle:
                    if self.hip.is_primary or self.mip.is_primary:#or HM_pred_hotfix(self.hip,self.hm_pred)!=HIP_HM or HM_pred_hotfix(self.mip,self.hm_pred)!=MIP_HM:
                        self.pass_failure+=[c]
                        #passed=False
                else:
                    raise Exception()

            if c=="Valid Interaction":
                checked=True
                if type(self.hip)==TruthParticle:
                    # assert type(interaction)==TruthInteraction
                    if self.truth_interaction_nu_id==-1:
                        self.pass_failure+=[c]
                        #passed=False


                elif type(self.hip)==RecoParticle:
                    if not self.is_flash_matched:# and best_dist>cuts[c]:
                        if self.truth: print("FAILED AT VALID INTERACTION")#,best_dist)
                        self.pass_failure+=[c]
                        #passed=False
                else:
                    raise Exception()
                
            if c=="HIP-MIP Order":
                checked=True
                if self.hip.pid in [MUON_PID,PION_PID] and self.mip.pid in [KAON_PID,PROT_PID]:
                    self.pass_failure+=[c]
                    #passed=False




            if c=="Max HIP-MIP Sep.":
                checked=True
                # if norm3d(self.mip.start_point-self.hip.start_point)>cuts[c]:
                if closest_distance(self.mip.start_point, self.mip.momentum, self.hip.start_point, self.hip.momentum)>cuts[c] or not np.isclose(np.linalg.norm(self.hip.start_point-self.mip.start_point),np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))):
                # if np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))>cuts[c]:
                    # if self.truth: print("Max HIP/MIP Sep.", norm3d(self.mip.start_point-self.hip.start_point))
                    # if passed:
                    self.pass_failure+=[c]
                    #passed=False

            if c=="No HIP or MIP Michel":
                checked=True
                for p in michels:
                    if norm3d(self.mip.end_point-p.start_point)<cuts[c] or norm3d(self.hip.end_point-p.start_point)<cuts[c]:
                    # if np.min(cdist(p.points, [self.mip.end_point]))<cuts[c]:
                        self.pass_failure+=[c]
                        #passed=False
                        break
                
            if c=="Impact Parameter":
                checked=True
                if self.vae>cuts[c]:
                    # if self.truth: print("VAE max", vae)
                    # if passed:
                    self.pass_failure+=[c]
                    #passed=False

            if c=="Min Decay Len":
                checked=True
                if ldl<cuts[c]:
                    # if self.truth: print("minimum decay len")
                    # if passed:
                    self.pass_failure+=[c]
                    #passed=False

            #TODO check if one of the particles has csda per whatever match the wrong one of proton or pion


            if c=="Parent Proximity":
                checked=True
                start_to_int=norm3d(guess_start-inter)
                for p in pot_parent:
                    if norm3d(p[0].start_point-inter)>2*min_len:
                        continue
                    # est_decay=(self.mip.start_point+self.hip.start_point)/2
                    if (norm3d(p[0].end_point-inter)>=start_to_int and 
                        norm3d(p[0].start_point-inter)>=start_to_int):continue
                    if norm3d(p[0].end_point-guess_start)<=min_len or norm3d(p[0].end_point-self.hip.start_point)<=min_len/2 or norm3d(p[0].end_point-self.mip.start_point)<=min_len/2:
                        # norm3d(p[0].end_point-inter)<= norm3d(self.hip.start_point-inter)+min_len and
                        # norm3d(p[0].end_point-inter)<= norm3d(self.mip.start_point-inter))+min_len:
                        # if self.truth: print("parent proximity")
                        # if passed:
                        self.pass_failure+=[c]
                        #passed=False
                        break
                    
            # if c==""
            
            if c=="# Children":
                checked=True
                hip_end_gs=norm3d(guess_start-self.hip.end_point)
                mip_end_gs=norm3d(guess_start-self.mip.end_point)
                local_child_count=0
                for p in extra_children:
                    # if is_primary_hotfix(p.child): continue
                    child_to_start=norm3d(p.start_point-guess_start)
                    if child_to_start>=min(min_len,norm3d(p.start_point-inter)):
                        continue
                    
                    if (child_to_start>=hip_end_gs or
                        child_to_start>=mip_end_gs):continue
                    local_child_count+=1
                    
                    # if (child_to_start<=min(min_len,norm3d(p.child.start_point-inter))):
                        # if self.truth: print("extra child")
                        # if passed:
                    self.pass_failure+=[c]
                    #passed=False
                    break
                # if local_child_count>=2 and c not in self.pass_failure:
                #     raise Exception()
                #     self.pass_failure+=[c]
                #     #passed=False


            if c==rf"Even # Primary $\gamma$":
                checked=True
                if (self.primary_particle_counts[PHOT_PID]%2==1 and 
                    primary_shapes[TRACK_SHP]<=3):
                    # if self.truth: 
                    #     print("odd photon cut")
                    # if passed:
                    self.pass_failure+=[c]
                    #passed=False

            if c=="Valid Len":
                checked=True
                assert type(self.hip)==RecoParticle
                if self.hip.reco_length<=cuts[c] and self.mip.reco_length<=cuts[c]:
                    self.pass_failure+=[c]
                    #passed=False




            if c=="":
                checked=True
                self.pass_failure+=[c]
            
            if not checked:
                raise Exception(c,"not found in lam cuts")


        # if not self.hip_children_contained_reco or not self.mip_children_contained_reco:
        #     if self.truth: print("prot or pi containment")
        #     if passed:
        #         self.pass_failure="prot or pi containment"
        #     passed=False

        # assert is_contained(interaction.vertex,mode=full_containment)

        # if not is_contained(inter,mode=full_containment):
        #     if self.truth: print("vertex OOB")
        #     if passed:
        #         self.pass_failure="vertex OOB"
        #     passed=False

        
        # assert norm3d(inter)<np.inf,inter

        # if norm3d(inter)==np.inf:
        #     if self.truth: print("prot or pi containment")
        #     if passed:
        #         self.pass_failure="prot or pi containment"
        #     passed=False

        # if momentum_from_children_ke_reco(self.hip,particles,KAON_MASS)

        # print(inter)
        
        
        # # print([(p.dist_to_parent,p.angle_to_parent) for p in self.prot_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0]])
        # childcut=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent,p.child.id) for p in self.prot_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and ((p.child_hm_pred==HIP_HM and p.child.reco_length>min_len and direction_acos(p.child.reco_start_dir,self.hip.reco_end_dir)>np.pi/12) or (p.child_hm_pred==MIP_HM and p.child.reco_length>min_len) or (not is_contained(p.child.points,mode=full_containment) and p.child_hm_pred in [HIP_HM,MIP_HM]))]
        # if len(childcut)!=0:
        #     if self.truth: print("proton child particle", childcut)
        #     if passed:
        #         self.pass_failure="proton child particle"
        #     passed=False

        
        # childcut=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent,p.child.id) for p in self.pi_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and ((p.child_hm_pred==MIP_HM and p.child.reco_length>min_len and direction_acos(p.child.reco_start_dir,self.mip.reco_end_dir)>np.pi/12) or (p.child_hm_pred==HIP_HM and p.child.reco_length>min_len) or (not is_contained(p.child.points,mode=full_containment) and p.child_hm_pred in [HIP_HM,MIP_HM]))]#or p.child_hm_pred in [SHOWR_HM,MICHL_HM]
        # # childcut2=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent,p.child.reco_length) for p in self.pi_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and ((direction_acos(p.child.reco_start_dir,self.mip.reco_end_dir)>np.pi/12 and p.child_hm_pred==MIP_HM and p.child.reco_length>min_len))]#or p.child_hm_pred in [SHOWR_HM,MICHL_HM]
        # if len(childcut)!=0:
        #     # if len(childcut2)>0:print("dcut2",childcut2)
        #     if self.truth: print("pion child particle", childcut)
        #     if passed:
        #         self.pass_failure="pion child particle"
        #     passed=False

        # if self.mip.reco_length<min_len:
        #     if self.truth: print("mip len")
        #     if passed:
        #         self.pass_failure="mip len"
        #     passed=False

        # if self.hip.reco_length<min_len:
        #     if self.truth: print("hip len")
        #     if passed:
        #         self.pass_failure="hip len"
        #     passed=False


        

        

        


        



        


        
        # base_len=self.decaylen*np.sin(min(self.vae,np.pi/2))
        # if vae1>cuts["VAE max new"]:
        #     # if self.truth: print("VAE max", vae)
        #     # if passed:
        #     self.pass_failure+=["Perp. Impact Parameter"]
        #     passed=False

        # if ldl>cuts["lam_decay_len_max"]:
        #     if self.truth: print("decay len max")
        #     if passed:
        #         self.pass_failure="decay len max"
        #     passed=False

        # mom_norm=norm3d(self.real_hip_momentum_reco + self.real_mip_momentum_reco)
        # tau=ldl/100/(2.998e8)*LAM_MASS/mom_norm*10**9
        # if tau>cuts["tau_max"]:
        #     if self.truth: print("tau cut")
        #     if passed:
        #         self.pass_failure="max tau"
        #     passed=False

        

        # if (interaction.primary_particle_counts[PHOT_PID]%2==1 and 
        #     interaction.primary_particle_counts[MUON_PID]<=2 and 
        #     interaction.primary_particle_counts[PROT_PID]<=1 and 
        #     interaction.primary_particle_counts[KAON_PID]<=1
        #     and "supress_single_photon" not in cuts):


        # if np.dot(dir1,self.hip.momentum)<0 and np.dot(dir1,self.mip.momentum)<0 and norm3d(dir1)>1:
        #     self.pass_failure+=[r"Valid $\Lambda$ Direction"]
        #     passed=False


        

        # for p in self.pot_parent:
        #     # est_decay=(self.mip.start_point+self.hip.start_point)/2
        #     v1=guess_start-p[0].end_point
        #     if norm3d(p[0].end_point-inter)>=norm3d(guess_start-inter):continue
        #     # if 
        #     if p[0].is_primary and np.dot(p[0].reco_end_dir,v1)>norm3d(v1)*norm3d(p[0].reco_end_dir)*np.cos(np.pi/8):
        #         # if self.truth: print("primary colinear parent")
        #         # if passed:
        #         self.pass_failure+=["Primary Colinear Parent"]
        #         passed=False
        #         break

        

        # interaction.flas

        

        #####
        #TODO decay time cut 
        ######

        # if self.momenta[0]>cuts["lam_pt max"]*LAM_PT_MAX:
        #     if self.truth: print("pt max", self.momenta[0])
        #     if passed:print("FAILED AT lam_pt max",self.truth)
        #     passed=False
        

        
        # masscut=abs((self.lam_mass2-PION_MASS**2-PROT_MASS**2)/(LAM_MASS**2-PION_MASS**2-PROT_MASS**2)-1)
        # if masscut>cuts["lam_percent_error_mass"]:
        #     if self.truth: print("mass", masscut)
        #     if passed:print("FAILED AT mass",self.truth,masscut)
        #     passed=False
        # # return True
        
        return len(self.pass_failure)==1
    @property
    def decay_len(self) -> float:
        """
        Returns distance from average start position of hip and mip to vertex location of the assocated interaction

        Returns
        -------
        float
            distance from decay point to vertex of interaction
        """
        # guess_start = get_pseudovertex(
        #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
        #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
        # )
        guess_start=(self.hip.start_point+self.mip.start_point)/2
        return float(norm3d(self.reco_vertex - guess_start))
    
    def mass_2(self) -> float:
        """
        Returns mass value constructed from the
        hip and mip candidate deposited energy and predicted direction

        Returns
        -------
        float
            reconstructed mass squared
        """
        assert (
            self.mip.ke > 0
        )  # print(mip.ke,"very bad",mip.id,mip.parent_pdg_code,mip.pid,mip.pdg_code,mip.energy_init)
        assert (
            self.hip.ke > 0
        )  # print(hip.ke,"very bad",hip.id,hip.parent_pdg_code,hip.pid,hip.pdg_code,hip.energy_init)
        mass2 = (self.mass1**2+
            self.mass2**2
            + 2 * (self.mip.reco_ke + self.mass2) * (self.hip.reco_ke + self.mass1)
            - 2 * np.dot(self.hip.reco_momentum, self.mip.reco_momentum)
        )
        return mass2
    
    # def vertex_angle_error(self) -> float:
    #     """
    #     Returns angle between the line constructed from the momenta of the hip and mip and
    #     the line constructed from the interaction vertex and the decay point

    #     Returns
    #     -------
    #     float
    #         distance from interaction vertex to line consructed from the momenta of the hip and mip
    #     """

    #     inter = reco_vert_hotfix(interaction)
    #     # guess_start = get_pseudovertex(
    #     #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
    #     #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
    #     # )
    #     guess_start=(self.hip.start_point+self.mip.start_point)/2
    #     assert norm3d(guess_start-guess_start)==0
    #     dir1 = guess_start - inter
    #     if np.isclose(norm3d(dir1),0):
    #         return 0

    #     dir2 = self.real_hip_momentum_reco + self.real_mip_momentum_reco

    #     if norm3d(dir2) == 0:
    #         return np.nan
    #     ret = np.arccos(
    #         np.dot(dir1, dir2) / norm3d(dir1) / norm3d(dir2)
    #     )
    #     assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,np.dot(dir1, dir2) , norm3d(dir1) , norm3d(dir2),np.dot(dir1, dir2) / norm3d(dir1) / norm3d(dir2))
    #     return ret
    
    # def momenta_projections(self) -> list[float]:
    #     """
    #     Returns the P_T and P_L of each particle relative to the measured from the decay

    #     Parameters
    #     ----------
    #     hip: spine.Particle
    #         spine particle object
    #     mip: spine.Particle
    #         spine particle object
    #     interactions: list[Interaction]
    #         list of interactions

    #     Returns
    #     -------
    #     list[float]
    #         shape(4) [hip transverse momentum, mip transverse momentum,hip long momentum, mip long momentum]
    #     """

    #     inter = interaction.vertex

    #     guess_start = get_pseudovertex(
    #         start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
    #         directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
    #     )
    #     dir = guess_start - inter

    #     p1 = self.hip.reco_momentum
    #     p2 = self.mip.reco_momentum

    #     dir = p1 + p2  # fix this hack #TODO

    #     dir_norm = norm3d(dir)
    #     if dir_norm == 0:
    #         return [np.nan, np.nan, np.nan, np.nan]

    #     dir = dir / dir_norm

    #     p1_long = np.dot(dir, p1)
    #     p2_long = np.dot(dir, p2)

    #     p1_transv = float(norm3d(p1 - p1_long * dir))
    #     p2_transv = float(norm3d(p2 - p2_long * dir))

    #     return [p1_transv, p2_transv, p1_long, p2_long]


#TODO This may be used for the CCNC separation
class PrimaryMIP:
    # truth:bool
    # mip:Particle

    # true_signal:bool

    def __init__(
        self,
        # pot_k: PotK,
        # K_hip: Particle,
        particles: list[Particle],
        # interactions: list[Interaction],
        hm_pred:dict[int,np.ndarray],

    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        ?????
        """

        # self.mip_len_base=mip.reco_length
        # self.mu_hm_acc = hm_acc[mip.id]
        # self.potential_michels=[]
        self.primarymips={}
        for p in particles:
            if p.interaction_id not in self.primarymips:
                self.primarymips[p.interaction_id]={}
            if HM_pred_hotfix(p,hm_pred)!=MIP_HM: continue
            if not is_primary_hotfix(p): continue
            self.primarymips[p.interaction_id][p.id]=[p,True]
        for t in particles:
            if HM_pred_hotfix(t,hm_pred) not in [HIP_HM,MIP_HM]: continue
            for pid in self.primarymips[t.interaction_id]:
                if norm3d(self.primarymips[t.interaction_id][pid][0].end_point-t.start_point)<min_len:
                    self.primarymips[t.interaction_id][pid][1]=False
        
            


def is_contained(pos: np.ndarray, mode: str =full_containment, margin: float = 3,define_con=True) -> bool:
    """
    Checks if a point is near dead volume of the detector
    ----------
    pos : np.ndarray
        (3) Vector position (cm)
    mode: str
        defined in spine Geometry class
    margin : np.ndarray/float
        Tolerance from module boundaries (cm)

    Returns
    -------
    Bool
        Point farther than eps away from all dead volume and in the detector
    """
    if define_con:
        Geo.define_containment_volumes(margin, mode=mode,include_limits=False)
    return bool(Geo.check_containment(pos))

# def tpc_module_id(pos: np.ndarray):
    
#     # Geo.define_containment_volumes(margin, mode=mode)
#     @Geo.get_closest_tpc(pos),
#     return Geo.get_closest_module(pos)


def HIPMIP_pred(particle: Particle, sparse3d_pcluster_semantics_HM: np.ndarray,perm=None,mode=True) -> np.ndarray:
    """
    Returns the semantic segmentation prediction encoded in sparse3d_pcluster_semantics_HM,
    where the prediction is not guaranteed unique for each cluster, for the particle object,
    decided by majority vote among the voxels in the cluster

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information and unique semantic segmentation prediction
    sparse3d_pcluster_semantics_HM : np.ndarray
        HIP/MIP semantic segmentation predictions for each voxel in an image

    Returns
    -------
    int
        Semantic segmentation prediction including HIP/MIP for a cluster
    """
    # if len(particle.depositions) == 0:
        # raise ValueError("No voxels")
    

    
    # assert np.max(idx)<len(perm),(np.max(idx),len(perm))

    idx=particle.index

    if perm is not None:
        idx=[i for i in particle.index if i<len(perm)] #TODO this has to be a bug
        idx=perm[idx]
    # else:


    if mode:
        
        HM_Pred = sparse3d_pcluster_semantics_HM[idx, -1]
        
        # print(HM_Pred,type(HM_Pred))
    else:
        # if idx>=len(sparse3d_pcluster_semantics_HM): HM_Pred=[]
        idx=idx[idx<len(sparse3d_pcluster_semantics_HM)]
        HM_Pred = np.argmax(sparse3d_pcluster_semantics_HM[idx], axis=1)
        # assert HM_pred==int(HM_pred),HM_pred
        # return HM_pred
        # raise Exception(sparse3d_pcluster_semantics_HM[particle.index],HM_pred)
        # HM_Pred = sparse3d_pcluster_semantics_HM[perm[particle.index], -1]
        # print(HM_Pred,type(HM_Pred))
    return np.bincount(HM_Pred.astype(np.int64),minlength=HIP_HM+1)#st.mode(HM_Pred).mode

    


# def HIPMIP_acc(particle: Particle, sparse3d_pcluster_semantics_HM: np.ndarray,perm=None,mode=True) -> float:
#     """
#     Returns the fraction of voxels for a particle whose HM semantic segmentation prediction agrees
#     with that of the particle itself as decided by HIPMIP_pred

#     Parameters
#     ----------
#     particle : spine.Particle
#         Particle object with cluster information and unique semantic segmentation prediction
#     sparse3d_pcluster_semantics_HM : np.ndarray
#         HIP/MIP semantic segmentation predictions for each voxel in an image

#     Returns
#     -------
#     int
#         Fraction of voxels whose HM semantic segmentation agrees with that of the particle
#     """
#     # if len(particle.depositions) == 0:
#         # raise ValueError("No voxels")
#     # slice a set of voxels for the target particle

#     idx=particle.index

#     if perm is not None:
#         idx=[i for i in particle.index if i<len(perm)] #TODO this has to be a bug
#         idx=perm[idx]



#     if mode:
#         HM_Pred = sparse3d_pcluster_semantics_HM[idx, -1]
        
#         # print(HM_Pred,type(HM_Pred))
#     else:
#         HM_Pred = np.argmax(sparse3d_pcluster_semantics_HM[idx], axis=1)

#     pred = st.mode(HM_Pred).mode
#     return Counter(HM_Pred)[pred] / len(HM_Pred) if len(HM_Pred)!=0 else 0


# def direction_acos(momenta: np.ndarray, direction=np.array([0.0, 0.0, 1.0])) -> float:
#     """
#     Returns angle between the beam-axis (here assumed in z) and the particle object's start direction

#     Parameters
#     ----------
#     momenta : np.ndarray[float]
#         Momenta of the particle
#     direction : np.ndarray[float]
#         Direction of beam

#     Returns
#     -------
#     float
#         Angle between particle direction and beam
#     """
#     assert np.isclose(norm3d(momenta),1),norm3d(momenta)
#     assert np.isclose(norm3d(direction),1),norm3d(direction)

#     return np.arccos(np.dot(momenta, direction))


# def collision_distance(particle1: Particle, particle2: Particle,orientation:list[str]=["start","start"]):
#     """
#     Returns for each particle, the distance from the start point to the point along the vector start direction
#     which is the point of closest approach to the other particle's corresponding line, along with the distance of closest approach.
#     The parameters, t1 and t2, are calculated by minimizing ||p1+v1*t1-p2-v2*t2||^2, where p1/p2 are the starting point of each particle
#     and v1/v2 are the start direction of each particle

#     Parameters
#     ----------
#     particle1 : spine.Particle
#         Particle object with cluster information
#     particle2 : spine.Particle
#         Particle object with cluster information

#     Returns
#     -------
#     [float,float,float]
#         [t1,t2, min_{t1,t2}(||p1+v1*t1-p2-v2*t2||^2)]
#     """
#     assert set(orientation).issubset(set(["start","end"]))

#     if orientation[0]=="start":
#         v1 = particle1.reco_start_dir
#         p1 = particle1.start_point
#     elif orientation[0]=="end":
#         v1 = particle1.reco_end_dir
#         p1 = particle1.end_point
#     else:
#         raise Exception()

#     if orientation[1]=="start":
#         v2 = particle2.reco_start_dir
#         p2 = particle2.start_point
#     elif orientation[1]=="end":
#         v2 = particle2.reco_end_dir
#         p2 = particle2.end_point
#     else:
#         raise Exception()
    

#     v11 = np.dot(v1, v1)
#     v22 = np.dot(v2, v2)
#     v12 = np.dot(v1, v2)
#     dp = p1 - p2

#     denom = v12**2 - v11 * v22

#     if denom == 0:
#         return [0, 0, norm3d(dp)]

#     t1 = (np.dot(v1, dp) * v22 - v12 * np.dot(v2, dp)) / denom
#     t2 = (v12 * np.dot(v1, dp) - np.dot(v2, dp) * v11) / denom

#     min_dist = np.dot(p1 + v1 * t1 - p2 - v2 * t2, p1 + v1 * t1 - p2 - v2 * t2)

#     return [t1, t2, min_dist]







# def lambda_AM(hip:Particle,mip:Particle)->list[float]:
#     '''
#     Returns the P_T and the longitudinal momentum asymmetry corresponding to the Armenteros-Podolanski plot https://www.star.bnl.gov/~gorbunov/main/node48.html

#     Parameters
#     ----------
#     hip: spine.Particle
#         spine particle object
#     mip: spine.Particle
#         spine particle object

#     Returns
#     -------
#     list[float]
#         shape(2) [hip pt + mip pt, hip vs mip longitudinal momentum assymmetry]
#     '''
#     # inter=interactions[hip.interaction_id].vertex


#     # guess_start=get_pseudovertex(start_points=np.array([hip.start_point,mip.start_point],dtype=float),
#     #                              directions=[hip.reco_start_dir,mip.reco_start_dir])
#     # Lvec=guess_start-inter
#     p1=hip.momentum
#     p2=mip.momentum

#     Lvec=p1+p2

#     Lvecnorm=norm3d(Lvec)
#     if Lvecnorm==0:
#         return [np.nan,np.nan,np.nan]

#     Lvec=Lvec/Lvecnorm

#     p1_L=np.dot(Lvec,p1)
#     p2_L=np.dot(Lvec,p2)

#     p1_T=float(norm3d(p1-p1_L*Lvec))
#     p2_T=float(norm3d(p2-p2_L*Lvec))

#     # asymm=abs((p1_L-p2_L)/(p1_L+p2_L))
#     # pt=p1_T+p2_T
#         # print("very good",asymm,p1_L,p2_L,Lvec)
#         # assert asymm>=-1 and asymm<=1, print("help me",asymm,p1_L,p2_L,Lvec)
#     assert norm3d((p1-p1_L*Lvec)+(p2-p2_L*Lvec))<=1e-3,print(norm3d((p1-p1_L*Lvec)+(p2-p2_L*Lvec)))
#     return [p1_T,p2_T,np.abs(p1_L-p2_L)/(p1_L+p2_L)]




def come_to_rest(p:Particle,mass=KAON_MASS)->float:
    # if not is_contained(p.points,mode=full_containment,margin=0):return False
    # if not is_contained(p.points,mode=full_containment,margin=0):
    # print(p.reco_length)
    p_csda=csda_ke_lar(p.reco_length, mass)
    # if not p_csda>0: return False
    # if (not p.calo_ke>0) and (not p.mcs_ke>0): return False

    if p.calo_ke<0: raise Exception(p.calo_ke)
    # check=p.calo_ke if p.calo_ke>0 else p.mcs_ke
    return p.calo_ke/p_csda-1


def all_children_reco(p:Particle,particles:list[Particle],dist=min_len/2,ignore=[])->list[Particle]:

    done=False
    children:list[Particle]=[p]
    while done==False:
        done=True
        for pd in particles:
            for d in children:
                if norm3d(pd.start_point-d.end_point)<dist and pd not in children and pd not in ignore and not is_primary_hotfix(pd):
                    children+=[pd]
                    done=False
    return children

# def all_children_contained_reco(p:Particle,particles:list[Particle],mode=full_containment,ignore=[])->bool:
#     ad=all_children_reco(p,particles,ignore=ignore)
#     assert p in ad
#     contained=True
#     for d in ad:
#         # if id not in larcv_id_to_spine_id: continue
#         contained*=is_contained(d.points,mode=mode)
#     return bool(contained)

def momentum_from_children_ke_reco(p:Particle,particles:list[Particle],mass,ignore=[])->float:
    # print("running mfdke")
    ad=all_children_reco(p,particles,ignore=ignore)
    assert p in ad
    tke=sum([d.calo_ke for d in ad])#if d.shape not in [MICHL_SHP]
    
    if type(p)==TruthParticle: assert tke>0,(tke,mass,len(p.points),p.num_voxels,p.shape,p.energy_deposit,p.depositions)
    assert len(p.reco_start_dir)==3
    return np.sqrt(tke**2+2*mass*tke)*p.reco_start_dir


def all_children(p:TruthParticle,particles:list[TruthParticle])->list[TruthParticle]:
    out:list[TruthParticle]=[]
    if type(p)==TruthParticle:
        assert type(p)==TruthParticle,type(p)
    if len(p.children_id)==0:
        return [p]
    to_explore:list[TruthParticle]=[p]
    while len(to_explore)!=0:
        curr=to_explore.pop()
        # if curr not in larcv_id_to_spine_id: continue
        # if abs(larcv_id_to_spine_id[curr].parent_pdg_code)==2112:
        #     continue
        out+=[curr]
        #TODO fix missing children in the particle record
        to_explore+=[particles[i] for i in curr.children_id if particles[i] not in out]
        # print(to_explore)
        # if curr.
    
    # start=
    return out

# def all_children_contained(p_tid:int,particles:list[TruthParticle],mode=full_containment)->bool:
#     # ad:list[TruthParticle]=all_children(p,particles)
#     # assert p in ad
#     contained=True
#     for d in particles:
#         if d.ancestor_track_id!=p_tid:continue
#         if d.parent_pdg_code==2112:continue
#         # if id not in larcv_id_to_spine_id: continue
#         contained*=is_contained(d.points,mode=mode)
#     return bool(contained)

def momentum_from_children_ke(p:TruthParticle,particles,mass)->float:
    # print("running mfdke")
    ad:list[TruthParticle]=all_children(p,particles)
    assert p in ad

    tke=sum([a.calo_ke for a in ad])
    return np.sqrt((tke+mass)**2-mass**2)*p.reco_start_dir


def point_to_plane_distance(p, p0, v1, v2):
    # Normal vector to the plane
    normal = np.cross(v1, v2)
    if norm3d(normal)==0: return 0
    normal_unit = normal / norm3d(normal)
    
    # Vector from p0 (plane) to p
    vec = p - p0
    
    # Distance is projection of vec onto normal
    distance = np.abs(np.dot(vec, normal_unit))
    return distance

def mom_to_mass(p1,p2,m1,m2):

    E1=np.sqrt(norm3d(p1)**2+m1**2)
    E2=np.sqrt(norm3d(p2)**2+m2**2)

    return np.sqrt(m1**2 + m2**2 + 2*E1*E2 - 2*np.dot(p1, p2))


def impact_parameter(vert,pos,mom):
    # dir2 = self.real_hip_momentum_reco + self.real_mip_momentum_reco

    # mom=mom/np.

    ret = angle_between(pos-vert,mom)
    assert ret==ret,(pos,vert,mom)
    # if passed: assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,dir1,dir2)
    
    return norm3d(pos-vert)*np.sin(min(ret,np.pi/2))

def get_tpc_id(point):
    return Geo.get_closest_tpc([point])[0]
    # return 


def cos_gamma_to_pip(E):
    return (PI0_MASS**2/2/E-E_PI0_Kp_Decay)/P_PI0_Kp_Decay


def cos_gamma_to_pip_bounds(E):
    out=np.clip([cos_gamma_to_pip(E*1.5),cos_gamma_to_pip(E/2)],0,1)
    # print(out[0],out[1])
    return out


def cos_gamma_to_E(vert,pos,mom):
    cost=angle_between(pos-vert,mom)
    return PI0_MASS**2/2/(E_PI0_Kp_Decay+cost*P_PI0_Kp_Decay)

def closest_distance(pos1, mom1, pos2, mom2):
    # a, u, b, v = map(np.array, (a, u, b, v))
    cross = np.cross(mom1, mom2)
    denom = np.linalg.norm(cross)
    if denom < 1e-12:  # parallel case
        return np.linalg.norm(np.cross(pos2 - pos1, mom1)) / np.linalg.norm(mom1)
    return abs(np.dot(pos2 - pos1, cross)) / denom

# def 

