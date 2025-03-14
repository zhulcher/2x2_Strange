""" 
This file contains output classes and cut functions useful for reconstructing kaons and 
lambdas in a liquid argon TPC using the reconstruction package SPINE https://github.com/DeepLearnPhysics/spine
"""
# from ast import Module
from collections import Counter

# from sympy import false                   
# import string
SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf

import sys
# Set software directory
sys.path.append(SOFTWARE_DIR)
from spine.data.out import TruthParticle,RecoParticle,RecoInteraction,TruthInteraction
from spine.utils.globals import MUON_PID, PROT_MASS, PION_MASS, KAON_MASS,PHOT_PID, PROT_PID,KAON_PID
from spine.utils.geo.manager import Geometry
# from spine.utils.vertex import get_pseudovertex
import numpy as np
from scipy import stats as st
from spine.utils.globals import MICHL_SHP,TRACK_SHP,SHOWR_SHP
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


analysis_type='icarus'
if analysis_type=='2x2':
    full_containment='detector'
else:
    full_containment='module'

min_len=2.5


HIP_HM = 7
MIP_HM = TRACK_SHP
SHOWR_HM = SHOWR_SHP
MICHL_HM=MICHL_SHP

LAM_MASS = 1115.683       # [MeV/c^2]

# LAM_PT_MAX = (np.sqrt((LAM_MASS**2 - (PION_MASS + PROT_MASS) ** 2) * (LAM_MASS**2 - (PION_MASS - PROT_MASS) ** 2))/2/LAM_MASS)


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



Geo = Geometry(detector=Model)

class ExtraChild:
    # child_id:int
    dist_to_parent:float
    # proj_dist_to_parent:float
    angle_to_parent:float
    # truth:bool
    child:Particle
    child_hm_pred: int

    """
    Storage class for extra children of any particle
    Attributes
    ----------
    dist_to_parent: float
        parent end to child start distance
    proj_dist_to_parent: float
        parent to child projected distance of closest approach
    angle_to_parent: float
        angle between child start direction and line from parent end to child start
    truth: bool
        is this a true child of the parent?
    child: Particle
        Particle object for the child
    child_hm_pred: int
        HM prediction for the child
    parent: None|Particle
        If the parent particle is specified, ignore the start point and use the point of closest approach from the parent (line) to the child (line)
    """
    def __init__(self, child:Particle,parent_end_pt:np.ndarray,hm_pred:list[int],parent:None|Particle=None):
        self.child=child
        
        par_end=parent_end_pt
        self.dist_to_parent=float(np.linalg.norm(child.start_point-par_end))
        # self.proj_dist_to_parent=np.nan
        # if parent!=None and not is_contained(parent.points,mode=full_containment):
            # print(parent.reco_end_dir,parent.reco_end_dir)
            # par_end = get_pseudovertex(
            #     start_points=np.array([parent.end_point, child.start_point], dtype=float),
            #     directions=[parent.reco_end_dir, child.reco_start_dir],
            # )
            # self.proj_dist_to_parent=collision_distance(parent,child,orientation=["end","start"])[-1]

        self.child_hm_pred=hm_pred[child.id]
        if self.dist_to_parent==0:
            self.angle_to_parent=0
        # else:
            # self.angle_to_parent=direction_acos((child.start_point-par_end)/self.dist_to_parent,child.reco_start_dir)
        # self.truth=False
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        child : Particle
            child particle
        parent_end_pt: list[float]
            end point of parent particle
        hm_pred: list[int]
            HM predictions for the particles
        """


# class PotK:
#     """
#     Storage class for primary Kaons and their cut parameters

#     Attributes
#     ----------
#     hip_id : int
#         id for hip associated to this class
#     hip_len: float
#         len attribute of the particle object
#     dir_acos: float
#         arccos of the particle direction with the beam direction
#     k_hm_acc:float
#         percent of the voxels of this particle
#         whose Hip/Mip semantic segmentation matches the overall prediction
#     """

#     hip_id: int
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
#         self.hip_id = kaon.id
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
        #             if abs(id_to_particle[id].pdg_code)==321 and np.linalg.norm(fake_hip.end_point-id_to_particle[id].start_point)<3 and id not in vis_part:
        #                 done=False
        #                 fake_hip=id_to_particle[id]
        #                 break
        #     self.truth_list=[abs(fake_hip.pdg_code)==321,(mip.parent_id==fake_hip.orig_id or (mip.parent_id in fake_hip.children_id and mip.parent_id not in vis_part)),abs(mip.parent_pdg_code)==321,(self.hip.is_primary or bool(np.linalg.norm(self.hip.start_point-interactions[mip.interaction_id].vertex)<min_len)),bool(np.linalg.norm(fake_hip.end_point-mip.start_point)<3)]
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
    """
    Storage class for primary Kaons with muon child and their cut parameters

    Attributes
    ----------
    mip_id : int
        id for hip associated to this class
    mip_len_base: float
        len attribute of the particle object
    mu_hm_acc:float
        percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
    potential_kaons: list[Particle]
        list of kaon candidates corresponding to this mip candidate
    potential_michels: list[Particle]
        list of michel candidates corresponding to this mip candidate
    truth: bool
        is this a truth muon coming from a kaon
    mip: Particle
        muon candidate Particle object
    require_kaon:
        does this mip require a kaon candidate
    require_michel:
        does this mip require a michel candidate
    """

    # mip_id: int
    # mip_len_base: float
    # mu_hm_acc: float
    # potential_kaons: list[PotK]
    # potential_michels: list[PotMich]
    # mu_extra_children:list[ExtraChild]
    truth:bool
    mip:Particle
    require_kaon:bool
    # require_michel:bool #TODO implement this

    # true_signal:bool

    def __init__(
        self,
        # pot_k: PotK,
        K_hip: Particle,
        particles: list[Particle],
        interactions: list[Interaction],
        hm_acc:list[float],
        hm_pred:list[int],
        
        truth:bool,
        reason:bool,
        truth_list
    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        pot_k:PotK
            PotK object with associated hip information
        mip_id : int
            id for hip associated to this class
        mip_len: float
            len attribute of the particle object
        dist_to_hip: float
            distance from mip start to hip end
        HM_acc_mu:float
            percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
        """

        # self.mip_id = mip.id
        # self.mip_len_base=mip.reco_length
        # self.mu_hm_acc = hm_acc[mip.id]
        # self.potential_michels=[]
        self.truth=truth
        self.reason=reason
        self.pass_failure=""
        # self.mip=mip
        # self.true_signal=False
        self.error=""
        self.truth_list=truth_list
        self.hip=K_hip
        self.accepted_mu=None
        if self.truth:
            assert abs(self.hip.pdg_code)==321,self.hip.pdg_code
        self.interaction=interactions[self.hip.interaction_id]
        self.potential_kaons=[[self.hip,[],0]]

        self.real_K_momentum=K_hip.reco_momentum
        if self.truth:
            assert type(K_hip)==TruthParticle
            self.real_K_momentum=momentum_from_daughter_ke(K_hip,KAON_MASS)
        assert hm_pred[K_hip.id]==HIP_HM
        done=False
        while not done: #this loop goes over all of the hips connected to the end of the kaon, and constructs a hadronic group which hopefully contains the kaon end. 
            done=True
            # print("looking")
            for p in particles:
                # print([r[0] for r in self.potential_kaons])
                if p not in [r[0] for r in self.potential_kaons] and hm_pred[p.id]==HIP_HM:
                    # print("getting here")
                    for k in list(self.potential_kaons).copy():
                        # print(k[0])
                        if np.linalg.norm(p.start_point-k[0].end_point)<min_len:
                            
                            self.potential_kaons+=[[p,[],0]]
                            done=False

        for k in self.potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group 
            for p in particles:
                if p in [r[0] for r in self.potential_kaons]:continue
                if hm_pred[p.id]==HIP_HM and np.linalg.norm(p.start_point-k[0].end_point)<min_len:
                    raise Exception("how",hm_pred[p.id],[r[0].id for r in self.potential_kaons])
                if hm_pred[p.id]==MICHL_HM:
                    continue
                if p.is_primary:
                    continue
                if hm_pred[p.id]==MIP_HM and np.linalg.norm(p.start_point-k[0].end_point)<min_len:
                    if is_contained(p.points):
                        add_it=True
                        for c in particles: #this loop looks for mips or hips at the end of this mip and rejects it if so
                            #TODO allow mips at the end of the mip, and add the lengths
                            #TODO add in counts for particles connecting at each end
                            if hm_pred[c.id] in [HIP_HM] and np.linalg.norm(c.start_point-p.end_point)<min_len:
                                add_it=False
                                break
                        if add_it:
                            k[1]+=[p]
                    else:
                        k[1]+=[-1]
                if hm_pred[p.id]==SHOWR_HM and np.linalg.norm(p.start_point-k[0].end_point)<14*4:
                    k[2]+=1
        # print(self.potential_kaons)





        # else: print("good end")
    def pass_cuts(self,cuts:dict)->bool:

        self.pass_failure=[]
        # passing_len=False
        # klens=[]
        # michlens=[]
        passed=True
        # print("TRYING A KAON",self.truth)

        # # potential_kaons=[]
        # done=False
        # while not done:
        #     done=True
        #     for r in self.particles:



        # for k in self.potential_kaons:
        #     if k.proj_dist_from_hip<cuts["par_child_dist max"][0] and len([(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in k.k_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and (p.child_hm_pred in [SHOWR_HM,MICHL_HM] or p.child.reco_length>min_len)])==0 and (Model=='2x2' or min(np.linalg.norm(k.hip.end_point-self.mip.start_point),np.linalg.norm(k.hip.start_point-self.mip.start_point))<cuts["par_child_dist max"][0]) and np.linalg.norm(k.hip.end_point-self.mip.start_point)<np.linalg.norm(k.hip.start_point-self.mip.start_point):
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

        # if self.interaction.id!==-1:
        #     if self.truth: 
        #         print("neutrino interaction")
        #     if passed:
        #         self.pass_failure="neutrino interaction"
        #         passed=False


        validlist=[]
        for k in self.potential_kaons:
            if len(k[1])!=1: validlist+=[False]
            for kk in k[1]:
                if kk==-1:
                    continue
                    # if self.truth: 
                    #     print("contained mips")
                    # if passed:
                    #     self.pass_failure="contained mips"
                    #     passed=False

                else:
                    validlist+=[((cuts["mu_len"][0]<kk.reco_length and kk.reco_length<cuts["mu_len"][1]) or 
                                (cuts["pi_len"][0]<kk.reco_length and kk.reco_length<cuts["pi_len"][1]))*(np.linalg.norm(kk.start_point-self.hip.end_point)<=np.linalg.norm(kk.start_point-self.hip.start_point))]

        # if self.hip.reco_length<min_len:
        #     if self.truth: 
        #         print("kaon_too_short")
        #     if passed:
        #         self.pass_failure="kaon_too_short"
        #         passed=False

        if (not self.hip.is_primary) or self.interaction.nu_id==-1:
            # if self.truth: 
            #     print("primary kaon")
            # if passed:
            self.pass_failure+=["primary kaon"]
            passed=False

        if not np.any(validlist):
            # self.good_mu=self.potential_kaons[np.argwhere(validlist)[0][0]][0]

            # assert type(self.good_mu)==Particle
            # if self.truth: 
            #     print("valid muon or pion decay",len(self.potential_kaons))#,[k[1:] for k in self.potential_kaons],validlist)
            # if passed:
            self.pass_failure+=["valid muon or pion decay"]
            passed=False

        if self.hip.reco_ke<40:
            # if self.truth: 
            #     print("min KE",len(self.potential_kaons))#,[k[1:] for k in self.potential_kaons],validlist)
            # if passed:
            self.pass_failure+=["min KE"]
            passed=False
            

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

        self.pass_failure+=[""]
        return passed


class Pred_L:
    """
    Storage class for Lambdas with hip and mip children

    Attributes
    ----------
    hip_id: int
        id for hip child
    mip_id: int
        id for mip child
    hip_len: float
        len attribute of the hip object
    mip_len: float
        len attribute of the mip object
    vae: float
        angle between the line constructed from the momenta of the hip and mip and
        the line constructed from the interaction vertex and the lambda decay point
    lam_mass2:float
        reconstructed mass squared of the Lambda
    lam_decay_len: float
        decay len of the lambda from the associated vertex
    momenta: list[float]
        shape(4) [hip transv. momentum, mip transv. momentum,hip long momentum, mip long momentum]
    # coll_dist: float
    #     shape(3) [t1,t2, dist]: the distance from the start point to the
    #     point along the vector start direction which is the point of
    #     closest approach to the other particle's corresponding line for the hip (t1) and mip (t2)
    #     along with the distance of closest approach of these lines (dist)
    lam_dir_acos:float
        arccos of the Lambda direction with the beam direction
    prot_hm_acc:float
        percent of the voxels for the hip whose Hip/Mip semantic segmentation matches the overall prediction
    pi_hm_acc:float
        percent of the voxels for the mip whose Hip/Mip semantic segmentation matches the overall prediction
    prot_extra_children: list[ExtraChild]
        extra children for the Lambda
    pi_extra_children: list[ExtraChild]
        extra children for the mip
    lam_extra_children: list[ExtraChild]
        extra children for the hip
    mip: Particle
        candidate pion
    hip: Particle
        candidate proton
    """

    hip_id: int
    mip_id: int
    hip_len: float
    mip_len: float
    vae: float
    lam_mass2: float
    lam_decay_len: float
    momenta: list[float]
    # coll_dist: list[float]
    lam_dir_acos: float
    prot_hm_acc: float
    pi_hm_acc: float
    # prot_extra_children: list[ExtraChild]
    # pi_extra_children: list[ExtraChild]
    lam_extra_children: list[ExtraChild]
    truth:bool
    mip: Particle
    hip: Particle

    def __init__(
        self,
        hip:Particle,
        mip:Particle,
        particles:list[Particle],
        interactions: list[Interaction],
        hm_acc:list[float],
        hm_pred:list[int],
        # hip_id,
        # mip_id,
        # hip_len,
        # mip_len,
        # vae,
        # lam_mass2,
        # lam_decay_len,
        # momenta: list[float],
        # coll_dist,
        # lam_extra_children,
        # prot_extra_children,
        # pi_extra_children,
        # lam_dir_acos,
        # prot_hm_acc,
        # pi_hm_acc,
        truth:bool,
        reason:bool
    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        hip: Particle
            hip particle object
        mip: Particle
            mip particle object
        particles: list[Particle]
            list of particle objects for the event
        interactions: list[Interaction]
            list of interactions for the event
        hm_acc: list[float]
            hip mip semantic segmentation accuracy for each particle
        hm_pred: list[int]
            hip mip semantic segmentation prediction for each particle
        truth:bool
            is this a signal lambda or not
        """

        # assert hm_pred[hip.id]==HIP_HM
        # assert hm_pred[mip.id]==MIP_HM
        self.hip_id = hip.id
        self.mip_id = mip.id
        self.hip_len =hip.reco_length
        self.mip_len = mip.reco_length
        self.interaction=interactions[hip.interaction_id]
        self.mip=mip
        self.hip=hip
        # self.lam_dir_acos = direction_acos((hip.reco_momentum+mip.reco_momentum)/np.linalg.norm(hip.reco_momentum+mip.reco_momentum))
        self.prot_hm_acc = hm_acc[hip.id]
        self.pi_hm_acc = hm_acc[mip.id]
        
        self.truth=truth
        self.reason=reason
        self.pass_failure=""
        self.error=""

        self.real_hip_momentum=hip.reco_momentum
        self.real_hip_momentum_reco=momentum_from_daughter_ke_reco(hip,particles,PROT_MASS,ignore=[mip])
        self.real_mip_momentum_reco=momentum_from_daughter_ke_reco(mip,particles,PION_MASS,ignore=[hip])
        # self.hip_daughters_contained_reco=all_daughters_contained_reco(hip,particles,ignore=[mip])
        # self.mip_daughters_contained_reco=all_daughters_contained_reco(mip,particles,ignore=[hip])
        if self.truth:
            assert type(hip)==TruthParticle
            self.real_hip_momentum=momentum_from_daughter_ke(hip,PROT_MASS)


        self.real_mip_momentum=mip.reco_momentum
        if self.truth:
            assert type(mip)==TruthParticle
            self.real_mip_momentum=momentum_from_daughter_ke(mip,PION_MASS)

        self.vae = self.vertex_angle_error()
        self.lam_mass2 = self.lambda_mass_2()
        self.lam_decay_len = self.lambda_decay_len()
        # self.momenta: list[float] = self.momenta_projections()
        # self.coll_dist = collision_distance(hip,mip)
        
        if self.truth:
            assert type(self.mip)==TruthParticle and type(self.hip)==TruthParticle
            # self.true_signal=abs(self.mip.parent_pdg_code)==3122 and abs(self.hip.pdg_code)==2212 and abs(self.mip.pdg_code)==211 and self.mip.parent_id==self.hip.parent_id and process_map[self.hip.creation_process]=='6::201' and process_map[self.mip.creation_process]=='6::201'
            # truth_parsed,self.reason=self.is_truth(particles)
            # assert truth_parsed==self.truth, (truth_parsed,self.reason)
        # if self.truth: print("We got a true lambda")
        # # self.lam_extra_children = lambda_children(hip,mip,[p for p in particles if hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]])

        # guess_start = get_pseudovertex(
        #     start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        #     directions=[hip.reco_start_dir, mip.reco_start_dir],
        # )
        guess_start=(hip.start_point+mip.start_point)/2

        # self.pi_extra_children=[]
        # self.prot_extra_children=[]
        self.lam_extra_children=[]
        self.pot_parent:list[tuple[Particle,bool]]=[]
        for p in particles:
            if p.interaction_id!=hip.interaction_id: continue
            if p.id not in [mip.id,hip.id] and hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM,MICHL_HM]:
                # self.prot_extra_children +=[ExtraChild(p,hip.end_point,hm_pred,hip)]
                # self.pi_extra_children +=[ExtraChild(p,mip.end_point,hm_pred,mip)]
                self.lam_extra_children += [ExtraChild(p, guess_start,hm_pred)]
                if hm_pred[p.id] in [MIP_HM,HIP_HM]: self.pot_parent+=[(p,False)]
    
    def pass_cuts(self,cuts:dict)->bool:
        passed=True

        self.pass_failure=[]
        # print("TRYING a lambda",self.truth)
        # if not (is_contained(self.hip.points,mode=full_containment,margin=2) and is_contained(self.mip.points,mode=full_containment,margin=2)):
        #     if self.truth: print("lam child containment issue")
        #     if passed:
        #         self.pass_failure="lam child containment issue"
        #     passed=False

        # if np.linalg.norm(self.mip.start_point-self.hip.start_point)>cuts["lam_cont_dist max"]:
        #     if self.truth: print("lam child contained start distance max", np.linalg.norm(self.mip.start_point-self.hip.start_point))
        #     if passed:
        #         self.pass_failure="lam child contained start distance max"
        #     passed=False

        # if self.coll_dist[-1]>cuts["lam_proj_dist max"]:
        #     if self.truth: print("lam_proj_dist max", self.coll_dist[-1])
        #     if passed:
        #         self.pass_failure="lam_proj_dist max"
        #     passed=False

        if self.hip.is_primary or self.mip.is_primary or self.interaction.nu_id==-1:
            # if self.truth: print("nonprimary hip/mip", np.linalg.norm(self.mip.start_point-self.hip.start_point))
            # if passed:
            self.pass_failure+=["nonprimary hip/mip"]
            passed=False


        
        
        if np.linalg.norm(self.mip.start_point-self.hip.start_point)>cuts["lam_dist max"]:
            # if self.truth: print("Max HIP/MIP Sep.", np.linalg.norm(self.mip.start_point-self.hip.start_point))
            # if passed:
            self.pass_failure+=["max HIP/MIP sep."]
            passed=False

        # if not self.hip_daughters_contained_reco or not self.mip_daughters_contained_reco:
        #     if self.truth: print("prot or pi containment")
        #     if passed:
        #         self.pass_failure="prot or pi containment"
        #     passed=False

        # assert is_contained(self.interaction.vertex,mode=full_containment)

        # if not is_contained(self.interaction.reco_vertex,mode=full_containment):
        #     if self.truth: print("vertex OOB")
        #     if passed:
        #         self.pass_failure="vertex OOB"
        #     passed=False

        assert np.linalg.norm(self.interaction.reco_vertex)>0,self.interaction.reco_vertex
        # assert np.linalg.norm(self.interaction.reco_vertex)<np.inf,self.interaction.reco_vertex

        # if np.linalg.norm(self.interaction.reco_vertex)==np.inf:
        #     if self.truth: print("prot or pi containment")
        #     if passed:
        #         self.pass_failure="prot or pi containment"
        #     passed=False

        # if momentum_from_daughter_ke_reco(self.hip,particles,KAON_MASS)

        # print(self.interaction.reco_vertex)
        
        
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

        guess_start=(self.hip.start_point+self.mip.start_point)/2
        ldl=float(np.linalg.norm(self.interaction.reco_vertex - guess_start))
        self.lam_decay_len=ldl



        for p in self.pot_parent:
            # est_decay=(self.mip.start_point+self.hip.start_point)/2
            if (np.linalg.norm(p[0].end_point-self.interaction.reco_vertex)>=np.linalg.norm(guess_start-self.interaction.reco_vertex) and 
                np.linalg.norm(p[0].start_point-self.interaction.reco_vertex)>=np.linalg.norm(guess_start-self.interaction.reco_vertex)):continue
            if np.linalg.norm(p[0].end_point-guess_start)<=min_len or np.linalg.norm(p[0].end_point-self.hip.start_point)<=min_len/2 or np.linalg.norm(p[0].end_point-self.mip.start_point)<=min_len/2:
                # np.linalg.norm(p[0].end_point-self.interaction.reco_vertex)<= np.linalg.norm(self.hip.start_point-self.interaction.reco_vertex)+min_len and
                # np.linalg.norm(p[0].end_point-self.interaction.reco_vertex)<= np.linalg.norm(self.mip.start_point-self.interaction.reco_vertex))+min_len:
                # if self.truth: print("parent proximity")
                # if passed:
                self.pass_failure+=["parent proximity"]
                passed=False
                break
        for p in self.pot_parent:
            # est_decay=(self.mip.start_point+self.hip.start_point)/2
            v1=guess_start-p[0].end_point
            if np.linalg.norm(p[0].end_point-self.interaction.reco_vertex)>=np.linalg.norm(guess_start-self.interaction.reco_vertex):continue
            # if 
            if p[0].is_primary and np.dot(p[0].reco_end_dir,v1)>np.linalg.norm(v1)*np.linalg.norm(p[0].reco_end_dir)*np.cos(np.pi/8):
                # if self.truth: print("primary colinear parent")
                # if passed:
                self.pass_failure+=["primary colinear parent"]
                passed=False
                break
                
        for p in self.lam_extra_children:
            if (np.linalg.norm(p.child.start_point-guess_start)>=np.linalg.norm(guess_start-self.hip.end_point) or
                np.linalg.norm(p.child.start_point-guess_start)>=np.linalg.norm(guess_start-self.mip.end_point)):continue
            # if p.child.is_primary:continue
            if (np.linalg.norm(p.child.start_point-guess_start)<=min(min_len,np.linalg.norm(p.child.start_point-self.interaction.reco_vertex))):
                # if self.truth: print("extra child")
                # if passed:
                self.pass_failure+=["number of children"]
                passed=False
                break


        inter = self.interaction.reco_vertex
        # guess_start = get_pseudovertex(
        #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
        #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
        # )
        assert np.linalg.norm(guess_start-guess_start)==0
        lam_dir1 = guess_start - inter
        if np.isclose(np.linalg.norm(lam_dir1),0):
            vae=0
        else:
            lam_dir2 = self.real_hip_momentum_reco + self.real_mip_momentum_reco

            # if np.linalg.norm(lam_dir2) == 0:
            #     return np.nan
            ret = np.arccos(
                np.dot(lam_dir1, lam_dir2) / np.linalg.norm(lam_dir1) / np.linalg.norm(lam_dir2)
            )
            if passed: assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,lam_dir1,lam_dir2)
            vae=ret

        self.vae=vae
        base_len=self.lam_decay_len*2*np.sin(self.vae/2)
        if base_len>cuts["lam_VAE max"]:
            # if self.truth: print("VAE max", vae)
            # if passed:
            self.pass_failure+=["max momentum angular diff."]
            passed=False

        if ldl<cuts["lam_decay_len"]:
            # if self.truth: print("minimum decay len")
            # if passed:
            self.pass_failure+=["minimum decay len"]
            passed=False

        # if ldl>cuts["lam_decay_len_max"]:
        #     if self.truth: print("decay len max")
        #     if passed:
        #         self.pass_failure="decay len max"
        #     passed=False

        # mom_norm=np.linalg.norm(self.real_hip_momentum_reco + self.real_mip_momentum_reco)
        # tau=ldl/100/(2.998e8)*LAM_MASS/mom_norm*10**9
        # if tau>cuts["tau_max"]:
        #     if self.truth: print("tau cut")
        #     if passed:
        #         self.pass_failure="max tau"
        #     passed=False

        if (self.interaction.primary_particle_counts[PHOT_PID]%2==1 and 
            self.interaction.primary_particle_counts[MUON_PID]<=2 and 
            self.interaction.primary_particle_counts[PROT_PID]<=1 and 
            self.interaction.primary_particle_counts[KAON_PID]<=1
            ):
            # if self.truth: 
            #     print("odd photon cut")
            # if passed:
            self.pass_failure+=["even primary photon count"]
            passed=False

        

        # self.interaction.flas

        self.pass_failure+=[""]

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

        return passed
    
    def lambda_decay_len(self) -> float:
        """
        Returns distance from average start position of hip and mip to vertex location of the assocated interaction

        Returns
        -------
        float
            distance from lambda decay point to vertex of interaction
        """
        # guess_start = get_pseudovertex(
        #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
        #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
        # )
        guess_start=(self.hip.start_point+self.mip.start_point)/2
        return float(np.linalg.norm(self.interaction.vertex - guess_start))
    
    def lambda_mass_2(self) -> float:
        """
        Returns lambda mass value constructed from the
        hip and mip candidate deposited energy and predicted direction

        Returns
        -------
        float
            reconstructed lambda mass squared
        """
        # LAM_MASS=1115.60 #lambda mass in MeV
        assert (
            self.mip.ke > 0
        )  # print(mip.ke,"very bad",mip.id,mip.parent_pdg_code,mip.pid,mip.pdg_code,mip.energy_init)
        assert (
            self.hip.ke > 0
        )  # print(hip.ke,"very bad",hip.id,hip.parent_pdg_code,hip.pid,hip.pdg_code,hip.energy_init)
        lam_mass2 = (
            PROT_MASS**2
            + PION_MASS**2
            + 2 * (self.mip.reco_ke + PION_MASS) * (self.hip.reco_ke + PROT_MASS)
            - 2 * np.dot(self.hip.reco_momentum, self.mip.reco_momentum)
        )
        return lam_mass2
    
    def vertex_angle_error(self) -> float:
        """
        Returns angle between the line constructed from the momenta of the hip and mip and
        the line constructed from the interaction vertex and the lambda decay point

        Returns
        -------
        float
            distance from interaction vertex to line consructed from the momenta of the hip and mip
        """

        inter = self.interaction.vertex
        # guess_start = get_pseudovertex(
        #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
        #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
        # )
        guess_start=(self.hip.start_point+self.mip.start_point)/2
        assert np.linalg.norm(guess_start-guess_start)==0
        lam_dir1 = guess_start - inter
        if np.isclose(np.linalg.norm(lam_dir1),0):
            return 0

        lam_dir2 = self.real_hip_momentum_reco + self.real_mip_momentum_reco

        if np.linalg.norm(lam_dir2) == 0:
            return np.nan
        ret = np.arccos(
            np.dot(lam_dir1, lam_dir2) / np.linalg.norm(lam_dir1) / np.linalg.norm(lam_dir2)
        )
        assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,np.dot(lam_dir1, lam_dir2) , np.linalg.norm(lam_dir1) , np.linalg.norm(lam_dir2),np.dot(lam_dir1, lam_dir2) / np.linalg.norm(lam_dir1) / np.linalg.norm(lam_dir2))
        return ret
    
    # def momenta_projections(self) -> list[float]:
    #     """
    #     Returns the P_T and P_L of each particle relative to the lambda measured from the decay

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

    #     inter = self.interaction.vertex

    #     guess_start = get_pseudovertex(
    #         start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
    #         directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
    #     )
    #     lam_dir = guess_start - inter

    #     p1 = self.hip.reco_momentum
    #     p2 = self.mip.reco_momentum

    #     lam_dir = p1 + p2  # fix this hack #TODO

    #     lam_dir_norm = np.linalg.norm(lam_dir)
    #     if lam_dir_norm == 0:
    #         return [np.nan, np.nan, np.nan, np.nan]

    #     lam_dir = lam_dir / lam_dir_norm

    #     p1_long = np.dot(lam_dir, p1)
    #     p2_long = np.dot(lam_dir, p2)

    #     p1_transv = float(np.linalg.norm(p1 - p1_long * lam_dir))
    #     p2_transv = float(np.linalg.norm(p2 - p2_long * lam_dir))

    #     return [p1_transv, p2_transv, p1_long, p2_long]


def is_contained(pos: np.ndarray, mode: str =full_containment, margin: float = 3) -> bool:
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
    Geo.define_containment_volumes(margin, mode=mode)
    return bool(Geo.check_containment(pos))


def HIPMIP_pred(particle: Particle, sparse3d_pcluster_semantics_HM: np.ndarray) -> int:
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
    if len(particle.depositions) == 0:
        raise ValueError("No voxels")
    # slice a set of voxels for the target particle
    HM_Pred = sparse3d_pcluster_semantics_HM[particle.index, -1]
    # print(HM_Pred,type(HM_Pred))
    return st.mode(HM_Pred).mode


def HIPMIP_acc(particle: Particle, sparse3d_pcluster_semantics_HM: np.ndarray) -> float:
    """
    Returns the fraction of voxels for a particle whose HM semantic segmentation prediction agrees
    with that of the particle itself as decided by HIPMIP_pred

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information and unique semantic segmentation prediction
    sparse3d_pcluster_semantics_HM : np.ndarray
        HIP/MIP semantic segmentation predictions for each voxel in an image

    Returns
    -------
    int
        Fraction of voxels whose HM semantic segmentation agrees with that of the particle
    """
    if len(particle.depositions) == 0:
        raise ValueError("No voxels")
    # slice a set of voxels for the target particle
    HM_Pred = sparse3d_pcluster_semantics_HM[particle.index, -1]
    pred = st.mode(HM_Pred).mode
    return Counter(HM_Pred)[pred] / len(HM_Pred)


def direction_acos(momenta: np.ndarray, direction=np.array([0.0, 0.0, 1.0])) -> float:
    """
    Returns angle between the beam-axis (here assumed in z) and the particle object's start direction

    Parameters
    ----------
    momenta : np.ndarray[float]
        Momenta of the particle
    direction : np.ndarray[float]
        Direction of beam

    Returns
    -------
    float
        Angle between particle direction and beam
    """
    assert np.isclose(np.linalg.norm(momenta),1),np.linalg.norm(momenta)
    assert np.isclose(np.linalg.norm(direction),1),np.linalg.norm(direction)

    return np.arccos(np.dot(momenta, direction))


def collision_distance(particle1: Particle, particle2: Particle,orientation:list[str]=["start","start"]):
    """
    Returns for each particle, the distance from the start point to the point along the vector start direction
    which is the point of closest approach to the other particle's corresponding line, along with the distance of closest approach.
    The parameters, t1 and t2, are calculated by minimizing ||p1+v1*t1-p2-v2*t2||^2, where p1/p2 are the starting point of each particle
    and v1/v2 are the start direction of each particle

    Parameters
    ----------
    particle1 : spine.Particle
        Particle object with cluster information
    particle2 : spine.Particle
        Particle object with cluster information

    Returns
    -------
    [float,float,float]
        [t1,t2, min_{t1,t2}(||p1+v1*t1-p2-v2*t2||^2)]
    """
    assert set(orientation).issubset(set(["start","end"]))

    if orientation[0]=="start":
        v1 = particle1.reco_start_dir
        p1 = particle1.start_point
    elif orientation[0]=="end":
        v1 = particle1.reco_end_dir
        p1 = particle1.end_point
    else:
        raise Exception()

    if orientation[1]=="start":
        v2 = particle2.reco_start_dir
        p2 = particle2.start_point
    elif orientation[1]=="end":
        v2 = particle2.reco_end_dir
        p2 = particle2.end_point
    else:
        raise Exception()
    

    v11 = np.dot(v1, v1)
    v22 = np.dot(v2, v2)
    v12 = np.dot(v1, v2)
    dp = p1 - p2

    denom = v12**2 - v11 * v22

    if denom == 0:
        return [0, 0, np.linalg.norm(dp)]

    t1 = (np.dot(v1, dp) * v22 - v12 * np.dot(v2, dp)) / denom
    t2 = (v12 * np.dot(v1, dp) - np.dot(v2, dp) * v11) / denom

    min_dist = np.dot(p1 + v1 * t1 - p2 - v2 * t2, p1 + v1 * t1 - p2 - v2 * t2)

    return [t1, t2, min_dist]







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

#     Lvecnorm=np.linalg.norm(Lvec)
#     if Lvecnorm==0:
#         return [np.nan,np.nan,np.nan]

#     Lvec=Lvec/Lvecnorm

#     p1_L=np.dot(Lvec,p1)
#     p2_L=np.dot(Lvec,p2)

#     p1_T=float(np.linalg.norm(p1-p1_L*Lvec))
#     p2_T=float(np.linalg.norm(p2-p2_L*Lvec))

#     # asymm=abs((p1_L-p2_L)/(p1_L+p2_L))
#     # pt=p1_T+p2_T
#         # print("very good",asymm,p1_L,p2_L,Lvec)
#         # assert asymm>=-1 and asymm<=1, print("help me",asymm,p1_L,p2_L,Lvec)
#     assert np.linalg.norm((p1-p1_L*Lvec)+(p2-p2_L*Lvec))<=1e-3,print(np.linalg.norm((p1-p1_L*Lvec)+(p2-p2_L*Lvec)))
#     return [p1_T,p2_T,np.abs(p1_L-p2_L)/(p1_L+p2_L)]




# def come_to_rest(p:Particle,percent_error=.2)->bool:
#     if not is_contained(p.points,mode=full_containment):return False
#     if not p.csda_ke>0: return False
#     if (not p.calo_ke>0) and (not p.mcs_ke>0): return False
#     check=p.mcs_ke if p.mcs_ke>0 else p.calo_ke
#     return abs(check/p.csda_ke-1)<percent_error


def all_daughters_reco(p:Particle,particles:list[Particle],dist=min_len/2,ignore=[])->list[Particle]:

    done=False
    daughters:list[Particle]=[p]
    while done==False:
        done=True
        for pd in particles:
            for d in daughters:
                if np.linalg.norm(pd.start_point-d.end_point)<dist and pd not in daughters and pd not in ignore and not pd.is_primary:
                    daughters+=[pd]
                    done=False
    return daughters

# def all_daughters_contained_reco(p:Particle,particles:list[Particle],mode=full_containment,ignore=[])->bool:
#     ad=all_daughters_reco(p,particles,ignore=ignore)
#     assert p in ad
#     contained=True
#     for d in ad:
#         # if id not in larcv_id_to_spine_id: continue
#         contained*=is_contained(d.points,mode=mode)
#     return bool(contained)

def momentum_from_daughter_ke_reco(p:Particle,particles:list[Particle],mass,ignore=[])->float:
    # print("running mfdke")
    ad=all_daughters_reco(p,particles,ignore=ignore)
    assert p in ad
    tke=sum([d.calo_ke for d in ad if d.shape not in [MICHL_SHP]])
    assert np.sqrt((tke+mass)**2-mass**2)>0,(tke,mass)
    assert len(p.reco_start_dir)==3
    return np.sqrt((tke+mass)**2-mass**2)*p.reco_start_dir


def all_daughters(p:TruthParticle)->list[TruthParticle]:
    out:list[TruthParticle]=[]
    if len(p.children_id)==0:
        return [p]
    to_explore:list[TruthParticle]=[p]
    while len(to_explore)!=0:
        curr=to_explore.pop()
        # if curr not in larcv_id_to_spine_id: continue
        # if abs(larcv_id_to_spine_id[curr].parent_pdg_code)==2112:
        #     continue
        out+=[curr]
        #TODO fix missing daughters in the particle record
        to_explore+=[i for i in curr.children_id if i not in out]
        # print(to_explore)
        # if curr.
    
    # start=
    return out

# def all_daughters_contained(p_tid:int,particles:list[TruthParticle],mode=full_containment)->bool:
#     # ad:list[TruthParticle]=all_daughters(p,particles)
#     # assert p in ad
#     contained=True
#     for d in particles:
#         if d.ancestor_track_id!=p_tid:continue
#         if d.parent_pdg_code==2112:continue
#         # if id not in larcv_id_to_spine_id: continue
#         contained*=is_contained(d.points,mode=mode)
#     return bool(contained)

def momentum_from_daughter_ke(p:TruthParticle,mass)->float:
    # print("running mfdke")
    ad:list[TruthParticle]=all_daughters(p)
    assert p in ad

    tke=sum([a.calo_ke for a in ad])
    return np.sqrt((tke+mass)**2-mass**2)*p.reco_start_dir