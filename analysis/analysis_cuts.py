""" 
This file contains output classes and cut functions useful for reconstructing kaons and 
lambdas in a liquid argon TPC using the reconstruction package SPINE https://github.com/DeepLearnPhysics/spine
"""
# from ast import Module
from collections import Counter
# import string
SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf

import sys
# Set software directory
sys.path.append(SOFTWARE_DIR)
from spine.data.out import TruthParticle,RecoParticle,RecoInteraction,TruthInteraction
from spine.utils.globals import PROT_MASS, PION_MASS
from spine.utils.geo.manager import Geometry
from spine.utils.vertex import get_pseudovertex
import numpy as np
from scipy import stats as st
from spine.utils.globals import MICHL_SHP,TRACK_SHP,SHOWR_SHP
from scipy.spatial.distance import cdist


# TODO things I would like added in truth:
# TODO spine.TruthParticle.children_id for ease of use
# TODO spine.TruthParticle.mass propagated to truth particles in larcv
# TODO spine.TruthParticle.parent_end_momentum
# TODO id vs parentid vs orig id vs the actual parentid that I want

# TODO things I would like at some point:
# TODO some sort of particle flow predictor
# TODO decay at rest predictor?

# TODO things I would like fixed:
# TODO michel timing issue
# TODO multiple contribs in samples

# TODO deal with particles with small scatters wih len cut

# TODO things I don't know that I need but may end up being useful
# TODO Kaon/ Michel flash timing?

# TODO remove some of the cuts on intermediate walls
# TODO add k0s info
# TODO if not truth, add in failure mode
# TODO if truth, and is a faiure, record faiure mode


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

LAM_PT_MAX = (np.sqrt((LAM_MASS**2 - (PION_MASS + PROT_MASS) ** 2) * (LAM_MASS**2 - (PION_MASS - PROT_MASS) ** 2))/2/LAM_MASS)


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
    for i in ["muIoni","conv","compt","eBrem","annihil","nCapture","muPairProd","muMinusCaptureAtRest","hIoni","muBrems","photonNuclear","eIoni"]:
        process_map[i]="other"
    process_map["muonNuclear"]="maybe_something"
    process_map["Decay"]="6::201"
    for i in ['pi+Inelastic', 'protonInelastic',"pi-Inelastic","kaon0LInelastic","kaon-Inelastic","kaon+Inelastic","lambdaInelastic","dInelastic","anti-lambdaInelastic","sigma-Inelastic","kaon0SInelastic","sigma+Inelastic","anti_neutronInelastic","neutronInelastic","anti_protonInelastic"]:
        process_map[i]="4::121"
    process_map["hBertiniCaptureAtRest"]="4::151"



Geo = Geometry(detector=Model)

class ExtraChild:
    child_id:int
    dist_to_parent:float
    proj_dist_to_parent:float
    angle_to_parent:float
    truth:bool
    child:Particle
    child_hm_pred: int

    """
    Storage class for extra children of any particle
    Attributes
    ----------
    child_id : int
        id for child
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
        self.child_id=child.id
        
        par_end=parent_end_pt
        self.dist_to_parent=float(np.linalg.norm(child.start_point-par_end))
        self.proj_dist_to_parent=np.nan
        if parent!=None and not is_contained(parent.points,mode=full_containment):
            # print(parent.end_dir,parent.reco_end_dir)
            # par_end = get_pseudovertex(
            #     start_points=np.array([parent.end_point, child.start_point], dtype=float),
            #     directions=[parent.reco_end_dir, child.start_dir],
            # )
            self.proj_dist_to_parent=collision_distance(parent,child,orientation=["end","start"])[-1]

        self.child_hm_pred=hm_pred[self.child_id]
        if self.dist_to_parent==0:
            self.angle_to_parent=0
        else:
            self.angle_to_parent=direction_acos((child.start_point-par_end)/self.dist_to_parent,child.start_dir)
        self.truth=False
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


class PotK:
    """
    Storage class for primary Kaons and their cut parameters

    Attributes
    ----------
    hip_id : int
        id for hip associated to this class
    hip_len: float
        len attribute of the particle object
    dir_acos: float
        arccos of the particle direction with the beam direction
    k_hm_acc:float
        percent of the voxels of this particle
        whose Hip/Mip semantic segmentation matches the overall prediction
    """

    hip_id: int
    hip_len: float
    dir_acos: float
    k_hm_acc: float
    dist_from_hip: float
    proj_dist_from_hip:float
    k_extra_children: list[ExtraChild]
    # dist_Hausdorff_from_hip:float
    pred_end: list[float]
    truth: bool
    hip: Particle
    truth_list:list[bool]

    def __init__(self, kaon:Particle,hm_acc:float,mip:Particle,particles:list[Particle],interactions:list[Interaction],set_truth:bool):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        kaon : Particle
            Kaon candidate particle object
        hm_acc:float
            percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
        """
        self.hip_id = kaon.id
        self.hip_len = kaon.reco_length
        self.dir_acos = direction_acos(kaon.start_dir)
        self.k_hm_acc = hm_acc
        self.k_extra_children=[]
        self.dist_from_hip=0
        self.proj_dist_from_hip=0
        # self.dist_Hausdorff_from_hip=0
        self.hip=kaon
        self.truth_list=[False]
        if set_truth:
            assert isinstance(mip,TruthParticle) and isinstance(self.hip,TruthParticle) and isinstance(kaon,TruthParticle)
            vis_part=[part.orig_id for part in particles if part.reco_length>3]
            # true_par_id=mip.parent_id
            fake_hip=kaon
            id_to_particle={}
            for par in particles:
                id_to_particle[par.orig_id]=par

            done=False
            while not done:
                done=True
                for id in fake_hip.children_id:
                    if id not in id_to_particle:
                        break
                    if abs(id_to_particle[id].pdg_code)==321 and np.linalg.norm(fake_hip.end_point-id_to_particle[id].start_point)<3 and id not in vis_part:
                        done=False
                        fake_hip=id_to_particle[id]
                        break
            self.truth_list=[abs(fake_hip.pdg_code)==321,(mip.parent_id==fake_hip.orig_id or (mip.parent_id in fake_hip.children_id and mip.parent_id not in vis_part)),abs(mip.parent_pdg_code)==321,(self.hip.is_primary or bool(np.linalg.norm(self.hip.start_point-interactions[mip.interaction_id].vertex)<min_len)),bool(np.linalg.norm(fake_hip.end_point-mip.start_point)<3)]
        self.truth=bool(np.all(self.truth_list))

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

    mip_id: int
    mip_len_base: float
    mu_hm_acc: float
    potential_kaons: list[PotK]
    # potential_michels: list[PotMich]
    mu_extra_children:list[ExtraChild]
    truth:bool
    mip:Particle
    require_kaon:bool
    require_michel:bool #TODO implement this

    # true_signal:bool

    def __init__(
        self,
        # pot_k: PotK,
        mip: Particle,
        particles: list[Particle],
        interactions: list[Interaction],
        hm_acc:list[float],
        hm_pred:list[int],
        
        assign_truth:bool
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

        self.mip_id = mip.id
        self.mip_len_base=mip.reco_length
        self.mu_hm_acc = hm_acc[mip.id]
        self.potential_kaons=[]
        self.potential_michels=[]
        self.truth=False
        self.reason=""
        self.pass_failure=""
        self.mip=mip
        # self.true_signal=False
        self.error=""
        self.mu_extra_children=[]
        if assign_truth:
            assert type(mip)==TruthParticle and type(self.mip)==TruthParticle
            if mip.pdg_code==-13:print("muon with parent",mip.parent_pdg_code,process_map[mip.creation_process])
            # self.true_signal=(mip.pdg_code==-13 and mip.parent_pdg_code==321 and process_map[mip.creation_process]=='6::201' and float(mip.p)>230.)
            # if self.true_signal: print("found a signal kaon")
            self.truth=(mip.pdg_code==-13 and mip.parent_pdg_code==321 and process_map[mip.creation_process]=='6::201' and float(mip.p)>230. and is_contained(self.mip.points,mode=full_containment) and is_contained(self.mip.points,mode=full_containment)) #TODO particles[mip.parent_id].is_primary primary cut
            if not is_contained(self.mip.points,mode=full_containment):
                self.reason="1"
            if not (mip.pdg_code==-13 and mip.parent_pdg_code==321):
                self.reason="2"
            if not float(mip.p)>230.:
                self.reason="3"
            if not process_map[mip.creation_process]=='6::201':
                self.reason="4"
            
            # if self.truth: print(mip.reco_length,mip.p)
            if self.truth:
                for p in particles:
                    assert type(p)==TruthParticle
                    if process_map[p.creation_process] in ['4::121','4::151'] and p.reco_length>min_len and abs(p.pdg_code)!=321:
                        if p.parent_id==mip.orig_id and direction_acos(mip.reco_end_dir,p.start_dir)>np.pi/12:
                            self.truth=False
                            self.reason="5"
                            break
                        if p.parent_id==mip.parent_id:
                            self.truth=False
                            self.reason="6"
                            break
                        # # if p.orig_id==mip.parent_id and np.linalg.norm(p.start_point-interactions[p.interaction_id].vertex)>min_len:
                        # #     self.truth=False
                        #     break
        # if self.truth: print("found a kaon in truth")
        self.require_kaon=((not self.mip.is_primary) and bool(np.linalg.norm(self.mip.start_point-interactions[self.mip.interaction_id].vertex)>min_len))
        # self.require_michel=(not is_contained(self.mip.points,mode=full_containment))
        self.require_michel=False
            

        for p in particles:  
            if hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]:
                self.mu_extra_children+=[ExtraChild(p,self.mip.end_point,hm_pred,self.mip)]
                
        for hip_candidate in particles:
            if not hip_candidate.is_primary and np.linalg.norm(hip_candidate.start_point-interactions[self.mip.interaction_id].vertex)>min_len:
                continue #PRIMARY
            if hm_pred[hip_candidate.id]!=HIP_HM:
                continue #HIP
            if hip_candidate.reco_length<min_len:
                continue
            if not is_contained(hip_candidate.points,mode=full_containment):
                continue
            self.potential_kaons+=[PotK(hip_candidate,hm_acc[hip_candidate.id],mip,particles,interactions,self.truth and assign_truth)]
            # if self.truth:print("got a true muon")
            if assign_truth and self.truth:
                assert isinstance(hip_candidate,TruthParticle) and isinstance(mip,TruthParticle)
                print("got a true muon",abs(hip_candidate.pdg_code)==321,(hip_candidate.orig_id,[mip.parent_id,mip.ancestor_track_id]),hip_candidate.is_primary,bool(np.linalg.norm(hip_candidate.start_point-interactions[self.mip.interaction_id].vertex)<min_len))#TODO figure out what to do with this ancestor, etc
                # self.potential_kaons[-1].truth= abs(hip_candidate.pdg_code)==321 and (mip.parent_id==hip_candidate.orig_id or (mip.parent_id in hip_candidate.children_id and mip.parent_id not in vis_part)) and abs(mip.parent_pdg_code)==321 and (hip_candidate.is_primary or bool(np.linalg.norm(hip_candidate.start_point-interactions[self.mip.interaction_id].vertex)<min_len)) and bool(np.linalg.norm(hip_candidate.end_point-mip.start_point)<3)
                #+[par.parent_id for par in particles if par.orig_id==mip.parent_id and abs(mip.parent_pdg_code)==321 and abs(par.parent_pdg_code)==321 and not par.is_primary]
                if self.potential_kaons[-1].truth:
                    print("WE GOT ONE")
                    for p in particles:
                        assert type(p)==TruthParticle
                        if process_map[p.creation_process] in ['4::121','4::151'] and p.parent_id==hip_candidate.orig_id and p.reco_length>min_len and direction_acos(hip_candidate.reco_end_dir,p.start_dir)>np.pi/12:
                            self.potential_kaons[-1].truth=False
                            break
            for p in particles:
                
                if p.id not in [mip.id,hip_candidate.id] and hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]:
                    self.potential_kaons[-1].k_extra_children+=[ExtraChild(p,hip_candidate.end_point,hm_pred,hip_candidate)]
                    if self.potential_kaons[-1].truth:
                        assert type(p) == TruthParticle
                        assert type(hip_candidate)==TruthParticle
                        assert abs(hip_candidate.pdg_code)==321,abs(hip_candidate.pdg_code)
                        if p.parent_id==hip_candidate.orig_id:
                            if abs(p.parent_pdg_code)!=321:
                                self.error=str((abs(p.parent_pdg_code),abs(hip_candidate.pdg_code),p.parent_id,hip_candidate.orig_id,p.id,p.orig_id,hip_candidate.parent_id,hip_candidate.track_id,hip_candidate.parent_track_id,p.track_id,p.parent_track_id,self.potential_kaons[-1].truth_list))
                            self.potential_kaons[-1].k_extra_children[-1].truth=True
            # hip_mip_pv=get_pseudovertex(start_points=np.array([hip_candidate.end_point, mip.start_point], dtype=float), directions=[hip_candidate.reco_end_dir, mip.start_dir])
            # self.potential_kaons[-1].dist_from_hip=float(np.linalg.norm(hip_mip_pv-mip.start_point))
        if assign_truth and self.require_kaon:
            if not np.any([p.truth for p in self.potential_kaons]):
                self.truth=False
                self.reason="7"

        # for mich_candidate in particles:
        #     if hm_pred[mich_candidate.id]!=MICHL_HM:
        #         continue #HIP
        #     self.potential_michels+=[PotMich(mich_candidate,hm_acc[mich_candidate.id])]
        #     # self.potential_michels[-1].mu_extra_children=children(mip, particles, ignore=[mich_candidate.id])
        #     for p in particles:
        #         if p.id not in [mip.id,mich_candidate.id] and hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]:
        #             self.potential_michels[-1].mu_extra_children+=[ExtraChild(p,mip.end_point,hm_pred,mip)]
        #             # print("just a check",mip.end_point-mip.end_position)
        #             if self.potential_michels[-1].truth:
        #                 assert type(mip)==TruthParticle and type(p)==TruthParticle
        #                 if p.parent_id==mip.orig_id:
        #                     assert abs(p.parent_pdg_code)==13,abs(p.parent_pdg_code)
        #                     self.potential_michels[-1].mu_extra_children[-1].truth=True
        #     # mich_mip_pv=get_pseudovertex(start_points=np.array([mich_candidate.start_point, mip.end_point], dtype=float), directions=[mich_candidate.start_dir, mip.reco_end_dir])
        #     # self.potential_michels[-1].dist_to_mich=float(np.linalg.norm(mip.end_point-mich_mip_pv))
        #     if assign_truth and self.truth:
        #         assert type(mip)==TruthParticle and type(mich_candidate)==TruthParticle
        #         self.potential_michels[-1].truth=abs(mich_candidate.pdg_code)==11 and mich_candidate.parent_id==mip.orig_id
        #         if self.potential_michels[-1].truth:
        #             for p in particles:
        #                 assert type(p)==TruthParticle
        #                 if process_map[p.creation_process] in ['4::121','4::151'] and p.parent_id==mip.orig_id and p.reco_length>min_len and direction_acos(mip.reco_end_dir,p.start_dir)>np.pi/12:
        #                     self.potential_michels[-1].truth=False
        #                     break
        # print("help me",mip.end_point,mip.start_point,mip.reco_end_dir, mip.reco_start_dir,mip.start_dir,mip.end_dir,mip.end_momentum,mip.ke,mip.reco_ke,[mip.csda_ke,mip.mcs_ke,mip.calo_ke])
        if analysis_type=='2x2':
            if not is_contained(mip.points,mode='tpc',margin=2):
                for hip in self.potential_kaons:
                    if not is_contained(mip.points,mode='tpc',margin=2):
                        pseudovert=get_pseudovertex(start_points=np.array([hip.hip.end_point, mip.start_point], dtype=float), directions=[hip.hip.reco_end_dir, mip.start_dir])
                        hip.dist_from_hip=float(np.linalg.norm(pseudovert-mip.start_point))
                        hip.proj_dist_from_hip=collision_distance(hip.hip, mip,["end","start"])[-1]
                        # hip.dist_Hausdorff_from_hip=float(np.min(cdist(hip.hip.points,mip.points)))
            # else: print("good start")
            if not is_contained(mip.points,mode='tpc',margin=2):
                for mich in self.potential_michels:
                    if not is_contained(mich.mich.points,mode='tpc',margin=2):
                        pseudovert=get_pseudovertex(start_points=np.array([mip.end_point,mich.mich.start_point], dtype=float), directions=[mip.reco_end_dir, mich.mich.start_dir])
                        mich.dist_to_mich=float(np.linalg.norm(pseudovert-mip.end_point))
                        mich.proj_dist_to_mich=collision_distance(mip, mich.mich,["end","start"])[-1]

        # else: print("good end")
    def pass_cuts(self,cuts:dict)->bool:
        passing_len=False
        klens=[]
        michlens=[]
        passed=True
        # print("TRYING A KAON",self.truth)

        for k in self.potential_kaons:
            if k.proj_dist_from_hip<cuts["par_child_dist max"][0] and len([(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in k.k_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and (p.child_hm_pred in [SHOWR_HM,MICHL_HM] or p.child.reco_length>min_len)])==0 and (Model=='2x2' or min(np.linalg.norm(k.hip.end_point-self.mip.start_point),np.linalg.norm(k.hip.start_point-self.mip.start_point))<cuts["par_child_dist max"][0]) and np.linalg.norm(k.hip.end_point-self.mip.start_point)<np.linalg.norm(k.hip.start_point-self.mip.start_point):
                klens+=[k.dist_from_hip]
        for mich in self.potential_michels:
            if mich.proj_dist_to_mich<cuts["par_child_dist max"][0]:
                michlens+=[mich.dist_to_mich]
        
        daughtercut=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in self.mu_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and p.child_hm_pred in [HIP_HM,MIP_HM] and ((direction_acos(p.child.start_dir,self.mip.reco_end_dir)>np.pi/12 and p.child.reco_length>min_len) or (not is_contained(p.child.end_point,mode=full_containment)))]
        if len(daughtercut)!=0:
            if self.truth: 
                print("muon daughter",self.truth, daughtercut)
            if passed:
                self.pass_failure="muon daughter"
                
            passed=False

        if cuts["require_kaon"] or self.require_kaon or not is_contained(self.mip.points,mode=full_containment):
            if len(klens)==0: 
                if self.truth:
                    print("missing required kaon")
                if passed:
                    self.pass_failure="missing required kaon"
                passed=False
        else:
            klens+=[0]

        if cuts["require_michel"] or self.require_michel:
            if len(self.potential_michels)==0: 
                if self.truth:
                    print("missing required michel")
                if passed:
                    self.pass_failure="missing required michel"
                passed=False
        else:
            michlens+=[0]

        for k in klens:
            for mich in michlens:
                estlen=self.mip_len_base+k+mich
                if cuts["mu_len"][0]<estlen and estlen<cuts["mu_len"][1]:
                    passing_len=True
                    break
            if passing_len: break
        if not passing_len:
            if self.truth:
                print("muon passing len",klens,michlens,self.mip_len_base)
            if passed:
                self.pass_failure="muon passing len"
            passed=False
        return passed #TODO more cuts


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
    coll_dist: float
        shape(3) [t1,t2, dist]: the distance from the start point to the
        point along the vector start direction which is the point of
        closest approach to the other particle's corresponding line for the hip (t1) and mip (t2)
        along with the distance of closest approach of these lines (dist)
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
    coll_dist: list[float]
    lam_dir_acos: float
    prot_hm_acc: float
    pi_hm_acc: float
    prot_extra_children: list[ExtraChild]
    pi_extra_children: list[ExtraChild]
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
        assign_truth:bool
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
        assign_truth:bool
            assign truth labels to true lambdas or not
        """

        assert hm_pred[hip.id]==HIP_HM
        assert hm_pred[mip.id]==MIP_HM
        self.hip_id = hip.id
        self.mip_id = mip.id
        self.hip_len =hip.reco_length
        self.mip_len = mip.reco_length
        self.interaction=interactions[hip.interaction_id]
        self.mip=mip
        self.hip=hip
        self.lam_dir_acos = direction_acos((hip.reco_momentum+mip.reco_momentum)/np.linalg.norm(hip.reco_momentum+mip.reco_momentum))
        self.prot_hm_acc = hm_acc[hip.id]
        self.pi_hm_acc = hm_acc[mip.id]
        
        self.truth=False
        self.reason=""
        self.pass_failure=""
        self.error=""

        self.vae = self.vertex_angle_error()
        self.lam_mass2 = self.lambda_mass_2()
        self.lam_decay_len = self.lambda_decay_len()
        # self.momenta: list[float] = self.momenta_projections()
        self.coll_dist = collision_distance(hip,mip)
        
        if assign_truth:
            assert type(self.mip)==TruthParticle and type(self.hip)==TruthParticle
            # self.true_signal=abs(self.mip.parent_pdg_code)==3122 and abs(self.hip.pdg_code)==2212 and abs(self.mip.pdg_code)==211 and self.mip.parent_id==self.hip.parent_id and process_map[self.hip.creation_process]=='6::201' and process_map[self.mip.creation_process]=='6::201'
            self.truth,self.reason=self.is_truth(particles)
        if self.truth: print("We got a true lambda")
        # self.lam_extra_children = lambda_children(hip,mip,[p for p in particles if hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM]])

        guess_start = get_pseudovertex(
            start_points=np.array([hip.start_point, mip.start_point], dtype=float),
            directions=[hip.start_dir, mip.start_dir],
        )
        self.pi_extra_children=[]
        self.prot_extra_children=[]
        self.lam_extra_children=[]
        self.pot_parent=[]
        for p in particles:
            if p.id not in [mip.id,hip.id] and hm_pred[p.id] in [SHOWR_HM,MIP_HM,HIP_HM,MICHL_HM]:
                self.prot_extra_children +=[ExtraChild(p,hip.end_point,hm_pred,hip)]
                self.pi_extra_children +=[ExtraChild(p,mip.end_point,hm_pred,mip)]
                self.lam_extra_children += [ExtraChild(p, guess_start,hm_pred)]
                if hm_pred[p.id] in [MIP_HM,HIP_HM]: self.pot_parent+=[[p,False]]
                if self.truth:
                    assert type(p)==TruthParticle and type(mip)==TruthParticle and type(hip)==TruthParticle

                    assert len(set([hip.parent_id==p.orig_id,mip.parent_id==p.orig_id]))==1
                    if hip.parent_id==p.orig_id and mip.parent_id==p.orig_id:
                        self.pot_parent[-1][1]=True

                    if p.parent_id==hip.orig_id:
                        # if abs(p.parent_pdg_code)!=2212:
                        #     print("2212 WARNING I HAVE NO IDEA WHAT IS GOING ON",abs(p.parent_pdg_code)-2212,p.id,p.orig_id,p.parent_id,p.parent_id)
                        self.prot_extra_children[-1].truth=True
                    if p.parent_id==mip.orig_id:
                        # if abs(p.parent_pdg_code)!=211:
                        #     print("211 WARNING I HAVE NO IDEA WHAT IS GOING ON",abs(p.parent_pdg_code),p.id,p.orig_id,p.parent_id,p.parent_id)
                        self.pi_extra_children[-1].truth=True
                    if p.parent_id==hip.parent_id:
                        assert p.parent_id==mip.parent_id#TODO do some stuff with parent_track_id for truth comparisons
                        if abs(p.parent_id)==np.iinfo(np.int64).max and set([abs(p.parent_pdg_code),abs(hip.parent_pdg_code),abs(mip.parent_pdg_code)])!= {3122}:
                            self.error=str((p.parent_pdg_code,hip.parent_pdg_code,mip.parent_pdg_code,p.parent_id,hip.parent_id,mip.parent_id,p.pdg_code,hip.pdg_code,mip.pdg_code))
                            continue
                        assert set([abs(p.parent_pdg_code),abs(hip.parent_pdg_code),abs(mip.parent_pdg_code)])=={3122},(p.parent_pdg_code,hip.parent_pdg_code,mip.parent_pdg_code,p.parent_id,hip.parent_id,mip.parent_id,p.pdg_code,hip.pdg_code,mip.pdg_code)
                        self.lam_extra_children[-1].truth=True

    def is_truth(self,particles:list[Particle]):
        assert type(self.mip)==TruthParticle and type(self.hip)==TruthParticle
        if abs(self.mip.parent_pdg_code)!=3122:
            return False,"1"
        if self.lam_decay_len<min_len:
            return False,"2"
        if abs(self.hip.pdg_code)!=2212:
            return False,"3"
        if abs(self.mip.pdg_code)!=211:
            return False,"4"
        if self.mip.interaction_id!=self.hip.interaction_id: 
            return False,"4.5"
        # if abs(self.mip.reco_length)<min_len:
        #     return False,"5"
        # if abs(self.hip.reco_length)<min_len:
        #     return False,"6"

        mip_extension=[]
        hip_extension=[]

        if process_map[self.hip.creation_process]!='6::201' or process_map[self.mip.creation_process]!='6::201':
            return False,"5"
        if self.mip.parent_id!=self.hip.parent_id:
            return False,"6"
        if not is_contained(self.hip.points,mode=full_containment,margin=2) or not is_contained(self.mip.points,mode=full_containment,margin=2):
            return False,"7"
        if not is_contained(self.hip.ancestor_position,mode=full_containment,margin=2):
            return False, "7.5"
        for p in particles:
            assert type(p)==TruthParticle
            if process_map[p.creation_process] in ['4::121','4::151'] and p.parent_id==self.hip.orig_id and ((p.reco_length>min_len and (direction_acos(p.start_dir,self.hip.reco_end_dir)>np.pi/12 or abs(p.pdg_code)==211)) or (not is_contained(p.points,mode=full_containment) and abs(p.pdg_code) not in  [11,22])):
                return False,"8"
            if process_map[p.creation_process] in ['4::121','4::151'] and p.parent_id==self.mip.orig_id and ((p.reco_length>min_len and (direction_acos(p.start_dir,self.mip.reco_end_dir)>np.pi/12 or p.pdg_code==2212)) or (not is_contained(p.points,mode=full_containment) and abs(p.pdg_code) not in  [11,22])):
                return False,"9"
            if process_map[p.creation_process] in ['4::121','4::151'] and p.parent_id==self.mip.parent_id and p.reco_length>min_len and p.parent_id!=np.iinfo(np.int64).max:
                return False,"10"
            
            if p.parent_id==self.hip.orig_id and direction_acos(p.start_dir,self.hip.reco_end_dir)<np.pi/12 and abs(p.pdg_code)==2212:
                hip_extension+=[p.reco_length]

            if p.parent_id==self.mip.orig_id and direction_acos(p.start_dir,self.mip.reco_end_dir)<np.pi/12 and abs(p.pdg_code)==211:
                mip_extension+=[p.reco_length]

        if len(np.array(mip_extension)[np.array(mip_extension)>min_len])>1 or (max(mip_extension,default=0)+self.mip.reco_length)<min_len:
            return False,"11"

        if len(np.array(hip_extension)[np.array(hip_extension)>min_len])>1 or (max(hip_extension,default=0)+self.hip.reco_length)<min_len:
            return False,"12"
        
        # if self.hip.nu_id==-1 or self.mip.nu_id==-1:
        #     return False,"13"
        
        if self.hip.parent_track_id!=self.mip.parent_track_id:
            return False,"13"
        
        if np.linalg.norm(self.hip.ancestor_position-self.hip.parent_position)>min_len:
            return False,"14"

        



            # if process_map[p.creation_process] in ['6::201'] and p.pdg_code==13 and p.parent_id==self.mip.orig_id and p.reco_length>15: #didn't decay at rest #TODO need to deal with pi to mu decay in flight with colinear.
            #     return False

        return True,""
    
    def pass_cuts(self,cuts:dict)->bool:
        passed=True
        # print("TRYING a lambda",self.truth)
        if is_contained(self.hip.points,mode=full_containment,margin=2) and is_contained(self.mip.points,mode=full_containment,margin=2) and np.linalg.norm(self.mip.start_point-self.hip.start_point)>cuts["lam_cont_dist max"]:
            if self.truth: print("lam daughter contained start distance max", np.linalg.norm(self.mip.start_point-self.hip.start_point))
            if passed:
                self.pass_failure="lam daughter contained start distance max"
            passed=False

        if self.coll_dist[-1]>cuts["lam_proj_dist max"]:
            if self.truth: print("lam_proj_dist max", self.coll_dist[-1])
            if passed:
                self.pass_failure="lam_proj_dist max"
            passed=False
        
        if np.linalg.norm(self.mip.start_point-self.hip.start_point)>cuts["lam_dist max"]:
            if self.truth: print("lam_dist max", np.linalg.norm(self.mip.start_point-self.hip.start_point))
            if passed:
                self.pass_failure="lam_dist max"
            passed=False

        if not is_contained(self.hip.points,mode=full_containment,margin=2) or not is_contained(self.mip.points,mode=full_containment,margin=2):
            if self.truth: print("prot/pi containment")
            if passed:
                self.pass_failure="prot/pi containment"
            passed=False


        if not is_contained(self.interaction.vertex,mode=full_containment):
            if self.truth: print("vertex OOB")
            if passed:
                self.pass_failure="vertex OOB"
            passed=False
        
        
        # print([(p.dist_to_parent,p.angle_to_parent) for p in self.prot_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0]])
        daughtercut=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent,p.child.id) for p in self.prot_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and ((p.child_hm_pred==HIP_HM and p.child.reco_length>min_len and direction_acos(p.child.start_dir,self.hip.reco_end_dir)>np.pi/12) or (p.child_hm_pred==MIP_HM and p.child.reco_length>min_len) or (not is_contained(p.child.points,mode=full_containment) and p.child_hm_pred in [HIP_HM,MIP_HM]))]
        if len(daughtercut)!=0:
            if self.truth: print("proton daughter particle", daughtercut)
            if passed:
                self.pass_failure="proton daughter particle"
            passed=False

        
        daughtercut=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent,p.child.id) for p in self.pi_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and ((p.child_hm_pred==MIP_HM and p.child.reco_length>min_len and direction_acos(p.child.start_dir,self.mip.reco_end_dir)>np.pi/12) or (p.child_hm_pred==HIP_HM and p.child.reco_length>min_len) or (not is_contained(p.child.points,mode=full_containment) and p.child_hm_pred in [HIP_HM,MIP_HM]))]#or p.child_hm_pred in [SHOWR_HM,MICHL_HM]
        # daughtercut2=[(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent,p.child.reco_length) for p in self.pi_extra_children if p.dist_to_parent<cuts["par_child_dist max"][0] and ((direction_acos(p.child.start_dir,self.mip.reco_end_dir)>np.pi/12 and p.child_hm_pred==MIP_HM and p.child.reco_length>min_len))]#or p.child_hm_pred in [SHOWR_HM,MICHL_HM]
        if len(daughtercut)!=0:
            # if len(daughtercut2)>0:print("dcut2",daughtercut2)
            if self.truth: print("pion daughter particle", daughtercut)
            if passed:
                self.pass_failure="pion daughter particle"
            passed=False

        if self.lam_decay_len<cuts["lam_decay_len"]:
            if self.truth: print("decay len")
            if passed:
                self.pass_failure="decay len"
            passed=False

        for p in self.pot_parent:
            if np.linalg.norm(p[0].end_point-self.mip.start_point)<cuts["par_child_dist max"][0] and np.linalg.norm(p[0].end_point-self.hip.start_point)<cuts["par_child_dist max"][0]:
                if self.truth: print("parent proximity",p)
                if passed:
                    self.pass_failure="parent proximity"
                passed=False
                break
        for p in self.pot_parent:
            v1=(self.mip.start_point+self.hip.start_point)/2-p[0].end_point
            # if 
            if p[0].is_primary and np.dot(p[0].reco_end_dir,v1)>np.linalg.norm(v1)*np.linalg.norm(p[0].reco_end_dir)*np.cos(cuts["lam_VAE max"]):
                if self.truth: print("primary colinear parent",p)
                if passed:
                    self.pass_failure="primary colinear parent"
                passed=False
                break
                
        for p in self.lam_extra_children:
            if np.linalg.norm(p.child.start_point-self.mip.start_point)<cuts["par_child_dist max"][0] and np.linalg.norm(p.child.start_point-self.hip.start_point)<cuts["par_child_dist max"][0]:
                if self.truth: print("extra child",p)
                if passed:
                    self.pass_failure="extra child"
                passed=False
                break

        if self.vae>cuts["lam_VAE max"]:
            if self.truth: print("VAE max", self.vae)
            if passed:
                self.pass_failure="VAE max"
                passed=False

        if self.lam_decay_len>2500:
            if self.truth: print("decay len max")
            if passed:
                self.pass_failure="decay len max"
            passed=False

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
        guess_start = get_pseudovertex(
            start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
            directions=[self.hip.start_dir, self.mip.start_dir],
        )
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
        guess_start = get_pseudovertex(
            start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
            directions=[self.hip.start_dir, self.mip.start_dir],
        )
        assert np.linalg.norm(guess_start-guess_start)==0
        lam_dir1 = guess_start - inter
        if np.isclose(np.linalg.norm(lam_dir1),0):
            return 0

        lam_dir2 = self.hip.reco_momentum + self.mip.reco_momentum

        if np.linalg.norm(lam_dir2) == 0:
            return np.nan
        ret = np.arccos(
            np.dot(lam_dir1, lam_dir2) / np.linalg.norm(lam_dir1) / np.linalg.norm(lam_dir2)
        )
        assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.start_dir,self.mip.start_dir,ret)
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
    #         directions=[self.hip.start_dir, self.mip.start_dir],
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


def is_contained(pos: np.ndarray, mode: str, margin: float = 3) -> bool:
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
        v1 = particle1.start_dir
        p1 = particle1.start_point
    elif orientation[0]=="end":
        v1 = particle1.reco_end_dir
        p1 = particle1.end_point
    else:
        raise Exception()

    if orientation[1]=="start":
        v2 = particle2.start_dir
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
#     #                              directions=[hip.start_dir,mip.start_dir])
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
