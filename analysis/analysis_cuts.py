""" 
This file contains output classes and cut functions useful for reconstructing kaons and 
lambdas in a liquid argon TPC using the reconstruction package SPINE https://github.com/DeepLearnPhysics/spine
"""
from __future__ import annotations
import copy
from re import A
from typing import Optional
from scipy.spatial.distance import cdist
SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf

import sys
# Set software directory
sys.path.append(SOFTWARE_DIR)

from spine.utils.globals import DELTA_SHP, MUON_PID,PHOT_PID, PION_PID, PROT_PID,KAON_PID,KAON_MASS,PROT_MASS, PION_MASS,MICHL_SHP,TRACK_SHP,SHOWR_SHP,LOWES_SHP
from spine.utils.geo.manager import Geometry
from spine.utils.energy_loss import csda_ke_lar
import numpy as np
from numba import njit


def is_truth(obj:object)->bool:
    if "nu_id" in dir(obj):
        return True
    if "orig_interaction_id" in dir(obj):
        return True
    return False
    


NUMI_BEAM_DIR=np.array([.388814672,-.058321970,.919468161])

NUMI_ROT=np.array([
    [ 0.921035925,  0.022715103,  0.388814672],
    [ 0.0,          0.998297825, -0.058321970],
    [-0.389477631,  0.053716629,  0.919468161]
]).T#Translation from icarus coord to numi coord

assert np.linalg.norm(NUMI_ROT@NUMI_ROT.T-np.eye(3))<.0001

@njit
def norm3d(arr):
    assert len(arr)==3
    return (arr[0]**2 + arr[1]**2 + arr[2]**2) ** 0.5



margin0=[[15.,15.],[15.,15.],[10.,60.]]

PI0_MASS=135.0 #[MeV/c^2]


E_PI0_Kp_Decay=(KAON_MASS**2+PI0_MASS**2-PION_MASS**2)/2/KAON_MASS
P_PI0_Kp_Decay=np.sqrt(E_PI0_Kp_Decay**2-PI0_MASS**2)


# TODO spine.TruthParticle.mass propagated to truth particles in larcv
# TODO spine.TruthParticle.parent_end_momentum
# TODO id vs parentid vs orig id vs the actual parentid that I want

# TODO some sort of particle flow predictor
# TODO decay at rest predictor?

# TODO deal with particles with small scatters wih len cut

# TODO Kaon/ Michel flash timing?
# TODO add k0s info

#TODO add sigma photon as a primary 

# drift_dir_map={}

analysis_type='icarus'
if analysis_type=='2x2':
    full_containment='detector'
else:
    full_containment='module'
    # drift_dir_map={0:np.array([-1,0,0]),
    #                1:np.array([1,0,0]),
    #                2:np.array([-1,0,0]),
    #                3:np.array([1,0,0]),}

min_len=3


HIP_HM = 7
MIP_HM = TRACK_SHP
SHOWR_HM = SHOWR_SHP
MICHL_HM=MICHL_SHP

LAM_MASS = 1115.683   # [MeV/c^2]
SIG0_MASS = 1192.642  # [MeV/c^2]

# from typing import TypeAlias
from typing import TYPE_CHECKING

from spine.data.out.interaction import RecoInteraction,TruthInteraction
from spine.data.out.particle import TruthParticle,RecoParticle
if TYPE_CHECKING:
    from typing import Union
    InteractionType = Union[RecoInteraction, TruthInteraction]
    ParticleType    = Union[TruthParticle, RecoParticle]


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
    for i in ['pi+Inelastic', 'protonInelastic',"pi-Inelastic","kaon0LInelastic","kaon-Inelastic","kaon+Inelastic","lambdaInelastic","dInelastic","anti-lambdaInelastic","sigma-Inelastic","kaon0SInelastic","sigma+Inelastic","anti_neutronInelastic","neutronInelastic","anti_protonInelastic","tInelastic",'He3Inelastic','anti_sigma-Inelastic','alphaInelastic','anti_xi0Inelastic','ionInelastic']:
        process_map[i]="4::121"
    process_map["hBertiniCaptureAtRest"]="4::151"


def reco_vert_hotfix(inter:"InteractionType"):
    if type(inter)==TruthInteraction:
        return inter.reco_vertex
    elif type(inter)==RecoInteraction:
        return inter.vertex
    else:
        raise Exception("type not allowed",type(inter))
    

def truth_interaction_id_hotfix(inter:"InteractionType"):
    if type(inter)==TruthInteraction:
        return inter.id
    elif type(inter)==RecoInteraction:
        if inter.is_matched:
            return inter.match_ids[0]
        return None
    else:
        raise Exception("type not allowed",type(inter))


def is_primary_hotfix(p:"ParticleType")->bool:

    if type(p)==TruthParticle:
        if abs(p.pdg_code) in [3222,3112] and process_map[p.creation_process]=="primary":
            return True
    return p.is_primary

def HM_pred_hotfix(p:"ParticleType",hm_pred:Optional[dict[int,np.ndarray]|list[np.ndarray]]=None,old=False)->int:

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
import math
@njit
def angle_between(a, b):
    anorm = math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    bnorm = math.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])
    # Zero vectors → define angle = 0 (arbitrary but safe)
    if anorm == 0.0 or bnorm == 0.0:
        return 0.0

    # Compute cosθ safely
    cos_theta = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) / (anorm * bnorm)

    # Manual clipping (np.clip is unsafe in njit sometimes)
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0

    return math.acos(cos_theta)



Geo = Geometry(detector=Model)

class PredKaonMuMich:
    __slots__ = ('event_number', 'truth','reason','pass_failure','error',
                'truth_list','hip','hm_pred','fm_interactions','particles',
                'potential_kaons','truth_interaction_vertex','truth_Kp','nu_id',
                'truth_hip','truth_michel','truth_pi0_gamma','real_K_momentum',
                'truth_interaction_overlap','decay_mip_dict','truth_interaction_id','is_flash_matched','match_overlaps','reco_vertex','primary_particle_counts','kaon_path','other_mip_dict')
    """
    Storage class for primary Kaons with muon child and their cut parameters

    Attributes
    ----------
    mip_len_base: float
        len attribute of the particle object
    potential_michels: list["ParticleType"]
        list of michel candidates corresponding to this mip candidate
    truth: bool
        is this a truth muon coming from a kaon
    """

    truth:bool

    # true_signal:bool

    def __init__(
        self,
        # pot_k: PotK,
        ENTRY_NUM:int,
        K_hip:"ParticleType",
        particles: list["ParticleType"],
        interactions: list["InteractionType"],
        # hm_acc:list[float],
        hm_pred:Optional[dict[int,np.ndarray]|list[np.ndarray[int]]],
        
        truth:bool,
        reason:str,
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
        self.pass_failure=[]
        # self.mip=mip
        # self.true_signal=False
        self.error=""
        self.truth_list=truth_list
        
        # self.accepted_mu=None
        # self.hm_pred=hm_pred
        

        self.fm_interactions=[(reco_vert_hotfix(i),i.is_flash_matched,i.id) for i in interactions]
        
        interaction=interactions[K_hip.interaction_id]
        self.particles:list["ParticleType"]=copy.deepcopy([p for p in interaction.particles if (len(p.points)>=3 or p.shape in [MICHL_SHP]) and is_contained(p.start_point,margin=-5) and (p.shape in [MICHL_SHP,SHOWR_SHP,LOWES_SHP,DELTA_SHP] or is_contained(p.end_point,margin=-5))])
        changed=emergency_option(self.particles,particles)
        ids=[p.id for p in self.particles]
        if K_hip.id in ids:
            self.hip=self.particles[ids.index(K_hip.id)]
        else:
            self.hip=copy.deepcopy(K_hip)
        self.potential_kaons:list[tuple["ParticleType",list[tuple["ParticleType",list[str],list[tuple["ParticleType",list[str]]],list[tuple[ParticleType,list[str]]]]]]]=[]

        if self.truth and type(self.hip)==TruthParticle:
            assert abs(self.hip.pdg_code)==321,self.hip.pdg_code

        self.is_flash_matched=interaction.is_flash_matched
        # interaction.


        self.hm_pred={}
        if hm_pred is not None:
            for i in range(len(hm_pred)):
                if particles[i].interaction_id==interaction.id:
                    self.hm_pred[particles[i].id]=hm_pred[particles[i].id]


        self.truth_interaction_vertex=[np.nan,np.nan,np.nan]
        self.nu_id=None
        self.truth_Kp={}

        self.truth_pi0_gamma:dict[int,TruthParticle]={}

        self.truth_michel:dict[int,TruthParticle]={}

        self.truth_interaction_id=truth_interaction_id_hotfix(interaction)

        # self.truth_mips={}

        self.truth_interaction_overlap=[[],[]]

        self.match_overlaps=interaction.match_overlaps

        self.reco_vertex=reco_vert_hotfix(interaction)

        self.primary_particle_counts=interaction.primary_particle_counts

        self.kaon_path:dict[int,TruthParticle]={}

        for n,p in enumerate(truth_particles):
            if p.pdg_code==321 and p.ancestor_pdg_code==321:
                self.kaon_path[n]=p



        # self.primaries=[]

        self.decay_mip_dict:dict[int,TruthParticle]={}

        self.other_mip_dict:dict[int,TruthParticle]={}

        if type(self.hip)==TruthParticle:
            assert type(interactions[0])==TruthInteraction,(type(interactions[0]))
            assert type(interaction)==TruthInteraction
            self.truth_interaction_vertex=interaction.vertex
            self.nu_id=interaction.nu_id
            self.truth_hip=self.hip
                

        else:
            assert type(interactions[0])==RecoInteraction,type(interactions[0])
            self.truth_hip=None
            if self.hip.is_matched:
                self.truth_hip=truth_particles[self.hip.match_ids[0]]
            if interaction.is_matched:
                self.truth_interaction_overlap=[truth_interactions[interaction.match_ids[0]].match_ids,truth_interactions[interaction.match_ids[0]].match_overlaps]
                self.truth_interaction_vertex=truth_interactions[interaction.match_ids[0]].vertex
                self.nu_id=truth_interactions[interaction.match_ids[0]].nu_id

        for n,k in enumerate(truth_particles):
            if k.is_primary and k.pdg_code==321:
                self.truth_Kp[n]=k
            if k.pdg_code==22 and k.parent_pdg_code==111 and k.ancestor_pdg_code==321 and process_map[k.parent_creation_process]=='6::201':
                self.truth_pi0_gamma[n]=k
            # if k.pdg_code in [-13,211] and k.parent_pdg_code==321 and process_map[k.creation_process]=='6::201':
            #     self.truth_mips[n]=k
            if abs(k.pdg_code) in [211,13] and process_map[k.creation_process]=='6::201' and k.parent_pdg_code==321 and k.ancestor_pdg_code==321:
                self.decay_mip_dict[n]=k
            if abs(k.pdg_code) in [211,13] and process_map[k.creation_process]=='6::201' and k.parent_pdg_code==321:
                self.other_mip_dict[n]=k
            if k.pdg_code==-11 and k.parent_pdg_code in [-13,211] and process_map[k.creation_process]=='6::201' and k.ancestor_pdg_code==321:
                self.truth_michel[n]=k
            # if k.is_primary:
                # self.primaries+=[n]

        
        # for n,i in enumerate(truth_particles):
            # i
        assert self.hip in changed or self.hip in interaction.particles,(self.hip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles],type(interaction))


        

        self.real_K_momentum=K_hip.reco_momentum
        if self.truth and type(K_hip)==TruthParticle:
            self.real_K_momentum=momentum_from_children_ke(K_hip,particles,KAON_MASS)
            assert HM_pred_hotfix(K_hip,hm_pred)==HIP_HM,(HM_pred_hotfix(K_hip,hm_pred),HIP_HM,K_hip.id,K_hip.pdg_code)
        


        # else: print("good end")
    @profile
    def pass_cuts(self,cuts:dict)->bool:#dict[str,dict[str,bool|list]|bool]
        self.pass_failure:list[str]=[]

        for p in self.particles:
            if p.id==self.hip.id:
                assert p.is_primary==self.hip.is_primary
        # primary_hip=(self.hip.is_primary)
        

        if type(self.hip)==RecoParticle and (self.is_flash_matched or self.truth):

            self.particles:list["ParticleType"]=[p for p in self.particles if (p.shape!=TRACK_SHP or p.id==self.hip.id or p.reco_length>min_len/2)]

            MIPS_AND_MICHELS(self.particles)
            MIPS_AND_MICHELS_2(self.particles,skip=[self.hip])

            michel_endify(self.particles)

            process_primary_tracks(self.particles)


            build_components_and_classify(self.particles)

            pop_obvious_cosmics(self.particles)

            PROBABLE_LEPTONS=[i for i in self.particles if i.pid in [MUON_PID,PHOT_PID] and i.is_primary and i.reco_ke>600]
            # CORRECT_BACKWARDS(self.particles,skip=[self.hip.id])

            PROBABLE_MIPS:list[ParticleType]=[]

            if "Valid MIP Len" in cuts:
                n=cuts["Valid MIP Len"]
                x1 = n % 100
                x2 = (n // 100) % 100
                x3 = (n // 100**2) % 100
                x4 = (n // 100**3) % 100
                pi_bounds=[x1,x2]
                mu_bounds=[x3,x4]
                potential_mips=[p for p in self.particles if p.pid in [MUON_PID,PION_PID]]
                STRICT_MICHLS:list["ParticleType"]=[p for p in self.particles if p.shape in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]
                for p1 in potential_mips:
                    if (pi_bounds[0]<p1.reco_length<pi_bounds[1] or mu_bounds[0]<p1.reco_length<mu_bounds[1]):
                        for m in STRICT_MICHLS:
                            if norm3d(m.start_point-p1.end_point)<2*min_len or norm3d(m.end_point-p1.end_point)<2*min_len:
                                p1.is_primary=False
                                PROBABLE_MIPS.append(p1)

            if False:
                LOOSE_MICHLS:list["ParticleType"]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred) in [MICHL_SHP,LOWES_SHP,SHOWR_SHP] and p.ke<50 and len(p.points)>3]#
                # PROBABLE_MIPS:list["ParticleType"]=[]
                for p in self.particles:
                    if p.pid in [MUON_PID,PION_PID]:
                        if (np.sum(np.array([norm3d(p.start_point-m.start_point) for m in LOOSE_MICHLS])<10*min_len)>0
                            and is_contained(p.start_point) and is_contained(p.end_point) 
                            # and np.sum(np.array([norm3d(p.end_point-m.start_point) for m in LOOSE_MICHLS])<10*min_len)==0 
                            and len([h for h in self.particles if (norm3d(h.start_point-p.start_point)<min_len or norm3d(h.end_point-p.start_point)<min_len) and h not in [p]])==np.sum(np.array([norm3d(p.start_point-m.start_point) for m in LOOSE_MICHLS])<min_len)
                            and len([h for h in self.particles if (h.shape!=TRACK_SHP or h.reco_length>min_len) and (norm3d(h.start_point-p.end_point)<min_len or norm3d(h.end_point-p.end_point)<min_len) and h not in [p]])==1):
                            flip_particle(p)
                            
                            if p.is_matched and p.match_ids[0] in self.decay_mip_dict and p.reco_length>20:
                                print("FLIPPED THE MIP for a SHOWER")
                                tp=self.decay_mip_dict[p.match_ids[0]]
                                if angle_between(p.start_point-p.end_point,tp.start_point-tp.end_point)<np.pi/2:
                                    print("AND IT'S RIGHT NOW")
                                else:
                                    print("AND IT'S WRONG NOW")
                            # if not p.is_primary:PROBABLE_MIPS+=[p]
                        # elif (not np.any(np.array([norm3d(p.start_point-m.start_point) for m in STRICT_MICHLS])<3*min_len)) and np.any(np.array([norm3d(p.end_point-m.start_point) for m in STRICT_MICHLS])<3*min_len):
                            # if not p.is_primary:PROBABLE_MIPS+=[p]
                PRIMARY_ELECTRONS:list["ParticleType"]=[p for p in self.particles if HM_pred_hotfix(p,self.hm_pred) in [SHOWR_SHP] and (is_primary_hotfix(p)) and p.ke>200]#
                for p in self.particles:
                    if p.pid in [MUON_PID,PION_PID,PROT_PID,KAON_PID]:
                        if np.any(np.array([norm3d(p.end_point-m.start_point) for m in PRIMARY_ELECTRONS])<min_len) and not np.any(np.array([norm3d(p.start_point-m.start_point) for m in PRIMARY_ELECTRONS])<min_len):
                            flip_particle(p)
                            
                            if p.is_matched and p.match_ids[0] in self.decay_mip_dict:
                                print("FLIPPED THE MIP for a PRIMARY SHOWER")
                                tp=self.decay_mip_dict[p.match_ids[0]]
                                if angle_between(p.start_point-p.end_point,tp.start_point-tp.end_point)<np.pi/2:
                                    print("AND IT'S RIGHT NOW")
                                else:
                                    print("AND IT'S WRONG NOW")

            merge_muons_and_pions(self.particles,self.hip,cuts)

            rot_hip_mom=(NUMI_ROT@self.hip.start_dir)
            if (rot_hip_mom[2]<0 and np.abs(rot_hip_mom[2])>np.linalg.norm(rot_hip_mom[:2])):# or (np.any(np.array([norm3d(self.hip.start_point-m.start_point) for m in PROBABLE_MIPS if m.id!=self.hip.id])<min_len) and not np.any(np.array([norm3d(self.hip.end_point-m.start_point) for m in PROBABLE_MIPS if m.id!=self.hip.id])<min_len)):# and self.hip.reco_length>25:#TODO fix this back if there is an obvious MIP next to it
                    flip_particle(self.hip)
            
            rot_hip_mom=(NUMI_ROT@self.hip.start_dir)
            theta=angle_between(self.hip.end_point-self.hip.start_point,-NUMI_BEAM_DIR)
            # if self.truth_hip is not None and self.truth_hip.pdg_code==321:
            #     print(theta,MAX_BACK_LEN(KAON_MASS,theta),self.hip.reco_length,)
            if rot_hip_mom[2]<0:
                
                
                    
                if theta<np.pi/2 and self.hip.reco_length>1.1*MAX_BACK_LEN(KAON_MASS,theta):
                    flip_particle(self.hip)
                    if self.truth:
                        print("NOW",((self.hip.end_point-self.hip.start_point)@self.truth_hip.momentum)>0)
                    # time_to_flip=[[p,self.hip] for p in self.particles if norm3d(p.end_point-self.hip.end_point)<min_len and norm3d(p.end_point-self.hip.end_point)<norm3d(p.start_point-self.hip.end_point) and p.shape==TRACK_SHP]
                    # while len(time_to_flip):
                    #     current=time_to_flip.pop()
                    #     p=current[0]
                    #     sh=current[1]
                    #     if norm3d(p.end_point-sh.end_point)<min_len and norm3d(p.end_point-sh.end_point)<norm3d(p.start_point-sh.end_point):
                    #         flip_particle(p)
                    #         print("flipping extra")
                    #         p.is_primary=False
                    #         time_to_flip+=[[q,p] for q in self.particles if norm3d(q.end_point-p.end_point)<min_len and norm3d(p.end_point-q.end_point)<norm3d(q.start_point-p.end_point) and q.shape==TRACK_SHP]

                    # for p in self.particles:
                    #     if self.hip.id==p.id:
                    #         # flip_particle(p)
                    #         assert (NUMI_ROT@p.start_dir)[2]>=0

            PROBABLE_MIPS_IDS=[p.id for p in PROBABLE_MIPS]

            for p in self.particles:
                if p.id!=self.hip.id and p.id not in PROBABLE_MIPS_IDS and p.is_primary:
                    rot_p_mom=(NUMI_ROT@p.start_dir)
                    if rot_p_mom[2]<0:
                        particles_near_start=0
                        for q in self.particles:
                            if q.shape==TRACK_SHP and q.reco_length>10:
                                rot_q_mom=(NUMI_ROT@q.start_dir)
                                if norm3d(p.end_point-q.start_point)<min(min_len,p.reco_length):
                                    particles_near_start+=(int(rot_q_mom[2]>0)*2)-1
                                if norm3d(p.end_point-q.end_point)<min(min_len,p.reco_length):
                                    particles_near_start+=(int(rot_q_mom[2]<0)*2)-1
                        if particles_near_start>=2:
                            flip_particle(p)
                            p.is_primary=False
                            # print("flipping")


            TRACKS=[i for i in self.particles if i.shape==TRACK_SHP]
        # if True:
            if True:
                done=False
                max_step=1
                while not done:
                    done=True


                    for p in TRACKS:
                        for q in [i for i in TRACKS if i.id!=p.id]:
                            # if p.shape==TRACK_SHP:

                            if norm3d(p.end_point-q.end_point)<min_len/2:
                                p_start_ending=0
                                q_start_ending=0
                                for r in [i for i in TRACKS if i.id!=p.id and i.id!=q.id]:
                                    if norm3d(r.end_point-q.start_point)<min_len/2:
                                        q_start_ending+=1
                                    if norm3d(r.end_point-p.start_point)<min_len/2:
                                        p_start_ending+=1
                                if p_start_ending and not q_start_ending:
                                    if q.id!=self.hip.id:
                                        flip_particle(q)
                                        done=False


                    for p in TRACKS:
                        if not p.is_primary: continue
                        for q in TRACKS:
                            if not q.is_primary:
                                est_end=(p.end_point+q.end_point)/2
                                if norm3d(p.end_point-q.end_point)>min_len/2: continue
                                # if (np.any(np.array([norm3d(o.end_point-est_end) for o in self.particles if o not in [p,q] and o.shape in [TRACK_SHP]])<min_len) or
                                    # np.any(np.array([norm3d(o.start_point-est_end) for o in self.particles if o not in [p,q] and o.shape not in [DELTA_SHP,LOWES_SHP]])<min_len)): continue

                                if q.id!=self.hip.id:
                                    flip_particle(q)
                                    done=False
                    
                    for p in TRACKS:
                        for q in TRACKS:
                            if q.is_primary:
                                est_end=(p.end_point+q.start_point)/2

                                if norm3d(p.end_point-q.start_point)>min_len: continue
                                if (np.any(np.array([norm3d(o.end_point-est_end) for o in self.particles if o.id not in [p.id,q.id] and o.shape in [TRACK_SHP]])<min_len) or
                                    np.any(np.array([norm3d(o.start_point-est_end) for o in self.particles if o.id not in [p.id,q.id] and o.shape not in [DELTA_SHP,LOWES_SHP]])<min_len)): continue
                                


                                if q.id!=self.hip.id:
                                    q.is_primary=False
                                    done=False
                    # if False:
                    for p in [j for j in TRACKS if not j.is_primary and j.reco_length>min_len and j.id!=self.hip.id]:
                        # other_tracks=
                        end_near_end=False
                        start_near_start=False
                        for q in [j for j in TRACKS if j.id!=p.id]:
                            if norm3d(q.end_point-p.end_point)<min_len:
                                end_near_end=True
                            if norm3d(q.start_point-p.start_point)<min_len:
                                start_near_start=True
                            if end_near_end and start_near_start:
                                flip_particle(p)

                    for p in self.particles:
                        p_rot=(NUMI_ROT@p.start_dir)
                        if p.shape==TRACK_SHP and p.reco_length>70 and p_rot[2]<0 and np.abs(p_rot[2])>np.linalg.norm(p_rot[:2]):
                            flip_particle(p)
                            # for p in self.particles:
                            if self.hip.id==p.id:
                                # flip_particle(self.hip)
                                # flip_particle(p)
                                h_rot=(NUMI_ROT@self.hip.start_dir)
                                assert h_rot[2]>=0,(p_rot,h_rot)
                                done=False
                    max_step-=1
                    if not max_step:
                        break

        # if True:
            ps=[(NUMI_ROT@p.start_point)[2] for p in self.particles if p.is_primary]
            if np.any(ps):
                primary_start=min(ps)

                for p in self.particles:
                    if p.is_primary and (NUMI_ROT@p.start_point)[2]>primary_start+30 and p.id!=self.hip.id:
                        assert p.id!=self.hip.id
                        p.is_primary=False
                        # print(p.id,self.hip.id,self.event_number,p.is_primary,p.reco_ke,p.pdg_code,self.hip.is_primary)

            for p in self.particles:
                if p.is_primary and p.shape in [SHOWR_SHP,DELTA_SHP,LOWES_SHP,MICHL_SHP] and p.reco_ke<100 and p.id!=self.hip.id:
                    p.is_primary=False
                    # if p==self.hip: raise Exception()

            primary_tracks=[i for i in self.particles if i.is_primary and i.shape==TRACK_SHP]

        
            for q in primary_tracks:
                q_true_start=(NUMI_ROT@q.start_point)
                for p in primary_tracks:
                
                    if p.id==q.id: continue
                    if q.id==self.hip.id: continue
                    
                    if min_endpoint_dist2(p,q)<(2*min_len)**2: continue
                    if impact_parameter(p.start_point,q.start_point,q.start_dir)<5 and q_true_start[2]>10+(NUMI_ROT@p.start_point)[2]:
                        q.is_primary=False
                        assert q.id!=self.hip.id
                    
                    if impact_parameter(p.end_point,q.start_point,q.start_dir)<5 and q_true_start[2]>10+(NUMI_ROT@p.end_point)[2]:
                        q.is_primary=False
                        assert q.id!=self.hip.id

            particles_near_start=0
            particles_near_end=0

            tracks=[p for p in self.particles if p.shape==TRACK_SHP]

            for p in tracks:
                if p.id==self.hip.id:continue
                # if p.shape==TRACK_SHP:
                if norm3d(p.start_point-self.hip.start_point)<min(min_len,self.hip.reco_length):
                    particles_near_start+=1
                if norm3d(p.end_point-self.hip.start_point)<min(min_len,self.hip.reco_length):
                    particles_near_start+=1

                if norm3d(p.start_point-self.hip.end_point)<min(min_len,self.hip.reco_length):
                    particles_near_end+=1
                if norm3d(p.end_point-self.hip.end_point)<min(min_len,self.hip.reco_length):
                    particles_near_end+=1

            if particles_near_end==0 and particles_near_start==1:
                for p in tracks:
                    if (impact_parameter(self.hip.end_point,p.start_point,p.momentum)<min_len and 
                        norm3d(p.start_point-self.hip.end_point)>10 and 
                        impact_parameter(self.hip.start_point,p.start_point,p.momentum)>3*min_len):
                        flip_particle(self.hip)
                        if self.truth:
                            assert self.truth_hip is not None
                            print("flipped in NC case",self.hip.momentum@self.truth_hip.momentum>0)

            for p in self.particles:
                if not is_contained(p.start_point,margin=3) and p.id!=self.hip.id:
                    p.is_primary=False
            
            

            # MIPS_AND_MICHELS_2(self.particles)
            new_vert=reconstruct_vertex_single(self.particles)
            # print(new_vert,self.truth)
            if new_vert is not None:
                self.reco_vertex=new_vert


            test_vertex=find_intersections(self.particles)
            if len(test_vertex)==1:
                if norm3d(test_vertex[0]-self.reco_vertex)>min_len:
                    
                    if self.truth or (self.truth_hip is not None and self.truth_hip.pdg_code==321):
                        print("NEW VERTEX TEST",norm3d(test_vertex[0]-self.truth_interaction_vertex),norm3d(self.reco_vertex-self.truth_interaction_vertex),
                            norm3d(test_vertex[0]-self.truth_interaction_vertex)<norm3d(self.reco_vertex-self.truth_interaction_vertex))
                    self.reco_vertex=test_vertex[0]

            # for p in PROBABLE_LEPTONS:
            #     if norm3d(p.start_point-self.hip.end_point)<norm3d(p.start_point-self.hip.start_point) and norm3d(p.start_point-self.hip.end_point)<min_len:
            #         flip_particle(self.hip)
            #         if self.truth: print("flipped for the lepton", self.hip.momentum@self.truth_hip.momentum>0)
            # if (NUMI_ROT@self.hip.start_dir)[2]<0:
            #     for p in PROBABLE_MIPS:
            #         if (NUMI_ROT@p.start_dir)[2]<0:
            #             if norm3d(self.hip.start_point-p.start_point)<min_len and len([i for i in TRACKS if norm3d(i.start_point-p.start_point)<min_len or norm3d(i.end_point-p.start_point)<min_len])<=2:
            #                 flip_particle(self.hip)
            #                 if self.truth:print("flipped for the MIP", self.hip.momentum@self.truth_hip.momentum>0)


            redirect([i for i in self.particles if i.shape==TRACK_SHP],self.reco_vertex)

        K_plus_cut_cascade(self,cuts,self.hip,self.potential_kaons,self.pass_failure,self.decay_mip_dict,self.kaon_path,stop_early=False)


        return len(self.pass_failure)==1


class Pred_Neut:
    __slots__ = ('event_number','mip','hip','hm_pred','truth','reason','pass_failure','error','mass1','mass2',
                'truth_hip','truth_mip','particles','fm_interactions','truth_interaction_vertex','truth_interaction_overlap',
                'nu_id','real_hip_momentum','real_mip_momentum','real_hip_momentum_reco','real_mip_momentum_reco',
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
    mip:"ParticleType"
        candidate pion
    hip:"ParticleType"
        candidate proton
    """

    vae: float
    mass1: float
    mass2: float
    momenta: list[float]
    # coll_dist: list[float]
    dir_acos: float
    truth:bool
    mip:"ParticleType"
    hip:"ParticleType"
    mass1:float
    mass2:float

    #TODO add in flag that the particle was added in special and check for this after
    

    def __init__(
        self,
        ENTRY_NUM:int,
        hip:"ParticleType",
        mip:"ParticleType",
        particles:list["ParticleType"],
        interactions: list["InteractionType"],
        hm_pred:Optional[dict[int,np.ndarray]|list[np.ndarray]],
        truth:bool,
        reason:str,
        mass1:float,
        mass2:float,
        truth_particles:list["ParticleType"],
        truth_interactions:list["InteractionType"]

    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        hip:"ParticleType"
            hip particle object for convenience, this will be identified with particle/mass1
        mip:"ParticleType"
            mip particle object
        particles: list["ParticleType"]
            list of particle objects for the event
        interactions: list["InteractionType"]
            list of interactions for the event
        hm_pred: list[np.ndarray]
            hip mip semantic segmentation prediction for each particle
        truth:bool
            is this a signal neutral or not
        """

        self.event_number=ENTRY_NUM
        interaction=interactions[hip.interaction_id]
        self.particles=copy.deepcopy([p for p in interaction.particles if (len(p.points)>=3 or p.shape in [MICHL_SHP]) and is_contained(p.start_point,margin=-5) and (p.shape in [MICHL_SHP,SHOWR_SHP,LOWES_SHP,DELTA_SHP] or is_contained(p.end_point,margin=-5))])
        changed=emergency_option(self.particles,particles)
        ids=[p.id for p in self.particles]
        if mip.id in ids:
            self.mip=self.particles[ids.index(mip.id)]
        else:
            self.mip=copy.deepcopy(mip)

        if hip.id in ids:
            self.hip=self.particles[ids.index(hip.id)]
        else:
            self.hip=copy.deepcopy(hip)

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
        self.nu_id=None

        self.fm_interactions=[(reco_vert_hotfix(i),i.is_flash_matched,i.id) for i in interactions]


        self.truth_interaction_overlap=[[],[]]

        self.truth_hip:Optional[TruthParticle]
        self.truth_mip:Optional[TruthParticle]

        if type(self.hip)==TruthParticle:
            assert type(interactions[0])==TruthInteraction,type(interactions[0])
            assert type(interaction)==TruthInteraction
            assert type(self.mip)==TruthParticle
            self.truth_hip=self.hip
            self.truth_mip=self.mip
            self.truth_interaction_vertex=interaction.vertex
            self.nu_id=interaction.nu_id
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
                self.nu_id=truth_interactions[interaction.match_ids[0]].nu_id
                self.truth_interaction_overlap=[truth_interactions[interaction.match_ids[0]].match_ids,truth_interactions[interaction.match_ids[0]].match_overlaps]
                # self.truth_interaction=truth_interactions[interaction.match_ids[0]]
            

        assert self.hip in changed or self.hip in interaction.particles,(self.hip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles])
        assert self.mip in changed or self.mip in interaction.particles,(self.mip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles])

        self.real_hip_momentum=hip.reco_momentum
        self.real_hip_momentum_reco=momentum_from_children_ke_reco(hip,particles,mass1,ignore=[mip])
        self.real_mip_momentum_reco=momentum_from_children_ke_reco(mip,particles,mass2,ignore=[hip])
        if self.truth and type(hip)==TruthParticle:
            assert type(hip)==TruthParticle
            self.real_hip_momentum=momentum_from_children_ke(hip,particles,mass1)


        self.real_mip_momentum=mip.reco_momentum
        if self.truth and type(mip)==TruthParticle:
            assert type(mip)==TruthParticle
            self.real_mip_momentum=momentum_from_children_ke(mip,particles,mass2)
        
        if self.truth and type(mip)==TruthParticle:
            assert type(self.mip)==TruthParticle and type(self.hip)==TruthParticle
            # self.true_signal=abs(self.mip.parent_pdg_code)==3122 and abs(self.hip.pdg_code)==2212 and abs(self.mip.pdg_code)==211 and self.mip.parent_id==self.hip.parent_id and process_map[self.hip.creation_process]=='6::201' and process_map[self.mip.creation_process]=='6::201'
            # truth_parsed,self.reason=self.is_truth(particles)
            # assert truth_parsed==self.truth, (truth_parsed,self.reason)
        # if self.truth: print("We got a true lambda")

        self.truth_interaction_id=truth_interaction_id_hotfix(interaction)

        self.is_flash_matched=interaction.is_flash_matched

        self.reco_vertex=reco_vert_hotfix(interaction)

        self.primary_particle_counts=interaction.primary_particle_counts
        
    # @profile
    def pass_cuts(self,cuts:dict)->bool:
        extra_children=[]
        self.pass_failure=[]

        pot_parent:list[tuple["ParticleType",bool]]=[]


        ms_hs=norm3d(self.mip.start_point-self.hip.start_point)
        me_hs=norm3d(self.mip.end_point-self.hip.start_point)

        ms_he=norm3d(self.mip.start_point-self.hip.end_point)
        me_he=norm3d(self.mip.end_point-self.hip.end_point)

        
        which_min=np.argmin([ms_hs,me_hs,ms_he,me_he])

        if which_min==1:
            flip_particle(self.mip)
        if which_min==2:
            flip_particle(self.hip)
        if which_min==3:
            flip_particle(self.mip)
            flip_particle(self.hip)

        guess_start=(self.hip.start_point+self.mip.start_point)/2

        MIPS_AND_MICHELS(self.particles)
        pop_obvious_cosmics(self.particles)

        
        for i in self.particles:
            if i.id==self.mip.id:
                self.mip=i
            if i.id==self.hip.id:
                self.hip=i

        new_vert=reconstruct_vertex_single(self.particles)
        if new_vert is not None:
            self.reco_vertex=new_vert

        test_vertex=find_intersections(self.particles)
        if len(test_vertex)==1:
            if norm3d(test_vertex[0]-self.reco_vertex)>min_len:
                
                if self.truth:
                    print("NEW VERTEX TEST",norm3d(test_vertex[0]-self.truth_interaction_vertex),norm3d(self.reco_vertex-self.truth_interaction_vertex),
                        norm3d(test_vertex[0]-self.truth_interaction_vertex)<norm3d(self.reco_vertex-self.truth_interaction_vertex))
                self.reco_vertex=test_vertex[0]

        for p in self.particles:
            # if p.interaction_id!=hip.interaction_id: continue
            if p.id not in [self.mip.id,self.hip.id] and HM_pred_hotfix(p,self.hm_pred) in [SHOWR_HM,MIP_HM,HIP_HM,MICHL_HM]:
                extra_children += [p]
                if HM_pred_hotfix(p,self.hm_pred) in [MIP_HM,HIP_HM]: pot_parent+=[(p,False)]

        # assert self.hip in particles,(self.hip.id,[i.id for i in particles])#,[i.id for i in interaction.primary_particles],type(interaction))
        # assert self.mip in particles,(self.mip.id,[i.id for i in particles])#,[i.id for i in interaction.primary_particles],type(interaction))
        michels:list["ParticleType"]=[p for p in self.particles if p.shape in [MICHL_SHP,LOWES_SHP]]


        self.vae=impact_parameter(self.reco_vertex,guess_start,self.real_hip_momentum_reco + self.real_mip_momentum_reco)

        #TODO make a cut of vae with other pid choices 

        primary_shapes=np.bincount([p.shape for p in self.particles if is_primary_hotfix(p)],minlength=10)


        for c in cuts:

            if (not self.truth) and len(self.pass_failure)>=2:
                continue

            checked=False

            if c=="MIP Length":
                checked=True
                if self.mip.reco_length<cuts[c]:
                    self.pass_failure+=[c]
                elif self.mip.reco_length>65:
                    self.pass_failure+=[c]
            if c=="HIP Length":
                checked=True
                if self.hip.reco_length<cuts[c]:
                    self.pass_failure+=[c]

            if c=="MIP Child":
                checked=True
                if self.mip.pid not in [MUON_PID,PION_PID,PROT_PID]:#:# and 
                    self.pass_failure+=[c]
                    continue
                if self.mip.pid in [MUON_PID] and self.mip.reco_length>100:
                    self.pass_failure+=[c]
                    continue

            if c=="HIP Child":
                checked=True
                if self.hip.pid not in [PION_PID,PROT_PID]:#:# and s
                    self.pass_failure+=[c]

            if c=="Primary HIP-MIP":
                checked=True
                if is_primary_hotfix(self.hip) or is_primary_hotfix(self.mip):
                    self.pass_failure+=[c]
                        

            if c=="Valid Interaction":
                checked=True
                if type(self.hip)==TruthParticle:
                    if self.nu_id==-1:
                        self.pass_failure+=[c]
                elif type(self.hip)==RecoParticle:
                    if not self.is_flash_matched:# and best_dist>cuts[c]:
                        if self.truth: print("FAILED AT VALID INTERACTION")#,best_dist)
                        self.pass_failure+=[c]       
                else:
                    raise Exception()
                
            if c=="HIP-MIP Order":
                checked=True
                if self.hip.pid in [MUON_PID,PION_PID] and self.mip.pid in [KAON_PID,PROT_PID]:
                    self.pass_failure+=[c]

            if c=="No Extended Michels":
                checked=True
                pot_mip_extensions=[]
                pot_hip_extensions=[]
                broken=False
                for p in self.particles:
                    if p.id==self.hip.id or p.id==self.mip.id:
                        continue
                    for q in [p.end_point,p.start_point]:
                        if norm3d(q-self.hip.end_point)<min_len:
                            pot_hip_extensions+=[p]
                        if norm3d(q-self.mip.end_point)<min_len:
                            pot_mip_extensions+=[p]

                for l in [pot_mip_extensions,pot_hip_extensions]:
                    if len(l)==1:
                        if l[0].shape!=TRACK_SHP: continue
                        for m in self.particles:
                            if m.shape in [MICHL_SHP,LOWES_SHP]:
                                for p in [m.start_point,m.end_point]:
                                    for q in [l[0].start_point,l[0].end_point]:
                                        if norm3d(p-q)<min_len:
                                            broken=False
                if broken:
                    self.pass_failure+=[c]

            if c=="Proj Max HIP-MIP Sep.":
                checked=True
                if closest_distance(self.mip.start_point, self.mip.end_point-self.mip.start_point, self.hip.start_point, self.hip.end_point-self.hip.start_point)>cuts[c]:
                    self.pass_failure+=[c]

            if c=="Starts Closest":
                checked=True
                if not np.isclose(norm3d(self.hip.start_point-self.mip.start_point)**2,min_endpoint_dist2(self.hip,self.mip)):
                    self.pass_failure+=[c]

            if c=="Max HIP-MIP Sep.":
                checked=True
                if norm3d(self.mip.start_point-self.hip.start_point)>cuts[c]:
                    self.pass_failure+=[c]

            if c=="No HIP or MIP Michel":
                checked=True
                for p in michels:
                    if norm3d(self.mip.end_point-p.start_point)<cuts[c] or norm3d(self.hip.end_point-p.start_point)<cuts[c]:
                        self.pass_failure+=[c]
                        break

            if c=="Closer to Decay Point":
                checked=True
                s=norm3d(self.reco_vertex-guess_start)
                if s>norm3d(self.reco_vertex-self.hip.end_point) or s>norm3d(self.reco_vertex-self.mip.end_point):
                    self.pass_failure+=[c]
                    continue
                if angle_between(self.hip.end_point-self.hip.start_point,guess_start-self.reco_vertex)>np.pi/2 or angle_between(self.mip.end_point-self.mip.start_point,guess_start-self.reco_vertex)>np.pi/2:
                    self.pass_failure+=[c]
                    continue

            if c=="Impact Parameter":
                checked=True
                if self.vae>cuts[c] or np.isclose(self.vae,norm3d(self.reco_vertex-guess_start)):
                    self.pass_failure+=[c]
                
            if c=="Not Back-to-Back":
                checked=True
                if angle_between(self.hip.momentum,self.mip.momentum)>cuts[c]:
                    self.pass_failure+=[c]


            if c=="Min Decay Len":
                checked=True
                for p in [self.hip,self.mip]:
                    if point_to_segment_distance(self.reco_vertex,p)<cuts[c]:
                        self.pass_failure+=[c]
                        break

            if c=="PID check":
                checked=True
                MIP_prot_KE_rat=abs(come_to_rest(self.mip,PROT_MASS))
                MIP_pi_KE_rat=abs(come_to_rest(self.mip,PION_MASS))
                if MIP_prot_KE_rat<min(.5,MIP_pi_KE_rat):
                    self.pass_failure+=[c]
                    

            #TODO check if one of the particles has csda per whatever match the wrong one of proton or pion


            if c=="Parent Proximity":
                checked=True
                # start_to_int=norm3d(guess_start-self.reco_vertex)
                for p in pot_parent:
                    # est_decay=(self.mip.start_point+self.hip.start_point)/2
                    # if (norm3d(p[0].end_point-self.reco_vertex)>=start_to_int and 
                    #     norm3d(p[0].start_point-self.reco_vertex)>=start_to_int):continue
                    if (norm3d(p[0].end_point-guess_start)<=min_len or
                        norm3d(p[0].end_point-self.hip.start_point)<=min_len/2 or
                        norm3d(p[0].end_point-self.mip.start_point)<=min_len/2) and (p[0].reco_length>2*min_len or norm3d(p[0].start_point-self.reco_vertex)<min_len):
                        # norm3d(p[0].end_point-self.reco_vertex)<= norm3d(self.hip.start_point-self.reco_vertex)+min_len and
                        # norm3d(p[0].end_point-self.reco_vertex)<= norm3d(self.mip.start_point-self.reco_vertex))+min_len:
                        # if self.truth: print("parent proximity")
                        # if passed:
                        self.pass_failure+=[c]
                        
                        break
                    if (norm3d(p[0].start_point-guess_start)<=min_len or
                        norm3d(p[0].start_point-self.hip.start_point)<=min_len/2 or
                        norm3d(p[0].start_point-self.mip.start_point)<=min_len/2) and p[0].reco_length>2*min_len and norm3d(p[0].end_point-self.reco_vertex)<min_len:
                        # norm3d(p[0].end_point-self.reco_vertex)<= norm3d(self.hip.start_point-self.reco_vertex)+min_len and
                        # norm3d(p[0].end_point-self.reco_vertex)<= norm3d(self.mip.start_point-self.reco_vertex))+min_len:
                        # if self.truth: print("parent proximity")
                        # if passed:
                        self.pass_failure+=[c]
                        
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
                    if child_to_start>=min(min_len,norm3d(p.start_point-self.reco_vertex)):
                        continue
                    
                    if (child_to_start>=hip_end_gs or
                        child_to_start>=mip_end_gs):continue
                    local_child_count+=1
                    
                    # if (child_to_start<=min(min_len,norm3d(p.child.start_point-self.reco_vertex))):
                        # if self.truth: print("extra child")
                        # if passed:
                    self.pass_failure+=[c]
                    
                    break
                # if local_child_count>=2 and c not in self.pass_failure:
                #     raise Exception()
                #     self.pass_failure+=[c]
                #     


            if c==rf"Even # Primary $\gamma$":
                checked=True
                if (self.primary_particle_counts[PHOT_PID]%2==1 and 
                    primary_shapes[TRACK_SHP]<=3):
                    self.pass_failure+=[c]
            
            elif c=="No HIP Deltas":
            
                checked=True
                if self.hip.shape!=TRACK_SHP:
                    self.pass_failure+=[c]

                DELTAS=[p for p in self.particles if p.shape==DELTA_SHP]
                for p in DELTAS:
                    # if p.shape==DELTA_SHP:
                    # if np.min(cdist(BASE_HIP.points, [p.start_point]))<min_len/2:
                    if point_to_segment_distance(p.start_point, self.hip)<min_len/2:
                        self.pass_failure+=[c]
                        break
            elif c=="Min Proj Decay Len":
                checked=True
                new_start=closest_point_between_lines(self.mip.start_point,
                                                      self.mip.end_point-self.mip.start_point,
                                                      self.hip.start_point,
                                                      self.hip.end_point-self.hip.start_point)[2]
                if norm3d(new_start-self.reco_vertex)<cuts[c]:
                    self.pass_failure+=[c]

            elif c=="Particle Though-Going":
                checked=True
                for p in self.particles:
                    if p==self.hip or p==self.mip:
                        continue
                    if p.shape!=TRACK_SHP:
                        continue
                    if point_to_segment_distance(guess_start,p)<min_len/2 and norm3d(p.start_point-guess_start)>min_len and norm3d(p.end_point-guess_start)>min_len:
                        self.pass_failure+=[c]
                        break

            if c=="":
                checked=True
                self.pass_failure+=[c]
            
            if not checked:
                raise Exception(c,"not found in lam cuts")
        
        assert len(self.pass_failure)==len(set(self.pass_failure))
        
        return len(self.pass_failure)==1
    
    def mass_2(self) -> float:
        """
        Returns mass value constructed from the
        hip and mip candidate deposited energy and predicted direction

        Returns
        -------
        float
            reconstructed mass squared
        """
        assert self.mip.ke > 0
        assert self.hip.ke > 0
        mass2 = (self.mass1**2+
            self.mass2**2
            + 2 * (self.mip.reco_ke + self.mass2) * (self.hip.reco_ke + self.mass1)
            - 2 * np.dot(self.hip.reco_momentum, self.mip.reco_momentum)
        )
        return mass2
    
    
    # def momenta_projections(self) -> list[float]:
    #     """
    #     Returns the P_T and P_L of each particle relative to the measured from the decay

    #     Parameters
    #     ----------
    #     hip: spine.Particle
    #         spine particle object
    #     mip: spine.Particle
    #         spine particle object
    #     interactions: list["InteractionType"]
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


def is_contained(pos: np.ndarray, mode: str =full_containment, margin:float|list[list[float]]|list[list[int]]|np.ndarray= 3,define_con=True) -> bool:
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


def HIPMIP_pred(particle:"ParticleType", sparse3d_pcluster_semantics_HM: np.ndarray,perm=None,mode=True) -> np.ndarray:
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


def come_to_rest(p:"ParticleType",mass=KAON_MASS)->float:
    p_csda=csda_ke_lar(p.reco_length, mass)
    assert type(p_csda)==float
    # if not p_csda>0: return False
    # if (not p.calo_ke>0) and (not p.mcs_ke>0): return False

    if p.calo_ke<0: raise Exception(p.calo_ke)
    # check=p.calo_ke if p.calo_ke>0 else p.mcs_ke
    return p.calo_ke/p_csda-1


def all_children_reco(p:"ParticleType",particles:list["ParticleType"],dist=min_len/2,ignore=[])->list["ParticleType"]:

    done=False
    children:list["ParticleType"]=[p]
    while done==False:
        done=True
        for pd in particles:
            for d in children:
                if norm3d(pd.start_point-d.end_point)<dist and pd not in children and pd not in ignore and not is_primary_hotfix(pd):
                    children+=[pd]
                    done=False
    return children

def momentum_from_children_ke_reco(p:"ParticleType",particles:list["ParticleType"],mass,ignore=[])->float:
    # print("running mfdke")
    ad=all_children_reco(p,particles,ignore=ignore)
    assert p in ad
    tke=sum([d.calo_ke for d in ad])#if d.shape not in [MICHL_SHP]
    
    if type(p)==TruthParticle: assert tke>0,(tke,mass,len(p.points),p.num_voxels,p.shape,p.energy_deposit,p.depositions)
    assert len(p.reco_start_dir)==3
    return np.sqrt(tke**2+2*mass*tke)*p.reco_start_dir


def all_children(p:"TruthParticle",particles:list["TruthParticle"])->list["TruthParticle"]:
    out:list["TruthParticle"]=[]
    if len(p.children_id)==0:
        return [p]
    to_explore:list["TruthParticle"]=[p]
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

def momentum_from_children_ke(p:"TruthParticle",particles,mass)->float:
    # print("running mfdke")
    ad:list["TruthParticle"]=all_children(p,particles)
    assert p in ad

    tke=sum([a.calo_ke for a in ad])
    return np.sqrt((tke+mass)**2-mass**2)*p.reco_start_dir

def mom_to_mass(p1,p2,m1,m2):

    E1=np.sqrt(norm3d(p1)**2+m1**2)
    E2=np.sqrt(norm3d(p2)**2+m2**2)

    return np.sqrt(m1**2 + m2**2 + 2*E1*E2 - 2*np.dot(p1, p2))

@njit
def impact_parameter(vert,pos,mom):
    # dir2 = self.real_hip_momentum_reco + self.real_mip_momentum_reco

    # mom=mom/np.

    ret = angle_between(pos-vert,mom)
    assert ret==ret,(pos,vert,mom)
    # if passed: assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,dir1,dir2)
    
    return norm3d(pos-vert)*np.sin(min(ret,np.pi/2))

# def get_tpc_id(point):
#     return Geo.get_closest_tpc([point])[0]
#     # return


# def cos_gamma_to_pip_bounds(E):
#     out=np.clip([cos_gamma_to_pip(E*1.5),cos_gamma_to_pip(E/2)],0,1)
#     # print(out[0],out[1])
#     return out

@njit
def cos_gamma_to_E(vert,pos,mom):
    cost=np.cos(angle_between(pos-vert,mom))
    return PI0_MASS**2/2/(E_PI0_Kp_Decay+cost*P_PI0_Kp_Decay)

# @njit
def closest_distance(pos1, mom1, pos2, mom2):
    # a, u, b, v = map(np.array, (a, u, b, v))
    cross = np.cross(mom1, mom2)
    denom = norm3d(cross)
    if denom < 1e-12:  # parallel case
        return norm3d(np.cross(pos2 - pos1, mom1)) / norm3d(mom1)
    return abs(np.dot(pos2 - pos1, cross)) / denom


def MCS_direction_prediction(p):
    from spine.utils.tracking import get_track_segments
    _, dirs, _ = get_track_segments(
            p.points, min_len, p.start_point)

    # Find the angles between successive segments
    costh = np.sum(dirs[:-1] * dirs[1:], axis=1)
    costh = np.clip(costh, -1, 1)
    theta = np.arccos(costh)
    if len(theta) < 5: return 1
    slope, intercept = np.polyfit(np.arange(len(theta)), theta, 1)
    return (slope)>0



def reconstruct_vertex_single(particles):
        
        """Post-processor which reconstructs one vertex for each interaction
        in the provided list.

        Parameters
        ----------
        inter : List[RecoInteraction, TruthInteraction]
            Reconstructed/truth interaction object
        """
        # Selected the set of particles to use as a basis for vertex prediction

        from spine.utils.vertex import get_vertex
        
        parts = [part for part in particles if (
                part.is_primary and
                # part.shape in self.include_shapes and
                part.size > 0) and part.shape not in [SHOWR_SHP]]
        if not len(parts):
            parts=[part for part in particles if (
                # part.shape in self.include_shapes and
                part.size > 0) and part.shape not in [SHOWR_SHP]]
        # print(len(particles))

        # if not self.use_primaries or not len(particles):
        #     particles = [part for part in particles if (
        #             part.shape in self.include_shapes and
        #             part.size > 0)]
        # if not len(particles):
        # particles = [part for part in particles if part.size > 0]

        if len(parts) > 0:
            # Collapse particle objects to start, end points and directions
            start_points = np.vstack([part.start_point for part in parts])
            end_points   = np.vstack([part.end_point for part in parts])
            directions   = np.vstack([part.start_dir for part in parts])
            shapes       = np.array([part.shape for part in parts])

            # Reconstruct the vertex for this interaction

            try:
                vtx, _ = get_vertex(
                    start_points, end_points, directions, shapes,return_mode=True)
            except np.linalg.LinAlgError:
                return None
            
            return vtx

def Bragg_Peak(p:ParticleType,len_bragg=10):#TODO this needs to be stored 
    assert p.reco_length>=len_bragg,(p.reco_length,len_bragg)
    dists = np.linalg.norm(p.points - p.end_point, axis=1)
    edeps=[]
    raw_edeps=[]
    check=[]
    # chi2=0
    for i in range(len_bragg):
        mask = (i<dists)&(dists<i+1)
        # dist_i=dists[mask]
        deps_i=p.depositions[mask]
        edeps+=[np.sum(deps_i)*np.power(i+.5,.42)]
        raw_edeps+=[np.sum(deps_i)]
        check+=[np.power(i+.5,-.42)]
    
    chi2=np.sum((np.array(raw_edeps)-np.median(edeps)*np.array(check))**2)

    return np.median(edeps),chi2
# 
# def correct_orientation(p,diff=3,len_bragg=10):
#     if p.reco_length<len_bragg: return 0
#     p_other=copy.deepcopy(p)
#     p_other.start_point,p_other.end_point=p_other.end_point,p_other.start_point
    

#     corr,_=Bragg_Peak(p,len_bragg)
#     wrong,_=Bragg_Peak(p_other,len_bragg)
#     if corr-wrong>diff:
#         return 1
#     if wrong-corr>diff:
#         return -1
#     return 0


def MIPS_AND_MICHELS(particles:list["ParticleType"]):
    STRICT_MICHLS:list["ParticleType"]=[p for p in particles if p.shape in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
    PROBABLE_MIPS:list["ParticleType"]=[]
    for p in particles:
        if p.pid in [MUON_PID,PION_PID,PROT_PID]:
            if np.any(np.array([np.min(cdist([p.start_point],m.points)) for m in STRICT_MICHLS])<3*min_len) and not np.any(np.array([np.min(cdist([p.end_point],m.points)) for m in STRICT_MICHLS])<3*min_len):
                flip_particle(p)

        if p.shape==TRACK_SHP:
            if np.any(np.array([np.min(cdist([p.end_point],m.points)) for m in STRICT_MICHLS])<min_len) and not np.any(np.array([np.min(cdist([p.start_point],m.points)) for m in STRICT_MICHLS])<3*min_len):
                if p.pid in [PROT_PID]:
                    p.pid=PION_PID
                    p.pdg_code=211 
                PROBABLE_MIPS.append(p)
    return PROBABLE_MIPS

def MIPS_AND_MICHELS_2(particles,skip:list["ParticleType"]=[]):
    STRICT_MICHLS:list["ParticleType"]=[p for p in particles if p.shape in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
    STRICT_MIPS:list["ParticleType"]=[p for p in particles if p.pid in [PION_PID,MUON_PID] ]#
    
    # STRICT_TRACKS:list["ParticleType"]=[p for p in particles if p.shape in [TRACK_SHP]]#

    for p in STRICT_MIPS:
        if p.id not in [i.id for i in skip]:
            if p.is_primary and np.any(np.array([np.min(cdist([p.end_point],m.points)) for m in STRICT_MICHLS])<3*min_len) and (NUMI_ROT@p.start_dir)[2]<0 and -(NUMI_ROT@p.start_dir)[2]>.5*norm3d(p.start_dir) and p.reco_length>40:
                p.is_primary=False
                # for q in STRICT_TRACKS:
                #     if q.id==p.id:
                #         continue
                #     if norm3d(q.start_point-p.start_point)<min_len and norm3d(q.end_point-p.start_point)>min_len:
                #         flip_particle(q)

    LOOSE_MIPS:list["ParticleType"]=[p for p in particles if p.shape in [TRACK_SHP] ]#
    for p in LOOSE_MIPS:
        if is_contained(p.end_point,margin=3): continue
        michels_near_start=0
        other_near_start=0
        for q in particles:
            if q.id==p.id: continue
            if norm3d(q.start_point-p.start_point)<3*min_len or norm3d(q.end_point-p.start_point)<3*min_len:
                if q.shape in [MICHL_SHP,LOWES_SHP,SHOWR_SHP] and (not is_primary_hotfix(q)):
                    michels_near_start+=1
                elif q.shape in [TRACK_SHP]:
                    other_near_start+=1
        if michels_near_start>=1 and not other_near_start:
            flip_particle(p)
            p.is_primary=False
            # print("flipping")
    
        
        


# def CORRECT_BACKWARDS(particles,skip:list["ParticleType"]=[]):
#     STRICT_MICHLS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p) in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
#     OBVIOUS_MUONS:list["ParticleType"]=[p for p in particles if p.pid in [MUON_PID,PION_PID] and np.any(np.array([np.min(cdist([p.end_point],m.points)) for m in STRICT_MICHLS])<3*min_len)]

#     all_starts_and_ends=[p.start_point for p in particles]+[p.end_point for p in particles]
#     for m in OBVIOUS_MUONS:
#         if m.start_dir[2]>0: continue
#         if m in skip:continue
#         if np.sum(cdist([m.start_point],all_starts_and_ends)<2*min_len)==2:
#             for p in particles:
#                 if p in skip: continue
#                 if m==p: continue
#                 if p.shape!=TRACK_SHP:continue
#                 if norm3d(p.start_point-m.start_point)<min_len and p.start_dir[2]<0:
#                     flip_particle(p)
#                     m.is_primary=False


def flip_particle(p:"ParticleType"):
    p.start_point,p.end_point=p.end_point,p.start_point
    p.start_dir=p.start_dir*(-1)
    p.end_dir=p.end_dir*(-1)
def merge_particles(p1,p2,particles):
    indices=[p.id for p in particles]
    p1.points=np.append(p1.points,p2.points,axis=0)
    p1.depositions=np.append(p1.depositions,p2.depositions)
    p1.end_point=p2.end_point
    p1.length=p1.length+p2.length
    p1.calo_ke=p1.calo_ke+p2.calo_ke
    p1.end_dir=p2.end_dir
    particles.pop(indices.index(p2.id))
    # particles.remove(p2)
    # print("merging",p1.id,p2.id)

# @profile
def K_plus_cut_cascade(obj,
                       cuts,
                       BASE_HIP:"ParticleType",
                       potential_kaons:Optional[list[tuple["ParticleType",list[tuple["ParticleType",list[str],list[tuple["ParticleType",list[str]]],list[tuple["ParticleType",list[str]]]]]]]]=None,
                       pass_failure:Optional[list[str]]=None,
                       decay_mip_dict:Optional[dict[int,TruthParticle]]=None,
                       kaon_path:Optional[dict[int,TruthParticle]]=None,
                       stop_early=True) -> bool:

    if potential_kaons is None:
        potential_kaons=[]
    if pass_failure is None:
        pass_failure = []
    if decay_mip_dict is None:
        decay_mip_dict = {}
    if kaon_path is None:
        kaon_path = {}

    RECO_VERTEX=obj.reco_vertex
    HM_PRED=None#self.hm_pred
    particles:list[ParticleType]=obj.particles

    if stop_early:
        if "Valid Interaction" in cuts:
            if type(BASE_HIP)==TruthParticle:
                if obj.nu_id==-1:
                    return False
            else:
                if not obj.is_flash_matched:# and best_dist>cuts[c]:
                    return False
                
        if "Primary $K^+$" in cuts:
            if not BASE_HIP.is_primary:
                return False
        if "Initial HIP" in cuts:
            if HM_pred_hotfix(BASE_HIP,HM_PRED)!=HIP_HM and BASE_HIP.pid not in  [3,4]:
                return False
            if BASE_HIP.reco_length<=0: 
                return False


    def update_mip(cut:str,k,n:int,p=None):
        if False and p!=None:
            assert p==k[1][n]
        
        assert cut not in k[1][n][1],(k[1][n][1],cut)
        k[1][n][1].append(cut)

    def update_mips(cut: str):
        assert potential_kaons is not None
        for k in potential_kaons:
            for n,p in enumerate(k[1]):   # SAFE: iterate over a copy
                update_mip(cut, k, n, p)
    # print(dir(obj))
    

    # if BASE_HIP.reco_length>4 and is_contained(BASE_HIP.start_point,margin=-5) and is_contained(BASE_HIP.end_point,margin=-5) and len(BASE_HIP.points)>=3: assert sum([p==BASE_HIP for p in particles])==1,(sum([p==BASE_HIP for p in particles]),sum([p.id==BASE_HIP.id for p in particles]))

    NON_PRIMARY_TRACKS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED) in [MIP_HM,HIP_HM] and p.reco_length>0]
    NON_PRIMARY_MIPS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED)==MIP_HM and p.reco_length>0]
    NON_PRIMARY_MICHLS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED) in [MICHL_SHP,LOWES_SHP,SHOWR_SHP,DELTA_SHP]]
    # NON_PRIMARY_SHWRS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED) in [SHOWR_SHP,LOWES_SHP]]
    done=False

    
    POT_GAMMA=[p for p in NON_PRIMARY_MICHLS]

    
    potential_kaons.append((BASE_HIP,[(i,
                            [],
                            [(j,[]) for j in NON_PRIMARY_MICHLS],
                            [(j,[]) for j in POT_GAMMA]) for i in NON_PRIMARY_MIPS 
                            if norm3d(i.start_point-BASE_HIP.end_point)<20 or 
                            (BASE_HIP.is_matched and i.is_matched and BASE_HIP.match_ids[0] in kaon_path and i.match_ids[0] in decay_mip_dict and 
                                (decay_mip_dict[i.match_ids[0]].parent_id==kaon_path[BASE_HIP.match_ids[0]].id or norm3d(decay_mip_dict[i.match_ids[0]].start_point-kaon_path[BASE_HIP.match_ids[0]].end_point)<min_len/2))
                            ]))
    
    # assert potential_kaons is not None
    while not done: #this loop goes over all of the hips connected to the end of the kaon, and constructs a hadronic group which hopefully contains the kaon end. 
        done=True
        # print("looking")
        
        for p in NON_PRIMARY_TRACKS:
            if p.id not in [r[0].id for r in potential_kaons]:
                # print("getting here")
                for k in potential_kaons:
                    # print(k[0])
                    n1=norm3d(p.start_point-k[0].end_point)
                    n2=norm3d(p.start_point-k[0].start_point)
                    if n1<min_len and n1<n2:
                        potential_kaons.append((p,[(i,
                                                    [],
                                                    [(j,[]) for j in NON_PRIMARY_MICHLS],
                                                    [(j,[]) for j in POT_GAMMA]) for i in NON_PRIMARY_MIPS 
                                                    if norm3d(i.start_point-p.end_point)<20 or 
                                                    (p.is_matched and i.is_matched and p.match_ids[0] in kaon_path and i.match_ids[0] in decay_mip_dict and 
                                                     (decay_mip_dict[i.match_ids[0]].parent_id==kaon_path[p.match_ids[0]].id or norm3d(decay_mip_dict[i.match_ids[0]].start_point-kaon_path[p.match_ids[0]].end_point)<min_len/2))
                                                    ]))
                        done=False
                        break
                    # elif n2<min_len and n2<n1:
                    #     potential_kaons+=[[p,copy.copy(MIP_CHAINS),[]]]
                    #     done=False
                    #     break

    failed=False
    for c in cuts:
        
        if cuts[c] is None: continue

        checked=False

        # if c=="Contained HIP":
        #     checked=True
        #     if (not is_contained(BASE_HIP.end_point,margin=3)) or (not is_contained(BASE_HIP.start_point,margin=3)):
        #         update_mips(c)
        #         failed=True

        if c=="Fiducialization":
            checked=True
            if (not is_contained(RECO_VERTEX,margin=margin0)):
                update_mips(c)
                failed=True

        if c=="Min HIP-MIP Angle":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    if angle_between(k[0].end_point-k[0].start_point,p[0].end_point-p[0].start_point)<cuts[c]:
                        update_mip(c,k,n,p)
                

        # elif c=="dedx chi2":
        #     checked=True
        #     for k in potential_kaons:
        #         if k[0].reco_length>10 and len(k[1]):
        #             bp=Bragg_Peak(k[0])[1]
        #             if bp>cuts[c]:
        #                 for n,p in enumerate(k[1]):
        #                     update_mip(c,k,n,p)


        elif c=="Connected Non-Primary MIP":
            # did_something=0
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])        

                    mip_start=p[0].start_point
                    # mip_end=p[0].end_point

                    n1=norm3d(mip_start-k[0].end_point)
                    if (n1>cuts[c] or n1>=norm3d(mip_start-k[0].start_point)):
                        # did_something=1
                        update_mip(c,k,n,p)
                    elif p[0].is_primary and norm3d(p[0].start_point-BASE_HIP.start_point)<min_len/2:
                        update_mip(c,k,n,p)
                    # elif norm3d(p[0].start_point-RECO_VERTEX)<min_len:
                    #     update_mip(c,k,n,p)

        elif c=="No HIP Deltas":
            
            checked=True
            deltas=False
            if BASE_HIP.shape!=TRACK_SHP:
                update_mips(c)
                continue

            DELTAS=[p for p in particles if p.shape==DELTA_SHP]
            for p in DELTAS:
                if point_to_segment_distance(p.start_point, BASE_HIP)<min_len/2:
                    deltas=True
                    update_mips(c)
                    failed=True
                    break
            if not deltas:
                for k in potential_kaons:
                    for n,p in enumerate(k[1]):
                        for d in DELTAS:
                        # for n,p in enumerate(k[1]):
                            if point_to_segment_distance(d.start_point, k[0])<min_len/2:
                            # if np.min(cdist(k[0].points, [d.start_point]))<min_len/2:
                        # deltas=True
                                update_mip(c,k,n,p)
                                break
        elif c=="No LOW E MIP Deltas":
            checked=True
            deltas=False

            DELTAS=[p for p in particles if p.shape==DELTA_SHP and p.reco_ke>10]
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    for d in DELTAS:
                    # for n,p in enumerate(k[1]):
                        # if np.min(cdist(p[0].points, [d.start_point]))<cuts[c]:
                        if point_to_segment_distance(d.start_point, p[0])<min_len/2:
                        # deltas=True
                            
                            update_mip(c,k,n,p)
                            break

            # if deltas:
                # if HM_pred_hotfix(BASE_HIP,HM_PRED)!=HIP_HM and BASE_HIP.pid not in  [3,4]:
                # update_mips(c)


        # elif c=="Contained MIP":
        #     checked=True
        #     for k in potential_kaons:
        #         for n,p in enumerate(k[1]):
        #                 # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
        #                 # plen=np.sum([i.reco_length for i in p])
        #                 # mip_start=p[0].start_point
        #                 mip_end=p[0].end_point

        #                 if not is_contained(mip_end,margin=1):
        #                     update_mip(c,k,n,p)
            
            # print(np.sum([len(k[1]) for k in potential_kaons]))
            # if np.sum([len(k[1]) for k in potential_kaons])==0:
            #     self.pass_failure+=[c]
                


        # elif c=="Nothing Before the Start":
        #     checked=True
        #     for p in particles:
        #         if p.shape==TRACK_SHP and p.is_primary and p.reco_length>5*min_len and p.pid!=MUON_PID:
        #             if norm3d(p.end_point-BASE_HIP.start_point)<min_len/2 and norm3d(p.start_point-BASE_HIP.start_point)>3*min_len:
        #                 update_mips(c)
        #                 failed=True

        elif c=="Initial HIP":
            checked=True
            if HM_pred_hotfix(BASE_HIP,HM_PRED)!=HIP_HM and BASE_HIP.pid not in  [3,4]:
                update_mips(c)
                failed=True
            if BASE_HIP.reco_length<=0: 
                update_mips(c)
                failed=True
                
        elif c==rf"Primary $K^+$":
            checked=True
            if not is_primary_hotfix(BASE_HIP):
                update_mips(c)
                failed=True     

        elif c=="Close to Vertex":
            checked=True
            if norm3d(BASE_HIP.start_point-RECO_VERTEX)>cuts[c] or norm3d(BASE_HIP.start_point-RECO_VERTEX)>norm3d(RECO_VERTEX-BASE_HIP.end_point):
                update_mips(c)
                failed=True

        elif c=="Kaon Len":
            checked=True
            if BASE_HIP.reco_length<cuts[c]:
                update_mips(c)
                failed=True

        elif c=="Valid Interaction":
            checked=True
            if type(BASE_HIP)==TruthParticle:
                if obj.nu_id==-1:
                    update_mips(c)
                    failed=True
            else:
                if not obj.is_flash_matched:
                    update_mips(c)
                    failed=True
        
        elif c=="":
            checked=True
            pass_failure+=[c]

        elif c=="Michel Child":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):

                    # mip_start=p[0].start_point
                    mip_end=p[0].end_point

                    # if p[0].reco_length>10:
                        # mich_child=Bragg_Peak(p[0])>5.5
                    for o in reversed(p[2]):
                        other=o[0]
                        if other.reco_ke>65:
                            o[1].append(c)
                            continue

                        check_dist=min(norm3d(mip_end-other.start_point),norm3d(mip_end-other.end_point))
                        if other.shape not in [DELTA_SHP,LOWES_SHP]:
                            if check_dist>cuts[c]:
                                # mich_child=True
                                o[1].append(c)
                                continue
                        else:
                            if check_dist>min_len/2:
                                # mich_child=True
                                o[1].append(c)
                                continue
                    if len([i for i in p[2] if not len(i[1])])==0:
                        update_mip(c,k,n,p)

        elif c==r"$\pi^0$ Quality":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    plen=p[0].reco_length

                    mip_start=p[0].start_point
                    mip_end=p[0].end_point

                    hip_to_mip=norm3d(BASE_HIP.start_point-mip_start)

                    for o in reversed(p[3]):
                        other=o[0]
                        if not (
                            norm3d(other.start_point-mip_end)>min_len
                            and norm3d(other.end_point-mip_end)>min_len
                            and (impact_parameter(mip_start,other.start_point,other.momentum)<impact_parameter(BASE_HIP.start_point,other.start_point,other.momentum) or hip_to_mip<3*min_len)
                            ):
                            o[1].append(c)
                    if plen<40:
                        if not np.any([o[1]==[] for o in p[3]]):
                            update_mip(c,k,n,p)

        elif c==r"$\pi^0$ Rel KE":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    # if other.reco_ke<10: continue# i think the minimal energy of a photon from the pi0 is around 20MeV
                    # if other.reco_ke>300: continue# i think the maximum energy is around 225 MeV
                    plen=p[0].reco_length
                    mip_start=p[0].start_point

                    for o in reversed(p[3]):
                        other=o[0]

                        ratio=(cos_gamma_to_E(mip_start,other.start_point,p[0].momentum)-other.reco_ke)/other.reco_ke
                        if ratio<cuts[c] or other.ke<10 or other.ke>300 or ratio>4: 
                            o[1].append(c)
                    if plen<40:
                        if not np.any([o[1]==[] for o in reversed(p[3])]):
                            update_mip(c,k,n,p)
        elif c==r"$\pi^0$ Impact Parameter":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    plen=p[0].reco_length
                    mip_start=p[0].start_point

                    for o in reversed(p[3]):
                        other=o[0]
                        if not (impact_parameter(mip_start,other.start_point,other.momentum)<cuts[c]):
                            o[1].append(c)
                    # if not has_pi0:
                    if plen<40:
                        if not np.any([o[1]==[] for o in reversed(p[3])]):
                            update_mip(c,k,n,p)
        elif c==r"Max Decay $\gamma$":
            checked=True
            if r"$\pi^0$ Rel KE" in cuts:
                assert list(cuts.keys())[-3]==c
            else:
                assert list(cuts.keys())[-2]==c
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    plen=p[0].reco_length
                    mip_start=p[0].start_point
                    gammas=[other for other in p[3] if other[0].shape==SHOWR_SHP and other[1]==[] and impact_parameter(mip_start,other[0].start_point,other[0].momentum)<impact_parameter(BASE_HIP.start_point,other[0].start_point,other[0].momentum)]
                    if plen>40 and len(gammas)>1:
                        update_mip(c,k,n,p)
                    if plen<40 and len(gammas)>3:
                        update_mip(c,k,n,p)

        elif c=="MIP Child At Most 1 Michel":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    mip_end=p[0].end_point
                    check=np.any([(norm3d(other.start_point-mip_end)<min_len or norm3d(other.end_point-mip_end)<min_len)*(other.reco_length>cuts[c]) for other in NON_PRIMARY_TRACKS if other.id!=p[0].id])
                    if check:
                        
                        update_mip(c,k,n,p)
                        continue
                    
        elif c=="Single MIP Decay":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    mip_start=p[0].start_point
                    if np.sum([(norm3d(other.start_point-mip_start)<min(min_len,norm3d(other.start_point-BASE_HIP.start_point)))*(other.reco_length>cuts[c]) for other in NON_PRIMARY_TRACKS if other.id!=BASE_HIP.id])>1:
                        update_mip(c,k,n,p)
        
        elif c=="Bragg Peak HIP":
            checked=True
            for k in potential_kaons:
                if k[0].reco_length>10 and len(k[1]):
                    bp=Bragg_Peak(k[0])[0]
                    if bp<cuts[c]:
                        for n,p in enumerate(k[1]):
                            update_mip(c,k,n,p)

        elif c=="Bragg Peak MIP":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    if p[0].reco_length>10:
                        bp=Bragg_Peak(p[0])[0]
                        if bp<cuts[c]:
                            update_mip(c,k,n,p)

        elif c=="Come to Rest MIP":
            checked=True
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    ctr=come_to_rest(p[0],mass=PION_MASS)+1
                    val=cuts[c]+1
                    if ctr<val or ctr>1/val:
                        update_mip(c,k,n,p)

        elif c=="Come to Rest":
            checked=True
            for k in potential_kaons:
                if  k[0].reco_length>5 and len(k[1]):
                    ctr=come_to_rest(k[0])
                    if ctr<cuts[c]:
                        for n,p in enumerate(k[1]):
                            update_mip(c,k,n,p)

        elif c=="Valid MIP Len":
            checked=True
            n=cuts["Valid MIP Len"]
            x1 = n % 100
            x2 = (n // 100) % 100
            x3 = (n // 100**2) % 100
            x4 = (n // 100**3) % 100
            pi_bounds=[x1,x2]
            mu_bounds=[x3,x4]
            for k in potential_kaons:
                for n,p in enumerate(k[1]):
                    plen=p[0].reco_length
                    if not ((pi_bounds[0]<plen and plen<pi_bounds[1]) or 
                            (mu_bounds[0]<plen and plen<mu_bounds[1])):
                        
                        update_mip(c,k,n,p)

        # elif c=="Correct MIP TPC Assoc.":
        #     checked=True
        #     for k in potential_kaons:
        #         for n,p in enumerate(k[1]):
        #             # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
        #             # plen=np.sum([i.reco_length for i in p])
        #             if not p[0].is_contained or not p[0].is_contained:
        #                 update_mip(c,k,n,p)

                        # if self.truth:
                            # print("Correct MIP Module Assoc.",get_tpc_id(p[0].start_point),set([i[1] for i in p[0].sources]),get_tpc_id(p[0].start_point),set([i[1] for i in p[0].sources]))
        test=np.any([len(l[1]) == 0 for k in potential_kaons for l in k[1]])
        if failed:
            assert not test

        if test:
            assert not failed
        else:
            if c!="":
                pass_failure+=[c]
            failed=True
            if stop_early:
                print("reasonable particle")
                return False
            # if not self.truth:
                # return False
                
        if not checked:
            raise Exception(c,"not found in K+ cuts")
    # print(len(pass_failure))
    assert len(pass_failure)==len(set(pass_failure))
    if pass_failure[0]==r"Max Decay $\gamma$":
        print("this nonsense actually worked for something")
    return pass_failure==[""]


def point_to_segment_distance(p0:np.ndarray, p:ParticleType):
    assert p.shape not in [DELTA_SHP,SHOWR_SHP,MICHL_SHP]
    p1=p.start_point
    p2=p.end_point

    v = p2 - p1       # segment direction
    w = p0 - p1       # point relative to p1

    seg_len_sq = np.dot(v, v)
    assert seg_len_sq==seg_len_sq,v
    assert seg_len_sq!=np.inf,v
    if seg_len_sq == 0:
        # p1 and p2 are the same point
        return norm3d(p0 - p1)

    # Project w onto v, normalized: t is how far along the segment the projection falls
    t = np.dot(w, v) / seg_len_sq

    # If projection falls before p1, clamp to p1
    if t <= 0:
        closest = p1
    # If projection falls after p2, clamp to p2
    elif t >= 1:
        closest = p2
    # Otherwise projection is within the segment
    else:
        closest = p1 + t * v

    return norm3d(p0 - closest)

def michel_endify(particles:list[ParticleType]):
    for p in particles:
        if p.shape==MICHL_SHP:
            d2 = np.sum((p.points - p.end_point)**2, axis=1)
            idx = np.argmax(d2)
            p.end_point=p.points[idx]




#TODO need PIDA variable
#TODO need an "end point for michel/delta/shower"


def closest_point_between_lines(p0, u, p1, v, eps=1e-12):
    """
    Closed-form closest approach between two 3D lines.
    No matrix solve needed.
    Returns: q0, q1, midpoint
    """
    p0 = np.asarray(p0, float)
    u  = np.asarray(u, float)
    p1 = np.asarray(p1, float)
    v  = np.asarray(v, float)

    w0 = p0 - p1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    D = a * c - b * b

    if abs(D) < eps:
        # parallel case
        t = -d / a
        s = 0.0
    else:
        t = (b * e - c * d) / D
        s = (a * e - b * d) / D

    q0 = p0 + t * u
    q1 = p1 + s * v
    midpoint = 0.5 * (q0 + q1)

    return q0, q1, midpoint

# @profile
def merge_muons_and_pions(particles,hip,cuts):
    done=False
    while not done:
        OBVIOUS_MUONS:list["ParticleType"]=[p for p in particles if p.pid in [MUON_PID] and p.id!=hip.id]
        PIONS=[p for p in particles if p.pid in [PION_PID] and p.id!=hip.id]
        STRICT_MICHLS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p) in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
        done=True
        for p1 in OBVIOUS_MUONS.copy():
            thresh2 = (2 * min_len) ** 2
            p1em = any(
                np.sum((m.points - p1.end_point) ** 2, axis=1).min() < thresh2
                for m in STRICT_MICHLS
            )
            p1sm = any(
                np.sum((m.points - p1.start_point) ** 2, axis=1).min() < thresh2
                for m in STRICT_MICHLS
            )
            for p2 in OBVIOUS_MUONS.copy()+PIONS:

                if p1.id==p2.id: continue

                min_dist=np.sqrt(min_endpoint_dist2(p1,p2))
                if min_dist>min_len/2: continue

                

                if abs(norm3d(p1.end_point-p2.start_point)-min_dist)<.001 and (angle_between(p1.end_point-p1.start_point,p2.end_point-p2.start_point)<np.pi/6):
                    # and np.all(np.array([norm3d(o.end_point-(p1.end_point+p2.start_point)/2) for o in self.particles if o not in [p1,p2] and o.shape in [TRACK_SHP]])>min_len)
                    # and np.all(np.array([norm3d(o.start_point-(p1.end_point+p2.start_point)/2) for o in self.particles if o not in [p1,p2] and o.shape not in [DELTA_SHP,LOWES_SHP]])>min_len)):
                    
                    # if p1.is_matched and p2.is_matched:
                    #     if p1.match_ids[0] in self.decay_mip_dict or p2.match_ids[0] in self.decay_mip_dict:
                    #         print("THESE TWO BELONG TOGETHER")
                    #         print("THE MATCH",p1.match_ids[0],p2.match_ids[0],p1.match_ids[0] in self.decay_mip_dict,p2.match_ids[0] in self.decay_mip_dict)
                    merge_particles(p1,p2,particles)
                    assert done
                    done=False
                    break
                
                if abs(norm3d(p1.start_point-p2.start_point)-min_dist)<.001 and (angle_between(p1.end_point-p1.start_point,p2.end_point-p2.start_point)>np.pi-np.pi/6):# or (not p1.is_primary) or (not p2.is_primary)):

                    
                    p2em = any(
                        np.sum((m.points - p2.end_point) ** 2, axis=1).min() < thresh2
                        for m in STRICT_MICHLS
                    )
                    p2sm = any(
                        np.sum((m.points - p2.start_point) ** 2, axis=1).min() < thresh2
                        for m in STRICT_MICHLS
                    )
                    
                    if p1em and not np.any([p1sm,p2em,p2sm]):
                        flip_particle(p2)
                        merge_particles(p2,p1,particles)
                        assert done
                        done=False
                    # if not done:  
                        break


            
                    
            if not done:
                    break
        if not done:continue
            
        if "Valid MIP Len" in cuts:
            n=cuts["Valid MIP Len"]
            x1 = n % 100
            x2 = (n // 100) % 100
            x3 = (n // 100**2) % 100
            x4 = (n // 100**3) % 100
            pi_bounds=[x1,x2]
            mu_bounds=[x3,x4]
            # potential_mips=[p for p in self.particles if p.pid in [MUON_PID,PION_PID] and p!=self.hip and ((pi_bounds[0]<p.reco_length and p.reco_length<pi_bounds[1]) or (mu_bounds[0]<p.reco_length and p.reco_length<mu_bounds[1]))]
            potential_mips=[p for p in particles if p.pid in [MUON_PID,PION_PID]]
            for p1 in potential_mips:
                for p2 in potential_mips:
                    if p1.id==p2.id:continue
                    if p1.id==hip.id or p2.id==hip.id: continue
                    if norm3d(p1.end_point-p2.start_point)>min_len:continue
                    if (pi_bounds[0]<p1.reco_length+p2.reco_length<pi_bounds[1] or mu_bounds[0]<p1.reco_length+p2.reco_length<mu_bounds[1]) and angle_between(p1.end_dir,p2.start_dir)<np.pi/12:
                        merge_particles(p1,p2,particles)
                        assert done
                        # print("trying")
                        done=False
                        break
                if not done:
                    break      
    assert done


    

# def diameter(vectors):
#     vecs = list(map(np.asarray, vectors))
#     max_d = 0.0
#     for i in range(len(vecs)):
#         for j in range(i+1, len(vecs)):
#             d = norm3d(vecs[i] - vecs[j])
#             if d > max_d:
#                 max_d = d
#     return max_d

# @njit
def min_endpoint_dist2(p, q):
    """Return the minimum squared distance between endpoints of particles p and q."""

    d1 = np.dot(p.start_point - q.start_point, p.start_point - q.start_point)
    d2 = np.dot(p.start_point - q.end_point,   p.start_point - q.end_point)
    d3 = np.dot(p.end_point   - q.start_point, p.end_point   - q.start_point)
    d4 = np.dot(p.end_point   - q.end_point,   p.end_point   - q.end_point)

    return min(d1, d2, d3, d4)



from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def find_intersections(particles, eps=min_len, min_samples=3, forward=NUMI_BEAM_DIR):
    f = forward
    pts = []
    dirs = []
    all_pts = []
    scores = []

    #  1) Collect endpoints 
    for p in particles:
        if p.shape != TRACK_SHP:
            continue

        sp = p.start_point
        ep = p.end_point
        v = ep - sp

        all_pts.append(ep); scores.append(ep @ f)
        all_pts.append(sp); scores.append(sp @ f)

        if norm3d(v) > 2 * eps:
            pts.append(sp); dirs.append(v)
            pts.append(ep); dirs.append(-v)

    if not pts:
        return []

    pts = np.asarray(pts)
    dirs = np.asarray(dirs)
    all_pts = np.asarray(all_pts)
    scores = np.asarray(scores)

    min_point = all_pts[np.argmin(scores)]
    N = len(pts)

    #  2) Build voxel hash grid 
    vox = np.floor(pts / eps).astype(np.int32)
    voxel_map = {}
    for i, (ix, iy, iz) in enumerate(vox):
        voxel_map.setdefault((ix, iy, iz), []).append(i)

    #  3) Build adjacency graph via radius search within voxel neighborhood 
    rows = []
    cols = []

    # neighbor voxel offsets
    neigh = [(dx, dy, dz)
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             for dz in (-1, 0, 1)]

    for i in range(N):
        ix, iy, iz = vox[i]
        pi = pts[i]

        neighbors = []
        # gather candidate neighbors from nearby voxels
        for dx, dy, dz in neigh:
            cell = (ix+dx, iy+dy, iz+dz)
            if cell in voxel_map:
                neighbors.extend(voxel_map[cell])

        # compute actual distances only for those candidates
        if len(neighbors) >= min_samples:
            nbr = np.array(neighbors, dtype=np.int32)

            d2 = np.sum((pts[nbr] - pi)**2, axis=1)
            good = nbr[d2 <= eps*eps]

            if len(good) >= min_samples:
                rows.extend([i]*len(good))
                cols.extend(good)

    if not rows:
        return []

    #  4) Connected components (cluster extraction) 
    graph = coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(N, N)
    )
    n_comp, labels = connected_components(graph, directed=False)

    proj = dirs @ f
    intersections = []

    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if len(idx) < min_samples:
            continue

        # downstream check
        if proj[idx].min() <= 0:
            continue

        intersections.append(pts[idx].mean(axis=0))

    #  5) Final cut 
    if len(intersections) == 1:
        if norm3d(min_point - intersections[0]) < min_len:
            return intersections

    return []


from scipy.spatial import cKDTree as KDTree
from collections import deque

def redirect(particles: list[ParticleType], vertex, min_len=min_len):
    #  1. Collect endpoints 
    endpoints = []
    for i, p in enumerate(particles):
        if p.shape != TRACK_SHP:
            continue
        endpoints.append((i, 0, p.start_point))
        endpoints.append((i, 1, p.end_point))

    if not endpoints:
        return particles

    coords = np.vstack([pt for _,_,pt in endpoints])
    tree = KDTree(coords)

    N = len(particles)
    adj = [[] for _ in range(N)]

    neighbors = [tree.query_ball_point(p, min_len) for p in coords]

    #  2. Build adjacency graph 
    for idx, neigh_list in enumerate(neighbors):
        pid_i, _, _ = endpoints[idx]
        for j in neigh_list:
            pid_j, _, _ = endpoints[j]
            if pid_i != pid_j:
                adj[pid_i].append(pid_j)
                adj[pid_j].append(pid_i)

    #  3. Find closest endpoint root 
    dists = np.linalg.norm(coords - vertex, axis=1)
    closest_idx = np.argmin(dists)
    root_pid, root_end_type, _ = endpoints[closest_idx]

    #  4. Seed orientations 
    orient = {}
    queue = deque()

    # Graph root
    orient[root_pid] = (root_end_type == 0)
    queue.append(root_pid)

    #  4b. Directional ray-based seeds 
    for i, p in enumerate(particles):
        if p.shape != TRACK_SHP or i in orient:
            continue

        s = p.start_point
        e = p.end_point
        d = e - s

        # particle clearly points away from vertex?
        v = s - vertex
        if np.linalg.norm(v) == 0:
            continue

        cosang = np.dot(d, v) / (np.linalg.norm(d) * np.linalg.norm(v))

        # If aligned to within ~18 degrees
        if cosang > 0.95:
            orient[i] = True
            queue.append(i)

    #  5. BFS propagation 
    while queue:
        u = queue.popleft()
        p_u = particles[u]
        u_S, u_E = p_u.start_point, p_u.end_point
        u_out = u_E if orient[u] else u_S

        for v in adj[u]:
            if v in orient:
                continue

            p_v = particles[v]
            v_S, v_E = p_v.start_point, p_v.end_point

            if np.linalg.norm(v_S - u_out) < min_len:
                orient[v] = True
            elif np.linalg.norm(v_E - u_out) < min_len:
                orient[v] = False
            else:
                continue

            queue.append(v)

    #  6. Apply flips 
    for pid, forward in orient.items():
        if not forward:
            flip_particle(particles[pid])
            # print("flipping",particles[pid].id)

    return particles

    
def process_primary_tracks(particles:list[ParticleType]):

    '''
    Function to fix forward going kaon decaying to backwards going muon looking instead like a backwards vertex
    '''


    #  1. Collect only primary track-shaped particles that point back along beam direction
    prim_tracks = [
        (i, p) for i, p in enumerate(particles)
        if p.shape == TRACK_SHP and (NUMI_ROT@(p.end_point-p.start_point))[2]<0 and p.reco_length>2*min_len
    ]
    if len(prim_tracks) < 2:
        return

    start_pts = np.array([p.start_point for _, p in prim_tracks])
    indices = [i for i, _ in prim_tracks]
    tree = KDTree(start_pts)

    #  2. Find pairs of primary forward-z track particles with start-start match 
    pairs = []
    for k, pt in enumerate(start_pts):
        neigh = tree.query_ball_point(pt, min_len)
        for n in neigh:
            if n <= k:
                continue
            pid1 = indices[k]
            pid2 = indices[n]
            pairs.append((pid1, pid2))

    #  3. Filter out pairs where another particle lies within min_len of the junction 
    # junction point = midpoint of start points
    for (i1, i2) in pairs:
        p1 = particles[i1]
        p2 = particles[i2]

        junction = 0.5 * (p1.start_point + p2.start_point)

        # check for other particles nearby
        any_other_near = False
        for j, q in enumerate(particles):
            if j == i1 or j == i2:
                continue
            if norm3d(q.start_point-junction) < 5 or norm3d(q.end_point-junction) < 5:
                any_other_near = True
                break

        if any_other_near:
            continue  # discard pair

        has_michl_near_end = False
        for j, q in enumerate(particles):
            if q.shape == MICHL_SHP and (norm3d(q.start_point-p1.end_point) < min_len or norm3d(q.end_point-p1.end_point) < min_len):
                has_michl_near_end = True
                break

        if has_michl_near_end:
            p1.is_primary = False
            flip_particle(p2)

    return

from spine.utils.energy_loss import csda_range_lar

def ke_from_momentum(p,m):
    return np.sqrt(p**2+m**2)-m

def lepton_momentum(E_nu, M, m_l, theta, M_X=0,M_nu=0):
    """
    Compute the lepton momentum p_l in a neutrino-nucleon interaction.

    Parameters
    ----------
    E_nu : float
        Incoming neutrino energy
    M : float
        Mass of nucleon at rest
    m_l : float
        Mass of outgoing lepton
    theta : float
        Scattering angle of lepton relative to incoming neutrino direction [rad]
    M_X : float
        Mass of recoil system

    Returns
    -------
    p_l : float
        Magnitude of lepton 3-momentum
    """
    mu = np.cos(theta)

    P_nu=np.sqrt(E_nu**2-M_nu**2)
    assert P_nu>0

    A = E_nu + M
    B = P_nu * mu
    D = A**2 - E_nu**2 +m_l**2 - M_X**2

    alpha = A**2 - B**2
    beta = D * B
    gamma = A**2 * m_l**2 - D**2/4

    discriminant = beta**2 - 4 * alpha * gamma
    if discriminant < 0:
        raise ValueError("No physical solution for given inputs")
    
    p_l = -2*gamma/(beta + np.sqrt(discriminant))
    return p_l



def MAX_BACK_LEN(m,t,M=PROT_MASS,E_nu=10**6,M_X=0):
    max_mom=lepton_momentum(E_nu,M,m,t,M_X)
    max_ke=ke_from_momentum(max_mom,m)
    return csda_range_lar(max_ke,m)


from collections import deque

def build_components_and_classify(particles: list[ParticleType], radius=2*min_len):
    """
    builds clusters using endpoint proximity 
    look at clusters with >3 tracks >10cm
    if one cluster is "forward" of another in terms of its start point and COM,
    everything in it is non_primary
    """

    # Filter tracks
    particles = [p for p in particles if p.shape == TRACK_SHP]
    n = len(particles)
    if n == 0:
        return [], []

    r2 = radius * radius

    # Pre-extract endpoints
    starts = np.array([p.start_point for p in particles])
    ends   = np.array([p.end_point   for p in particles])

    graph = [[] for _ in range(n)]

    for i in range(n):
        s1 = starts[i]
        e1 = ends[i]

        ds_ss = np.sum((starts[i+1:] - s1)**2, axis=1)
        ds_se = np.sum((ends[i+1:]   - s1)**2, axis=1)
        ds_es = np.sum((starts[i+1:] - e1)**2, axis=1)
        ds_ee = np.sum((ends[i+1:]   - e1)**2, axis=1)

        d2 = np.minimum.reduce((ds_ss, ds_se, ds_es, ds_ee))
        close = np.where(d2 <= r2)[0] + i + 1

        for j in close:
            graph[i].append(j)
            graph[j].append(i)

    visited = np.zeros(n, dtype=bool)
    components = []

    for i in range(n):
        if visited[i]:
            continue

        q = deque([i])
        visited[i] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        components.append(comp)

    comp_info = []

    for comp in components:
        # Rotate all points at once
        pts = np.concatenate(
            [particles[k].points @ NUMI_ROT.T for k in comp],
            axis=0
        )

        com = pts.mean(axis=0)
        idx_up = np.argmin(pts[:, 2])

        comp_info.append({
            "particles": comp,
            "com": com,
            "start": pts[idx_up],
            "n_long": sum(particles[i].reco_length > 10 for i in comp)
        })

    for A in comp_info:
        if A["n_long"] <= 3:
            continue

        Azs = A["start"][2]
        Azc = A["com"][2]

        for B in comp_info:
            if A is B:
                continue

            if Azs < B["start"][2] and Azc < B["com"][2]:
                for idx in B["particles"]:
                    particles[idx].is_primary = False

    return components, comp_info


def pop_obvious_cosmics(particles:list[ParticleType]):
    for p in particles:
        if not p.is_time_contained:
            p.is_primary=False

    deltas=[p for p in particles if p.shape in [DELTA_SHP,SHOWR_SHP]]
    tracks=[p for p in particles if p.shape==TRACK_SHP and not (is_contained(p.start_point,margin=5) and is_contained(p.end_point,margin=5)) and p.is_primary]
    # print([i.id for i in tracks])
    all_tracks=[p for p in particles if p.shape==TRACK_SHP]

    valid_tracks=[]
    for p1 in list(tracks.copy()):
        valid=True
        for p2 in list(all_tracks):
            if p1.id==p2.id:
                continue
            if (norm3d(p1.end_point-p2.end_point)<min_len or 
                norm3d(p1.start_point-p2.end_point)<min_len or 
                norm3d(p1.end_point-p2.start_point)<min_len or 
                norm3d(p1.start_point-p2.start_point)<min_len
                ):
                valid=False
                # print(p1.id,"not under consideration",p2.id)
                break
        if valid:
            valid_tracks.append(p1)
    tracks=valid_tracks
    for p in tracks:
        if p.is_primary:
            delta_count=0
            for d in deltas:
                if point_to_segment_distance(d.start_point,p)<min_len:
                    delta_count+=1
            if delta_count>=2:
                # print("removing",p.id,p.pid,p.start_point,p.end_point)
                p.is_primary=False
            else:
                if not np.any([norm3d(p.end_point-p2.end_point)<30 or 
                    norm3d(p.start_point-p2.end_point)<30 or 
                    norm3d(p.end_point-p2.start_point)<30 or 
                    norm3d(p.start_point-p2.start_point)<30 for p2 in all_tracks if p2.id!=p.id]):
                    p.is_primary=False
    for p in all_tracks:
        if p.is_primary:
            if not np.any([norm3d(p.end_point-p2.end_point)<30 or 
                    norm3d(p.start_point-p2.end_point)<30 or 
                    norm3d(p.end_point-p2.start_point)<30 or 
                    norm3d(p.start_point-p2.start_point)<30 for p2 in all_tracks if p2.id!=p.id]):
                    p.is_primary=False

def get_cathode_pos(p:ParticleType):
    # tpc_id=Geo.get_closest_tpc([point]).cathode
    modules, tpcs =Geo.get_contributors(p.sources)
    return Geo.tpc[modules[0], tpcs[0]].cathode_pos

# import os
# @profile
def emergency_option(particles:list[ParticleType],all_particles:list[ParticleType]):#TODO apparently this is a SPINE preprocessor
    changed=[]
    all_tracks=[p for p in particles if p.shape==TRACK_SHP]
    to_check=[p for p in particles if p.reco_length>50 and p.is_primary and 
                  not np.any(
                      [norm3d(p.start_point-q.start_point)<3*min_len or 
                      norm3d(p.start_point-q.end_point)<3*min_len or 
                      norm3d(p.end_point-q.start_point)<3*min_len or 
                      norm3d(p.end_point-q.end_point)<3*min_len
                      for q in all_tracks if p.id!=q.id])]
    if len(to_check):
        cath_pos=get_cathode_pos(to_check[0])
        for p in to_check:
            translated_start=p.start_point+np.array([2*(cath_pos-p.start_point[0]),0,0])
            translated_end=p.end_point+np.array([2*(cath_pos-p.end_point[0]),0,0])

            for q in all_particles:
                if q.id==p.id:
                    continue
                if q.shape==TRACK_SHP and q.reco_length>50:
                    if (norm3d(translated_start-q.start_point)<min_len or 
                        norm3d(translated_start-q.end_point)<min_len or 
                        norm3d(translated_end-q.start_point)<min_len or 
                        norm3d(translated_end-q.end_point)<min_len):
                        print("found a cathode crosser")
                        p.is_primary=False
                        changed.append(p)
    return changed