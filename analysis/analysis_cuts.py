""" 
This file contains output classes and cut functions useful for reconstructing kaons and 
lambdas in a liquid argon TPC using the reconstruction package SPINE https://github.com/DeepLearnPhysics/spine
"""

import copy
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
    


NUMI_BEAM_DIR=[.388814672,-.058321970,.919468161]

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

from typing import TypeAlias
from typing import TYPE_CHECKING

from spine.data.out.interaction import RecoInteraction,TruthInteraction
from spine.data.out.particle import TruthParticle,RecoParticle
if TYPE_CHECKING:
    
    InteractionType:TypeAlias= "RecoInteraction | TruthInteraction"
    ParticleType:TypeAlias = "TruthParticle|RecoParticle"


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

def HM_pred_hotfix(p:"ParticleType",hm_pred:Optional[dict[int,np.ndarray]]=None,old=False)->int:

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
        assert cos_theta==cos_theta,(cos_theta,a,b)
        assert float(cos_theta)==cos_theta,(cos_theta,a,b)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  



Geo = Geometry(detector=Model)

class PredKaonMuMich:
    __slots__ = ('event_number', 'truth','reason','pass_failure','error',
                'truth_list','hip','hm_pred','fm_interactions','particles',
                'potential_kaons','truth_interaction_vertex','truth_Kp','truth_interaction_nu_id',
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
    potential_kaons:list[tuple["ParticleType",list[tuple["ParticleType",list[str]]]]]

    def __init__(
        self,
        # pot_k: PotK,
        ENTRY_NUM:int,
        K_hip:"ParticleType",
        particles: list["ParticleType"],
        interactions: list["InteractionType"],
        # hm_acc:list[float],
        hm_pred:dict[int,np.ndarray],
        
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
        ids=[p.id for p in self.particles]
        if K_hip.id in ids:
            self.hip=self.particles[ids.index(K_hip.id)]
        else:
            self.hip=copy.deepcopy(K_hip)
        self.potential_kaons=[]

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

        self.kaon_path={}

        for n,p in enumerate(truth_particles):
            if p.pdg_code==321 and p.ancestor_pdg_code==321:
                self.kaon_path[n]=p



        # self.primaries=[]

        self.decay_mip_dict={}

        self.other_mip_dict={}

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
            if abs(k.pdg_code) in [211,13] and process_map[k.creation_process]=='6::201' and k.parent_pdg_code==321:
                self.other_mip_dict[n]=k
            if k.pdg_code==-11 and k.parent_pdg_code in [-13,211] and process_map[k.creation_process]=='6::201' and k.ancestor_pdg_code==321:
                self.truth_michel[n]=k
            # if k.is_primary:
                # self.primaries+=[n]

        
        # for n,i in enumerate(truth_particles):
            # i
        assert self.hip in interaction.particles,(self.hip.id,[i.id for i in interaction.particles])#,[i.id for i in interaction.primary_particles],type(interaction))


        

        self.real_K_momentum=K_hip.reco_momentum
        if self.truth and type(K_hip)==TruthParticle:
            self.real_K_momentum=momentum_from_children_ke(K_hip,particles,KAON_MASS)
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
    def pass_cuts(self,cuts:dict)->bool:#dict[str,dict[str,bool|list]|bool]
        # original_hip_id=self.hip.id
        # if self.hip in self.particles:
        #     h=self.particles.index(self.hip)
        # self.particles=copy.deepcopy(self.particles)

        # if self.hip in self.particles:
        #     self.hip=self.particles[h]
        # else:
        #     self.hip=copy.deepcopy(self.hip)


        

        self.pass_failure:list[str]=[]

        if self.hip.reco_length>4 and is_contained(self.hip.start_point,margin=-5) and is_contained(self.hip.end_point,margin=-5) and len(self.hip.points)>=3: assert sum([p==self.hip for p in self.particles])==1,(sum([p==self.hip for p in self.particles]),sum([p.id==self.hip.id for p in self.particles]),self.hip.reco_length,len(self.hip.points),self.hip.pid,self.hip.shape)

        for p in self.particles:
            if p==self.hip:
                assert p.is_primary==self.hip.is_primary
        primary_hip=(self.hip.is_primary)
        

        if type(self.hip)==RecoParticle and self.is_flash_matched:


            MIPS_AND_MICHELS(self.particles)
            MIPS_AND_MICHELS_2(self.particles,skip=[self.hip])
            # CORRECT_BACKWARDS(self.particles,skip=[self.hip.id])

            self.particles:list["ParticleType"]=[p for p in self.particles if (p.shape!=TRACK_SHP or p==self.hip or p.reco_length>min_len/2)]
                
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
                for p in self.particles:
                    if p.pid in [PION_PID,PROT_PID,KAON_PID]:
                        if np.any(np.array([norm3d(p.start_point-m.start_point) for m in PROBABLE_MIPS])<min_len) and not np.any(np.array([norm3d(p.end_point-m.start_point) for m in PROBABLE_MIPS])<min_len):
                            flip_particle(p)
                            
                            if p.is_matched and p.match_ids[0] in self.kaon_path:
                                print("FLIPPED THE HIP")
                                tp=self.kaon_path[p.match_ids[0]]
                                if angle_between(p.start_point-p.end_point,tp.start_point-tp.end_point)<np.pi/2:
                                    print("AND IT'S RIGHT NOW")
                                else:
                                    print("AND IT'S WRONG NOW")
                        # PROBABLE_MIPS+=[p]
                    # elif (not np.any(np.array([norm3d(p.start_point-m.start_point) for m in STRICT_MICHLS])<3*min_len)) and np.any(np.array([norm3d(p.end_point-m.start_point) for m in STRICT_MICHLS])<3*min_len):
                    #     PROBABLE_MIPS+=[p]

            done=False
            while not done:
                OBVIOUS_MUONS:list["ParticleType"]=[p for p in self.particles if p.pid in [MUON_PID] and p!=self.hip]
                PIONS=[p for p in self.particles if p.pid in [PION_PID] and p!=self.hip]
                STRICT_MICHLS:list["ParticleType"]=[p for p in self.particles if HM_pred_hotfix(p) in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
                done=True
                for p1 in OBVIOUS_MUONS.copy():
                    p1em=np.any(np.array([np.min(cdist([p1.end_point],m.points)) for m in STRICT_MICHLS])<2*min_len)
                    p1sm=np.any(np.array([np.min(cdist([p1.start_point],m.points)) for m in STRICT_MICHLS])<2*min_len)
                    for p2 in OBVIOUS_MUONS.copy()+PIONS:

                        if p1.id==p2.id: continue
                        # if p1.pid!=MUON_PID and p2.pid!=MUON_PID:
                            # continue

                        min_dist=np.min(cdist([p1.start_point,p1.end_point],[p2.start_point,p2.end_point]))
                        if min_dist>min_len/2: continue

                        

                        if abs(norm3d(p1.end_point-p2.start_point)-min_dist)<.001 and (angle_between(p1.end_point-p1.start_point,p2.end_point-p2.start_point)<np.pi/6):
                            # and np.all(np.array([norm3d(o.end_point-(p1.end_point+p2.start_point)/2) for o in self.particles if o not in [p1,p2] and o.shape in [TRACK_SHP]])>min_len)
                            # and np.all(np.array([norm3d(o.start_point-(p1.end_point+p2.start_point)/2) for o in self.particles if o not in [p1,p2] and o.shape not in [DELTA_SHP,LOWES_SHP]])>min_len)):
                            
                            if p1.is_matched and p2.is_matched:
                                if p1.match_ids[0] in self.decay_mip_dict or p2.match_ids[0] in self.decay_mip_dict:
                                    print("THESE TWO BELONG TOGETHER")
                                    print("THE MATCH",p1.match_ids[0],p2.match_ids[0],p1.match_ids[0] in self.decay_mip_dict,p2.match_ids[0] in self.decay_mip_dict)
                            merge_particles(p1,p2,self.particles)
                            assert done
                            done=False
                            break
                        
                        if abs(norm3d(p1.start_point-p2.start_point)-min_dist)<.001 and (angle_between(p1.end_point-p1.start_point,p2.end_point-p2.start_point)>np.pi-np.pi/6):# or (not p1.is_primary) or (not p2.is_primary)):

                            
                            p2em=np.any(np.array([np.min(cdist([p2.end_point],m.points)) for m in STRICT_MICHLS])<2*min_len)
                            p2sm=np.any(np.array([np.min(cdist([p2.start_point],m.points)) for m in STRICT_MICHLS])<2*min_len)
                            
                            if p1em and not np.any([p1sm,p2em,p2sm]):
                                flip_particle(p2)
                                merge_particles(p2,p1,self.particles)
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
                    potential_mips=[p for p in self.particles if p.pid in [MUON_PID,PION_PID]]
                    for p1 in potential_mips:
                        for p2 in potential_mips:
                            if p1==p2:continue
                            if p1==self.hip or p2==self.hip: continue
                            if norm3d(p1.end_point-p2.start_point)>min_len:continue
                            if (pi_bounds[0]<p1.reco_length+p2.reco_length<pi_bounds[1] or mu_bounds[0]<p1.reco_length+p2.reco_length<mu_bounds[1]) and angle_between(p1.end_dir,p2.start_dir)<np.pi/12:
                                merge_particles(p1,p2,self.particles)
                                assert done
                                # print("trying")
                                done=False
                                break
                        if not done:
                            break      
            assert done

            rot_hip_mom=(NUMI_ROT@self.hip.start_dir)
            if (rot_hip_mom[2]<0 and np.abs(rot_hip_mom[2])>np.linalg.norm(rot_hip_mom[:2])):# or (np.any(np.array([norm3d(self.hip.start_point-m.start_point) for m in PROBABLE_MIPS if m.id!=self.hip.id])<min_len) and not np.any(np.array([norm3d(self.hip.end_point-m.start_point) for m in PROBABLE_MIPS if m.id!=self.hip.id])<min_len)):# and self.hip.reco_length>25:#TODO fix this back if there is an obvious MIP next to it
                    flip_particle(self.hip)

                    for p in self.particles:
                        if self.hip.id==p.id:
                            # flip_particle(p)
                            assert (NUMI_ROT@p.start_dir)[2]>=0
        # if True:
            if True:
                done=False
                max_step=1
                while not done:
                    done=True


                    for p in [i for i in self.particles if i.shape==TRACK_SHP]:
                        for q in [i for i in self.particles if i.shape==TRACK_SHP and i.id!=p.id]:
                            # if p.shape==TRACK_SHP:

                            if norm3d(p.end_point-q.end_point)<min_len/2:
                                p_start_ending=0
                                q_start_ending=0
                                for r in [i for i in self.particles if i.shape==TRACK_SHP and i.id!=p.id and i.id!=q.id]:
                                    if norm3d(r.end_point-q.start_point)<min_len/2:
                                        q_start_ending+=1
                                    if norm3d(r.end_point-p.start_point)<min_len/2:
                                        p_start_ending+=1
                                if p_start_ending and not q_start_ending:
                                    if q!=self.hip:
                                        flip_particle(q)
                                        done=False


                    for p in self.particles:
                        if p.shape!=TRACK_SHP: continue
                        if not p.is_primary: continue
                        for q in self.particles:
                            if q.shape!=TRACK_SHP:continue
                            if not q.is_primary:
                                est_end=(p.end_point+q.end_point)/2
                                if norm3d(p.end_point-q.end_point)>min_len/2: continue
                                # if (np.any(np.array([norm3d(o.end_point-est_end) for o in self.particles if o not in [p,q] and o.shape in [TRACK_SHP]])<min_len) or
                                    # np.any(np.array([norm3d(o.start_point-est_end) for o in self.particles if o not in [p,q] and o.shape not in [DELTA_SHP,LOWES_SHP]])<min_len)): continue

                                if q!=self.hip:
                                    flip_particle(q)
                                    done=False

                    for p in self.particles:
                        if p.shape!=TRACK_SHP: continue
                        for q in self.particles:
                            if q.shape!=TRACK_SHP:continue
                            if q.is_primary:
                                est_end=(p.end_point+q.start_point)/2

                                if norm3d(p.end_point-q.start_point)>min_len/2: continue
                                if (np.any(np.array([norm3d(o.end_point-est_end) for o in self.particles if o not in [p,q] and o.shape in [TRACK_SHP]])<min_len) or
                                    np.any(np.array([norm3d(o.start_point-est_end) for o in self.particles if o not in [p,q] and o.shape not in [DELTA_SHP,LOWES_SHP]])<min_len)): continue
                                


                                if q!=self.hip:
                                    q.is_primary=False
                                    done=False
                    # if False:
                    for p in [j for j in self.particles if j.shape==TRACK_SHP and not j.is_primary and j.reco_length>min_len and j!=self.hip]:
                        # other_tracks=
                        end_near_end=False
                        start_near_start=False
                        for q in [j for j in self.particles if j.shape==TRACK_SHP and j.id!=p.id]:
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
                # p=None
                # for p in [:
                if self.hip.shape==TRACK_SHP and self.hip.reco_length>10 and False:
                    dists_end = np.linalg.norm(self.hip.points - self.hip.end_point, axis=1)
                    mask_end = dists_end< 10
                    max_edep_end = np.median(self.hip.depositions[mask_end]*np.power(dists_end[mask_end],.42))


                    dists_start = np.linalg.norm(self.hip.points - self.hip.start_point, axis=1)
                    mask_start = dists_start< 10
                    max_edep_start = np.median(self.hip.depositions[mask_start]*np.power(dists_start[mask_start],.42))

                    if max_edep_start>1.5*max_edep_end:
                        flip_particle(self.hip)

                    
                        if self.hip.is_matched and self.hip.match_ids[0] in self.kaon_path:
                            print("FLIPPED THE HIP")
                            if angle_between(self.hip.start_point-self.hip.end_point,self.kaon_path[self.hip.match_ids[0]].start_point-self.kaon_path[self.hip.match_ids[0]].end_point)<np.pi/2:
                                print("AND IT'S RIGHT NOW")
                            else:
                                print("AND IT'S WRONG NOW",self.hip.is_matched and self.hip.match_ids[0] in self.kaon_path)

        # if True:
            ps=[(NUMI_ROT@p.start_dir)[2] for p in self.particles if p.is_primary]
            if np.any(ps):
                primary_start=min(ps)

                for p in self.particles:
                    if p.is_primary and (NUMI_ROT@p.start_dir)[2]>primary_start+100 and p!=self.hip:
                        assert p.id!=self.hip.id
                        p.is_primary=False
                        # print(p.id,self.hip.id,self.event_number,p.is_primary,p.reco_ke,p.pdg_code,self.hip.is_primary)

            if False:
                if self.hip.reco_length>min_len:
                    dists = np.linalg.norm(self.hip.points - self.hip.end_point, axis=1)
                    slope, intercept = np.polyfit(dists ,self.hip.depositions, 1)
                    if slope>.1:
                        flip_particle(self.hip)
                        if self.hip.is_matched and self.truth_hip is not None and self.truth_hip.pdg_code==321 and self.truth:
                            print("CHANGED THE HIP DIRECTION WITH SLOPE",angle_between(self.hip.momentum,self.truth_hip.momentum)<np.pi/2,slope)



            # if "Valid MIP Len" in cuts:
            #     pi_bounds=cuts["Valid MIP Len"]["pi_len"]
            #     mu_bounds=cuts["Valid MIP Len"]["mu_len"]
            #     # potential_mips=[p for p in self.particles if p.pid in [MUON_PID,PION_PID] and p!=self.hip and ((pi_bounds[0]<p.reco_length and p.reco_length<pi_bounds[1]) or (mu_bounds[0]<p.reco_length and p.reco_length<mu_bounds[1]))]
            #     potential_mips=[p for p in self.particles if p.pid in [MUON_PID,PION_PID]]
            #     for 
                # if len(potential_mips)==1:
                #     if norm3d(potential_mips[0].start_point-self.hip.start_point)<min_len and norm3d(potential_mips[0].start_point-self.hip.end_point)>min_len and len([l for l in self.particles if norm3d(l.start_point-potential_mips[0].start_point)<min_len and len(l.points>3)])==2 and len([l for l in self.particles if norm3d(l.end_point-potential_mips[0].start_point)<min_len and len(l.points>3)])==0:
                #         self.hip.start_point,self.hip.end_point=self.hip.end_point,self.hip.start_point
                #         # raise Exception(ship_copy.momentum,type(ship_copy.momentum))
                #         self.hip.start_dir=self.hip.start_dir*(-1)
                #         if self.hip.is_matched and self.truth_hip is not None and self.truth_hip.pdg_code==321 and self.truth:
                #             print("CHANGED THE HIP DIRECTION WITH LENGTH",angle_between(self.hip.momentum,self.truth_hip.momentum)<np.pi/2)


                #     if norm3d(potential_mips[0].end_point-self.hip.end_point)<min_len and norm3d(potential_mips[0].start_point-self.hip.end_point)>min_len and len([l for l in self.particles if norm3d(l.start_point-potential_mips[0].start_point)<min_len and len(l.points>3)])==0 and len([l for l in self.particles if norm3d(l.end_point-potential_mips[0].start_point)<min_len and len(l.points>3)])==2:
                #             potential_mips[0].start_point,potential_mips[0].end_point=potential_mips[0].end_point,potential_mips[0].start_point
                #             # raise Exception(ship_copy.momentum,type(ship_copy.momentum))
                #             potential_mips[0].start_dir=potential_mips[0].start_dir*(-1)
                    

            for p in self.particles:
                if p.is_primary and p.shape in [SHOWR_SHP,DELTA_SHP,LOWES_SHP,MICHL_SHP] and p.reco_ke<100 and p!=self.hip:
                    p.is_primary=False
                    # if p==self.hip: raise Exception()

        

            if True:
                for p in self.particles:
                    if not p.is_primary: continue
                    if not p.shape==TRACK_SHP:continue
                    for q in self.particles:
                        if p==q: continue
                        if not q.is_primary: continue
                        if not q.shape==TRACK_SHP:continue
                        if q==self.hip: continue
                        if np.min(cdist([q.start_point,q.end_point],[p.start_point,p.end_point]))<2*min_len: continue
                        if impact_parameter(p.start_point,q.start_point,q.start_dir)<5 and (NUMI_ROT@q.start_dir)[2]>10+(NUMI_ROT@p.start_dir)[2]:
                            q.is_primary=False
                            assert q.id!=self.hip.id

        if True:    
            new_vert=reconstruct_vertex_single(self.particles)
            if new_vert is not None:
                self.reco_vertex=new_vert

        # if self.truth:
            # print("help",self.hip.id,self.hip.is_primary)

        assert (self.hip.is_primary)==primary_hip
        K_plus_cut_cascade(self,cuts,self.hip,self.truth_interaction_nu_id,self.potential_kaons,self.pass_failure,self.decay_mip_dict,self.kaon_path)
        assert (self.hip.is_primary)==primary_hip
        
        # assert self.hip.id==original_hip_id
        # if self.truth:
            # print(self.hip.id,self.hip.is_primary)



        if len(self.pass_failure) and primary_hip and self.pass_failure[0]=="Primary $K^+$":
            raise Exception(self.hip.is_primary,primary_hip)
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
        hm_pred:dict[int,np.ndarray],
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
        ids=[p.id for p in self.particles]
        if mip.id in ids:
            self.mip=self.particles[ids.index(mip.id)]
        else:
            self.mip=copy.deepcopy(mip)

        if hip.id in ids:
            self.hip=self.particles[ids.index(hip.id)]
        else:
            self.hip=copy.deepcopy(hip)


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

        # guess_start = get_pseudovertex(
        #     start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        #     directions=[hip.reco_start_dir, mip.reco_start_dir],
        # )

        

        # self.pi_extra_children=[]
        # self.prot_extra_children=[]
        
        # self.pot_parent:list[tuple["ParticleType",bool]]=[]

        self.truth_interaction_id=truth_interaction_id_hotfix(interaction)

        self.is_flash_matched=interaction.is_flash_matched

        self.reco_vertex=reco_vert_hotfix(interaction)

        self.primary_particle_counts=interaction.primary_particle_counts
        
    
    def pass_cuts(self,cuts:dict)->bool:
        # passed=True
        extra_children=[]
        self.pass_failure=[]

        pot_parent:list[tuple["ParticleType",bool]]=[]

        guess_start=(self.hip.start_point+self.mip.start_point)/2

        MIPS_AND_MICHELS(self.particles)
        for i in self.particles:
            if i.id==self.mip.id:
                self.mip=i
            if i.id==self.hip.id:
                self.hip=i

        

        new_vert=reconstruct_vertex_single(self.particles)
        if new_vert is not None:
            self.reco_vertex=new_vert







        # particles:list["ParticleType"]=interaction.particles

        for p in self.particles:
            # if p.interaction_id!=hip.interaction_id: continue
            if p.id not in [self.mip.id,self.hip.id] and HM_pred_hotfix(p,self.hm_pred) in [SHOWR_HM,MIP_HM,HIP_HM,MICHL_HM]:
                extra_children += [p]
                if HM_pred_hotfix(p,self.hm_pred) in [MIP_HM,HIP_HM]: pot_parent+=[(p,False)]

        # assert self.hip in particles,(self.hip.id,[i.id for i in particles])#,[i.id for i in interaction.primary_particles],type(interaction))
        # assert self.mip in particles,(self.mip.id,[i.id for i in particles])#,[i.id for i in interaction.primary_particles],type(interaction))
        michels:list["ParticleType"]=[p for p in self.particles if p.shape in [MICHL_SHP,LOWES_SHP]]
        # assert norm3d(self.reco_vertex)>0,inter


        self.vae=impact_parameter(self.reco_vertex,guess_start,self.real_hip_momentum_reco + self.real_mip_momentum_reco)

        #TODO make a cut of vae with other pid choices 


        primary_shapes=np.bincount([p.shape for p in self.particles if is_primary_hotfix(p)],minlength=10)


        MIP_prot_KE_rat=abs(come_to_rest(self.mip,PROT_MASS))
        MIP_pi_KE_rat=abs(come_to_rest(self.mip,PION_MASS))
        # HIP_prot_KE_rat=abs(come_to_rest(self.hip,PROT_MASS))
        # HIP_pi_KE_rat=abs(come_to_rest(self.hip,PION_MASS))


        for c in cuts:

            # if len(self.pass_failure)>=3: break
            checked=False


            if c=="MIP Child":
                checked=True
                if HM_pred_hotfix(self.mip,self.hm_pred)!=MIP_HM and self.mip.pid not in [PION_PID,PROT_PID]:#:# and 
                    self.pass_failure+=[c]
                    
                if MIP_prot_KE_rat<min(.5,MIP_pi_KE_rat):
                    self.pass_failure+=[c]
                    

            if c=="HIP Child":
                checked=True
                if HM_pred_hotfix(self.hip,self.hm_pred)!=HIP_HM and self.hip.pid not in [PION_PID,PROT_PID]:#:# and s
                    self.pass_failure+=[c]
                    
                # if HIP_pi_KE_rat<min(.5,HIP_prot_KE_rat):
                #     self.pass_failure+=[c]
                #     

                    # PROT_PID

            if c=="Primary HIP-MIP":
                checked=True
                # if type(self.hip)==TruthParticle:
                if is_primary_hotfix(self.hip) or is_primary_hotfix(self.mip):# or HM_pred_hotfix(self.hip,self.hm_pred)!=HIP_HM or HM_pred_hotfix(self.mip,self.hm_pred)!=MIP_HM:
                    # if self.truth: print("nonprimary hip/mip", norm3d(self.mip.start_point-self.hip.start_point))
                    # if passed:
                    self.pass_failure+=[c]
                        

            if c=="Valid Interaction":
                checked=True
                if type(self.hip)==TruthParticle:
                    if self.truth_interaction_nu_id==-1:
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
                    


            if c=="Proj Max HIP-MIP Sep.":
                checked=True
                # if norm3d(self.mip.start_point-self.hip.start_point)>cuts[c]:
                # if closest_distance(self.mip.start_point, self.mip.momentum, self.hip.start_point, self.hip.momentum)>cuts[c] or not np.isclose(np.linalg.norm(self.hip.start_point-self.mip.start_point),np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))):

                if closest_distance(self.mip.start_point, self.mip.momentum, self.hip.start_point, self.hip.momentum)>cuts[c]:
                # if np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))>cuts[c]:
                    # if self.truth: print("Max HIP/MIP Sep.", norm3d(self.mip.start_point-self.hip.start_point))
                    # if passed:
                    self.pass_failure+=[c]


            if c=="Starts Closest":
                checked=True
                # if norm3d(self.mip.start_point-self.hip.start_point)>cuts[c]:
                # if closest_distance(self.mip.start_point, self.mip.momentum, self.hip.start_point, self.hip.momentum)>cuts[c] or :

                if not np.isclose(np.linalg.norm(self.hip.start_point-self.mip.start_point),np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))):
                # if np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))>cuts[c]:
                    # if self.truth: print("Max HIP/MIP Sep.", norm3d(self.mip.start_point-self.hip.start_point))
                    # if passed:
                    self.pass_failure+=[c]




            if c=="Max HIP-MIP Sep.":
                checked=True
                # if norm3d(self.mip.start_point-self.hip.start_point)>cuts[c]:

                if norm3d(self.mip.start_point-self.hip.start_point)>cuts[c]:
                # if np.min(cdist([self.hip.start_point,self.hip.end_point] ,[self.mip.start_point,self.mip.end_point]))>cuts[c]:
                    # if self.truth: print("Max HIP/MIP Sep.", norm3d(self.mip.start_point-self.hip.start_point))
                    # if passed:
                    self.pass_failure+=[c]
                    

            if c=="No HIP or MIP Michel":
                checked=True
                for p in michels:
                    if norm3d(self.mip.end_point-p.start_point)<cuts[c] or norm3d(self.hip.end_point-p.start_point)<cuts[c]:
                    # if np.min(cdist(p.points, [self.mip.end_point]))<cuts[c]:
                        self.pass_failure+=[c]
                        
                        break
                
            if c=="Impact Parameter":
                checked=True
                if self.vae>cuts[c]:
                    # if self.truth: print("VAE max", vae)
                    # if passed:
                    self.pass_failure+=[c]
                    

            if c=="Min Decay Len":
                checked=True
                if float(norm3d(self.reco_vertex - guess_start))<cuts[c]:
                    # if self.truth: print("minimum decay len")
                    # if passed:
                    self.pass_failure+=[c]
                    

            #TODO check if one of the particles has csda per whatever match the wrong one of proton or pion


            if c=="Parent Proximity":
                checked=True
                start_to_int=norm3d(guess_start-self.reco_vertex)
                for p in pot_parent:
                    if norm3d(p[0].start_point-self.reco_vertex)>2*min_len:
                        continue
                    # est_decay=(self.mip.start_point+self.hip.start_point)/2
                    if (norm3d(p[0].end_point-self.reco_vertex)>=start_to_int and 
                        norm3d(p[0].start_point-self.reco_vertex)>=start_to_int):continue
                    if norm3d(p[0].end_point-guess_start)<=min_len or norm3d(p[0].end_point-self.hip.start_point)<=min_len/2 or norm3d(p[0].end_point-self.mip.start_point)<=min_len/2:
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
                    # if self.truth: 
                    #     print("odd photon cut")
                    # if passed:
                    self.pass_failure+=[c]
                    

            if c=="":
                checked=True
                self.pass_failure+=[c]
            
            if not checked:
                raise Exception(c,"not found in lam cuts")
        
        
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

# def HIPMIP_acc(particle:"ParticleType", sparse3d_pcluster_semantics_HM: np.ndarray,perm=None,mode=True) -> float:
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


# def collision_distance(particle1:"ParticleType", particle2:"ParticleType",orientation:list[str]=["start","start"]):
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







# def lambda_AM(hip:"ParticleType",mip:"ParticleType")->list[float]:
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
        
        particles = [part for part in particles if (
                part.is_primary and
                # part.shape in self.include_shapes and
                part.size > 0) and part.shape not in [SHOWR_SHP]]
        # if not self.use_primaries or not len(particles):
        #     particles = [part for part in particles if (
        #             part.shape in self.include_shapes and
        #             part.size > 0)]
        # if not len(particles):
        # particles = [part for part in particles if part.size > 0]

        if len(particles) > 0:
            # Collapse particle objects to start, end points and directions
            start_points = np.vstack([part.start_point for part in particles])
            end_points   = np.vstack([part.end_point for part in particles])
            directions   = np.vstack([part.start_dir for part in particles])
            shapes       = np.array([part.shape for part in particles])

            # Reconstruct the vertex for this interaction
            vtx, _ = get_vertex(
                start_points, end_points, directions, shapes,return_mode=True)
            
            return vtx

def Bragg_Peak(p,len_bragg=10):#TODO this needs to be stored 
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


def MIPS_AND_MICHELS(particles):
    STRICT_MICHLS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p) in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
    # PROBABLE_MIPS:list["ParticleType"]=[]
    for p in particles:
        if p.pid in [MUON_PID,PION_PID]:
            if np.any(np.array([np.min(cdist([p.start_point],m.points)) for m in STRICT_MICHLS])<3*min_len) and not np.any(np.array([np.min(cdist([p.end_point],m.points)) for m in STRICT_MICHLS])<3*min_len):
                flip_particle(p)
def MIPS_AND_MICHELS_2(particles,skip:list["ParticleType"]=[]):
    STRICT_MICHLS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p) in [MICHL_SHP,LOWES_SHP] and (not is_primary_hotfix(p))]#
    for p in particles:
        if p.pid in [MUON_PID,PION_PID] and p not in skip:
            if p.is_primary and np.any(np.array([np.min(cdist([p.end_point],m.points)) for m in STRICT_MICHLS])<3*min_len) and (NUMI_ROT@p.start_dir)[2]<0 and -(NUMI_ROT@p.start_dir)[2]>.5*np.linalg.norm(p.start_dir) and p.reco_length>40:
                p.is_primary=False

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
    p1.points=np.append(p1.points,p2.points,axis=0)
    p1.depositions=np.append(p1.depositions,p2.depositions)
    p1.end_point=p2.end_point
    p1.length=p1.length+p2.length
    p1.calo_ke=p1.calo_ke+p2.calo_ke
    p1.end_dir=p2.end_dir
    particles.remove(p2)
    # print("merging",p1.id,p2.id)


def K_plus_cut_cascade(obj,cuts,BASE_HIP:"ParticleType",nu_id,potential_kaons:list[tuple["ParticleType",list[tuple["ParticleType",list[str]]]]],pass_failure,decay_mip_dict={},kaon_path={}):
    def update_mip(cut:str,k,p):
        idx = k[1].index(p)
        assert cut not in k[1][idx][1]
        if not len(p[1]):
            k[1][idx][1].append(cut)
        elif (not p[0].is_matched) or (p[0].match_ids[0] not in decay_mip_dict) or (not k[0].is_matched) or (k[0].match_ids[0] not in kaon_path):
            k[1].remove(p)
        elif decay_mip_dict[p[0].match_ids[0]].parent_id==k[0].id or norm3d(decay_mip_dict[p[0].match_ids[0]].start_point-kaon_path[k[0].match_ids[0]].end_point)<min_len:
            k[1][idx][1].append(cut)
        # elif 
        else:
            k[1].remove(p)

    # BASE_HIP=copy.deepcopy(BASE_HIP)
    
    def update_mips(cut: str):
        for k in potential_kaons:
            for p in list(reversed(k[1])):   # SAFE: iterate over a copy
                update_mip(cut, k, p)


    RECO_VERTEX=obj.reco_vertex
    HM_PRED=None#self.hm_pred
    particles=obj.particles

    if BASE_HIP.reco_length>4 and is_contained(BASE_HIP.start_point,margin=-5) and is_contained(BASE_HIP.end_point,margin=-5) and len(BASE_HIP.points)>=3: assert sum([p==BASE_HIP for p in particles])==1,(sum([p==BASE_HIP for p in particles]),sum([p.id==BASE_HIP.id for p in particles]))

    NON_PRIMARY_TRACKS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED) in [MIP_HM,HIP_HM] and p.reco_length>0]
    NON_PRIMARY_MIPS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED)==MIP_HM and p.reco_length>0]
    NON_PRIMARY_MICHLS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED) in [MICHL_SHP,LOWES_SHP,SHOWR_SHP,DELTA_SHP]]
    NON_PRIMARY_SHWRS:list["ParticleType"]=[p for p in particles if HM_pred_hotfix(p,HM_PRED) in [SHOWR_SHP,LOWES_SHP]]
    done=False

    potential_kaons+=[(BASE_HIP,[(i,[]) for i in NON_PRIMARY_MIPS])]
    

    while not done: #this loop goes over all of the hips connected to the end of the kaon, and constructs a hadronic group which hopefully contains the kaon end. 
        done=True
        # print("looking")
        
        for p in NON_PRIMARY_TRACKS:
            if p not in [r[0] for r in potential_kaons]:
                # print("getting here")
                for k in potential_kaons:
                    # print(k[0])
                    n1=norm3d(p.start_point-k[0].end_point)
                    n2=norm3d(p.start_point-k[0].start_point)
                    if n1<min_len and n1<n2:

                        potential_kaons+=[(p,[(i,[]) for i in NON_PRIMARY_MIPS])]
                        done=False
                        break
                    # elif n2<min_len and n2<n1:
                    #     potential_kaons+=[[p,copy.copy(MIP_CHAINS),[]]]
                    #     done=False
                    #     break

    failed=False
    for c in cuts:
        # if len(self.pass_failure)>=3: break

        checked=False

        if c=="Contained HIP":
            checked=True
            if (not is_contained(BASE_HIP.end_point,margin=3)) or (not is_contained(BASE_HIP.start_point,margin=3)):
                update_mips(c)
                failed=True

        if c=="Fiducialization":
            checked=True
            if (not is_contained(RECO_VERTEX,margin=margin0)):
                update_mips(c)
                failed=True
                

        elif c=="dedx chi2":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                if k[0].reco_length>10 and len(k[1]):
                    bp=Bragg_Peak(k[0])[1]
                    if bp>cuts[c]:
                        for p in list(reversed(k[1])):
                            update_mip(c,k,p)


        elif c=="Connected Non-Primary MIP":
            # did_something=0
            checked=True
            for k in potential_kaons:
                for p in list(reversed(k[1])):
                        # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                        # plen=np.sum([i.reco_length for i in p])

                    

                        mip_start=p[0].start_point
                        # mip_end=p[0].end_point



                        n1=norm3d(mip_start-k[0].end_point)
                        if (n1>min_len*2 or n1>=norm3d(mip_start-k[0].start_point)):
                            # did_something=1
                            update_mip(c,k,p)
                        elif p[0].is_primary and norm3d(p[0].start_point-BASE_HIP.start_point)<min_len/2:
                            update_mip(c,k,p)

                
        
        elif c=="No HIP Deltas":
            checked=True
            deltas=False

            DELTAS=[p for p in particles if p.shape==DELTA_SHP]
            for p in DELTAS:
                # if p.shape==DELTA_SHP:
                if np.min(cdist(BASE_HIP.points, [p.start_point]))<min_len/2:
                    deltas=True
                    update_mips(c)
                    failed=True
                    break
            if not deltas:
                for k in potential_kaons:
                    for p in list(reversed(k[1])):
                        for d in DELTAS:
                        # for p in list(reversed(k[1])):
                            if np.min(cdist(k[0].points, [d.start_point]))<min_len/2:
                        # deltas=True
                                update_mip(c,k,p)
                                break
        elif c=="No LOW E MIP Deltas":
            checked=True
            deltas=False

            DELTAS=[p for p in particles if p.shape==DELTA_SHP and p.reco_ke>10]
            for k in potential_kaons:
                for p in list(reversed(k[1])):
                    for d in DELTAS:
                    # for p in list(reversed(k[1])):
                        if np.min(cdist(p[0].points, [d.start_point]))<cuts[c]:
                        # deltas=True
                            
                            update_mip(c,k,p)
                            break

            # if deltas:
                # if HM_pred_hotfix(BASE_HIP,HM_PRED)!=HIP_HM and BASE_HIP.pid not in  [3,4]:
                # update_mips(c)


        elif c=="Contained MIP":
            checked=True
            for k in potential_kaons:
                for p in list(reversed(k[1])):
                        # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                        # plen=np.sum([i.reco_length for i in p])

                        

                        # mip_start=p[0].start_point
                        mip_end=p[0].end_point

                        if not is_contained(mip_end,margin=1):
                            update_mip(c,k,p)
            
            # print(np.sum([len(k[1]) for k in potential_kaons]))
            # if np.sum([len(k[1]) for k in potential_kaons])==0:
            #     self.pass_failure+=[c]
                


        elif c=="Nothing Before the Start":
            checked=True
            for p in particles:
                if p.shape==TRACK_SHP and p.is_primary and p.reco_length>5*min_len and p.pid!=MUON_PID:
                    if norm3d(p.end_point-BASE_HIP.start_point)<min_len/2 and norm3d(p.start_point-BASE_HIP.start_point)>3*min_len:
                        update_mips(c)
                        failed=True


        

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
            if not is_primary_hotfix(BASE_HIP):# and norm3d(inter-BASE_HIP.end_point)>cuts[c]):# or norm3d(BASE_HIP.start_point-inter)>cuts[c]:# or HM_pred_hotfix(BASE_HIP,hm_pred)!=HIP_HM:
                update_mips(c)
                failed=True
                
            # else:
            #     if not is_primary_hotfix(BASE_HIP):#norm3d(BASE_HIP.start_point-inter)>cuts[c] and 
            #         self.pass_failure+=[c]
            #         

        elif c=="Close to Vertex":
            checked=True
            # others_dist_start=min([np.linalg.norm(p.start_point-BASE_HIP.start_point) for p in self.particles if p!=BASE_HIP])
            # others_dist_end=min([np.linalg.norm(p.start_point-BASE_HIP.start_point) for p in self.particles if p!=BASE_HIP])
            if np.linalg.norm(BASE_HIP.start_point-RECO_VERTEX)>cuts[c]:# and norm3d(inter-BASE_HIP.end_point)>cuts[c]):# or norm3d(BASE_HIP.start_point-inter)>cuts[c]:# or HM_pred_hotfix(BASE_HIP,hm_pred)!=HIP_HM:
                # if others_dist<2*min_len:
                update_mips(c)
                failed=True
                # else:
                #     if impact_parameter(RECO_VERTEX,BASE_HIP.start_point,BASE_HIP.start_dir)<min_len/2:



        elif c=="Correct HIP TPC Assoc.":
            checked=True
            # if get_tpc_id(BASE_HIP.start_point) not in set([i[1] for i in BASE_HIP.sources]) and get_tpc_id(BASE_HIP.end_point) not in set([i[1] for i in BASE_HIP.sources]):#BASE_HIP.module_ids:
            if not BASE_HIP.is_contained:
                
                update_mips(c)
                failed=True
                

                # if self.truth:
                    # print("Correct HIP Module Assoc.",get_tpc_id(BASE_HIP.start_point),get_tpc_id(BASE_HIP.end_point),BASE_HIP.sources)


        elif c=="Kaon Len":
            checked=True
            if BASE_HIP.reco_length<cuts[c]:
                update_mips(c)
                failed=True
                

        elif c=="Valid Interaction":
            checked=True
            if type(BASE_HIP)==TruthParticle:
                if nu_id==-1:
                    update_mips(c)
                    failed=True
            elif type(BASE_HIP)==RecoParticle:
                # print()
                if not obj.is_flash_matched:# and best_dist>cuts[c]:
                    # if self.truth: print("FAILED AT VALID INTERACTION")#,best_dist)
                    update_mips(c)
                    failed=True
                    
            else:
                raise Exception(type(BASE_HIP))
            

        elif c=="Forward HIP":
            checked=True
            if angle_between(BASE_HIP.momentum,NUMI_BEAM_DIR)>cuts[c] and len([p for p in particles if p.shape==TRACK_SHP])==2 and len([p for p in particles if p.shape==SHOWR_SHP and p.is_primary])==0:
                # found_another=False
                # for p in self.particles:
                #     if p.id!=BASE_HIP.id and norm3d(p.start_point-BASE_HIP.start_point)<min_len:
                #         found_another=True
                #         break
                # if not found_another:
                update_mips(c)
                failed=True
        elif c=="More Than Downgoing":
            checked=True
            ending=[]
            starting=[]
            for p in particles:
                if norm3d(p.end_point-RECO_VERTEX)<min_len/2:
                    ending+=[p]
                if norm3d(p.start_point-RECO_VERTEX)<min_len/2:
                    starting+=[p]
                if len(ending)>1 or len(starting)>1:
                    break
                # if len()
            if len(starting)==1 and len(ending)==1:
                # if norm3d(starting[0].start_point-ending[0].end_point)<min_len:
                if np.abs(angle_between(starting[0].momentum,ending[0].end_dir)-np.pi/2)>np.pi/2-cuts[c][1] and np.abs(angle_between(starting[0].momentum,np.array([0,-1,0]))-np.pi/2)>np.pi/2-cuts[c][0]:
                    update_mips(c)
                    failed=True
                    

            elif len(starting)==1 and len(ending)<2 and angle_between(starting[0].momentum,np.array([0,-1,0]))<cuts[c][0]:
                update_mips(c)
                failed=True
                
            
        elif c=="":
            checked=True
            # 

            pass_failure+=[""]

        elif c=="Michel Child":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):

                    # mip_start=p[0].start_point
                    mip_end=p[0].end_point

                    mich_child=False
                    # if p[0].reco_length>10:
                        # mich_child=Bragg_Peak(p[0])>5.5
                    for other in NON_PRIMARY_MICHLS:
                        if other.reco_ke>90: continue
                        check_dist=np.min(cdist([mip_end],other.points))
                        if other.shape not in [DELTA_SHP,LOWES_SHP]:
                            if check_dist<4*min_len:
                                mich_child=True
                                break
                        else:
                            if check_dist<min_len/2:
                                mich_child=True
                                break

                    if not mich_child:
                        
                        update_mip(c,k,p)

        # elif c==r"HIP $K^+ or >0 Scatter$":
                            
        #     if k[0]==BASE_HIP and BASE_HIP.pid in [2,3]:
        #         
        #         update_mip(c,k,p)

        elif c==r"Low MIP len $\pi^0$ Tag":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    plen=p[0].reco_length

                    mip_start=p[0].start_point
                    mip_end=p[0].end_point
                    
    
                    has_pi0=(plen>=40)
                    if not has_pi0:
                        for other in NON_PRIMARY_SHWRS:
                            if other.reco_ke<10: continue# i think the minimal energy of a photon from the pi0 is around 20MeV
                            if other.reco_ke>300: continue# i think the maximum energy is around 225 MeV
                            if (#norm3d(other.start_point-mip_start)<cuts[c][0]*np.inf
                                # and norm3d(other.start_point-mip_end)>min_len 
                                np.min(cdist(other.points, [mip_end]))>min_len
                                and (impact_parameter(mip_start,other.start_point,other.momentum)<impact_parameter(BASE_HIP.start_point,other.start_point,other.momentum) or np.linalg.norm(BASE_HIP.start_point-mip_start)<3*min_len)
                                and (impact_parameter(mip_start,other.start_point,other.momentum)<cuts[c])):#or angle_between(other.start_point-mip_start,other.momentum)<cuts[c][2])
                                # and cos_gamma_to_pip_bounds(other.reco_ke)[0]<np.cos(angle_between(other.start_point-mip_start,p[0].momentum)) and np.cos(angle_between(other.start_point-mip_start,p[0].momentum))<cos_gamma_to_pip_bounds(other.reco_ke)[1]):
                                # and abs((cos_gamma_to_E(mip_start,other.start_point,p[0].momentum)-other.reco_ke)/other.reco_ke)<cuts[c][2]):
                                has_pi0+=1 #TODO apparently a 3112 doesnt get a reco length?
                    if not has_pi0:
                        
                        update_mip(c,k,p)

        elif c=="MIP Child At Most 1 Michel":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    plen=p[0].reco_length

                    mip_start=p[0].start_point
                    mip_end=p[0].end_point
                            
                    # for other in NON_PRIMARY_HIPS:
                    # check1=np.any([(norm3d(other.start_point-mip_end)<min_len)*(other.reco_length>min_len) for other in NON_PRIMARY_HIPS if other!=p[-1]])
                    # check2=np.any([(norm3d(other.start_point-mip_end)<min_len)*(other.reco_length>min_len) for other in NON_PRIMARY_MIPS if other!=p[-1]])

                    check=np.any([(norm3d(other.start_point-mip_end)<min_len)*(other.reco_length>min_len/2) for other in NON_PRIMARY_TRACKS if other.pid==PROT_PID])
                    if check:

                        
                        update_mip(c,k,p)
                        continue
                    check2=np.any([(norm3d(other.start_point-mip_end)<min_len)*(other.reco_length>min_len) for other in NON_PRIMARY_TRACKS])
                    if check2:

                        
                        update_mip(c,k,p)
                        continue
                    check3=np.any([(norm3d(other.end_point-mip_end)<min_len)*(other.reco_length>min_len) for other in NON_PRIMARY_TRACKS if other.id!=p[0].id])
                    if check3:

                        
                        update_mip(c,k,p)
                        continue
                    
        elif c=="Single MIP Decay":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    plen=p[0].reco_length

                    mip_start=p[0].start_point
                    mip_end=p[0].end_point

                    if np.sum([(norm3d(other.start_point-mip_start)<min_len)*(other.reco_length>min_len)*(norm3d(other.start_point-mip_start)<norm3d(other.start_point-BASE_HIP.start_point)) for other in NON_PRIMARY_TRACKS if other!=BASE_HIP])>1:
                        
                        update_mip(c,k,p)
        
        elif c=="Separable MIP":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    if angle_between(BASE_HIP.end_dir,p[0].start_dir)<cuts[c]:
                        update_mip(c,k,p)

                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    # plen=p[0].reco_length

                            # if other.shape==TRACK_SHP and norm3d(other.start_point-p.start_point)<min_len and other.reco_length>min_len:
                                # add_it=False
                                # ???????
        elif c=="Bragg Peak HIP":
            checked=True
            # all_particle_starts=[p.start_point for p in self.particles]
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                if k[0].reco_length>10 and len(k[1]):
                    # starts = np.array(all_particle_starts)
                    # v0 = np.array(k[0].end_point)

                    # # Compute Euclidean distances from each point to v0
                    # dists = np.linalg.norm(starts - v0, axis=1)

                    # # Count how many are within 'dist'
                    # num_within = np.sum(dists <= 2*min_len)
                    # if num_within>1: continue

                    bp=Bragg_Peak(k[0])[0]
                    if bp<cuts[c]:
                        for p in list(reversed(k[1])):
                            update_mip(c,k,p)

        elif c=="Bragg Peak MIP":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    if p[0].reco_length>10:
                        bp=Bragg_Peak(p[0])[0]
                        if bp<cuts[c]:
                            # for p in list(reversed(k[1])):
                            update_mip(c,k,p)
        elif c=="Come to Rest MIP":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    ctr=come_to_rest(p[0],mass=PION_MASS)+1
                    val=cuts[c]+1
                    if ctr<val or ctr>1/val:
                        # for p in list(reversed(k[1])):
                        update_mip(c,k,p)

        elif c=="Come to Rest":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    # plen=p[0].reco_length

                    # mip_start=p[0].start_point
                    # mip_end=p[0].end_point
                    if  k[0].reco_length>5 and len(k[1]):

                        ctr=come_to_rest(k[0])
                        if ctr<cuts[c]:# and k[0].reco_length>10:# or abs(csda_ke_lar(k[0].reco_length, PROT_MASS)-k[0].calo_ke)<.25*abs(csda_ke_lar(k[0].reco_length, KAON_MASS)-k[0].calo_ke) or abs(csda_ke_lar(k[0].reco_length, PION_MASS)-k[0].calo_ke)<.25*abs(csda_ke_lar(k[0].reco_length, KAON_MASS)-k[0].calo_ke):
                            
                            for p in list(reversed(k[1])):
                                update_mip(c,k,p)

        elif c=="Valid MIP Len":
            checked=True
            n=cuts["Valid MIP Len"]
            x1 = n % 100
            x2 = (n // 100) % 100
            x3 = (n // 100**2) % 100
            x4 = (n // 100**3) % 100
            pi_bounds=[x1,x2]
            mu_bounds=[x3,x4]
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    plen=p[0].reco_length

                    # mip_start=p[0].start_point
                    # mip_end=p[0].end_point
                            
                    if not ((pi_bounds[0]<plen and plen<pi_bounds[1]) or 
                            (mu_bounds[0]<plen and plen<mu_bounds[1])):
                        
                        update_mip(c,k,p)

        elif c=="Correct MIP TPC Assoc.":
            checked=True
            for k in potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                for p in list(reversed(k[1])):
                    # assert type(p)==list["ParticleType"],(type(p),type[p[0]])
                    # plen=np.sum([i.reco_length for i in p])
                    plen=p[0].reco_length

                    mip_start=p[0].start_point
                    mip_end=p[0].end_point

                    if not p[0].is_contained or not p[0].is_contained:
                        
                        
                        update_mip(c,k,p)

                        # if self.truth:
                            # print("Correct MIP Module Assoc.",get_tpc_id(p[0].start_point),set([i[1] for i in p[0].sources]),get_tpc_id(p[0].start_point),set([i[1] for i in p[0].sources]))
        if failed:
            assert not np.any([len(l[1]) == 0 for k in potential_kaons for l in k[1]])

        if np.any([len(l[1]) == 0 for k in potential_kaons for l in k[1]]):
            assert not failed
        else:
            pass_failure+=[c]
            failed=True
            # if not self.truth:
                # return False
                
        if not checked:
            raise Exception(c,"not found in K+ cuts")



#TODO need PIDA variable
#TODO need an "end point for michel/delta/shower"
