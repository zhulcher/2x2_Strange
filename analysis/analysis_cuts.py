from spine.utils.geo.base import *
import spine
from spine.utils.globals import *
from spine.utils.vertex import *
import numpy as np
from collections import Counter
from scipy import stats as st


#TODO things I would like added in truth:
    #children_id

#TODO do I need the Kaon/ Michel flash timing?

HIP_HM=5
MIP_HM=1
SHOWR_HM=0


Particle=spine.RecoParticle|spine.TruthParticle
Interaction=spine.RecoInteraction|spine.TruthInteraction
Met=spine.Meta
Geo=Geometry(detector='2x2')

class Pot_K:
    hip_id: int
    hip_len: float
    dir_acos: float

    '''
    This is a storage class for primary Kaons and their cut parameters
    '''

    def __init__(self, hip_id,hip_len,dir_acos):
        self.hip_id = hip_id 
        self.hip_len = hip_len
        self.dir_acos=dir_acos

    def apply_cuts_K(self):
        pass

    def output(self):
        return [self.hip_id,self.hip_len,self.dir_acos]

class Pred_K(Pot_K):
    mip_id: int
    mip_len: float
    dist_to_hip: float
    K_closest_kids: list[float]

    '''
    This is a storage class for primary Kaons with muon daughter and their cut parameters
    '''

    def __init__(self, pot_k:Pot_K,mip_id:int,mip_len:float,dist_to_hip:float,K_closest_kids:list[float]):
        self.hip_id = pot_k.hip_id 
        self.hip_len = pot_k.hip_len
        self.dir_acos=pot_k.dir_acos

        self.mip_id = mip_id 
        self.mip_len = mip_len
        self.dist_to_hip=dist_to_hip
        self.K_closest_kids=K_closest_kids

    def apply_cuts_mu(self):
        pass

    def output(self):
        return [self.hip_id,self.hip_len,self.dir_acos,self.mip_id,self.mip_len,self.dist_to_hip,self.K_closest_kids]

class Pred_K_Mich(Pred_K):
    mich_id: int
    dist_to_mich: float
    Mu_closest_kids: list[float]

    '''
    This is a storage class for primary Kaons with muon daughter and michel and their cut parameters
    '''

    def __init__(self, pred_k:Pred_K,mich_id,dist_to_mich,Mu_closest_kids):

        self.hip_id = pred_k.hip_id 
        self.hip_len = pred_k.hip_len
        self.dir_acos=pred_k.dir_acos

        self.mip_id = pred_k.mip_id 
        self.mip_len = pred_k.mip_len
        self.dist_to_hip=pred_k.dist_to_hip
        self.K_closest_kids=pred_k.K_closest_kids

        self.mich_id=mich_id
        self.dist_to_mich=dist_to_mich
        self.Mu_closest_kids=Mu_closest_kids

    def apply_cuts_mich(self):
        pass

    def output(self):
        return [self.hip_id,self.hip_len,self.dir_acos,self.mip_id,self.mip_len,self.dist_to_hip,self.K_closest_kids,self.mich_id,self.dist_to_mich,self.Mu_closest_kids]

class Pred_L:
    hip_id: int
    hip_len: float
    VAE: float
    L_mass2:float
    L_decay_len: float
    AM: float
    coll_dist: float
    closest_kids: list[float]

    '''
    This is a storage class for primary Lambdas and their cut parameters
    '''

    def __init__(self, hip_id,mip_id,VAE,L_mass2,L_decay_len,AM,coll_dist,closest_kids):
        self.hip_id=hip_id
        self.mip_id=mip_id
        self.VAE=VAE
        self.L_mass2=L_mass2
        self.L_decay_len=L_decay_len
        self.AM=AM
        self.coll_dist=coll_dist
        self.closest_kids=closest_kids

    def apply_cuts_K(self):
        pass

    def output(self):
        return [self.hip_id,self.mip_id,self.VAE,self.L_mass2,self.L_decay_len,self.AM,self.coll_dist,self.closest_kids]


def is_contained(pos:np.ndarray,mode:str,margin:float=2)->bool:
    '''
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
        '''
    Geo.define_containment_volumes(margin,mode=mode)
    return bool(Geo.check_containment(pos))

#TODO is contained unit test a million points each 


def HIPMIP_pred(particle:Particle,sparse3d_pcluster_semantics_HM:np.ndarray)->int:
    '''
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
    '''
    if len(particle.depositions)==0:raise ValueError("No voxels")
    #slice a set of voxels for the target particle
    HM_Pred=sparse3d_pcluster_semantics_HM[particle.index,-1]
    # print(HM_Pred,type(HM_Pred))
    return st.mode(HM_Pred).mode

def HM_score(particle:Particle,sparse3d_pcluster_semantics_HM:np.ndarray)->float:
    '''
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
    '''
    if len(particle.depositions)==0:raise ValueError("No voxels")
    #slice a set of voxels for the target particle
    HM_Pred=sparse3d_pcluster_semantics_HM[particle.index,-1]
    pred=max(set(HM_Pred), key = HM_Pred.count)
    return Counter(HM_Pred)[pred]/len(HM_Pred)
    
def direction_acos(particle:Particle, direction=np.array([0.,0.,1.])) -> float:
    '''
    Returns angle between the beam-axis (here assumed in z) and the particle object's start direction

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information
    direction : np.ndarray[float]
        Direction of beam
    
    Returns
    -------
    float
        Angle between particle direction and beam
    '''
    return np.arccos(np.dot(particle.start_dir,direction))

def collision_distance(particle1:Particle,particle2:Particle):
    '''
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
    '''
    v1=particle1.start_dir
    v2=particle2.start_dir

    p1=particle1.start_point
    p2=particle2.start_point

    v11=np.dot(v1,v1)
    v22=np.dot(v2,v2)
    v12=np.dot(v1,v2)
    dp=p1-p2

    denom=v12**2-v11*v22

    if denom==0: return [0,0,np.linalg.norm(dp)]

    t1=(np.dot(v1,dp)*v22-v12*np.dot(v2,dp))/denom
    t2=(v12*np.dot(v1,dp)-np.dot(v2,dp)*v11)/denom

    min_dist=np.dot(p1+v1*t1-p2-v2*t2,p1+v1*t1-p2-v2*t2)

    return [t1,t2,min_dist]

    
def dist_end_start(particle:Particle,parent_candidates:list[Particle])->list[list[float]]:
    
    '''
    Returns distance between the start of child particle and the end
    of every parent candidate supplied, along with the parent candidate identified.

    Parameters
    ----------
    particle : spine.Particle
        Particle object
    parent_candidates: List(spine.Particle)
        List of spine particle objects corresponding to potential parents of particle
    
    Returns
    -------
    [list[float,float]]
        Distance from parent end to child start and corresponding entry in parent candidate list

    '''
    out=[]
    for n in range(len(parent_candidates)):
        out+=[[float(np.linalg.norm(parent_candidates[n].end_point-particle.start_point)),n]]
    return out



def is_child_eps_angle(parent_end:np.ndarray,child:Particle)->tuple[float,float]:
    '''
    Returns separation from parent particle end to child particle start and
    angle between child start direction and direction from parent end to child start 

    Parameters
    ----------
    parent_end : np.ndarray(3)
        parent end location
    child: spine.Particle
        Particle object
    
    Returns
    -------
        [float,float,bool]
            distance from child start to parent end, angle between child start direction and direction from parent end to child start 
    '''
    true_dir=child.start_point-parent_end
    separation=float(np.linalg.norm(true_dir))
    if separation==0:
        angle=0
    else:
        angle=np.arccos(np.dot(true_dir,child.start_dir)/separation)
    return (separation,angle)

def children(parent:Particle,particle_list:list[Particle],ignore:list[int])->list[float]:
    '''
    Returns children candidates

    Parameters
    ----------
    parent : spine.Particle
        parent particle
    particle_list: List(spine.Particle)
        List of spine particle objects
    ignore: List(int)
        list of particle ids to ignore
    
    Returns
    -------
        [float,float]:
            minimum distance and angle as defined in 'is_child_eps_angle' between a parent and any potential child other than the particles in ignore
    '''
    children=[np.inf,np.inf]
    for p in particle_list:
        if p.id in ignore: continue
        is_child=is_child_eps_angle(parent.end_point,p)
        children[0]=min(children[0],is_child[0])
        children[1]=min(children[1],is_child[1])
    return children
    

def lambda_children(hip:Particle,mip:Particle,particle_list:list[Particle])->list[float]:
    '''
    Returns children candidates for lambda particle

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    particle_list: List(spine.Particle)
        List of spine particle objects
    max_dist: float
        max dist from child start to parent end 
    max_angle: float
        max angle between line pointing from parent end to child start and child initial direction
    min_dist: float
        if child start closer than this from parent end, return True
    
    Returns
    -------
        [float,float]:
            minimum distance and angle as defined in 'is_child_eps_angle' between the potential lambda and any potential child other than the hip and mip
    '''
    children=[]
    guess_start=get_pseudovertex(start_points=np.array([hip.start_point,mip.start_point],dtype=float),
                                 directions=[hip.start_dir,mip.start_dir])
    children=[np.inf,np.inf]
    for p in particle_list:
        if p.id==hip.id or p.id==mip.id: continue
        is_child=is_child_eps_angle(guess_start,p)
        children[0]=min(children[0],is_child[0])
        children[1]=min(children[1],is_child[1])
    return children

def lambda_decay_len(hip:Particle,mip:Particle,interactions:list[Interaction])->float:
    '''
    Returns distance from average start position of hip and mip to vertex location of the assocated interaction

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    interactions:
        list of spine interactions
    
    Returns
    -------
    float
        distance from lambda decay point to vertex of interaction
    '''
    guess_start=get_pseudovertex(start_points=np.array([hip.start_point,mip.start_point],dtype=float),
                                 directions=[hip.start_dir,mip.start_dir])
    idx=hip.interaction_id
    return float(np.linalg.norm(interactions[idx].vertex-guess_start))


def lambda_AM(hip:Particle,mip:Particle,interactions:list[Interaction])->list[float]:
    '''
    Returns the P_T and the longitudinal momentum asymmetry corresponding to the Armenteros-Podolanski plot https://www.star.bnl.gov/~gorbunov/main/node48.html

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    list[float]
        shape(2) [hip pt + mip pt, hip vs mip longitudinal momentum assymmetry]
    '''
    inter=interactions[hip.interaction_id].vertex

    guess_start=get_pseudovertex(start_points=np.array([hip.start_point,mip.start_point],dtype=float),
                                 directions=[hip.start_dir,mip.start_dir])
    Lvec=guess_start-inter

    Lvecnorm=np.linalg.norm(Lvec)
    if Lvecnorm==0: 
        asymm=np.nan
        pt=np.nan
    else: 
        p1=hip.momentum
        p2=mip.momentum
        Lvec=Lvec/Lvecnorm

        p1_L=np.dot(Lvec,p1)
        p2_L=np.dot(Lvec,p2)

        p1_T=float(np.linalg.norm(p1-p1_L*Lvec))
        p2_T=float(np.linalg.norm(p2-p2_L*Lvec))

        asymm=abs((p1_L-p2_L)/(p1_L+p2_L))
        pt=p1_T+p2_T
    return [asymm,pt]


   
def lambda_mass_2(hip:Particle,mip:Particle)->float:
    '''
    Returns lambda mass value constructed from the hip and mip candidate deposited energy and predicted direction

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    float
        reconstructed lambda mass squared
    '''
    # LAM_MASS=1115.60 #lambda mass in MeV
    L_mass2=2*(PROT_MASS*mip.ke+hip.ke*PION_MASS-np.dot(hip.momentum,mip.momentum))+(PROT_MASS+PION_MASS)**2
    return L_mass2

def vertex_angle_error(hip:Particle,mip:Particle,interactions:list[Interaction])->float:
    '''
    Returns angle between the line constructed from the momenta of the hip and mip and 
    the line constructed from the interaction vertex and the lambda decay point

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    float
        distance from interaction vertex to line consructed from the momenta of the hip and mip
    '''


    
    inter=interactions[hip.interaction_id].vertex
    guess_start=get_pseudovertex(start_points=np.array([hip.start_point,mip.start_point],dtype=float),
                                 directions=[hip.start_dir,mip.start_dir])
    Lvec1=guess_start-inter

    Lvec2=hip.momentum+mip.momentum

    if np.linalg.norm(Lvec1)==0 or np.linalg.norm(Lvec2)==0: return np.nan
    ret=np.arccos(np.dot(Lvec1,Lvec2)/np.linalg.norm(Lvec1)/np.linalg.norm(Lvec2))
    assert ret==ret
    return ret


# def true_k_with_mu(particle_list):
#     '''
#     Returns track_ids for kaons which are contained and which only have a muon which is both contained and full range

#     Parameters
#     ----------
#     particle_list: List(spine.Particle)
#         List of spine particle objects
    
#     Returns
#     -------
#     List
#         Shape (n) track ids for true kaons satisfying cuts
#     '''
#     K_pdgs={}
#     for p in range(particle_list):
#         if p.parent_pdg_code==321 and ((p.is_contained and abs(p.pdg)==13) or p.processid=="4::121"):
#             if p.parent_id not in K_pdgs: K_pdgs[p.parent_id]=[]
#             K_pdgs[p.parent_id].append(abs(p.pdg)*(p.processid!="4::121"))
#     for i in list(K_pdgs.keys()):
#         if set(K_pdgs[i])!=set([13]):
#             K_pdgs.pop(i)
#     return list(K_pdgs.keys())

# def true_lambda(particle_list):
#     '''
#     Returns track_ids for true p/pi pairs which are both contained and full range

#     Parameters
#     ----------
#     particle_list: List(spine.Particle)
#         List of spine particle objects
    
#     Returns
#     -------
#     List([int,int])
#         List of contained pion/proton pairs which originate from true lambdas
#     '''
#     lambda_pdgs={}
#     for p in range(particle_list):
#         if p.parent_pdg_code==3122 and ((p.is_contained and abs(p.pdg) in [2212,211]) or p.processid=="4::121"):
#             if p.parent_id not in lambda_pdgs: lambda_pdgs[p.parent_id]=[]
#             lambda_pdgs[p.parent_id].append(abs(p.pdg)*(p.processid!="4::121"))
#     for i in list(lambda_pdgs.keys()):
#         if set(lambda_pdgs[i])!=set([2212,211]):
#             lambda_pdgs.pop(i)
#     return list(lambda_pdgs.keys())
