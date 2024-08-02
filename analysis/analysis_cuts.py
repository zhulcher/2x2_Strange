# from pandas import Float32Dtype
from spine.utils.geo.base import *
import spine
from spine.utils.globals import *
from spine.utils.vertex import *
import numpy as np


#TODO things I would like added in truth:
    #children_id

HIP=2
MIP=3


Particle=spine.RecoParticle|spine.TruthParticle
Interaction=spine.RecoInteraction|spine.TruthInteraction
Met=spine.Meta
Geo=Geometry(detector='2x2')


def is_contained(pos:np.ndarray,mode:str,margin:float=0)->bool:
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
    return max(set(HM_Pred), key = HM_Pred.count)
    
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

    
def dist_end_start(particle:Particle,parent_candidates:list[Particle])->list[float]:
    
    '''
    Returns minimum distance between the start of child particle and the end
    of every parent candidate supplied, along with the parent candidate identified.
    Returns (np.inf, None) if the parent_candidate list is empty

    Parameters
    ----------
    particle : spine.Particle
        Particle object
    parent_candidates: List(spine.Particle)
        List of spine particle objects corresponding to potential parents of particle
    
    Returns
    -------
    [float,spine.Particle]
        Distance from parent end to child start and corresponding parent

    '''
    idp=np.nan
    N=0
    shortest_dist=np.inf
    for p in parent_candidates:
        if np.linalg.norm(p.end_point-particle.start_point)<shortest_dist:
            v=p.end_point-particle.start_point
            shortest_dist=np.dot(v,v)
            idp=N
        N+=1
    return [shortest_dist,idp]


def is_child_eps_angle(parent_end:np.ndarray,child:Particle,max_dist:float=np.inf,max_angle:float=np.pi,min_dist:float=0)->tuple[bool,float,float]:
    '''
    Returns True iff the child particle start is within dist from the parent particle end and the child particle points back
    to the parent particle end with an angular deviation smaller than angle

    Parameters
    ----------
    parent_end : np.ndarray(3)
        parent end location
    child: spine.Particle
        Particle object
    max_dist: float
        max dist from child start to parent end 
    max_angle: float
        max angle between line pointing from parent end to child start and child initial direction
    min_dist: float
        if child start closer than this from parent end, return True

    
    Returns
    -------
        bool
            this is a child of the parent, according to the prescription outlined
    '''
    true_dir=child.start_point-parent_end
    separation=float(np.linalg.norm(true_dir))
    if separation==0:
        angle=0
    else:
        angle=np.arccos(np.dot(true_dir,child.start_dir)/separation)
    if separation<min_dist: return (True, separation,angle)
    if separation>max_dist: return (False,separation,angle)
    if angle>max_angle: return (False,separation,angle)
    return (True,separation,angle)

def children(parent_end:np.ndarray,particle_list:list[Particle],max_dist:float=np.inf,max_angle:float=np.pi,min_dist:float=0)->list[tuple[Particle,float,float]]:
    '''
    Returns children candidates

    Parameters
    ----------
    parent_end : np.ndarray(3)
        parent end location
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
        list[Particle]
            children candidates
    '''
    children=[]
    for p in particle_list:
        is_child=is_child_eps_angle(parent_end,p,max_dist,max_angle,min_dist)
        if is_child[0]:
            children+=[(p,is_child[1],is_child[2])]
    return children
    

def lambda_children(hip:Particle,mip:Particle,particle_list:list[Particle],max_dist:float=np.inf,max_angle:float=np.pi,min_dist:float=0)->list[tuple[Particle,float,float]]:
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
        list[Particle]
            children candidates
    '''
    children=[]
    
    guess_start=get_pseudovertex(start_points=[hip.start_point,mip.start_point],
                                 directions=[hip.start_dir,mip.start_dir])
    children=[]
    for p in particle_list:
        is_child=is_child_eps_angle(guess_start,p,max_dist,max_angle,min_dist)
        if is_child[0]:
            children+=[(p,is_child[1],is_child[2])]
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
    guess_start=get_pseudovertex(start_points=[hip.start_point,mip.start_point],
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

    guess_start=get_pseudovertex(start_points=[hip.start_point,mip.start_point],
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
    guess_start=get_pseudovertex(start_points=[hip.start_point,mip.start_point],
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
