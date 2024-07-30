# from pandas import Float32Dtype
import spine
from spine.utils.globals import *
import numpy as np
from scipy import stats


#TODO things I would like added in truth:
    #children_id

#TODO label functions with keyword from slides
# import moduleboundaries

moduleboundaries=np.array([[-63.931,  63.931],
                           [-62.076,  62.076],
                           [-64.538,  64.538]])

HIP=2
MIP=3

FinalParticle=spine.RecoParticle|spine.TruthParticle

def is_contained(pos:np.ndarray,eps=30)->bool:
    '''
    Checks if a position is within the defined module boundaries within some tolerance, eps
    #TODO specify type with colon and units 
    Parameters
    ----------
    pos : np.ndarray
        (3) Vector position (cm)
    M : float/np.ndarray
        Tolerance from module boundaries (cm)

    Returns
    -------
    Bool
        Point at least eps inside of boundaries 
        '''
    if np.any(pos<moduleboundaries.T[0]+eps): return False
    if np.any(pos>moduleboundaries.T[1]-eps): return False
    return True


def HIPMIP_pred(particle:FinalParticle,sparse3d_pcluster_semantics_HM:?????):
    '''
    Returns the semantic segmentation prediction encoded in sparse3d_pcluster_semantics_HM,
    where the prediction is not guaranteed unique for each cluster, for the particle object,
    decided by majority vote among the voxels in the cluster

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information and unique semantic segmentation prediction
    sparse3d_pcluster_semantics_HM : ????????
        New semantic segmentation predictions for each voxel in an image

    #TODO do quality check 

    Returns
    -------
    int
        Semantic segmentation prediction including HIP/MIP for a cluster
    '''
    if len(particle.depositions)==0:return None
    HM_Pred=sparse3d_pcluster_semantics_HM[particle.index[0]:particle.index[1]]
    return stats.mode(HM_Pred)
    
def direction_acos(particle:FinalParticle, beam_dir=np.array([0.,0.,1.])) -> float:
    '''
    Returns angle between the beam-axis (here assumed in z) and the particle object's momentum

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information
    beam_dir : np.ndarray[float]
    
    Returns
    -------
    float
        Angle between particle direction and beam
    '''
    return np.arccos(np.dot(particle.start_dir,beam_dir))

def collision_distance(particle1:FinalParticle,particle2:FinalParticle):
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

#TODO angular resolution check
#TODO direction estimate check 

    
def dist_hipend_mipstart(particle:FinalParticle,hip_candidates:list[list[FinalParticle]]):
    
    '''
    Returns minimum distance between the start of a mip candidate and the end
    of every hip candidate supplied, along with the hip candidate identified.
    Returns (np.inf, None) if the mip candidate is not a mip or the hip_candidate
    list is empty

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information
    hip_candidates: List(List(spine.Particle))
        List of spine particle objects corresponding to identified kaons
    
    Returns
    -------
    [float,[spine.Particle,spine.Particle]]
        Distance from hip to mip candidate and identified kaon candidate 

    '''
    shortest_dist=[np.inf, None]
    for h in hip_candidates:
        if np.linalg.norm(h[0].end_point-particle.start_point)<shortest_dist:
            shortest_dist=np.linalg.norm(h[0].end_point-particle.start_point)
            hfinal=h
    return [shortest_dist,[hfinal,particle]]

def daughters(particle:FinalParticle,particle_list:List(FinalParticle)):
    '''
    Returns number of daughter candidates with each semantic segmentation prediction.

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information
    particle_list: List(spine.Particle)
        List of spine particle objects
    
    Returns
    -------
        np.ndarray
            shape (6) number of daughters with particular semantic segmentation prediction 
    '''
    #TODO the logic from one isn't right yet. I need to find daughters which are only from hard scattering (probably hip), etc. 
        #something like the logic in true_k_with_mu, but without truth information obviously
        #might have to do something about particles which arent close which point back
        #or particles which are connected near ends but are just the product of the busy environment
    # daughter_dist=np.inf
    # for p in particle_list:
    #     if p.id==particle.id: continue
    #     daughter_dist=np.minimum(daughter_dist,np.linalg.norm(p.position-particle.end_point))
    # return daughter_dist
    pass
    
def MIP_to_michel(michel:FinalParticle,kmupairs:list[list[FinalParticle]]):
    '''
    Returns minimum distance between the start of a michel candidate and the end
    of every mip candidate supplied, along with the hip/mip pair identified.
    Returns (np.inf, None) if the michel candidate is not a michel or the mip_candidate
    list is empty

    Parameters
    ----------
    michel : spine.Particle
        Particle object with cluster information
    kmupairs: List([spine.Particle,spine.Particle])
        List of spine particle objects corresponding to identified k/mu pairs along
    
    Returns
    -------
    [float,[spine.Particle,spine.Particle]]
        Distance from michel to mip candidate and a list of identified kaon candidate, identified muon candidate
    '''
    mindist=np.inf
    pfinal=[]
    for p in kmupairs:
        dist=np.linalg.norm(michel.start_point-p[1].end_point)
        mindist=np.minimum(mindist,dist)
        pfinal=p
    return [mindist,pfinal+[michel]]
    

def potential_lambda():
    pass
    '''
    Returns true if various checks (TBD) on a potential lambda pass
    #TODO something else to check what other particles begin near the lambda start point.
    # I only want a proton and a pion, nothing else
    Parameters
    ----------
    Returns
    -------
    '''
    return True

def lambda_decay_len(hip:FinalParticle,mip:FinalParticle,interactions:List[spine.interactions????]):
    pass
    '''
    Returns distance from average start position of hip and mip to vertex location of the assocated interaction

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    float
        distance from lambda decay point to vertex of interaction
    '''
    guess_start=(hip.start_point+mip.start_point)/2
    idx=hip.interaction_id
    return np.linalg.norm(interactions[idx].vertex-guess_start)
   
def lambda_kinematic(hip:FinalParticle,mip:FinalParticle):
    '''
    Returns lambda mass value constructed from the hip and mip candidate deposited energy and predicted direction
    #TODO it will probably be useful to use other kinematic quantities

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    float
        difference between reconstructed lambda mass and true lambda mass
    '''
    LAM_MASS=1115.60 #lambda mass in MeV
    return 2*(PROT_MASS*mip.KE+hip.KE*PION_MASS-np.dot(hip.momentum,mip.momentum))+(PROT_MASS+PION_MASS)**2-LAM_MASS**2



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
