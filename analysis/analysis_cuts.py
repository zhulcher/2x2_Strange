from spine.utils.globals import *
import numpy as np
from scipy import stats

#TODO things I would like added in truth:
    #children_id


import moduleboundaries

HIP=2
MIP=3

def is_contained(pos,eps=30):
    '''
    Checks if a position is within the defined module boundaries within some tolerance, eps

    Parameters
    ----------
    pos : np.ndarray
        (3) Vector position
    M : float/np.ndarray
        Tolerance from module boundaries

    Returns
    -------
    Bool
        Point at least eps inside of boundaries 
        '''
    for i in range(3):
        if pos[i]<moduleboundaries[i][0]+eps: return False
        if pos[i]>moduleboundaries[i][1]-eps: return False
    return True


def HIPMIP_pred(particle,sparse3d_pcluster_semantics_HM,cluster3d_pcluster):
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
    cluster3d_pcluster: ????????
        Map from particle object to voxels associated with that particle object

    Returns
    -------
    int
        Semantic segmentation prediction including HIP/MIP for a cluster
    '''

    #TODO I don't believe for a second I have the logic for this one right yet.
    #TODO I think I want to rework this to do the semantic segmentation all at once,
        # and store the value in a dictionary or something I can pass between functions that need it. 

    if len(particle.depositions)==0:return None
    shapelist=[]
    voxelmap = cluster3d_pcluster
    # truthinfo = particle_pcluster
    semantics_HM = sparse3d_pcluster_semantics_HM
    semanticsdict={}
    for i in range(len(semantics_HM.as_vector())): semanticsdict[semantics_HM.as_vector()[i].id()]=i
    for k in range(len(voxelmap.as_vector()[particle.id])):
        id=voxelmap.as_vector()[particle.id].as_vector()[k].id()
        # print(semanticsdict[id],id, semantics_HM.as_vector()[semanticsdict[id]].value())
        shapelist+=semantics_HM.as_vector()[semanticsdict[id]].value()

    return stats.mode(shapelist)

def HIP_range(particle): 
    '''
    Returns the track length for a contained particle object identified as a HIP through
    semantic segmentation prediction, with particles not satisfying these assumptions
    assigned zero range.

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information

    Returns
    -------
    float
        Range of particle
    '''
    if HIPMIP_pred(particle,etc)!=HIP: return 0
    if not particle.is_contained: return 0
    return np.linalg.norm(particle.position-particle.end_position)
    
def forwardness(particle):
    '''
    Returns 'forwardness' defined as the angle between the beam-axis (here assumed in z) and the
    particle object's momentum

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information
    
    Returns
    -------
    float
        Angle between particle direction and beam
    '''
    return np.arccos(particle.momentum[2]/np.linalg.norm(particle.momentum))
    
def dist_hipend_mipstart(particle,hip_candidates):
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
        List of spine particle objects corresponding to identified kaons, with the associated event number
    
    Returns
    -------
    (float,[int, spine.Particle])
        Distance from hip to mip candidate and a list of event number and identified kaon candidate 

    '''
    if HIPMIP_pred(particle,etc)!=MIP: return (np.inf, None)
    shortest_dist=(np.inf, None)
    for h in hip_candidates:
        if np.linalg.norm(h.end_position-particle.position)<shortest_dist:
            shortest_dist=np.linalg.norm(hip_candidates.end_position-particle.position)
            hfinal=h
    return (shortest_dist,hfinal)
   
def MIP_range(particle):
    '''
    Returns the range of a mip candidate, with uncontained particles recieving zero length

    Parameters
    ----------
    particle : spine.Particle
        Particle object with cluster information
    
    Returns
    -------
    float
        Range of particle
    '''
    if not particle.is_contained: return 0
    return np.linalg.norm(particle.end_position-particle.position)

def daughters(particle,particle_list):
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
    daughter_dist=np.inf
    for p in particle_list:
        if p.id==particle.id: continue
        daughter_dist=np.min(daughter_dist,np.linalg.norm(p.position-particle.end_position))
    return daughter_dist
    
def MIP_to_michel(michel,kmupairs):
    '''
    Returns minimum distance between the start of a michel candidate and the end
    of every mip candidate supplied, along with the hip/mip pair identified.
    Returns (np.inf, None) if the michel candidate is not a michel or the mip_candidate
    list is empty

    Parameters
    ----------
    michel : spine.Particle
        Particle object with cluster information
    kmupairs: List([int,spine.Particle,spine.Particle])
        List of spine particle objects corresponding to identified k/mu pairs along with the associated event number
    
    Returns
    -------
    (float,[int, spine.Particle,spine.Particle])
        Distance from michel to mip candidate and a list of event number,identified kaon candidate, identified muon candidate
    '''
    mindist=np.inf
    if michel.shape!=MICHL_SHP:return np.inf
    for p in kmupairs:
        dist=np.linalg.norm(michel.position-p[1].end_position)
        mindist=np.min(mindist,dist)
    return (mindist,p)
    
def true_k_with_mu(particle_list):
    '''
    Returns track_ids for kaons which are contained and which only have a muon which is both contained and full range

    Parameters
    ----------
    particle_list: List(spine.Particle)
        List of spine particle objects
    
    Returns
    -------
    List
        Shape (n) track ids for true kaons satisfying cuts
    '''
    K_pdgs={}
    for p in range(particle_list):
        if p.parent_pdg_code==321 and ((p.is_contained and abs(p.pdg)==13) or p.processid=="4::121"):
            if p.parent_id not in K_pdgs: K_pdgs[p.parent_id]=[]
            K_pdgs[p.parent_id].append(abs(p.pdg)*(p.processid!="4::121"))
    for i in list(K_pdgs.keys()):
        if set(K_pdgs[i])!=set([13]):
            K_pdgs.pop(i)
    return list(K_pdgs.keys())

def true_lambda(particle_list):
    '''
    Returns track_ids for true p/pi pairs which are both contained and full range

    Parameters
    ----------
    particle_list: List(spine.Particle)
        List of spine particle objects
    
    Returns
    -------
    List([int,int])
        List of contained pion/proton pairs which originate from true lambdas
    '''
    lambda_pdgs={}
    for p in range(particle_list):
        if p.parent_pdg_code==3122 and ((p.is_contained and abs(p.pdg) in [2212,211]) or p.processid=="4::121"):
            if p.parent_id not in lambda_pdgs: lambda_pdgs[p.parent_id]=[]
            lambda_pdgs[p.parent_id].append(abs(p.pdg)*(p.processid!="4::121"))
    for i in list(lambda_pdgs.keys()):
        if set(lambda_pdgs[i])!=set([2212,211]):
            lambda_pdgs.pop(i)
    return list(lambda_pdgs.keys())

def potential_lambda_hip(hip):
    '''
    Returns true iff a particle object is a contained predicted hip

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    
    Returns
    -------
    bool
        True iff particle is a contained predicted hip
    '''
    if not hip.is_contained: return False
    if HIPMIP_pred(hip,etc)!=HIP: return False
    return True

def potential_lambda_mip(mip):
    '''
    Returns true iff a particle object is a contained predicted mip

    Parameters
    ----------
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    bool
        True iff particle is a contained predicted mip
    '''
    if not mip.is_contained: return False
    if HIPMIP_pred(mip,etc)!=MIP: return False
    return True

def potential_lambda(hip,mip):
    '''
    Returns distance from hip candidate start to mip candidate start

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    
    Returns
    -------
    float
        distance from start point of two given particles
    '''
    return np.linalg.norm(hip.position-mip.position)

#TODO something else to check what other particles begin near the lambda start point.
# I only want a proton and a pion, nothing else
    
def lambda_decay_len(hip,mip):
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
    guess_start=(hip.position+mip.position)/2
    neutrino_int=(neutrino associated to interaction/hip/mip).position
    return np.linalg.norm(guess_start-neutrino_int)
   
def lambda_kinematic(hip,mip):
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
    return 2*(PROT_MASS*mip.KE+hip.KE*PION_MASS-np.dot(hip.momentum,mip.momentum))+(PROT_MASS+PI_MASS)**2-LAM_MASS**2
