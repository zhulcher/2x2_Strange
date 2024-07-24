from spine.utils.globals import *
import numpy as np
from scipy import stats

#things I would like added in truth:
    #parent_pdg
    #children_id


# import moduleboundaries

HIP=2
MIP=3

# def is_contained(pos,eps):

#     for i in range(3):
#         if pos[i]<moduleboundaries[i][0]+eps: return False
#         if pos[i]>moduleboundaries[i][1]-eps: return False
#     return True

    #if this is not defined per particle, check if a position is within the detector bounds


def HIPMIP_pred(particle,sparse3d_pcluster_semantics_HM,cluster3d_pcluster):

    #returns a semantic segmentation prediction for a cluster bc not guaranteed unique
    #majority vote
    #I don't want to write this one seriously til I can look at a file.
    #Strategy:
        #I need to read in pcluster_semantics_HM structure and cluster3d_pcluster
            #The Hip/Mip semantics per voxel are stored in the pcluster_semantics_HM structure

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
    #returns HIP length if HIP candidate which satisfying containment cuts
    #returns 0 if not HIP/ not contained
    if HIPMIP_pred(particle,etc)!=HIP: return 0
    if not particle.is_contained: return 0
    return np.linalg.norm(particle.position-particle.end_position)
    
def forwardness(particle):
    #assumes beam in z
    #for cut on forwardness of kaon in detector? dunno yet
    #returns angle between beam direction and kaon
    return np.arccos(particle.momentum[2]/np.linalg.norm(particle.momentum))
    
def dist_hipend_mipstart(particle,hip_candidates):
     #if particle is not a MIP, return inf
    #return distance to nearest hip end
    if HIPMIP_pred(particle,etc)!=MIP: return np.inf
    shortest_dist=np.inf,None
    hfinal=None
    for h in hip_candidates:
        if np.linalg.norm(h.end_position-particle.position)<shortest_dist:
            shortest_dist=np.linalg.norm(hip_candidates.end_position-particle.position)
            hfinal=h
    return shortest_dist,hfinal.trackid
   
def MIP_range(particle):
    #if not contained, return 0
    #returns mip range
    if not particle.is_contained: return 0
    return np.linalg.norm(particle.end_position-particle.position)

def extra_daughters(particle,particle_list):
    #TODO this one isn't right. I need to find daughters which are only from hard scattering, etc. 
    #something like the logic in true_k_with_mu
    #I need help thinking how to associate daughters from a possible hard scattering, etc?
    #as a first pass return range to start of nearest daughter which isnt the associated MIP
    #might have to do something about particles which arent close which point back or (not now, eventually, im not looking at this now) do some more complicated parentage prediction like a directed GNN to catch photons or longer neutrals
    #or particles which are connected but are just the product of the busy environment
    daughter_dist=np.inf
    for p in particle_list:
        if p.id==particle.id: continue
        daughter_dist=np.min(daughter_dist,np.linalg.norm(p.position-particle.end_position))
    return daughter_dist
    
def MIP_to_michel(michel,kmupairs):
    #returns distance from MIP end to start of nearest michel
    mindist=np.inf
    if michel.shape!=MICHL_SHP:return np.inf
    for p in kmupairs:
        dist=np.linalg.norm(michel.position-p[1].end_position)
        mindist=np.min(mindist,dist)
    return mindist,p
    
def true_k_with_mu(particle_list):
    #returns true if cluster is a contained muon with true kaon as parent
    K_pdgs={}
    for p in range(particle_list):
        if p.parent_pdg==321 and ((p.is_contained and abs(p.pdg)==13) or p.processid=="4::121"):
            if p.parent_id not in K_pdgs: K_pdgs[p.parent_id]=[]
            K_pdgs[p.parent_id].append(abs(p.pdg)*(p.processid!="4::121"))
    for i in list(K_pdgs.keys()):
        if set(K_pdgs[i])!=set([13]):
            K_pdgs.pop(i)
    return list(K_pdgs.keys())

def true_lambda(particle_list):
    #returns pdgcodes for true lambda (pi+p) contained pairs
    #if particle.parent_pdg==3122: return particle.parent_id
    lambda_pdgs={}
    for p in range(particle_list):
        if p.parent_pdg==3122 and ((p.is_contained and abs(p.pdg) in [2212,211]) or p.processid=="4::121"):
            if p.parent_id not in lambda_pdgs: lambda_pdgs[p.parent_id]=[]
            lambda_pdgs[p.parent_id].append(abs(p.pdg)*(p.processid!="4::121"))
    for i in list(lambda_pdgs.keys()):
        if set(lambda_pdgs[i])!=set([2212,211]):
            lambda_pdgs.pop(i)
    return list(lambda_pdgs.keys())

def potential_lambda_hip(hip):
    #check if a contained hip
    if not hip.is_contained: return False
    if HIPMIP_pred(hip,etc)!=HIP: return False
    return True

def potential_lambda_mip(mip):
    #check if a contained mip
    if not mip.is_contained: return False
    if HIPMIP_pred(mip,etc)!=MIP: return False
    return True

def potential_lambda(hip,mip):
    # returns distance from hip candidate start to mip candidate start
    return np.linalg.norm(hip.position-mip.position)

#TODO something else to check what other particles begin near the lambda start point.
# I only want a proton and a pion, nothing else
    
    
def lambda_decay_len(hip,mip):
    #return distance from some notion of mutual start point to vertex
    guess_start=(hip.position+mip.position)/2
    neutrino_int=(neutrino associated to interaction/hip/mip).position
    return np.linalg.norm(guess_start-neutrino_int)
   

def lambda_kinematic(hip,mip):
    #other kinematic values, i.e. Lambda Kaon separation observable
    LAM_MASS=1115.60 #lambda mass in MeV
    return 2*(PROT_MASS*mip.KE+hip.KE*PION_MASS-np.dot(hip.momentum,mip.momentum))+(PROT_MASS+PI_MASS)**2-LAM_MASS**2
