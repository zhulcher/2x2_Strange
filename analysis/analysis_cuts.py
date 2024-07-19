def HIPMIP_pred(voxels,sparse3d_pcluster_semantics_HM_or_something_like_that):
    return
    #returns a semantic segmentation prediction for a cluster bc not guaranteed unique
    #majority vote

def HIP_candidate(particle): 
    return 
    #returns true if HIP candidate which satisfying containment cuts, nothing fancy
    #cut on forwardness of kaon in detector? dunno yet

def MIP_candidate(particle,hip_candidates,range=[400,600]): 
    return
    #returns true if MIP candidate if a HIP candidate is their mother
        #mother determined as start point of MIP epsilon away from end point of MIP (or identified)
    #cuts MIPS with range outside of set bounds
    #cuts candidates with any extra daughters associated to their parent
    #also cuts on containment of daughter particle

def MIP_michel(particle,hip_candidates,range=[400,600]):
    return 
    #returns true if MIP candidates above has a single electron daughter of this MIP
    #parentage determined as start point of daughter epsilon away from end point of Mother (or identified)
    

def true_k_with_mu(particle):#returns true if cluster is a contained muon with true kaon as parent
    return

def true_lambda(particle): #returns identifiers for true lambdas
    return

def potential_lambda(particle):
    return
    # returns back to back hip mip with
        #the connection point separated from the vertex of the event
        #the hip and mip full range
        #some mass or kinematic cut I need to workshop