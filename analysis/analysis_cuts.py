from collections import Counter
import spine
from spine.utils.globals import PROT_MASS, PION_MASS
from spine.utils.geo.base import Geometry
from spine.utils.vertex import get_pseudovertex
import numpy as np
from scipy import stats as st


# TODO things I would like added in truth:
# spine.TruthParticle.children_id for ease of use
# truth length so truthparticle.length works
# spine.TruthParticle.mass propagated to truth particles in larcv
# spine.TruthParticle.parent_end_momentum
# every particle's spine.TruthParticle.end_momentum
# spine.TruthParticle.reco_momentum
# spine.TruthParticle.reco_ke
# spine.TruthParticle.reco_end_momentum
# spine.RecoParticle.end_momentum

# TODO things I would like at some point:
# some sort of particle flow predictor

# TODO things I would like fixed:
# michel timing issue

# TODO things I don't know that I need but may end up being useful
# Kaon/ Michel flash timing?

HIP_HM = 5
MIP_HM = 1
SHOWR_HM = 0


Particle = spine.RecoParticle | spine.TruthParticle
Interaction = spine.RecoInteraction | spine.TruthInteraction
Met = spine.Meta
Geo = Geometry(detector="2x2")


class PotK:
    """
    Storage class for primary Kaons and their cut parameters

    Attributes
    ----------
    hip_id : int
        id for hip associated to this class
    hip_len: float
        length attribute of the particle object
    dir_acos: float
       arccos of the particle direction with the beam direction
    HM_acc_K:float
        percent of the voxels of this particle
        whose Hip/Mip semantic segmentation matches the overall prediction
    """

    hip_id: int
    hip_len: float
    dir_acos: float
    HM_acc_K: float

    def __init__(self, hip_id, hip_len, dir_acos, HM_acc_K):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        hip_id : int
            id for hip associated to this class
        hip_len: float
            length attribute of the particle object
        dir_acos: float
            arccos of the particle direction with the beam direction
        HM_acc_K:float
            percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
        """

        self.hip_id = hip_id
        self.hip_len = hip_len
        self.dir_acos = dir_acos
        self.HM_acc_K = HM_acc_K

    def apply_cuts_K(self):
        pass

    # def output(self):
    #     return [self.hip_id,self.hip_len,self.dir_acos]


class PredK(PotK):
    """
    Storage class for primary Kaons with muon child and their cut parameters

    Attributes
    ----------
    mip_id : int
        id for hip associated to this class
    mip_len: float
        length attribute of the particle object
    dist_to_hip: float
       distance from mip start to hip end
    K_extra_children: list[float]
        extra children parameters as defined in 'children' function
    HM_acc_mu:float
        percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
    """

    mip_id: int
    mip_len: float
    dist_to_hip: float
    K_extra_children: list[float]
    HM_acc_mu: float

    def __init__(
        self,
        pot_k: PotK,
        mip_id: int,
        mip_len: float,
        dist_to_hip: float,
        K_extra_children: list[float],
        HM_acc_mu,
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
            length attribute of the particle object
        dist_to_hip: float
            distance from mip start to hip end
        K_extra_children: list[float]
            extra children parameters as defined in 'children' function for the hip
        HM_acc_mu:float
            percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
        """
        self.hip_id = pot_k.hip_id
        self.hip_len = pot_k.hip_len
        self.dir_acos = pot_k.dir_acos
        self.HM_acc_K = pot_k.HM_acc_K

        self.mip_id = mip_id
        self.mip_len = mip_len
        self.dist_to_hip = dist_to_hip
        self.K_extra_children = K_extra_children
        self.HM_acc_mu = HM_acc_mu

    def apply_cuts_mu(self):
        pass

    # def output(self):
    #     return [self.hip_id,self.hip_len,self.dir_acos,self.mip_id,self.mip_len,self.dist_to_hip,self.K_extra_children]


class PredK_Mich(PredK):
    """
    Storage class for primary Kaons with muon child and michel and their cut parameters

    Attributes
    ----------
    mich_id : int
        id for hip associated to this class
    dist_to_mich: float
       distance from mich start to mip end
    mu_extra_children: list[float]
        extra children parameters as defined in 'children' function for the mip
    HM_acc_mich:float
        percent of the voxels of this particle whose Hip/Mip semantic segmentation matches the overall prediction
    """

    mich_id: int
    dist_to_mich: float
    mu_extra_children: list[float]
    # decay_t_to_dist: float
    HM_acc_mich: float

    def __init__(
        self, pred_k: PredK, mich_id, dist_to_mich, mu_extra_children, HM_acc_mich
    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        pred_K:PredK
            PredK object with associated hip and mip information
        mich_id : int
            id for hip associated to this class
        dist_to_mich: float
        distance from mich start to mip end
        mu_extra_children: list[float]
            extra children parameters as defined in 'children' function for the mip
        HM_acc_mich:float
            percent of the voxels of this particle whose Hip/Mip semantic segmentation
            matches the overall prediction
        """

        self.hip_id = pred_k.hip_id
        self.hip_len = pred_k.hip_len
        self.dir_acos = pred_k.dir_acos
        self.HM_acc_K = pred_k.HM_acc_K

        self.mip_id = pred_k.mip_id
        self.mip_len = pred_k.mip_len
        self.dist_to_hip = pred_k.dist_to_hip
        self.K_extra_children = pred_k.K_extra_children
        self.HM_acc_mu = pred_k.HM_acc_mu

        self.mich_id = mich_id
        self.dist_to_mich = dist_to_mich
        self.mu_extra_children = mu_extra_children
        self.HM_acc_mich = HM_acc_mich

        # self.decay_t=decay_t
        # self.decay_sep=decay_sep

    def apply_cuts_mich(self):
        pass

    # def output(self):
    #     return [self.hip_id,self.hip_len,self.dir_acos,self.mip_id,self.mip_len,self.dist_to_hip,self.K_extra_children,self.mich_id,self.dist_to_mich,self.mu_extra_children]


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
        length attribute of the hip object
    mip_len: float
        length attribute of the mip object
    VAE: float
        angle between the line constructed from the momenta of the hip and mip and
        the line constructed from the interaction vertex and the lambda decay point
    lam_mass2:float
        reconstructed mass squared of the Lambda
    lam_decay_len: float
        decay length of the lambda from the associated vertex
    momenta: list[float]
        shape(4) [hip transv. momentum, mip transv. momentum,hip long momentum, mip long momentum]
    coll_dist: float
        shape(3) [t1,t2, dist]: the distance from the start point to the
        point along the vector start direction which is the point of
        closest approach to the other particle's corresponding line for the hip (t1) and mip (t2)
        along with the distance of closest approach of these lines (dist)
    dir_acos_L:float
        arccos of the Lambda direction with the beam direction
    HM_acc_prot:float
        percent of the voxels for the hip whose
        Hip/Mip semantic segmentation matches the overall prediction
    HM_acc_pi:float
        percent of the voxels for the mip whose
        Hip/Mip semantic segmentation matches the overall prediction
    prot_extra_children:list[float]
        extra children parameters as defined in 'lambda_children' function for the Lambda
    Pi_extra_children:list[float]
        extra children parameters as defined in 'children' function for the mip
    lam_extra_children:list[float]
        extra children parameters as defined in 'children' function for the hip
    """

    hip_id: int
    mip_id: int
    hip_len: float
    mip_len: float
    VAE: float
    lam_mass2: float
    lam_decay_len: float
    momenta: list[float]
    coll_dist: list[float]
    dir_acos_L: float
    HM_acc_prot: float
    HM_acc_pi: float
    prot_extra_children: list[float]
    Pi_extra_children: list[float]
    lam_extra_children: list[float]

    def __init__(
        self,
        hip_id,
        mip_id,
        hip_len,
        mip_len,
        VAE,
        lam_mass2,
        lam_decay_len,
        momenta: list[float],
        coll_dist,
        lam_extra_children,
        prot_extra_children,
        Pi_extra_children,
        dir_acos_L,
        HM_acc_prot,
        HM_acc_pi,
    ):
        """
        Initialize with all of the necessary particle attributes

        Parameters
        ----------
        hip_id: int
            id for hip child
        mip_id: int
            id for mip child
        hip_len: float
            length attribute of the hip object
        mip_len: float
            length attribute of the mip object
        VAE: float
            angle between the line constructed from the momenta of the hip and mip and
            the line constructed from the interaction vertex and the lambda decay point
        lam_mass2:float
            reconstructed mass squared of the Lambda
        lam_decay_len: float
            decay length of the lambda from the associated vertex
        momenta: list[float]
            shape(4) [hip transverse momentum, mip transverse momentum,hip long momentum, mip long momentum]
        coll_dist: list[float]
            shape(3) [t1,t2, dist]: the distance from the start pointto the point along the vector start direction
            which is the point of closest approach to the other particle's corresponding line for the hip (t1) and mip (t2)
            along with the distance of closest approach of these lines (dist)
        dir_acos_L:float
            arccos of the Lambda direction with the beam direction
        HM_acc_prot:float
            percent of the voxels for the hip whose Hip/Mip semantic segmentation matches the overall prediction
        HM_acc_pi:float
            percent of the voxels for the mip whose Hip/Mip semantic segmentation matches the overall prediction
        prot_extra_children:list[float]
            extra children parameters as defined in 'lambda_children' function for the Lambda
        Pi_extra_children:list[float]
            extra children parameters as defined in 'children' function for the mip
        lam_extra_children:list[float]
            extra children parameters as defined in 'children' function for the hip
        """
        self.hip_id = hip_id
        self.mip_id = mip_id
        self.hip_len = hip_len
        self.mip_len = mip_len
        self.VAE = VAE
        self.lam_mass2 = lam_mass2
        self.lam_decay_len = lam_decay_len
        self.momenta: list[float] = momenta
        self.coll_dist = coll_dist
        self.lam_extra_children = lam_extra_children
        self.prot_extra_children = prot_extra_children
        self.Pi_extra_children = Pi_extra_children
        self.dir_acos_L = dir_acos_L
        self.HM_acc_prot = HM_acc_prot
        self.HM_acc_pi = HM_acc_pi

    def apply_cuts_K(self):
        pass

    # def output(self):
    #     return [self.hip_id,self.mip_id,self.VAE,self.lam_mass2,self.lam_decay_len,self.momenta,self.coll_dist,self.extra_children]


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


# TODO is contained unit test a million points each


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


def HM_score(particle: Particle, sparse3d_pcluster_semantics_HM: np.ndarray) -> float:
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

    return np.arccos(np.dot(momenta, direction))


def collision_distance(particle1: Particle, particle2: Particle):
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
    v1 = particle1.start_dir
    v2 = particle2.start_dir

    p1 = particle1.start_point
    p2 = particle2.start_point

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


def dist_end_start(
    particle: Particle, parent_candidates: list[Particle]
) -> list[list[float]]:
    """
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

    """
    out = []
    for n in range(len(parent_candidates)):
        out += [
            [
                float(
                    np.linalg.norm(
                        parent_candidates[n].end_point - particle.start_point
                    )
                ),
                n,
            ]
        ]
    return out


def is_child_eps_angle(parent_end: np.ndarray, child: Particle) -> tuple[float, float]:
    """
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
    """
    true_dir = child.start_point - parent_end
    separation = float(np.linalg.norm(true_dir))
    if separation == 0:
        angle = 0
    else:
        angle = np.arccos(np.dot(true_dir, child.start_dir) / separation)
    return (separation, angle)


def children(
    parent: Particle, particle_list: list[Particle], ignore: list[int]
) -> list[float]:
    """
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
    """
    children = [np.inf, np.inf]
    for p in particle_list:
        if p.id in ignore:
            continue
        is_child = is_child_eps_angle(parent.end_point, p)
        children[0] = min(children[0], is_child[0])
        children[1] = min(children[1], is_child[1])
    return children


def lambda_children(
    hip: Particle, mip: Particle, particle_list: list[Particle]
) -> list[float]:
    """
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
    """
    children = []
    guess_start = get_pseudovertex(
        start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        directions=[hip.start_dir, mip.start_dir],
    )
    children = [np.inf, np.inf]
    for p in particle_list:
        if p.id in (hip.id, mip.id):
            continue
        is_child = is_child_eps_angle(guess_start, p)
        children[0] = min(children[0], is_child[0])
        children[1] = min(children[1], is_child[1])
    return children


def lambda_decay_len(
    hip: Particle, mip: Particle, interactions: list[Interaction]
) -> float:
    """
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
    """
    guess_start = get_pseudovertex(
        start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        directions=[hip.start_dir, mip.start_dir],
    )
    idx = hip.interaction_id
    return float(np.linalg.norm(interactions[idx].vertex - guess_start))


def momenta_projections(
    hip: Particle, mip: Particle, interactions: list[Interaction]
) -> list[float]:
    """
    Returns the P_T and P_L of each particle relative to the lambda measured from the decay

    Parameters
    ----------
    hip: spine.Particle
        spine particle object
    mip: spine.Particle
        spine particle object
    interactions: list[Interaction]
        list of interactions

    Returns
    -------
    list[float]
        shape(4) [hip transverse momentum, mip transverse momentum,hip long momentum, mip long momentum]
    """

    inter = interactions[hip.interaction_id].vertex

    guess_start = get_pseudovertex(
        start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        directions=[hip.start_dir, mip.start_dir],
    )
    lam_dir = guess_start - inter

    p1 = hip.momentum
    p2 = mip.momentum

    lam_dir = p1 + p2  # fix this hack #TODO

    lam_dir_norm = np.linalg.norm(lam_dir)
    if lam_dir_norm == 0:
        return [np.nan, np.nan, np.nan, np.nan]

    lam_dir = lam_dir / lam_dir_norm

    p1_long = np.dot(lam_dir, p1)
    p2_long = np.dot(lam_dir, p2)

    p1_transv = float(np.linalg.norm(p1 - p1_long * lam_dir))
    p2_transv = float(np.linalg.norm(p2 - p2_long * lam_dir))

    return [p1_transv, p2_transv, p1_long, p2_long]


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


def lambda_mass_2(hip: Particle, mip: Particle) -> float:
    """
    Returns lambda mass value constructed from the
    hip and mip candidate deposited energy and predicted direction

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
    """
    # LAM_MASS=1115.60 #lambda mass in MeV
    assert (
        mip.ke > 0
    )  # print(mip.ke,"very bad",mip.id,mip.parent_pdg_code,mip.pid,mip.pdg_code,mip.energy_init)
    assert (
        hip.ke > 0
    )  # print(hip.ke,"very bad",hip.id,hip.parent_pdg_code,hip.pid,hip.pdg_code,hip.energy_init)

    lam_mass2 = (
        PROT_MASS**2
        + PION_MASS**2
        + 2 * (mip.ke + PION_MASS) * (hip.ke + PROT_MASS)
        - 2 * np.dot(hip.momentum, mip.momentum)
    )

    # if lam_mass2<0 or lam_mass2>2*1115.6**2: print(lam_mass2,hip.parent_track_id,mip.parent_track_id,hip.parent_pdg_code,mip.ke,hip.ke,np.dot(hip.momentum,mip.momentum),[hip.pdg_code,hip.id,hip.creation_process],[mip.pdg_code,mip.id,mip.creation_process])
    return lam_mass2


def vertex_angle_error(
    hip: Particle, mip: Particle, interactions: list[Interaction]
) -> float:
    """
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
    """

    inter = interactions[hip.interaction_id].vertex
    guess_start = get_pseudovertex(
        start_points=np.array([hip.start_point, mip.start_point], dtype=float),
        directions=[hip.start_dir, mip.start_dir],
    )
    lam_dir1 = guess_start - inter

    lam_dir2 = hip.momentum + mip.momentum

    if np.linalg.norm(lam_dir1) == 0 or np.linalg.norm(lam_dir2) == 0:
        return np.nan
    ret = np.arccos(
        np.dot(lam_dir1, lam_dir2) / np.linalg.norm(lam_dir1) / np.linalg.norm(lam_dir2)
    )
    assert ret == ret
    return ret


# def true_k_with_mu(particle_list):
#     '''
#     Returns track_ids for kaons which are contained
#     and which only have a muon which is both contained and full range

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
