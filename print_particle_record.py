from larcv import larcv
import tempfile

import sys
import numpy as np
import os
from analysis.analysis_cuts import *

from collections import defaultdict,Counter
import copy
'''



The way to run this script is

python3 add_HIPMIP.py "input_file.larcv.root" "output_file.larcv.root"

'''


latextoplot={
        130:r'$K^0_L$',
        310:r'$K^0_S$',
        311:r'$K^0$',
        321:r'$K^+$',
        # -321:r'$K^-$',
        3122:r'$\Lambda^0$',
        3222:r'$\Sigma^+$',
        3212:r'$\Sigma^0$',
        3112:r'$\Sigma^-$',
        3322:r'$\Xi^0$',
        3312:r'$\Xi^-$',
        3334:r'$\Omega^-$',
        221:r'$\eta$',
        331:r'$\eta^{\prime}$',
        431:r'$D^+_s$',
        333:r'$\phi(1020)$',
        433:r'$D_s^{*+}$',
        313:r'$K^*(892)^0$',
        323:r'$K^*(892)^+$',
        9000311:r'$K_0^*(800)^0$',
        9000321:r'$K_0^*(800)^+$',
        531:r'$B_s^0$',
        533:r'$B_s^{*0}$',
        100333:r'$\phi(1680)$',
        3214:r'$\Sigma^{*0}$',
        3114:r'$\Sigma^{*-}$',
        3224:r'$\Sigma^{*+}$',
        3324:r'$\Xi^{*0}$',
        3314:r'$\Xi^{*-}$',
        # 13:r'$\mu$',
        # 111:r'$\pi^0$',
        # 211:r'$\pi^+$',
        # 2212:r'$p$',
        # 22:r'$\gamma$',
        # 11:r'$e^-$',
        # 411:r'$D^+$',
        # 421:r'$D^0$',
        # 2112:r'$n$',
}

def merge_dicts(*dicts):
    merged_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            merged_dict[key].extend(value)  # Extend the list for shared keys
    return merged_dict


def point_to_line_distance(p1, v, p2):
    p1 = np.array(p1)
    v = np.array(v)
    p2 = np.array(p2)

    diff = p2 - p1
    cross = np.cross(diff, v)
    distance = np.linalg.norm(cross) / np.linalg.norm(v)
    return distance


def load_cfg(cfg):
    """"
    Prepare larcv io for a given config string.

    Parameters
    ----------
    cfg: str
        Config string for larcv io
    
    Returns:
    io: larcv.IOManager
        larcv io interface
    """
    
    tmp = tempfile.NamedTemporaryFile('w')
    tmp.write(cfg)
    tmp.flush()

    io = larcv.IOManager(tmp.name)
    io.initialize()

    return io

# def make_labels(col1, labels):
#     flags = []
#     labels.meta(col1.meta())
#     for v1 in col1.as_vector():
#         label = larcv.Voxel(v1.id(), v1.value())
#         labels.insert(label)
#     return flags

if __name__ == "__main__":
    interesting_pdgs=list(latextoplot.keys())
    interesting_pdgs+=[-i for i in interesting_pdgs]
    interesting_pdgs=list(set(interesting_pdgs))
    print("looking for",interesting_pdgs)
    # final_lambda=[]
    # final_kp=[]
    # final_km=[]
    # final_k0s=[]
    # final_sig0s=[]
    # final_sigps=[]
    # finalsigms=[]
    
    
    pdg_dict=defaultdict(list)
    neutrinos=[[],[]]
    final_interactions=defaultdict(list)
    # final_particles=defaultdict(list)

    print("attempting to read",sys.argv[2])

    cfg = '''
        IOManager: {
            Verbosity    : 2
            Name         : "MainIO"
            IOMode       : 0
            InputFiles   : [%s]
        }
        '''% sys.argv[2]
    #"MiniRun5_1E19_RHC.flow.0000001.larcv.root"
    # sys.argv[1]
    io = load_cfg(cfg)

    n_evts = io.get_n_entries()#min(n_max, 10000)


    # outcfg= '''
    #     IOManager: {
    #         Verbosity    : 2
    #         Name         : "OUTIO"
    #         IOMode       : 2
    #         InputFiles   : [%s]
    #         OutFileName  : %s
    #     }
    #     '''% (sys.argv[1], sys.argv[2])
    # #"output_HM-larcv.root"
    # io_out = load_cfg(outcfg)





    print(n_evts)

    for i_entry in range(n_evts):
        print(i_entry)
        if i_entry % 100 == 0:
            print('Processing', i_entry)

        # io_out.read_entry(i_entry)
        
        # j_entry = (i_entry + 1) % io.get_n_entries()
        io.read_entry(i_entry)

        # =openroot.Get("_pcluster_tree")
        # truthinfo=openroot.Get("particle_pcluster_tree")
        # voxelmap = io.get_data('cluster3d', 'pcluster')
        truthinfo_old = io.get_data('particle', 'pcluster')
        truthinfo = io.get_data('particle', 'mpv')
        neutrinoinfo = io.get_data('neutrino', 'mpv')
        # semantics = io.get_data('sparse3d', 'pcluster_semantics')
        # semantics2 = io.get_data('sparse3d', 'pcluster_lowE')

        # semantics_HM = io_out.get_data('sparse3d', 'pcluster_semantics_HM')

        # semanticsdict={}
        # semantics_HM=copy.deepcopy(semantics)

        # make_labels(semantics, semantics_HM)

        # print("dear god help me",len(semantics_HM.as_vector()))
        # for i in range(len(semantics2.as_vector())): semanticsdict[semantics2.as_vector()[i].id()]=i
            # raise Exception(dir(semantics.as_vector()[i]),semantics.as_vector()[i].id(),semantics.as_vector()[i].value())
            
        # print(len(truthinfo))
        # print(len(truthinfo_old))
        # raise Exception(set(semanticsdict.values()))
        
        # break
        # lambdas=[]
        # kaons=[]
        # k0s=[]
        # sig0s=[]
        # final_lambda=[]
        # final_kp=[]
        # final_km=[]
        # final_k0s=[]
        # final_sig0s=[]
        # final_sigps=[]
        # ginalsigms=[]
        # particles=defaultdict(list)#defaultdict(lambda: defaultdict(list))
        neut=neutrinoinfo.as_vector()
        for j in neut:
            if not is_contained([j.position().x(),j.position().y(),j.position().z()],margin=0): continue
            neutrinos[0]+=[j.energy_init()]
            neutrinos[1]+=[[j.position().x(),j.position().y(),j.position().z(),j.position().t()]]
            # print(is_contained([j.position().x(),j.position().y(),j.position().z()],margin=0),is_contained([j.position().x(),j.position().y(),j.position().z()],mode='detector',margin=0))
            # print([j.position().x(),j.position().y(),j.position().z()])
        print("number of neut",len(neutrinos[0]))
        # assert len(neutrinoinfo.as_vector())==1,len(neutrinoinfo.as_vector())

        # pdg_list=[]

        pos_to_int_id={}

        interactions=defaultdict(list)

        
                # 
                # for k in range(len(truthinfo_old)):
                
                #     y=truthinfo_old.as_vector()[k]
                #     if y.ancestor_pdg_code() in [3122,321] and pos2==pos:



            # if x.pdg_code() in [321]:
            #     print("found one", x.pdg_code())
            #     # print("found one", x.pdg_code())
            #     kaons+=[val]

            # if x.pdg_code() in [310,311]:
            #     print("found one", x.pdg_code())
            #     k0s+=[val]

                
                # pos=[x.position().x(),x.position().y(),x.position().z(),x.position().t()]
                # for k in range(len(truthinfo_old)):
        # for i in list(interactions.keys()):
        #     if 123456789 not in interactions[i] and -123456789 not in interactions[i]:
        #         interactions.pop(i)
        # if len(lambdas)==0 and len(kaons)==0 and len(k0s)==0: continue
        # if len(particles)==0: continue
        # for i in particles:
        #     if len(particles[i])>1: raise Exception()
        
        # if
        # if False:

        pospdg_2_int_id={}
        pospdg_2_track_id={}

        for k in range(len(truthinfo_old)):

            
            
            y=truthinfo_old.as_vector()[k]
            pdg=y.ancestor_pdg_code()

            

            #
            pos2=(y.ancestor_position().x(),y.ancestor_position().y(),y.ancestor_position().z(),y.ancestor_position().t())
            if not is_contained(pos2[:3],margin=0): continue
            if pos2[1]==2000 and abs(pdg) in [321,130]: continue

            
            pospdg_2_int_id[pos2]=y.interaction_id()

            key=(i_entry,y.interaction_id())

            

            if len(interactions[key])==0:
                interactions[key]=[pos2,defaultdict(list),None,None,None,None]

            # print("secondaries:",y.nu_current_type(),y.pdg_code())
            
            if pdg in interesting_pdgs:
                pospdg_2_track_id[(pos2,y.ancestor_pdg_code())]=y.ancestor_track_id()

                # if pdg==-11:
                    # if 


                

                
                # assert pos2 in pos_to_int_id,(pos_to_int_id.keys(),pos2)
                
                # assert abs(y.interaction_id())<2**30
                
                # pos3=[y.position().x(),y.position().y(),y.position().z()]
                # pos4=[y.end_position().x(),y.end_position().y(),y.end_position().z()]


            # for pdg in interesting_pdgs:
            
                # if y.ancestor_pdg_code()==pdg:
                assert key in interactions, (key,interactions.keys(),pos2,y.ancestor_pdg_code(),pdg)

                if y.creation_process()=='Decay' and y.parent_pdg_code()==pdg:
                    # and ( or (y.ancestor_pdg_code()==pdg and y.parent_pdg_code() in [-13,211] and y.pdg_code()==-11))
                    s_pos=y.position()
                    e_pos=y.end_position()
                    interactions[key][1][(y.ancestor_track_id(),y.ancestor_pdg_code())]+=[(y.pdg_code(),y.p(),y.track_id(),[[s_pos.x(),s_pos.y(),s_pos.z()],[e_pos.x(),e_pos.y(),e_pos.z()]])]
                    print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)
                    # raise Exception("something else afoot",interactions,pos2,y.ancestor_pdg_code(),pdg)

        print("interactions post secondaries",interactions)
        for j in range(len(truthinfo)):
            # raise Exception(dir(neutrinoinfo.as_vector()[j]))

            
            
            x=truthinfo.as_vector()[j]
            # if x.pdg_code() not in interesting_pdgs:
            #     continue

            if abs(x.pdg_code()) not in [12,14]+interesting_pdgs: continue
            
            
            pos=(x.position().x(),x.position().y(),x.position().z(),x.position().t())

            if abs(x.pdg_code()) in [12,14] and pos not in pospdg_2_int_id: continue

            # if abs(x.pdg_code())
            #     pos=(x.position().x(),x.position().y(),x.position().z(),x.position().t())

            # pos_to_int_id[pos]=x.interaction_id()
            if not is_contained([x.position().x(),x.position().y(),x.position().z()],margin=0): continue

            if pos[1]==2000 and abs(x.pdg_code()) in [321,130]: continue

            assert pos in pospdg_2_int_id,(pos,x.pdg_code(),pospdg_2_int_id.keys())
            intid=pospdg_2_int_id[pos]

            key=(i_entry,intid)

            


            # if key not in interactions:
            # assert x.nu_current_type() in [0,1]
            if abs(x.pdg_code()) in [12,14] and x.nu_current_type() in [0,1]:
                # interactions[key][0]=pos
                interactions[key][-4]=x.nu_current_type()
                interactions[key][-3]=x.nu_interaction_type()
                interactions[key][-2]=x.creation_process()
                interactions[key][-1]=x.p()

                # interactions[key]=[pos,{},x.nu_current_type(),x.nu_interaction_type(),x.creation_process(),x.p()]

            # interactions[key]+=[x.pdg_code()]
            # anc_pos=[x.ancestor_position().x(),x.ancestor_position().y(),x.ancestor_position().z(),x.ancestor_position().t()]
            # if x.nu_current_type():
            # pdg_list+=[[x.nu_current_type(),x.pdg_code()]]
            
                # interactions[key][2]=x.nu_current_type()
                # interactions[key][3]=x.nu_interaction_type()
                # interactions[key]+=[123456789*int(-.5)*2)]
            
            # interactions[pos]+=[[]]
            
            # val=[x.p(),pos,[],x.creation_process(),, x.pdg_code(),True,x.track_id()]

            if x.pdg_code() not in interesting_pdgs:
                continue

            if x.pdg_code() not in [z[1] for z in pospdg_2_track_id]:
                pospdg_2_track_id[(pos,x.ancestor_pdg_code())]=x.track_id()

            
            assert (pos,x.ancestor_pdg_code()) in pospdg_2_track_id,(pospdg_2_track_id.keys(),(pos,x.ancestor_pdg_code()))

            atid=pospdg_2_track_id[(pos,x.ancestor_pdg_code())]

            new_key=(atid,x.pdg_code())

            # assert set([len(z) for z in interactions[key][1]])==set([2]),interactions[key][1].keys()
            if x.pdg_code() not in [z[1] for z in interactions[key][1]]:
                interactions[key][1][new_key]=[]

            #and Counter([z for z in truthinfo.as_vector() if z.pdg_code()==x.pdg_code()])[x.pdg_code()]==1
            if new_key not in interactions[key][1] and Counter([z[1] for z in interactions[key][1]])[x.pdg_code()]==1 :
                for old_key in interactions[key][1]:
                    if x.pdg_code()==old_key[1]:# and old_key[0]>10000:
                        new_key=copy.deepcopy(old_key)

                # if x.pdg_code() in [z[1] for z in interactions[key][1]]

            
            assert new_key in interactions[key][1],(new_key,interactions[key],[(z.pdg_code(),z.p(),z.track_id()) for z in truthinfo.as_vector() if z.pdg_code()==x.pdg_code()])

            
            print("adding",x.pdg_code(),x.p())
            pdg_dict[x.pdg_code()]+=[x.p()]

            s_pos=x.position()
            e_pos=x.end_position()
            interactions[key][1][new_key]+=[(x.pdg_code(),x.p(),x.track_id(),[[s_pos.x(),s_pos.y(),s_pos.z()],[e_pos.x(),e_pos.y(),e_pos.z()]])]
        for key in interactions:
            for prim in interactions[key][1]:
                d=[z for z in interactions[key][1][prim] if z[0]==prim[1]]
                if len(d)==0:
                    sys.stderr.write("unknown error"+str(interactions[key][1][prim]))
                    continue
                # assert len(d)>0, 


                # assert prim[0]>10000 or prim[0]==d[0][2],(prim,d)
                if prim[0]<10000 and prim[0]!=d[0][2]:
                    print("The track ids don't match up at all!!!!!!!",prim,d)
            if len(interactions[key][1])>0:
                assert interactions[key][2]!=-1, [(z.pdg_code(),z.nu_current_type()) for z in truthinfo.as_vector()]
                #TODO this is valid, but not for mvmpr
                # assert interactions[key][2] is not None, (key,interactions,[(z.pdg_code(),z.nu_current_type(),is_contained([z.position().x(),z.position().y(),z.position().z()],margin=0)) for z in truthinfo.as_vector()])
                # for d in interactions[key][1]

                
                # particles[x.pdg_code()]+=[val]

                # interactions[key][1][x.track_id()]=val
                # if len(particles[pdg])==0: raise Exception()
                # entry=-100
                # for i in range(len(particles[pdg])):
                #     if pos2==particles[pdg][i][1]:
                #         if entry!=-100:
                #             raise Exception("duplicates",(particles,pos2))
                #         entry=i
                        
                # assert entry!=-100,(particles,pos2)
                    

                # if particles[pdg][entry][1]==pos2:
                    
                #         particles[pdg][entry][2]+=[y.pdg_code()]
                        
                    # if y.parent_pdg_code()!=2112:
                        # particles[pdg][entry][-1]*=is_contained(pos3)
                        # particles[pdg][entry][-1]*=is_contained(pos4)
                    # print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)
                # else: raise Exception()
                
                # if pos2 not in interactions:
                #     assert not is_contained(pos2),(interactions,pos2,y.ancestor_pdg_code())
                # if pos2 in interactions:
                # interactions[pos2][0][pdg]=particles[pdg][entry]
                

            # if y.ancestor_pdg_code()==3122:
            #     if len(lambdas)==0: raise Exception()
            #     if lambdas[0][1]==pos2:
            #         if y.parent_pdg_code()==3122 and y.creation_process()=='Decay':
            #             lambdas[0][2]+=[y.pdg_code()]
            #             print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)
            #         if y.parent_pdg_code!=2112:
            #             lambdas[0][-1]*=is_contained(pos3)
            #             lambdas[0][-1]*=is_contained(pos4)
            #         # print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)
            #     else: raise Exception()
            #     assert pos2 in interactions
            #     interactions[pos2][0]["lam"]=lambdas[0]

            # if y.ancestor_pdg_code() in [310,311]:
            #     if len(k0s)==0: raise Exception()
            #     if k0s[0][1]==pos2:
            #         if y.parent_pdg_code()==310 and y.creation_process()=='Decay':
            #             k0s[0][2]+=[y.pdg_code()]
            #             print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)

            #         if y.parent_pdg_code!=2112:
            #             k0s[0][-1]*=is_contained(pos3)
            #             k0s[0][-1]*=is_contained(pos4)
            #     else: raise Exception()
            #     assert pos2 in interactions
            #     interactions[pos2][0]["k0s"]=k0s[0]
        # final_lambda+=lambdas
        # final_kaon+=kaons
        # final_k0s+=k0s
        # final_particles=merge_dicts(final_particles,particles)
        # assert not set(interactions.keys()).intersection(final_interactions.keys())
        final_interactions.update(interactions)
    SAVEDIR=os.path.dirname(sys.argv[1]).replace("_files","_larcv_truth")
    # os.makedirs(SAVEDIR, exist_ok=True)
    # print("SAVEDIR",SAVEDIR)
    print([pdg_dict,neutrinos,final_interactions])
    np.save(SAVEDIR,np.array([pdg_dict,neutrinos,final_interactions], dtype=object))


                
                    #     y=truthinfo_old.as_vector()[k]

        # print(lambdas,kaons)
                # if x.ancestor_pdg_code() in [3122,321] and  x.pdg_code() not in [3122,321]: print("very good",x.ancestor_pdg_code(),x.pdg_code())
                # if x.ancestor_creation_process()!='primary':continue
                # if x.ancestor_pdg_code()!=x.pdg_code():continue

                # children_id
                # pos=[x.position().x(),x.position().y(),x.position().z(),x.position().t()]
                # print("found",x.pdg_code(),x.id(),pos,x.parent_track_id(),x.ancestor_track_id(),x.creation_process(),x.p(),x.ancestor_creation_process(), x.ancestor_pdg_code(),x.children_id(),dir(x))

                # for k in range(len(truthinfo_old)):
                    
                # y=truthinfo_old.as_vector()[k]
                # pos2=[y.ancestor_position().x(),y.ancestor_position().y(),y.ancestor_position().z(),y.ancestor_position().t()]
                # if y.ancestor_pdg_code() in [3122,321] and pos2==pos:
                    
                    # if y.ancestor_track_id()==x.track_id():
                    #     print("very good",y.pdg_code(),x.pdg_code())


            
            # if len(voxelmap.as_vector()[j])==0: continue
            # print(abs(truthinfo.as_vector()[j].pdg_code()))
            # print("got one")
            # print(truthinfo.as_vector()[j].pdg_code(),len(voxelmap.as_vector()[j]))
            # for k in range(len(voxelmap.as_vector()[j])):
                # id=voxelmap.as_vector()[j].as_vector()[k].id()
                # print(semanticsdict[id],id, semantics_HM.as_vector()[semanticsdict[id]].value())
                # semantics_HM.as_vector()[semanticsdict[id]].set(id,7)
                # raise Exception(semantics_HM.as_vector()[semanticsdict[id]].id(),semantics_HM.as_vector()[semanticsdict[id]].value())
        # io_out.save_entry()
    # io_out.finalize()