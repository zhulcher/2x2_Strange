from larcv import larcv
import tempfile

import sys

from analysis.analysis_cuts import *

'''



The way to run this script is

python3 add_HIPMIP.py "input_file.larcv.root" "output_file.larcv.root"

'''

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

    print("attempting to read",sys.argv[1])

    cfg = '''
        IOManager: {
            Verbosity    : 2
            Name         : "MainIO"
            IOMode       : 0
            InputFiles   : [%s]
        }
        '''% sys.argv[1]
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
        lambdas=[]
        kaons=[]
        if len(lambdas)>1: raise Exception(lambdas)
        if len(kaons)>1: raise Exception(kaons)
        for j in range(len(truthinfo)):
            x=truthinfo.as_vector()[j]
            if x.pdg_code() in [3122]:
                print("found one", x.pdg_code())
                pos=[x.position().x(),x.position().y(),x.position().z(),x.position().t()]
                lambdas+=[[x.p(),pos,[],x.creation_process(),True,pos]]
                # 
                # for k in range(len(truthinfo_old)):
                
                #     y=truthinfo_old.as_vector()[k]
                #     if y.ancestor_pdg_code() in [3122,321] and pos2==pos:



            if x.pdg_code() in [321]:
                print("found one", x.pdg_code())
                # print("found one", x.pdg_code())
                pos=[x.position().x(),x.position().y(),x.position().z(),x.position().t()]
                kaons+=[[x.p(),pos,[],x.creation_process(),True,pos]]
                
                # pos=[x.position().x(),x.position().y(),x.position().z(),x.position().t()]
                # for k in range(len(truthinfo_old)):
        if len(lambdas)==0 and len(kaons)==0: continue
        if len(lambdas)>1 or len(kaons)>1: raise Exception()
        # if
        # if False:
        for k in range(len(truthinfo_old)):
            y=truthinfo_old.as_vector()[k]
            pos2=[y.ancestor_position().x(),y.ancestor_position().y(),y.ancestor_position().z(),y.ancestor_position().t()]
            

            if y.ancestor_pdg_code()==321:
                if len(kaons)==0: raise Exception()
                if kaons[0][1]==pos2:
                    if y.parent_pdg_code()==321 and y.creation_process()=='Decay':
                        kaons[0][2]+=[y.pdg_code()]
                    print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)
                else: raise Exception()

            if y.ancestor_pdg_code()==3122:
                if len(lambdas)==0: raise Exception()
                if lambdas[0][1]==pos2:
                    if y.parent_pdg_code()==3122 and y.creation_process()=='Decay':
                        lambdas[0][2]+=[y.pdg_code()]
                    print("ok its something",y.creation_process(),y.pdg_code(),y.parent_pdg_code(),y.ancestor_pdg_code(),y.track_id(),y.parent_track_id(),y.ancestor_track_id(),pos2)
                else: raise Exception()
                    #     y=truthinfo_old.as_vector()[k]

        print(lambdas,kaons)
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