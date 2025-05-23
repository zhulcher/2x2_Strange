from larcv import larcv
import tempfile

import sys

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

def make_labels(col1, labels):
    flags = []
    labels.meta(col1.meta())
    for v1 in col1.as_vector():
        label = larcv.Voxel(v1.id(), v1.value())
        labels.insert(label)
    return flags

if __name__ == "__main__":

    print("attempting to convert",sys.argv[1],"to",sys.argv[2])

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


    outcfg= '''
        IOManager: {
            Verbosity    : 2
            Name         : "OUTIO"
            IOMode       : 2
            InputFiles   : [%s]
            OutFileName  : %s
        }
        '''% (sys.argv[1], sys.argv[2])
    #"output_HM-larcv.root"
    io_out = load_cfg(outcfg)





    print(n_evts)

    for i_entry in range(n_evts):
        print(i_entry)
        if i_entry % 100 == 0:
            print('Processing', i_entry)

        io_out.read_entry(i_entry)
        
        # j_entry = (i_entry + 1) % io.get_n_entries()
        io.read_entry(i_entry)

        # =openroot.Get("_pcluster_tree")
        # truthinfo=openroot.Get("particle_pcluster_tree")
        voxelmap = io.get_data('cluster3d', 'pcluster')
        truthinfo = io.get_data('particle', 'pcluster')
        semantics = io.get_data('sparse3d', 'pcluster_semantics')
        truthinfo_primaries = io.get_data('particle', 'mpv')

        semantics_HM = io_out.get_data('sparse3d', 'pcluster_semantics_HM')

        semanticsdict={}
        # semantics_HM=copy.deepcopy(semantics)

        make_labels(semantics, semantics_HM)

        # print("dear god help me",len(semantics_HM.as_vector()))
        for i in range(len(semantics.as_vector())): semanticsdict[semantics.as_vector()[i].id()]=i
            # raise Exception(dir(semantics.as_vector()[i]),semantics.as_vector()[i].id(),semantics.as_vector()[i].value())
            

        # raise Exception(set(semanticsdict.values()))
        # break

        pdgs=[truthinfo.as_vector()[j].pdg_code() for j in range(len(truthinfo.as_vector()))]

        # assert len(truthinfo)==len(truthinfo.as_vector())

        for j in range(len(truthinfo)):

            part=truthinfo.as_vector()[j]


            if part.pdg_code() in [-2212,3222,3112] and part.energy_deposit()>0:
                print("found 1 in larcv file a ",[part.pdg_code(),part.parent_pdg_code(),part.ancestor_pdg_code()],part.shape(),[part.track_id(),part.parent_track_id(),part.ancestor_track_id()],[part.id(),part.parent_id()],part.group_id(),pdgs[part.group_id()],i_entry,sys.argv[1],part.energy_init(),part.energy_deposit(),len(voxelmap.as_vector()[j]),part.num_voxels(),file=sys.stderr, flush=True)

            if part.shape()==larcv.kShapeLEScatter: continue
            # if len(voxelmap.as_vector()[j])==0: continue
            # if part.energy_deposit()==0: continue
            # if part.num_voxels()==0: continue
            # assert len(voxelmap.as_vector()[j])==part.num_voxels(),(len(voxelmap.as_vector()[j]),part.num_voxels())

            # if abs(part.pdg_code())==321:
            #     raise Exception(part.pdg_code())

            if abs(part.pdg_code()) in [13,211,2212,321,3222,3112]:

                assert part.shape()==larcv.kShapeTrack,([part.pdg_code(),part.parent_pdg_code(),part.ancestor_pdg_code()],part.shape(),[part.track_id(),part.parent_track_id(),part.ancestor_track_id()],[part.id(),part.parent_id()],part.group_id(),pdgs[part.group_id()],i_entry,sys.argv[1],part.energy_init(),part.energy_deposit())

            if part.shape()!=larcv.kShapeTrack: continue
            if abs(part.pdg_code()) in [13,211]:continue
            if abs(part.pdg_code()) not in [2212,321,3222,3112] and abs(part.pdg_code())< 1000000000:
                raise Exception(part.pdg_code())
            
            # if (abs(part.pdg_code()) not in [2212,321] and abs(part.pdg_code())< 1000000000):continue
            
            # if len(voxelmap.as_vector()[j])==0: continue
            # print(abs(part.pdg_code()))
            # print("got one")
            # print(part.pdg_code(),len(voxelmap.as_vector()[j]))
            # voxels=
            # assert len(voxelmap.as_vector()[j])==len()

            for k in range(voxelmap.as_vector()[j].as_vector().size()):
                id=voxelmap.as_vector()[j].as_vector()[k].id()
                # print(semanticsdict[id],id, semantics_HM.as_vector()[semanticsdict[id]].value())
                semantics_HM.as_vector()[semanticsdict[id]].set(id,7)
                # raise Exception(semantics_HM.as_vector()[semanticsdict[id]].id(),part.pdg_code(),semantics_HM.as_vector()[semanticsdict[id]].value())
        for j in range(len(truthinfo_primaries)):
            part=truthinfo_primaries.as_vector()[j]
            if part.pdg_code() in [-2212,3222,3112] and part.energy_deposit()>0:
                print("found 2 in larcv file a ",[part.pdg_code(),part.parent_pdg_code(),part.ancestor_pdg_code()],part.shape(),[part.track_id(),part.parent_track_id(),part.ancestor_track_id()],[part.id(),part.parent_id()],part.group_id(),i_entry,sys.argv[1],part.energy_init(),part.energy_deposit(),part.num_voxels(),file=sys.stderr, flush=True)

            if part.shape()==larcv.kShapeLEScatter: continue
            # if len(voxelmap.as_vector()[j])==0: continue
            # if part.energy_deposit()==0: continue
            # if part.num_voxels()==0: continue
            # assert len(voxelmap.as_vector()[j])==part.num_voxels(),(len(voxelmap.as_vector()[j]),part.num_voxels())

            # if abs(part.pdg_code())==321:
            #     raise Exception(part.pdg_code())

            if abs(part.pdg_code()) in [13,211,2212,321,3222,3112]:

                assert part.shape()==larcv.kShapeTrack,([part.pdg_code(),part.parent_pdg_code(),part.ancestor_pdg_code()],part.shape(),[part.track_id(),part.parent_track_id(),part.ancestor_track_id()],[part.id(),part.parent_id()],part.group_id(),pdgs[part.group_id()],i_entry,sys.argv[1],part.energy_init(),part.energy_deposit())

            # if part.shape()!=larcv.kShapeTrack: continue
            # if abs(part.pdg_code()) in [13,211]:continue
            # if abs(part.pdg_code()) not in [2212,321,3222,3112] and abs(part.pdg_code())< 1000000000:
            #     raise Exception(part.pdg_code())
            
            # # if (abs(part.pdg_code()) not in [2212,321] and abs(part.pdg_code())< 1000000000):continue
            
            # # if len(voxelmap.as_vector()[j])==0: continue
            # # print(abs(part.pdg_code()))
            # # print("got one")
            # # print(part.pdg_code(),len(voxelmap.as_vector()[j]))
            # # voxels=
            # # assert len(voxelmap.as_vector()[j])==len()

            # for k in range(len(voxelmap.as_vector()[j])):
            #     id=voxelmap.as_vector()[j].as_vector()[k].id()
            #     # print(semanticsdict[id],id, semantics_HM.as_vector()[semanticsdict[id]].value())
            #     semantics_HM.as_vector()[semanticsdict[id]].set(id,7)



        io_out.save_entry()
    io_out.finalize()