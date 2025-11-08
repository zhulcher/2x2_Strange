# import sys



# import matplotlib.pyplot as plt
# from spine.utils.globals import *
# from analysis.analysis_cuts import *
import numpy as np
# def delete_only_files(directory_path):
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)  # delete file or symlink
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')

import os

def save_plotlies(directory0,entry=-np.inf,mode="truth"):
    SOFTWARE_DIR = '/sdf/group/neutrino/zhulcher/spine' #or wherever on sdf
    import sys
    sys.path.append(SOFTWARE_DIR)
    import yaml
    import plotly.graph_objs as go

    if 'Drawer' not in globals():
        from spine.driver import Driver
        from spine.vis.out import Drawer

    which_analysis="both"
    # draw_mode="truth"
    assert mode in ["truth","reco"]
    if mode=="reco":
        which_analysis="reco"
        # draw_mode="both"

    DATA_PATH = directory0 + f'/analysis_{which_analysis}.h5' # Replace the source file with the correct path
    anaconfig = 'configs/anaconfig_read.cfg'
    anaconfig = yaml.safe_load(open(anaconfig, 'r').read().replace('DATA_PATH', DATA_PATH))

    driver = Driver(anaconfig)
    os.makedirs(directory0+"/eventdisplays/",exist_ok=True)

    if entry==-np.inf:
        raise Exception("i need an entry in save_plotlies")
    
    if entry>=len(driver):
        print("this entry does not exist",entry,len(driver))

        # delete_only_files(directory0)

    # for ENTRY in range(len(driver)):
    #     if ENTRY!=entry:
    #         continue

    #     # ENTRY = 156
    #     print(ENTRY)
    if True:
        ENTRY=entry
        data = driver.process(entry=ENTRY)
        data.keys()
        # if not os.path.exists(directory+"/eventdisplays/"):
        
        outfile=directory0+"/eventdisplays/"+str(ENTRY)+".html"
        

        # openroot = ROOT.TFile.Open(directory+"/output_0_0000-edepsim.root", "READ")
        # events=openroot.Get("EDepSimEvents")
        # entry = ROOT.TG4Event()
        # events.SetBranchAddress("Event",ROOT.AddressOf(entry))
        # events.GetEntry(ENTRY)

        # print("event length",events.GetEntries())
        # print(entry.RunId,entry.EventId)
        # for i in range(1000000):
        #     events.GetEntry(i)
        #     entry.Trajectories
        #     print(i)
        # raise Exception()

        # run_info=data['run_info']
        # change=0
        # while entry.EventId!=run_info.event:

        #     events.GetEntry(change)
        #     # print("intermediate to",(entry.EventId))
        #     change+=1
        # # print("changed to",(entry.EventId,run_info.event,change))
        drawer = Drawer(data, detector='icarus',draw_mode="truth",width=1700) # Try to replace none with 'icarus', 'sbnd' or '2x2'!
        fig2 = drawer.get('particles', ['shape','id','match_ids','match_overlaps','parent_id','children_id','orig_id','orig_parent_id','orig_children_id','orig_group_id','pdg_code','parent_pdg_code','ancestor_pdg_code','creation_process','parent_creation_process','ancestor_creation_process','track_id','parent_track_id','ancestor_track_id','reco_length','length','is_primary','is_contained','end_momentum','nu_id','position','parent_position','ancestor_position','is_valid','momentum','reco_momentum','ke','reco_ke','start_dir','reco_start_dir','end_dir','reco_end_dir','start_point','end_point'], draw_end_points=True,split_traces=True) # Try to replace 'id' with 'shape'!
        fig = drawer.get('interactions', ['topology','match_ids','match_overlaps','is_fiducial','primary_particle_counts','current_type','interaction_mode','interaction_type','target','nucleon','quark','energy_init','lepton_pdg_code','pdg_code','hadronic_invariant_mass','bjorken_x','momentum_transfer','is_contained','reco_vertex','vertex'], draw_vertices=True,split_traces=True) # Try to replace 'id' with 'shape'!
        for trace in fig2.data:
            fig.add_trace(trace)
        
        if mode=="reco":
            drawer = Drawer(data, detector='icarus',draw_mode="reco",width=1700) # Try to replace none with 'icarus', 'sbnd' or '2x2'!
            fig3 = drawer.get('particles', ['shape','id','match_ids','match_overlaps','pdg_code','reco_length','length','is_primary','is_contained','is_valid','momentum','reco_momentum','ke','reco_ke','start_dir','reco_start_dir','end_dir','reco_end_dir','start_point','end_point','is_time_contained'], draw_end_points=True,split_traces=True) # Try to replace 'id' with 'shape'!
            fig4 = drawer.get('interactions', ['id','match_ids','match_overlaps','topology','is_fiducial','primary_particle_counts','is_contained','vertex','is_flash_matched','is_time_contained'], draw_vertices=True,split_traces=True) # Try to replace 'id' with 'shape'!
            for trace in fig3.data:
                fig.add_trace(trace)
            for trace in fig4.data:
                fig.add_trace(trace)
        
        # if mode=="reco":
            # fig3=drawer.get('reco_particles', ['shape','id','parent_id','children_id','orig_id','orig_parent_id','orig_children_id','orig_group_id','pdg_code','parent_pdg_code','ancestor_pdg_code','creation_process','parent_creation_process','ancestor_creation_process','track_id','parent_track_id','ancestor_track_id','reco_length','is_primary','is_contained','end_momentum','nu_id','position','parent_position','ancestor_position','is_valid','momentum','reco_momentum','ke','reco_ke','start_dir','reco_start_dir','end_dir','reco_end_dir','start_point','end_point'], draw_end_points=True,split_traces=True)
            # fig4 = drawer.get('reco_interactions', ['topology','is_fiducial','primary_particle_counts','current_type','interaction_mode','interaction_type','target','nucleon','quark','energy_init','lepton_pdg_code','pdg_code','hadronic_invariant_mass','bjorken_x','momentum_transfer','is_contained','reco_vertex','vertex'], draw_vertices=True,split_traces=True) # Try to replace 'id' with 'shape'!
        # for t in entry.Trajectories:
        #     # print("number of trajectories", entry.Event.Trajectories.size())
        #     # print(len(t.Points))
        #     xlist=[]
        #     ylist=[]
        #     zlist=[]
        #     idlist=[]
        #     trajid=[]
        #     parentid=[]
        #     proc=[]
        #     subproc=[]


        #     if t.GetPDGCode() in[-13,13]:
        #         for p in range(len(t.Points)-1):
        #             pos=(t.Points[p+1].GetPosition().Vect()+t.Points[p].GetPosition().Vect())/2
        #             def p_to_KE(p):
        #                 return np.sqrt(p.GetMomentum().Mag()**2+105.7**2)-105.7
        #             dx=(t.Points[p+1].GetPosition().Vect()-t.Points[p].GetPosition().Vect()).Mag()
        #             dKE=p_to_KE(t.Points[p])-p_to_KE(t.Points[p+1])
        #             print(dKE/dx,[pos[0],pos[1],pos[2]],is_contained([np.array([pos[0],pos[1],pos[2]])/10],mode='module',margin=0),dx,dKE,t.GetTrackId())


        #     for p in t.Points:
        #         if t.GetPDGCode() not in [11,2112,22] and abs(t.GetPDGCode())<1000000:
        #             xlist+=[p.GetPosition().X()/10]
        #             ylist+=[p.GetPosition().Y()/10]
        #             zlist+=[p.GetPosition().Z()/10]
        #             idlist+=[t.GetPDGCode()]
        #             trajid+=[t.GetTrackId()]
        #             parentid+=[t.GetParentId()]
        #             proc+=[t.Points[0].GetProcess()]
        #             subproc+=[t.Points[0].GetSubprocess()]

        #         # print(t.GetInitialPosition().X())
        #         # print(t.GetName())
        #         # l=dir(t)
        #         # print(l)
        #     hover_texts = [f"PDG: {group}<br>Trajectory: {trajid}<br>Parent: {parentid}<br>Creation: {pr}::{subpr}" for group,trajid,parentid,pr,subpr in zip(idlist,trajid,parentid,proc,subproc)]
        #     if len(xlist)>0:
        #         sc3d = go.Scatter3d(
        #             x=xlist,
        #             y=ylist,
        #             z=zlist,
        #             text=hover_texts,
        #             name="Trajectory:"+str(t.GetTrackId()) + " , PDG:" + str(t.GetPDGCode())+", Creation:"+str(t.Points[0].GetProcess())+"::"+str(t.Points[0].GetSubprocess()),
        #             marker=dict(size=5, color=idlist, colorscale="Viridis", opacity=0.8),
        #             line=dict(color=idlist,width=5)
        #         )
        #         fig.add_trace(sc3d)
        xlist=[]
        ylist=[]
        zlist=[]
        color=[]
        # print(data.keys())
        if not len(data["depositions_g4"])==len(data['points_g4']): raise Exception(len(data["depositions_g4"]),len(data['points_g4']))
        for p in range(len(data["depositions_g4"])):

            point=data['points_g4'][p]

            xlist+=[point[0]]
            ylist+=[point[1]]
            zlist+=[point[2]]
            color+=[data["depositions_g4"][p]]

        hover_text = [f"G4 Edep: {c:.2f}" for c in color]
        if len(xlist)>0:
            sc3d = go.Scatter3d(
                x=xlist,
                y=ylist,
                z=zlist,
                hovertext=hover_text,        # Display color value in hovertext
                hoverinfo='text',             # Only show the custom hovertext
                name="G4 Edep",
                mode="markers",
                marker=dict(
                    size=1.5,
                    color=color,  # set color to an array/list of desired values
                    # colorscale="Viridis",  # choose a colorscale
                ),
            )
            fig.add_trace(sc3d)

        if "depositions" in data:
            xlist=[]
            ylist=[]
            zlist=[]
            color=[]
            # print(data.keys())
            if not len(data["depositions"])==len(data['points']): raise Exception(len(data["depositions"]),len(data['points']))
            for p in range(len(data["depositions"])):

                point=data['points'][p]

                xlist+=[point[0]]
                ylist+=[point[1]]
                zlist+=[point[2]]
                color+=[data["depositions"][p]]

            hover_text = [f"Edep: {c:.2f}" for c in color]
            if len(xlist)>0:
                sc3d = go.Scatter3d(
                    x=xlist,
                    y=ylist,
                    z=zlist,
                    hovertext=hover_text,        # Display color value in hovertext
                    hoverinfo='text',             # Only show the custom hovertext
                    name="Edep",
                    mode="markers",
                    marker=dict(
                        size=1.5,
                        color=color,  # set color to an array/list of desired values
                        # colorscale="Viridis",  # choose a colorscale
                    ),
                )
                fig.add_trace(sc3d)

        # for det in entry.SegmentDetectors:
        #     # print(det.first)
        #     for dep in det.second:

        #         if dep.Contrib[0] in validmu:
        #             pos=(dep.GetStart().Vect()+dep.GetStop().Vect())/2
        #             print(dep.GetEnergyDeposit()/dep.GetTrackLength(),[pos[0],pos[1],pos[2]],is_contained([np.array([pos[0],pos[1],pos[2]])/10],mode='module',margin=.1))

        #         xlist+=[dep.GetStart().X()/10]
        #         ylist+=[dep.GetStart().Y()/10]
        #         zlist+=[dep.GetStart().Z()/10]

        #         xlist+=[dep.GetStop().X()/10]
        #         ylist+=[dep.GetStop().Y()/10]
        #         zlist+=[dep.GetStop().Z()/10]

        #         xlist+=[np.nan]
        #         ylist+=[np.nan]
        #         zlist+=[np.nan]
        # if len(xlist)>0:
        #     sc3d = go.Scatter3d(
        #         x=xlist,
        #         y=ylist,
        #         z=zlist,
        #         name="Edep",
        #         marker=dict(size=1.5, opacity=0.5),
        #         line=dict(width=1)
        #     )
        #     fig.add_trace(sc3d)

        # raise Exception(data.keys(),data['points_g4'],data['depositions'])

        # print("event number", event.EventId)
        # print("number of trajectories", event.Trajectories.size())

        # raise Exception(outfile)
        print("writing to file",outfile)

        fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.write_html(outfile,full_html=True,
            # include_plotlyjs='cdn',
            config={'responsive': True})

# save_plotlies("2024-08-14-lambdas")
# save_plotlies("2024-08-14-kaons")

# parent_directory="/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_files/"
if __name__ == "__main__":
    import os
    parent_directory="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_files/"
    for item in os.listdir(parent_directory):
        full_path = os.path.join(parent_directory, item)
        if os.path.isdir(full_path):  # Check if it's a directory
            save_plotlies(full_path)
        # print(f"Directory: {full_path}")


# from pathlib import Path
# import os
# path="beam/processed"
# directories = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
# for entry in sorted(directories):
#     full_path = os.path.join(path, entry)
#     if os.path.isdir(full_path):
#         print(entry)
#         # print(path)
#         Path("event_displays/eventdisplays/beam/processed/"+entry).mkdir(parents=True, exist_ok=True)
#         save_plotlies("beam/processed/"+entry)
# save_plotlies("2024-08-16-lambdas")
# save_plotlies("2024-08-16-kaons")

#


# # from bragganalysis.showeranalysis.showerdirection import *
# def plotedep2(myroot, eventnum):
#     xlist = []
#     ylist = []
#     zlist = []
#     idlist = []

#     numpts = 0
#     trackptlist = []
#     pdglist = []

#     numdep = 0
#     pdgpardict = {}
#     trackx = []
#     segxlist = []
#     segylist = []
#     segzlist = []

#     par = {}
#     pdg = {}

#     par[-1] = -1
#     pdg[-1] = -1

#     segts = {}
#     segcolorlist = []
#     tracksegE = {}
#     pt2mom = {}
#     pdgtomass = {}
#     # depx=[]

#     # shower_start = [-125, -225, -225]
#     for entry in eventlist:

#         # print(entry.Event.EventId)
#         if entry.Event.EventId < eventnum:
#             continue
#         if entry.Event.EventId != eventnum:
#             print("went past it", entry.Event.EventId, "!=", eventnum)
#             raise ValueError
#         print("starting event number:", entry.Event.EventId)
#         if entry.Event.Trajectories.size() == 0:
#             print("empty event, moving on")
#             raise ValueError
#         vtx = [
#             entry.Event.Primaries[0].GetPosition().X(),
#             entry.Event.Primaries[0].GetPosition().Y(),
#             entry.Event.Primaries[0].GetPosition().Z(),
#         ]
#         print("vertex pos", vtx[0], vtx[1], vtx[2])
#         # if not inbounds(vtx[0],vtx[1],vtx[2]):continue

#         for p in entry.Event.Primaries[0].Particles:
#             mom = p.GetMomentum()
#             print(
#                 "primaries", p.GetTrackId(), p.GetPDGCode(), mom.X(), mom.Y(), mom.Z()
#             )
#         skip = 0

#         # print("Number of Trajectories/Tracks:",entry.Event.Trajectories.size())
#         # print("Number of Primary Vertices:",entry.Event.Primaries.size())
#         print("number of vertices", entry.Event.Primaries.size())
#         for v in entry.Event.Primaries:
#             print("num of primary particles:", len(v.Particles))
#             print(re.split(";|,", v.GetReaction()))
#             print(v.GetReaction()[:6])
#             print(v.GetReaction())
#             # if v.GetReaction()[:6]=="nu:14;":# or v.GetReaction()[:6]=="nu:-14" or v.GetReaction()[:6]=="nu:12;":# or v.GetReaction()[:6]=="nu:-12":
#             #     skip=1
#             #     break
#             for prim in v.Particles:
#                 print(
#                     prim.GetPDGCode(),
#                     prim.GetMomentum().X(),
#                     prim.GetMomentum().Y(),
#                     prim.GetMomentum().Z(),
#                     prim.GetMomentum().E(),
#                 )
#                 if prim.GetPDGCode() in [14, -14, 12, -12]:
#                     print("bad primary particle")
#                     # skip =1
#         if skip == 1:
#             continue
#         groupid = {}
#         parentid = {}
#         parentid[-1] = -1
#         groupid[-1] = [-1, -1]

#         posmap = {}

#         for idx, t in enumerate(entry.Event.Trajectories):
#             # if t.GetParentId()==-1:

#             # print(t.GetTrackId(),t.GetParentId(),"hi",t.GetPDGCode(),t.GetInitialMomentum().E(),t.Points[0].GetProcess(),t.Points[0].GetSubprocess(),pinbounds(t.Points[0].GetPosition()),t.Points[0].GetPosition().X(),t.Points[0].GetProcess(),t.Points[0].GetSubprocess())
#             #             if t.GetTrackId()==0:
#             #                 if abs(t.GetPDGCode())==13 or abs(t.GetPDGCode())==22:
#             #                     skip=1
#             #                     break
#             #                 print(t.GetParentId(),"hi",t.GetPDGCode(),t.GetInitialMomentum().X(),t.GetInitialMomentum().Y(),t.GetInitialMomentum().Z())
#             # if t.GetTrackId()==-1: print(t.GetPDGCode())
#             # print(t.GetParentId())
#             # if t.GetTrackId()==1 and abs(t.GetPDGCode()) not in [321,311]: raise Exception("help me",t.GetPDGCode())
#             if t.GetParentId() == 1:
#                 print("initial", t.GetTrackId(), t.GetParentId(), t.GetPDGCode())
#             groupid[t.GetTrackId()] = [t.GetPDGCode(), t.GetTrackId()]
#             parentid[t.GetTrackId()] = t.GetParentId()

#             posmap[t.GetTrackId()] = np.array(
#                 [
#                     [
#                         t.Points[0].GetPosition().X(),
#                         t.Points[0].GetPosition().Y(),
#                         t.Points[0].GetPosition().Z(),
#                     ],
#                     [
#                         t.Points[-1].GetPosition().X(),
#                         t.Points[-1].GetPosition().Y(),
#                         t.Points[-1].GetPosition().Z(),
#                     ],
#                 ]
#             )
#             # print(t.GetTrackId())
#         test = 0
#         # print("parent of ",t.GetTrackId(),"is",t.GetParentId())
#         while test == 0:
#             test = 1
#             for i in groupid.keys():
#                 pi = parentid[i]
#                 if groupid[pi][0] in [
#                     11,
#                     -11,
#                     13,
#                     -13,
#                     211,
#                     -211,
#                     2212,
#                     -2212,
#                     2112,
#                     -2112,
#                     321,
#                     -321,
#                     22,
#                 ] and (
#                     groupid[i][0] != groupid[pi][0] or groupid[i][1] != groupid[pi][1]
#                 ):
#                     groupid[i][0] = groupid[pi][0]
#                     groupid[i][1] = groupid[pi][1]
#                     # print(groupid[N][i],groupid[N][parentid[N][i]])
#                     test = 0
#         if skip == 1:
#             print("muon or photon event, skipping")
#             continue
#         # for t in entry.Event.Trajectories:
#         #     if abs(groupid[t.GetTrackId()][0])==2112:
#         #         print([t.GetTrackId(),t.GetPDGCode()],[t.GetParentId()],"hi",t.GetInitialMomentum().E(),t.Points[0].GetProcess(),t.Points[0].GetSubprocess(),pinbounds(t.Points[0].GetPosition()),t.Points[0].GetPosition().X(),t.Points[0].GetProcess(),t.Points[0].GetSubprocess())
#         print("ending tracks")

#         print("number of trajectories", entry.Event.Trajectories.size())
#         for t in entry.Event.Trajectories:
#             if t.GetPDGCode() not in pdgtomass.keys():
#                 pdgtomass[t.GetPDGCode()] = t.GetInitialMomentum().M()
#             # if t.GetTrackId()!=5:continue
#             # if t.GetParentId()==0:print(t.GetInitialMomentum().E())
#             if filter != 1234567890 and abs(groupid[t.GetTrackId()][0]) != filter:
#                 continue
#             # print(t.GetPDGCode())
#             # if t.GetPDGCode()==2112:continue
#             pdgpardict[t.GetTrackId()] = [t.GetParentId(), t.GetName()]

#             par[t.GetTrackId()] = t.GetParentId()
#             pdg[t.GetTrackId()] = t.GetPDGCode()

#             # print(t.GetName())
#             # print("number of trajectories", entry.Event.Trajectories.size())
#             # print(len(t.Points))
#             # if t.GetParentId()==-1:print("primary PDG:",t.GetPDGCode())
#             # if t.GetParentId()!=-1: continue
#             # if abs(t.GetPDGCode())!=13:continue
#             # print(t.GetInitialMomentum().X(),t.GetInitialMomentum().Y(),t.GetInitialMomentum().Z())
#             tracksegE[t.GetTrackId()] = []
#             pt2mom[t.GetTrackId()] = {}

#             if abs(pdg[t.GetParentId()]) == 2212 and t.Points[0].GetProcess() == 6:
#                 pos = t.Points[0].GetPosition()

#                 print(
#                     "found one",
#                     t.GetTrackId(),
#                     t.GetParentId(),
#                     "break",
#                     pdg[par[t.GetTrackId()]],
#                     pdg[t.GetTrackId()],
#                     t.Points[0].GetProcess(),
#                     t.Points[0].GetSubprocess(),
#                     [pos.X(), pos.Y(), pos.Z()],
#                     [
#                         posmap[t.GetTrackId()][0] - posmap[t.GetParentId()][0],
#                         posmap[t.GetTrackId()][0] - posmap[t.GetParentId()][1],
#                     ],
#                 )
#                 if t.GetParentId() != -1:
#                     print(par[t.GetParentId()], pdg[t.GetParentId()])
#                 if par[t.GetParentId()] != -1:
#                     print(par[par[t.GetParentId()]], pdg[par[t.GetParentId()]])

#             #             if par[t.GetTrackId()]==1:
#             #                 pos=t.Points[0].GetPosition()
#             #                 print("that proton",t.GetTrackId(),t.GetParentId(),"break",pdg[par[t.GetTrackId()]],pdg[t.GetTrackId()],t.Points[0].GetProcess(),t.Points[0].GetSubprocess(),[pos.X(),pos.Y(),pos.Z()])

#             # if abs(t.GetPDGCode())==13:
#             # print("track init E:",t.GetInitialMomentum().E())
#             for p in t.Points:
#                 if abs(t.GetPDGCode()) == 13 and t.GetParentId() == -1:
#                     print(p.GetMomentum().Mag(), pinbounds(p.GetPosition()))
#                 tracksegE[t.GetTrackId()] += [
#                     [p.GetPosition().X(), p.GetPosition().Y(), p.GetPosition().Z()]
#                 ]
#                 pt2mom[t.GetTrackId()][
#                     p.GetPosition().X()
#                     + 1000 * p.GetPosition().Y()
#                     + 1000000 * p.GetPosition().Z()
#                 ] = [p.GetMomentum().X(), p.GetMomentum().Y(), p.GetMomentum().Z()]

#                 trackx += [
#                     p.GetPosition().X()
#                     + 1000 * p.GetPosition().Y()
#                     + 1000000 * p.GetPosition().Z()
#                 ]
#                 xlist += [p.GetPosition().X()]
#                 ylist += [p.GetPosition().Y()]
#                 zlist += [p.GetPosition().Z()]
#                 idlist += [t.GetPDGCode()]
#                 trackptlist += [t.GetTrackId()]
#                 pdglist += [t.GetPDGCode()]

#                 # print(t.GetInitialPosition().X())
#                 # l=dir(t)
#                 # print(l)
#                 numpts += 1

#         # show=Shower()
#         for det in entry.Event.SegmentDetectors:

#             print(det.first)
#             for dep in det.second:
#                 start = [
#                     dep.GetStart().X(),
#                     dep.GetStart().Y(),
#                     dep.GetStart().Z(),
#                     dep.GetStart().T(),
#                 ]
#                 stop = [
#                     dep.GetStop().X(),
#                     dep.GetStop().Y(),
#                     dep.GetStop().Z(),
#                     dep.GetStop().T(),
#                 ]
#                 # show.addEdep(start,stop,dep.GetEnergyDeposit())

#                 good = 0
#                 if dep.Contrib[0] not in segts.keys():
#                     segts[dep.Contrib[0]] = []
#                 if len(dep.Contrib) > 1:
#                     raise Exception("contrib too long", len(dep.Contrib))
#                 for i in dep.Contrib:
#                     if i in trackptlist:
#                         good = 1
#                 if good == 0:
#                     continue
#                 numdep += 1

#                 # depx+=[dep.GetStart().X()+1000*dep.GetStart().Y()+1000000*dep.GetStart().Z()]
#                 # depx+=[dep.GetStop().X()+1000*dep.GetStop().Y()+1000000*dep.GetStop().Z()]
#                 segxlist += [dep.GetStart().X(), dep.GetStop().X(), None]
#                 segylist += [dep.GetStart().Y(), dep.GetStop().Y(), None]
#                 segzlist += [dep.GetStart().Z(), dep.GetStop().Z(), None]
#                 segts[dep.Contrib[0]] += [dep.GetStart().T(), dep.GetStop().T()]
#                 if numdep in [1]:
#                     print("start for ", numdep, dep.GetStart().Z())
#                 # print(dep.GetSecondaryDeposit()/dep.GetEnergyDeposit())
#                 # if len(segxlist)>3:
#                 #    if segxlist[-3]!=segxlist[-5] or segylist[-3]!=segylist[-5] or segzlist[-3]!=segzlist[-5]:
#                 # print(segxlist[-5],segylist[-5],segzlist[-5],"to",segxlist[-3],segylist[-3],segzlist[-3])
#                 # print ([segxlist[-5],segylist[-5],segzlist[-5]] in tracksegE[dep.Contrib[0]])
#                 # print ([segxlist[-3],segylist[-3],segzlist[-3]] in tracksegE[dep.Contrib[0]])
#                 # print(pt2mom[dep.Contrib[0]])
#                 # print(pt2mom[dep.Contrib[0]][segxlist[-5]+1000*segylist[-5]+1000000*segzlist[-5]])
#                 # print(pt2mom[dep.Contrib[0]][segxlist[-3]+1000*segylist[-3]+1000000*segzlist[-3]])

#                 segcolorlist += [dep.Contrib[0] + 100, dep.Contrib[0] + 100, 0]
#         print("num energy deposits:", numdep)
#         # print(dep.GetPrimaryId(),dep.Contrib)
#         # shower_start=show.start[:3]
#         # print("shower start",show.start)
#         if numdep == 0:
#             print("no edep")
#             continue
#         break
#         """
#         store=0
#         showg={}
#         for tid in pdgpardict.keys():
#             if pdgpardict[tid][1]=="e-":
#                 showg[tid]=tid
#             else: showg[tid]=-2
#         print(type(pdgpardict[0][1]))
#         while store==0:
#             print("went through")
#             store=1
#             for tid in pdgpardict.keys():
#                 if pdgpardict[tid][0]==-1: continue
#                 #print(pdgpardict[tid][1]=='e-',pdgpardict[tid][1]=='gamma',pdgpardict[pdgpardict[tid][0]][1]=='e-',pdgpardict[pdgpardict[tid][0]][1]=='gamma')
#                 if (pdgpardict[tid][1]=='e-' or pdgpardict[tid][1]=='gamma') and (pdgpardict[pdgpardict[tid][0]][1]=='e-' or pdgpardict[pdgpardict[tid][0]][1]=='gamma'):
#                     #print("changed")
#                     pdgpardict[tid]=pdgpardict[pdgpardict[tid][0]]
#                     showg[tid]=pdgpardict[tid][0]
#                     store=0

#         showglist=[showg[trackptlist[i]] for i in range(numpts)]
#         #print(showglist)
#         showglistdict=dict([(y,x+1) for x,y in enumerate(sorted(set(showglist)))])
#         showlist=[showglistdict[i] for i in showglist]
#         #print(2,len(showglist))
#         #print(showglist)
#         if entry.Event.EventId==eventnum: break

#     for i in range(len(idlist)):
#         if idlist[i] in pdgmap.keys():
#             idlist[i]=pdgmap[idlist[i]]
#         else:
#             pdgmap[idlist[i]]=count
#             idlist[i]=count
#             count+=1
#     sizemax=[10 for i in xlist]
#     """
#     print("Number of Points:", numpts)
#     # fig=px.scatter_3d(x=xlist,y=ylist,z=zlist,color=idlist,size=sizemax)
#     # points=np.transpose(np.array([xlist,ylist,zlist]))

#     cleanx = []
#     cleany = []
#     cleanz = []
#     cleanpdg = []

#     for i in range(len(xlist)):
#         if (
#             xlist[i] > -20000
#             and xlist[i] < 20000
#             and ylist[i] > -20000
#             and ylist[i] < 20000
#             and zlist[i] > -20000
#             and zlist[i] < 20000
#         ):
#             cleanx += [xlist[i]]
#             cleany += [ylist[i]]
#             cleanz += [zlist[i]]
#             cleanpdg += [pdglist[i]]
#     points = np.transpose(np.array([cleanx, cleany, cleanz]))

#     # print(points.shape)
#     # print(set(depx)&set(trackx))
#     # print(len(set(depx)&set(trackx)),len(set(depx)))
#     # color=np.array(showglist)
#     # color=np.array(idlist)
#     # color=np.array(trackptlist)
#     # print(trackx)
#     # print(depx)

#     # bounds=[[-300*10,400*10],[-220*10,94.5*10],[416*10,916*10]]
#     # cleanpdg=np.array([])
#     # points=np.array([])
#     # graphs=scatter_points(points=points,color=np.array(cleanpdg),markersize=2)
#     # graphs.append(go.Scatter3d(x = segxlist, y = segylist, z = segzlist,
#     #                                    mode = 'lines',
#     #                                    name = 'Graph edges',
#     #                                    line = dict(
#     #                                        color = 'black',
#     #                                        width = 5
#     #                                    ),
#     #                                    hoverinfo = 'none'))

#     graphs = go.Scatter3d(
#         x=segxlist,
#         y=segylist,
#         z=segzlist,
#         mode="lines",
#         name="Graph edges",
#         line=dict(color="black", width=5),
#         hoverinfo="none",
#     )

#     #     z1 = np.linspace(bounds[2][0], bounds[2][1], 2)
#     #     y1 = np.linspace(bounds[1][0], bounds[1][1], 2)

#     #     if boundaries==1:

#     #         x1 =np.full(2,bounds[0][0])
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+1000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+2000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+3000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+4000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+5000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+6000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+7000);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))

#     #         x1 =np.full(2,bounds[0][0])
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+1500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+2500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+3500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+4500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+5500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))
#     #         x1 =np.full(2,bounds[0][0]+6500);
#     #         graphs.append(go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)),opacity=0.1))

#     #         y1 = np.linspace(bounds[0][0], bounds[0][1], 2)

#     #         x1 =np.full(2,bounds[1][0]);
#     #         graphs.append(go.Surface(x=y1, y=x1, z=np.array([z1] * len(x1)).T,opacity=0.1))
#     #         x1 =np.full(2,bounds[1][1]);
#     #         graphs.append(go.Surface(x=y1, y=x1, z=np.array([z1] * len(x1)).T,opacity=0.1))

#     #         zlev=bounds[2][0]
#     #         graphs.append(go.Surface(x=bounds[0],y=bounds[1],z=[[zlev, zlev],[zlev, zlev]],opacity=0.1))
#     #         zlev=bounds[2][0]+1000
#     #         graphs.append(go.Surface(x=bounds[0],y=bounds[1],z=[[zlev, zlev],[zlev, zlev]],opacity=0.1))
#     #         zlev=bounds[2][0]+2000
#     #         graphs.append(go.Surface(x=bounds[0],y=bounds[1],z=[[zlev, zlev],[zlev, zlev]],opacity=0.1))
#     #         zlev=bounds[2][0]+3000
#     #         graphs.append(go.Surface(x=bounds[0],y=bounds[1],z=[[zlev, zlev],[zlev, zlev]],opacity=0.1))
#     #         zlev=bounds[2][0]+4000
#     #         graphs.append(go.Surface(x=bounds[0],y=bounds[1],z=[[zlev, zlev],[zlev, zlev]],opacity=0.1))
#     #         zlev=bounds[2][0]+5000
#     #         graphs.append(go.Surface(x=bounds[0],y=bounds[1],z=[[zlev, zlev],[zlev, zlev]],opacity=0.1))

#     # bounds=[[-300*10,400*10],[-218.19099999999994*10,96.33*10],[415.7559*10,915.7559000000001*10]] #bad dump tree bad

#     fig.update_layout(height=1000, width=1000)
#     fig.show()
#     for i in segts.keys():
#         if not segts[i] == sorted(segts[i]):

#             plt.plot(segts[i])

#             raise Exception("segts out of order", i, segts[i], segxlist)
#         plt.plot(segts[i])
#     plt.show()
#     print(pdgtomass)


# def plotedep(myroot,eventnum):
#     xlist=[]
#     ylist=[]
#     zlist=[]
#     idlist=[]
#     openroot = ROOT.TFile.Open(myroot, "READ")
#     eventlist = openroot.Get("EDepSimEvents")
#     numpts=0
#     trackdict={}
#     count=0
#     pdgmap={}
#     for entry in eventlist:
#         #print(entry.Event.EventId)
#         if entry.Event.EventId!=eventnum: continue
#         print("Number of Trajectories/Tracks:",entry.Event.Trajectories.size())
#         for p in entry.Event.Primaries:
#             print(p.Particles[0].GetTrackId())
#         for t in entry.Event.Trajectories:
#             #print("number of trajectories", entry.Event.Trajectories.size())
#             #print(len(t.Points))
#             for p in t.Points:
#                 xlist+=[p.GetPosition().X()]
#                 ylist+=[p.GetPosition().Y()]
#                 zlist+=[p.GetPosition().Z()]
#                 idlist+=[t.GetPDGCode()]
#                 trackdict[t.GetTrackId()]=1
#                 #print(t.GetInitialPosition().X())
#                 #print(t.GetName())
#                 #l=dir(t)
#                 #print(l)
#                 numpts+=1
#         if entry.Event.EventId==eventnum: break
#     print("Number of Points:",numpts)
#     #print("Number of Tracks:",len(trackdict.keys()))
#     for i in range(len(idlist)):
#         if idlist[i] in pdgmap.keys():
#             idlist[i]=pdgmap[idlist[i]]
#         else:
#             pdgmap[idlist[i]]=count
#             idlist[i]=count
#             count+=1

#     fig=px.scatter_3d(x=xlist,y=ylist,z=zlist,color=idlist,size_max=5)
#     fig.show()


# print("done")
