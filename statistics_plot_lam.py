
from statistics_plot_base import *

import os

from analysis.analysis_cuts import *


lam_pass_order_truth={
            # "lam child contained start distance max",
            # "lam_proj_dist max",
            # "Non-Primary HIP or MIP",
            "Valid Interaction":True,
            "MIP Child": True,
            "HIP Child": True,
            
            "Primary HIP-MIP":True, #TODO this is actually opposite in truth
            
            "Max HIP-MIP Sep.":1.7,#cm
            "No HIP or MIP Michel":2.5,
            "Parent Proximity":True,
            "# Children":True,
            "Impact Parameter":4,#cm
            # "vertex not found",
            # "prot or pi containment",
            # "mip len",
            # "hip len",
            # "proton child particle",
            # "pion child particle",
            # "decay len",
            
            # "Perp. Impact Parameter",
            
            "Min Decay Len": 1, #cm,
            # "decay len max",
            # "max tau",

            # r"Valid $\Lambda$ Direction",
            
            rf"Even # Primary $\gamma$":True,
            # "Primary Colinear Parent",
            
            
            
            "":True
}


lam_pass_order_reco={
            # "lam child contained start distance max",
            # "lam_proj_dist max",
            # "Non-Primary HIP or MIP",
            "Valid Interaction":400,
            "MIP Child": True,
            "HIP Child": True,
            "Valid Len":2*min_len,
            "HIP-MIP Order":True,
            "No HIP or MIP Michel":2.5,

            "Starts Closest":True,
            

            "Proj Max HIP-MIP Sep.":2.5,#cm
            
            "Max HIP-MIP Sep.":5,#cm


            
            
            
            
            "Min Decay Len": 4, #cm,
            
            "Impact Parameter":2,#cm
            
            # "vertex not found",
            # "prot or pi containment",
            # "mip len",
            # "hip len",
            # "proton child particle",
            # "pion child particle",
            # "decay len",
            
            # "Perp. Impact Parameter",
            
            
            # "decay len max",
            # "max tau",

            # r"Valid $\Lambda$ Direction",
            
            # rf"Even # Primary $\gamma$":True,
            # "Primary Colinear Parent",

            "Parent Proximity":True,
            "# Children":True,

            # "Primary HIP-MIP":True, #TODO this is actually opposite in truth
            
            
            
            "":True
}


fakecuts={
    "VAE max new":1.75,#cm
    "tau_max":2.6,
}



from concurrent.futures import ProcessPoolExecutor

print("cpu count",os.cpu_count())

def load_single_file(args:tuple)->tuple[str,list[Pred_Neut],list]:
    file0, d0, directory2 = args

    lfile = os.path.join(d0, file0)
    lfile_truth = os.path.join(directory2, file0)
    lfile_truth = lfile_truth[:-1] + "y"

    with np.load(lfile, allow_pickle=True) as lambdas, open(lfile_truth, 'rb') as f2:
        particles = np.load(f2, allow_pickle=True)
        predl = lambdas['PREDLAMBDA']

    return file0, predl, particles

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Script to plot Lam stats')
    parser.add_argument('--mode', type=str, choices=["truth", "reco"], help='Reco or Truth running mode')
    parser.add_argument('--N', type=int, default=sys.maxsize, help='Number of files to run')
    parser.add_argument('--single_file',type=str, default="",help="if set, just run this file, and don't plot")

    args = parser.parse_args()


    

    assert args.mode

    if args.mode=="truth":
        lam_pass_order=lam_pass_order_truth
    else:
        lam_pass_order=lam_pass_order_reco


    from collections import defaultdict


    # num_nu_from_file= np.load('num_nu.npy')
    num_nu_from_file=0

    lam_reason_map=[defaultdict(int),defaultdict(int)]
    lam_pass_failure=[defaultdict(int),defaultdict(int)]

    # current_type_map=defaultdict(int)


    PLOTSDIR="plots/" + str(FOLDER) + "/lambda/"+str(args.mode)+"/"

    assert os.path.isdir(PLOTSDIR), f"{PLOTSDIR} does not exist"

    MAXFILES=args.N

    d0=directory.replace("_analysis", f"_analysis_{args.mode}")


    files = os.listdir(d0)
    # FILES = [os.path.join(directory, f) for f in files]
    if len(files)>MAXFILES:
        files=files[:MAXFILES]

    lam_mass_fixed = defaultdict(lambda: defaultdict(list))
    lam_mass_disc_total=defaultdict(lambda: defaultdict(list))
    # momenta = [[[], []], [[], []]]
    lam_decay_len = defaultdict(lambda: defaultdict(list))
    lam_decay_len_disc_total = defaultdict(lambda: defaultdict(list))


    lam_dir_acos = defaultdict(lambda: defaultdict(list))

    HM_acc_prot =[]
    HM_acc_pi = []

    prot_primary=[[],[]]
    pi_primary=[[],[]]

    prot_mom=[[],[]]
    pi_mom=[[],[]]

    pi_dir=[]
    prot_dir=[]
    


    ProtPi_dist = defaultdict(lambda: defaultdict(list))
    ProtPi_dist_disc_total = defaultdict(lambda: defaultdict(list))

    mip_len=defaultdict(lambda: defaultdict(list))
    hip_len=defaultdict(lambda: defaultdict(list))
    vae = defaultdict(lambda: defaultdict(list))

    lam_momentum=defaultdict(lambda: defaultdict(list))
    lam_true_momentum=defaultdict(lambda: defaultdict(list))

    lam_tau0_est=defaultdict(lambda: defaultdict(list))
    lam_tau0_est_fixed=defaultdict(lambda: defaultdict(list))


    # base_len_vae=defaultdict(lambda: defaultdict(list))
    base_len_vae_disc_total=defaultdict(lambda: defaultdict(list))
    base_len_vae_disc_perp=defaultdict(lambda: defaultdict(list))
    base_len_vae_disc_parr=defaultdict(lambda: defaultdict(list))

    sigma_mass=defaultdict(list)

    
    vertex_displacement=[[],[]]
    vertex_dz=[[],[]]

    TrueLambdas=0
    correctlyselectedlambdas=0
    selectedlambdas=0

    print("only including", MAXFILES, "files")


    # num_nu=0
    filecount=0

    LOCAL_EVENT_DISPLAYS=event_display_new_path+"/lambda/"+args.mode+'/'

    assert os.path.isdir(LOCAL_EVENT_DISPLAYS), f"{LOCAL_EVENT_DISPLAYS} does not exist"




    NONLOCAL_EVENT_DISPLAYS=base_directory+FOLDER+"_files_"+args.mode
    
    assert os.path.isdir(NONLOCAL_EVENT_DISPLAYS), f"{NONLOCAL_EVENT_DISPLAYS} does not exist"

    # raise Exception("")

    # LOCAL

    if args.single_file=="":
        args_list = [(file0, d0, directory2) for file0 in files]
    else:
        # assert os.path.isfile(args.single_file)
        assert args.single_file in files
        args_list = [(args.single_file, d0, directory2)]


        results:list[tuple[str,list[Pred_Neut],list]] = []

    from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor

    results = []
    max_cpus = os.cpu_count()
    # with ProcessPoolExecutor(max_workers=max_cpus) as executor:
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_single_file, arg) for arg in args_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())


    # for args0 in tqdm(args_list, total=len(args_list)):
    #     result = load_single_file(args0)
    #     results.append(result)

    for file0, predl, particles in results:
        if filecount == 0 and args.single_file=="":
            clear_html_files(LOCAL_EVENT_DISPLAYS)
        # if filecount % 500 == 0:
        if filecount%500==0:print("filecount ",filecount,np.divide(correctlyselectedlambdas,TrueLambdas),np.divide(correctlyselectedlambdas,selectedlambdas),[correctlyselectedlambdas,selectedlambdas,TrueLambdas])
        filecount+=1


        if filecount == MAXFILES:
            break
        # filecount += 1
        lfile=os.path.join(d0, file0)
        lfile_truth=os.path.join(directory2, file0)
        lfile_truth=lfile_truth[:-1]+"y"

        def save_my_html(newpath,name,name2):
            copy_and_rename_file(NONLOCAL_EVENT_DISPLAYS,lfile,newpath,name,name2,args.mode)


        both_there=True
        for f in [lfile_truth]:
            both_there*=os.path.exists(f)#, f"File not found: {f}"
        if not both_there:
            print("had to skip because",f"File not found: {file0,lfile,lfile_truth}")
            continue


        # with np.load(lfile, allow_pickle=True) as lambdas, open(lfile_truth, 'rb') as f2:
        if True:
            num_nu_from_file+=len(particles[1][0])

            interactions=particles[2]





            interaction_pos_dict=defaultdict(list)
            for I in interactions:
                i=interactions[I]
                if is_contained(i[0][:3],margin=margin0):
                    primpdgs=[z for z in i[1].keys() if z[1]==3122]
                    if len(primpdgs):
                        assert len(primpdgs)==1
                        print("found lambda",primpdgs)
                        testlam=i[1][primpdgs[0]]
                        ppi=[z for z in testlam if z[0] in [-211, 2212]]
                        if len(ppi)==2:
                            print("found valid lambda in truth")
                            add_it=True
                            TrueLambdas+=1
                            pot_reason=None

                            # print(predl.keys(),I[0])

                            # if I[0] in predl:
                            for l in predl:
                                # if l.reason=="":

                                if l.reason=="" and l.truth_interaction_id==I[1] and l.event_number==I[0]:
                                    add_it=False
                                    print("found the lambda in events")
                                    
                            if add_it:
                                
                                print("completely missed a lambda")
                                # try:
                                save_my_html(LOCAL_EVENT_DISPLAYS+'completely_missed',create_html_filename(I[0], lfile,extra=f"Interaction_id: {I[1]}"),I[0])
                            # else:
                            #     print("didn't have to add it because I found it")

                        elif len(ppi)!=0:
                            print("missed valid lambda",ppi)

                        # raise Exception()

                    # elif:
                    #     print("missed lambda",primpdgs)

                # else:
                #     print("missed the interactions",(key,l.interaction.id),interactions.keys())
                #         l=i[0][3122]
                #         if set(l[2])==set([-211, 2212]):
                #             add_it=True
                #             TrueLambdas+=1
                #             if i[-1][0] in predl:
                #                 for l in predl[i[-1][0]]:
                #                     if l.reason=="" and np.linalg.norm(l.interaction.vertex-I[:3])<.001:
                #                         add_it=False
                #                         print("found the lambda in truth")
                #             if add_it:
                                
                #                 print("completely missed a lambda")
                #                 # try:
                #                 save_my_html(LOCAL_EVENT_DISPLAYS+'/completely_missed',create_html_filename(i[-1][0], lfile,extra=""),i[-1][0])
                #                 # except AssertionError as e:
                #                     # print("something wrong with the base file",lfile,LOCAL_EVENT_DISPLAYS+'/completely_missed',create_html_filename(i[-1][0], lfile,extra=""),i[-1][0],lfile)



    # test_key=(key,l.interaction.id)
    #                     if test_key in interactions and is_contained(I[:3],margin=margin0):
    #                         print("found the interactions",test_key)
    #                         





            # for key in predl:
            if True:
                for l in predl:

                    if l.hip.id==l.mip.id:
                        continue
                    key=l.event_number

                    inter = l.reco_vertex.astype(np.float64)

                    decaylen=np.linalg.norm(l.reco_vertex-(l.hip.start_point+l.mip.start_point)/2)

                    is_true=l.truth*is_contained(l.truth_interaction_vertex,margin=margin0)
                    truth_hip=l.truth_hip
                    truth_mip=l.truth_mip
                    # if args.mode=="truth":
                    #     truth_mip=l.mip
                    #     truth_hip=l.hip

                    

                    def quick_save(dir):
                        save_my_html(LOCAL_EVENT_DISPLAYS+dir,create_html_filename(key, lfile,extra=str(l.mip.id)+"_"+str(l.hip.id)),key)
                    

                    if l.error!="":
                        print("VERY BAD ERROR",key,l.error)
                        quick_save('/error')
                    # print("event",key)
                    # if l.true_signal:
                    #     # print("found a true_signal lambda")
                    #     lam_mass[2] += [np.sqrt(l.mass2)/LAM_MASS]
                    # if l.mass2 < 0:
                    #     print("THIS IS VERY BAD")

                    fixed_prot_mom=np.float64(l.real_hip_momentum_reco)
                    fixed_pi_mom=np.float64(l.real_mip_momentum_reco)

                    if is_true:
                        assert truth_hip is not None
                        assert truth_mip is not None
                        if len(l.hm_pred): HM_acc_prot += [l.hm_pred[l.hip.id][HIP_HM]/(l.hm_pred[l.hip.id][HIP_HM]+l.hm_pred[l.hip.id][MIP_HM])]
                        if len(l.hm_pred): HM_acc_pi += [l.hm_pred[l.mip.id][MIP_HM]/(l.hm_pred[l.mip.id][HIP_HM]+l.hm_pred[l.mip.id][MIP_HM])]
                        prot_primary[int(l.hip.is_primary)]+=[decaylen]
                        pi_primary[int(l.mip.is_primary)]+=[decaylen]

                        prot_mom[0]+=[np.linalg.norm(fixed_prot_mom)]
                        prot_mom[1]+=[truth_hip.p]
                        pi_mom[0]+=[np.linalg.norm(fixed_pi_mom)]
                        pi_mom[1]+=[truth_mip.p]


                        pi_dir+=[angle_between(l.mip.momentum,truth_mip.momentum)]
                        prot_dir+=[angle_between(l.hip.momentum,truth_hip.momentum)]

                        vertex_displacement[0]+=[np.linalg.norm(l.truth_interaction_vertex-inter)]
                        vertex_dz[0]+=[l.truth_interaction_vertex[2]-inter[2]]

                    pc=l.pass_cuts(lam_pass_order)

                    


                    # if pc and not is_true:
                    #     if not is_contained(inter,margin=margin0):
                    #         continue

                    lpf=l.pass_failure[0]

                    # lam_mass[is_true][lpf==""] += [np.sqrt(l.mass2)]#[(l.mass2-PION_MASS**2-PROT_MASS**2)/(LAM_MASS**2-PION_MASS**2-PROT_MASS**2)]


                    


                    


                    if type(l.hip)==TruthParticle:
                        assert type(truth_mip)==TruthParticle
                        assert type(truth_hip)==TruthParticle
                        if truth_mip.ancestor_pdg_code in [3122,3212] and truth_hip.ancestor_pdg_code in [3122,3212] and l.hip.pdg_code==2212 and truth_mip.pdg_code==-211 and truth_hip.parent_pdg_code==3122 and truth_mip.parent_pdg_code==3122 and process_map[truth_hip.creation_process]=='6::201' and process_map[truth_mip.creation_process]=='6::201':
                            primary_phot=[p for p in l.particles if p.is_primary and p.pdg_code==22]
                            for p in primary_phot:
                                assert type(p)==TruthParticle
                                sigma_mass[p.ancestor_pdg_code==3212]+=[mom_to_mass(fixed_prot_mom+fixed_pi_mom,p.reco_momentum,LAM_MASS,0)]


                    # fixed_prot_E=np.sqrt(np.linalg.norm(fixed_prot_mom)**2+PROT_MASS**2)
                    # fixed_pi_E=np.sqrt(np.linalg.norm(fixed_pi_mom)**2+PION_MASS**2)

                    


                    lam_mass_fixed[is_true][lpf==""]+=[mom_to_mass(fixed_prot_mom,fixed_pi_mom,PROT_MASS,PION_MASS)]

                    if is_true and (lam_mass_fixed[is_true][lpf==""][-1]<1.08*1000 ):
                        print("MESSED UP MASS",key,l)
                        quick_save('/error/lowmass')

                    if is_true and (lam_mass_fixed[is_true][lpf==""][-1]>1.15*1000 ):
                        print("MESSED UP MASS",key,l)
                        quick_save('/error/highmass')
                        

                    if not is_true and lpf=="" and np.dot((l.hip.start_point+l.mip.start_point)/2-inter,fixed_prot_mom + fixed_pi_mom)<0:
                        print("MESSED UP direction",key,l)
                        quick_save('/error/direction')

                    if is_true:
                        if np.linalg.norm((l.hip.start_point+l.mip.start_point)/2-inter)>10**7:
                            print("infinite vertex",key,l)
                            quick_save('/error/infinite_vertex')







                    #########################
                    if (not is_true) and lpf=="":
                        if truth_mip is None:
                            quick_save('/backgrounds/no_MIP_match')
                        elif truth_hip is None:
                            quick_save('/backgrounds/no_HIP_match')
                        else:
                            assert type(truth_hip)==TruthParticle
                            assert type(truth_mip)==TruthParticle


                            if truth_mip.ancestor_pdg_code==3212:
                                quick_save('/backgrounds/sigma0')

                            elif truth_mip.parent_pdg_code==3122 and truth_hip.parent_pdg_code==2212:
                                quick_save('/backgrounds/missing_proton_parent')
                            elif truth_mip.parent_pdg_code==3122 and truth_hip.parent_pdg_code==-211 and truth_hip.creation_process=="hBertiniCaptureAtRest":
                                quick_save('/backgrounds/pioncapture')
                            elif truth_mip.parent_pdg_code==2112:
                                if truth_mip.pdg_code==-211:
                                    quick_save('/backgrounds/neutron_pim')
                                elif truth_mip.pdg_code==211:
                                    quick_save('/backgrounds/neutron_pip')
                                else:
                                    quick_save('/backgrounds/neutron_random')
                                
                            elif truth_mip.creation_process=="lambdaInelastic":
                                quick_save('/backgrounds/lambdaInelastic')

                            elif truth_mip.creation_process in ["protonInelastic","pi+Inelastic","kaon0LInelastic","muonNuclear"]:
                                quick_save('/backgrounds/otherInelastic')
                            elif truth_mip.ancestor_creation_process=="lambdaInelastic":
                                quick_save('/backgrounds/lambdaInelasticancestor')
                            elif truth_mip.ancestor_creation_process in ["pi+Inelastic","pi-Inelastic","kaon0LInelastic"]:
                                quick_save('/backgrounds/secondarylambdainelastic')
                            # elif l.reason=="11":
                            #     quick_save('/backgrounds/pi0_containment')
                            elif not is_contained(l.truth_interaction_vertex,margin=margin0):
                                quick_save('/backgrounds/reco_vertex_out_of_bounds')

                            elif truth_hip.ancestor_pdg_code==311 and truth_mip.ancestor_pdg_code==311:
                                quick_save('/backgrounds/K0311')
                            elif truth_hip.ancestor_pdg_code==3112 and truth_mip.ancestor_pdg_code==3112:
                                quick_save('/backgrounds/sigmaminus')

                            elif l.truth_interaction_nu_id==-1:
                                quick_save('/backgrounds/cosmics')

                            elif truth_mip.is_primary and truth_hip.is_primary and truth_hip.shape==TRACK_SHP and truth_mip.shape==TRACK_SHP:
                                quick_save('/backgrounds/two_truth_primary_tracks')

                            elif truth_mip.is_primary and truth_hip.parent_pdg_code==3122:
                                quick_save('/backgrounds/correct_hip_primary_mip')

                            elif truth_mip.pdg_code==2212 and truth_hip.pdg_code==-211 and truth_mip.parent_pdg_code==3122 and truth_hip.parent_pdg_code==3122 and truth_hip.ancestor_pdg_code==3122:
                                quick_save('/backgrounds/correct_lambda_but_reversed')

                            elif truth_mip.id==truth_hip.id:
                                quick_save(f'/backgrounds/same_truth_particle_{truth_mip.pdg_code}_from_{truth_mip.parent_pdg_code}_w_ancestor_{truth_mip.ancestor_pdg_code}')

                            else:
                                quick_save('/backgrounds/unknown_'+l.reason)
                        # if is_true and lpf!="":
                        #     if type(l.interaction)==RecoInteraction:
                        #         pass
                        #         #TODO this clustering code
                        #     #     assert len(l.truth_interaction.match_overlaps[l.truth_interaction.match_ids==l.interaction.id])==1
                        #     # if l.truth_interaction.match_overlaps[l.truth_interaction.match_ids==l.interaction.id]<.75 and np.min(cdist(np.concatenate([i.points for i in l.particles]), [l.truth_interaction_vertex]))>5:
                        #     #     quick_save('/missing/lambda_not_clustered')
                        #     # if np.min(cdist([i.points for i in l.particles], [truth_interaction.vertex]))>5:
                        #     else:
                        #         quick_save('/missing/unknown_'+lpf)







                    ##########################

                


                    # momenta[0][0] += [l.momenta[0]]
                    # momenta[0][1] += [l.momenta[1]]


                    ppi_dist=np.linalg.norm(l.hip.start_point-l.mip.start_point)
                    # ppi_dist=np.min(cdist([l.hip.start_point,l.hip.end_point] ,[l.mip.start_point,l.mip.end_point]))
                    
                    lam_decay_len[is_true][lpf==""] += [decaylen]
                    lam_decay_len_disc_total[is_true][(l.pass_failure==[""])+(l.pass_failure==["Min Decay Len",""])]+=[decaylen]
                    # lam_dir_acos[is_true][lpf==""] += [l.lam_dir_acos]
                    

                    


                    ProtPi_dist[is_true][lpf==""]+= [ppi_dist]

                    lam_mass_disc_total[is_true][(l.pass_failure==[""])]+=[mom_to_mass(fixed_prot_mom,fixed_pi_mom,PROT_MASS,PION_MASS)]


                    ProtPi_dist_disc_total[is_true][(l.pass_failure==[""])+(l.pass_failure==["Max HIP-MIP Sep.",""])]+= [ppi_dist]
                    
                    vae[is_true][lpf==""] += [np.arcsin(l.vae / decaylen if decaylen != 0 else 0)]
                    hip_len[is_true][lpf==""]+=[l.hip.reco_length]
                    mip_len[is_true][lpf==""]+=[l.mip.reco_length]
                    mom_norm=np.linalg.norm(l.hip.reco_momentum+l.mip.reco_momentum)
                    # mom_norm=np.linalg.norm(fixed_prot_mom+fixed_pi_mom)
                    
                    mom_norm_fixed=np.linalg.norm(fixed_prot_mom+fixed_pi_mom)
                    lam_momentum[is_true][lpf==""]+=[mom_norm_fixed]
                    lam_true_momentum[is_true][lpf==""]+=[np.linalg.norm(l.hip.momentum+l.mip.momentum)]

                    lam_tau0_est[is_true][lpf==""]+=[decaylen/100/(2.998e8)*LAM_MASS/mom_norm*10**9]
                    lam_tau0_est_fixed[is_true][lpf==""]+=[decaylen/100/(2.998e8)*LAM_MASS/mom_norm_fixed*10**9]

                    # if lpf=="":print(decaylen/100/(2.998e8)*LAM_MASS/mom_norm*10**9,decaylen/100/(2.998e8)*LAM_MASS/mom_norm_fixed*10**9)
                    # if is_true:
                        # print("lam tau est",lam_tau0_est[is_true][lpf==""],decaylen,np.sqrt(l.mass2),mom_norm)
                    
                    # print(pc,l.mip.id,l.hip.id)

                    # base_len_vae[is_true][lpf==""]+=[l.vae]

                    # base_len_vae_disc_total[is_true][(l.pass_failure==[""])+(l.pass_failure==["Impact Parameter",""])]+=[decaylen*np.sin(min(l.vae,np.pi/2))]

                    # v1=l.mip.momentum
                    # v2=l.hip.momentum
                    # p0=l.mip.start_point
                    # p=l.interaction.reco_vertex

                    
                    guess_start=((l.hip.start_point+l.mip.start_point)/2).astype(np.float64)
                    # guess_start = get_pseudovertex(
                    #     start_points=np.array([self.hip.start_point, self.mip.start_point], dtype=float),
                    #     directions=[self.hip.reco_start_dir, self.mip.reco_start_dir],
                    # )
                    # assert np.linalg.norm(guess_start-guess_start)==0
                    vae0=0

                    dir1 = (guess_start - inter).astype(np.float64)
                    # if not np.isclose(np.linalg.norm(dir1),0):


                    
                        
                    # vae1=0
                    # vae2=0
                    # ret=0
                    # # dir1 = guess_start - inter
                    # if (not np.isclose(np.linalg.norm(dir1),0)) and (not np.isclose(np.linalg.norm(fixed_pi_mom),0)) and (not np.isclose(np.linalg.norm(fixed_prot_mom),0)):

                    #     dir2 = fixed_prot_mom + fixed_pi_mom

                    #     ret = angle_between(dir1,dir2)
                    #     # if passed: assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,dir1,dir2)
                    #     vae0=decaylen*np.sin(min(ret,np.pi/2))
                    #     # dir2 = fixed_prot_mom + fixed_pi_mom
                    #     vae1=point_to_plane_distance(inter,guess_start,fixed_prot_mom,fixed_pi_mom)
                    #     if np.dot(dir2,dir1)<0:
                    #         vae1=vae0
                    #     # vae2=point_to_plane_distance_in_plane(inter,guess_start,fixed_prot_mom,fixed_pi_mom)
                    #     assert vae1<=vae0 or np.isclose(vae1-vae0,0,rtol=.001, atol=.001),(vae1,vae0,vae1-vae0,inter,guess_start,fixed_prot_mom,fixed_pi_mom)
                    #     vae2=np.sqrt(abs(vae0**2-vae1**2))
                    #     if np.dot(dir2,dir1)<0:
                    #         vae2=vae0

                        # vae1=np.arccos(vae1/decaylen)
                        # vae2=np.arccos(vae2/decaylen)


                        # ret = np.arccos(
                        #     np.dot(dir1, dir2) / np.linalg.norm(dir1) / np.linalg.norm(dir2)
                        # )
                        # # if passed: assert ret == ret,(self.hip.start_point,self.mip.start_point,self.hip.reco_start_dir,self.mip.reco_start_dir,ret,dir1,dir2)
                        # vae=ret

                    base_len_vae_disc_total[is_true][(l.pass_failure==[""])+(l.pass_failure==["Impact Parameter",""])]+=[l.vae]#[decaylen*np.sin(min(ret,np.pi/2))]

                    # base_len_vae_disc_perp[is_true][(l.pass_failure==[""])+(l.pass_failure==["Impact Parameter","12344lsfdhalsfhk"])]+=[vae1]
                    # base_len_vae_disc_parr[is_true][(l.pass_failure==[""])+(l.pass_failure==["Impact Parameter","sklhjafklhdfjklhafhjkfshjk"])]+=[vae2]









                    if not is_true:
                        lam_pass_failure[1][lpf]+=1
                        lam_reason_map[1][l.reason]+=1
                    
                    if is_true:
                        lam_pass_failure[0][lpf]+=1
                        lam_reason_map[0][l.reason]+=1
                        # print("MOMENTUM COMPARISON",l.mip.momentum,l.fi)

                        vertex_displacement[1]+=[np.linalg.norm(l.truth_interaction_vertex-inter)]
                        vertex_dz[1]+=[l.truth_interaction_vertex[2]-inter[2]]

                        # current_type_map[l.interaction.current_type]+=1
                        # print("current_type",current_type_map)
                        

                    if pc:
                        print("MOM_NORM",[mom_norm,mom_norm_fixed],[np.linalg.norm(l.hip.momentum+l.mip.momentum),np.linalg.norm(l.hip.reco_momentum+l.mip.reco_momentum)])
                        selectedlambdas+=1
                        if not is_true:
                            quick_save('/false_found/'+l.reason)
                            print("GOT A FALSE LAMBDA",l.mip.id,l.hip.id,key,lfile,
                            "    ",
                            # np.sqrt(l.mass2)/LAM_MASS,#(l.mass2-PION_MASS**2-PROT_MASS**2)/(LAM_MASS**2-PION_MASS**2-PROT_MASS**2),
                            decaylen,
                            # l.lam_dir_acos,
                            # l.prot_hm_acc,
                            # l.pi_hm_acc,
                            # l.coll_dist[-1],
                            # l.momenta[0] / LAM_PT_MAX,
                            l.vae,
                            l.mip.reco_length,
                            l.hip.reco_length,
                            np.linalg.norm(l.hip.start_point-l.mip.start_point),
                            [l.reason,lpf]
                        )

                    
                    if is_true:
                        
                        # TrueLambdas+=1
                        if pc:
                            quick_save('/true_found')
                            correctlyselectedlambdas+=1
                            print("true lambda")
                        else:
                            quick_save('/true_missed/'+lpf)
                            print("MISSED A GOOD LAMBDA",key,lfile,l.hip.pdg_code, l.pass_failure)
                        # if l.mass2 < 0:
                        #     raise Exception("THIS IS VERY BAD")
                        # if np.linalg.norm(l.interaction.vertex-l.interaction.reco_vertex)>1: print("VERY VERY VERY BAD",np.linalg.norm(l.interaction.vertex-l.interaction.reco_vertex),lfile,key)
                        # lam_mass[1] += [np.sqrt(l.mass2)/LAM_MASS]#[(l.mass2-PION_MASS**2-PROT_MASS**2)/(LAM_MASS**2-PION_MASS**2-PROT_MASS**2)]
                        # momenta[1][0] += [l.momenta[0]]
                        # momenta[1][1] += [l.momenta[1]]
                        # hip_len[1]+=[l.hip_len]
                        # mip_len[1]+=[l.mip_len]
                        # lam_decay_len[1] += [decaylen]
                        # lam_dir_acos[1] += [l.lam_dir_acos]
                        # print(l.lam_dir_acos)
                        # prot_hm_acc[1] += [l.prot_hm_acc]
                        # HM_acc_pi[1] += [l.pi_hm_acc]
                        # ProtPi_dist += [l.coll_dist[-1]]
                        # for prc in l.prot_extra_children:
                        #     prot_extra_children[0][0] += [prc.dist_to_parent]
                        #     prot_extra_children[0][1] += [prc.angle_to_parent]
                        #     if prc.truth:
                        #         prot_extra_children[1][0] += [prc.dist_to_parent]
                        #         prot_extra_children[1][1] += [prc.angle_to_parent]
                        # for pic in l.pi_extra_children:
                        #     pi_extra_children[0][0] += [pic.dist_to_parent]
                        #     pi_extra_children[0][1] += [pic.angle_to_parent]
                        #     if pic.truth:
                        #         pi_extra_children[1][0] += [pic.dist_to_parent]
                        #         pi_extra_children[1][1] += [pic.angle_to_parent]
                        # for lc in l.lam_extra_children:
                        #     lam_extra_children[0][0] += [lc.dist_to_parent]
                        #     lam_extra_children[0][1] += [lc.angle_to_parent]
                        #     if lc.truth:
                        #         lam_extra_children[1][0] += [lc.dist_to_parent]
                        #         lam_extra_children[1][1] += [lc.angle_to_parent]
                        # vae[1] += [l.vae]
                        # if l.vae>.5: print("bad VAE",l.vae,key)
                        # print(
                        #     "true lambda:",
                        #     key,
                        #     "    ",
                        #     np.sqrt(l.mass2)/LAM_MASS,#[(l.mass2-PION_MASS**2-PROT_MASS**2)/(LAM_MASS**2-PION_MASS**2-PROT_MASS**2)]
                        #     decaylen,
                        #     l.lam_dir_acos,
                        #     l.prot_hm_acc,
                        #     l.pi_hm_acc,
                        #     # l.coll_dist[-1],
                        #     # l.momenta[0] / LAM_PT_MAX,
                        #     l.vae,
                        #     l.mip_len,
                        #     l.hip_len,
                        #     np.linalg.norm(l.hip.start_point-l.mip.start_point)
                        # )
                        print("")

                    # momenta[1][2]+=[k.momenta[2]]
                    # if k.AM[2]<.3: print("bad lambda mom?",key)
        # print("results",[correctlyselectedkaons,selectedkaons,TrueKaons],[correctlyselectedlambdas,selectedlambdas,TrueLambdas])
        # print("kaon eff/pur",np.divide(correctlyselectedkaons,TrueKaons),np.divide(correctlyselectedkaons,selectedkaons),[correctlyselectedkaons,selectedkaons,TrueKaons])
        # print("kaon reasons",kaon_reason_map)
        # print("kaon pass_failure",kaon_pass_failure)
    print("num neutrinos",num_nu_from_file)
    print("lam eff/pur",np.divide(correctlyselectedlambdas,TrueLambdas),np.divide(correctlyselectedlambdas,selectedlambdas),[correctlyselectedlambdas,selectedlambdas,TrueLambdas])
    print("lambda reasons",lam_reason_map)
    print("lambda pass_failure",lam_pass_failure)

    if args.single_file!="":
        print(f"returning from {args.single_file}")
        return


    # print()


    # def round_to_2(x):
    #     if x!=x: return np.nan
    #     # if x==np.inf: raise Exception()
    #     # if type(x)==np.float32:x=float(x)/
    #     # if type(x)!=float: raise Exception(x,-int(np.floor(np.log10(np.abs(x)))) + 3)
    #     # if x<0: raise Exception()
    #     # print(x)
    #     if x==0: return 0
    #     return round(x, -int(np.floor(np.log10(np.abs(x)))) + 3)


    # plt.rcParams['text.usetex'] = True #do this for final plots


    # plt.scatter(pt1,lam_decay_len[0],label="Before parameter cuts")
    # if OT:
    #     plt.scatter(pt1_truth,lam_decay_len[1],label=r"True $\Lambda$ decay pairs")
    # plt.xlabel("Dist to closest child [cm]")
    # plt.ylabel("angle to closest child []")
    # plt.xscale('log')
    # ##plt.yscale('log')
    # plt.tight_layout();plt.savefig("plots/"+str(FOLDER)+"/lam_prot_children")
    # plt.clf()
    # (n, bins, patches) = plt.hist(muon_len[0], bins=list(np.linspace(40,60,40)), label="all")
    # plt.hist(muon_len[1], bins=list(bins), label=r"all $K^+-\mu$")
    # # plt.hist(muon_len[2], bins=list(bins), label="signal events",alpha=.5)
    # plt.axvline(x=54, color="r", label="ideal KDAR muon")
    # plt.xlabel("Muon Len [cm]")
    # plt.ylabel("Freq")
    # plt.legend()
    # plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/Muon_Len")
    # plt.clf()


    # mla30_0=[i for i in muon_len_adjusted[0] if i>40 and i<60]
    # mla30_1=[i for i in muon_len_adjusted[1] if i>40 and i<60]
    # print( muon_len_adjusted,"MLA")
    # (n, bins, patches) = plt.hist(
    #     mla30_0,
    #     bins=25,
    #     label=r"True $K^+-\mu^+$ (only $\mu^+$): $\mu (40cm-60cm)$= "
    #     + str(round_to_2(np.mean(mla30_0)))
    #     + r", $\sigma (40cm to 60cm)$= "
    #     + str(round_to_2(np.std(mla30_0))),
    # )
    # plt.hist(
    #     mla30_1,
    #     bins=list(bins),
    #     label=r"True $K^+-\mu^+$ (adjusted): $\mu (40cm-60cm)$= "
    #     + str(round_to_2(np.mean(mla30_1)))
    #     + r", $\sigma (40cm to 60cm)$= "
    #     + str(round_to_2(np.std(mla30_1))),
    #     alpha=.5
    # )

    # for x in [
    #     [k_extra_children,r"signal $K^+$ extra children","kaonkids"],
    #     [mu_extra_children,r"signal $\mu$ extra children","mukids"],
    #     [pi_extra_children,r"signal $\Lambda-\pi^+$ extra children","pikids"],
    #     [prot_extra_children,r"signal $\Lambda-p$ extra children","protkids"],
    #     [lam_extra_children,r"signal $\Lambda$ extra children","lamkids"]
    #     ]:


    #     plt.scatter(x[0][0][0],x[0][0][1], label="Before parameter cuts")

    #     plt.scatter(x[0][1][0],x[0][1][1], label=x[1])
    #     plt.xlabel("parent-child end-to-start distance [cm]")
    #     plt.ylabel("child start dir to end-to-start angle")
    #     # plt.xscale("log")
    #     # plt.yscale("log")
    #     plt.legend()
    #     plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/"+x[2])
    #     plt.clf()

    import math

    def rsf(x, sig=2):
        """Rounds a number to a specified number of significant figures.
        """
        if x == 0:
            return 0.0
        return round(x, sig-int(math.floor(math.log10(abs(x))))-1)
    

    #2D plots

    for s in [(prot_primary,r"Decay Len [cm]","prot_primary"),
        (pi_primary,r"Decay Len [cm]","pi_primary"),
        (vertex_dz,r"Vertex Truth-Reco $\Delta$z [cm]","vertex_dz"),
        (vertex_displacement,r"Vertex Displacement [cm]","vertex_displacement"),]:
        hist_values, bins = np.histogram(s[0][True]+s[0][False], bins=10)
        plt.clf()

        if s[2] in ["prot_primary","pi_primary"]:
            l0="Primary"
            l1="Non-Primary"
            plt.hist(s[0][True], label=l0,alpha=.5,bins=bins.tolist())
            plt.hist(s[0][False], label=l1,alpha=.5,bins=bins.tolist())

        elif s[2] in ["vertex_dz","vertex_displacement"]:
            l0="Out-of-the-box: "+f"mean={np.mean(s[0][0]):.2f}, std={np.std(s[0][0]):.2f}"
            l1="Clustering Patch: "+f"mean={np.mean(s[0][1]):.2f}, std={np.std(s[0][1]):.2f}"
            plt.hist(s[0][0], label=l0,alpha=.5,bins=bins.tolist())
            plt.hist(s[0][1], label=l1,alpha=.5,bins=bins.tolist())
        else:
            raise Exception(s[2])
        
        # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
        plt.xlabel(s[1])
        plt.ylabel("Freq")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout();plt.savefig(PLOTSDIR+"/"+s[2])
        plt.clf()


    


    for s in [
            [HM_acc_pi, r"$\pi$ HM Accuracy", "lam_HM_pi"],
            [HM_acc_prot, r"Proton HM Accuracy", "lam_HM_prot"],
            [pi_dir, r"$\pi$ $\theta$ from True Dir.", "lam_pi_dir"],
            [prot_dir, r"Proton $\theta$ from True Dir.", "lam_prot_dir"]]:
        
        print("running",s[2])
        #prot_primary+=[decaylen]
                    # pi_primary+=[decaylen]

                    # prot_mom[0]+=[fixed_prot_mom]
                    # prot_mom[1]+=[truth_hip.p]
                    # pi_mom[0]+=[fixed_pi_mom]
                    # pi_mom[1]+=[truth_mip.p]


                    # pi_dir+=[angle_between(l.mip.momentum,truth_mip.p)]
                    # prot_dir+=[angle_between(l.hip.momentum,truth_hip.p)]

        # print(s[0],np.isnan(s[0]))

        rmnans=[z for z in s[0] if not np.isnan(z)]

        plt.hist(s[0], label=f"mean={np.mean(rmnans):.2f}, std={np.std(rmnans):.2f}",bins=50)
        # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
        plt.xlabel(s[1])
        plt.ylabel("Freq")
        # if s[2] in ["vertex_dz","vertex_displacement"]:
        plt.yscale("log")
        plt.legend()
        plt.tight_layout();plt.savefig(PLOTSDIR+"/"+s[2])
        plt.clf()



    tau_max=fakecuts["tau_max"]
    tau_min=0.2631
    # from scipy.optimize import curve_fit

    # def exp_con_cdf0(t, tau,s):
    #     return -s*np.exp(-t/tau)#(1-s)*t/(tau_max-tau_min)

    # def exp_con_cdf(t, tau,s):
    #     return (exp_con_cdf0(t, tau,s)-exp_con_cdf0(tau_min, tau,s))/(exp_con_cdf0(tau_max, tau,s)-exp_con_cdf0(tau_min, tau,s))

    def exp_con_pdf(t, tau,s):
        return s/tau*np.exp(-t/tau)#(1-s)/(tau_max-tau_min)+

    # for tau in [[lam_tau0_est,r"$\tau_0~[ns]$","t0_est_total_fit"],
    #             [lam_tau0_est_fixed,r"$\tau_0~[ns]$","t0_est_fixed_total_fit"]
    #             ]:

    #     big_val=tau[0][True][True]+tau[0][False][True]
    #     big_val=np.array(big_val)
    #     bigval=big_val[(big_val>tau_min)&(big_val<tau_max)]
    #     # bins=np.logspace(np.log10(min(big_val)), np.log10(max(big_val)), 20)
    #     # (n, bins, patches) =plt.hist(big_val, bins=20, label=f"remaining after cuts")
    #     (n, bins, patches) =plt.hist(big_val,label=f"All after cuts",bins=30)
    #     plt.hist(tau[0][False][True],bins=bins,label=f"Background after cuts")
    #     # plt.hist(tau[0][actual][True], bins=list(bins), label=f"remaining after cuts")

    #     # points=np.logspace(np.log10(min(big_val)), np.log10(max(big_val)), 200)
    #     # points=np.linspace(0,max(tau[0][actual][True]),100)
    #     # plt.plot(points,np.exp(-points/0.2631)/0.2631*len(tau[0][actual][True])*(bins[1]-bins[0]),label=r" Expected $t_{decay}$ Distribution [ns]")
    #     sorted_data = np.sort(big_val)
    #     cdf = np.arange(1, len(big_val) + 1) / len(big_val)

    #     popt, pcov = curve_fit(exp_con_cdf, sorted_data, cdf, bounds=([.0000000001,0],[np.inf,1]),p0=(1,.9))
    #     perr = np.sqrt(np.diag(pcov))

    #     # Generate fitted CDF values
    #     # fitted_cdf = gaussian_cdf(sorted_data, *popt)


    #     plt.axvline(x = 0.2631, color = 'r', label = r'$\tau_{0\Lambda}~[0.2631~ns]$')
    #     points=np.linspace(0,tau_max,100)
    #     plt.plot(points,exp_con_pdf(points,popt[0],popt[1])*len(big_val)*(bins[1]-bins[0]),label=r"Cons+Exp fit: $t_{decay}=$"+str(round_to_2(popt[0]))+r"\pm"+str(round_to_2(perr[0])))
    #     # plt.xscale('log')
    #     plt.ylim(0, 1.1*len(big_val)*(bins[1]-bins[0]))
    #     plt.xlabel(tau[1])
    #     plt.ylabel("Freq")
    #     # plt.yscale("log")
    #     plt.legend()
    #     plt.tight_layout();plt.savefig(PLOTSDIR+tau[2])
    #     plt.clf()


    plt.hist(sigma_mass[False],label=r"Other Primary $\gamma$",bins=np.linspace(LAM_MASS, 1.3*SIG0_MASS,31).tolist(),alpha=.5)
    plt.hist(sigma_mass[True],label=r"Primary $\gamma$ from $\Sigma^0$",bins=np.linspace(LAM_MASS, 1.3*SIG0_MASS, 31).tolist(),alpha=.5)
    

    plt.xlabel(r"$\Sigma^0$ Candidate Mass [MeV]")
    plt.ylabel("freq")
    plt.grid(True) # Enables grid lines
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.axvline(x = SIG0_MASS, color = 'r', label = r'$M_{\Sigma^0} [1192.642 MeV]$')
    
    plt.tight_layout();plt.savefig(PLOTSDIR+"/sigma0_mass")
    plt.clf()


    for actual in [True,False]:


        # for pred in [True,False]:




        plt.scatter(vae[actual][True]+vae[actual][False],lam_decay_len[actual][True]+lam_decay_len[actual][False],label=f"all {actual}")
        plt.scatter(vae[actual][True],lam_decay_len[actual][True],label=f"{actual} after cuts",alpha=.75)

        # plt.scatter(vae[1],lam_decay_len[1],label="only truth")
        # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
        plt.xlabel(r"Angle between $\Lambda$ momentum estimates")
        plt.ylabel(r"$\Lambda$ decay length [cm]")
        plt.grid(True) # Enables grid lines
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.tight_layout();plt.savefig(PLOTSDIR+"/lam_vae_"+str(actual))
        plt.clf()


    for actual in [True,False]:

        for help in [
            # [dist_to_hip,"Extra mu dist from hip [cm]","hip_mip_dist"],
            # [dist_to_mich,"Extra mu dist from mich [cm]","mip_mich_dist"],
            # [kaon_len,"Kaon Len [cm]","Kaon_Len"],
            # [HM_acc_mu, r"$\mu$ HM scores", "mu_HM"],
            # [HM_acc_K, r"K HM scores", "K_HM"],
            # [HM_acc_mich, r"Michel HM scores", "michel_HM"],
            # [HM_acc_pi, r"$\Lambda$ $\pi$ HM scores", "lam_HM_pi"],
            # [HM_acc_prot, r"$\Lambda$ Proton HM scores", "lam_HM_prot"],
            [ProtPi_dist,r"$\Lambda$ p-$\pi$ Proj. dist [cm]","lam_prot_pi_dist"],
            [lam_decay_len,r"$\Lambda$ Decay Length [cm]","lam_decay_len"],
            # [lam_dir_acos,"Lambda Acos momentum to beam","lam_Acos"],
            [lam_momentum,r"$\Lambda$ Momentum [MeV/c]","lam_mom"],
            [lam_true_momentum,r"Geant4 $\Lambda$ Momentum [MeV/c]","lam_mom_g4"],
            # [base_len_vae,"Impact Parameter [cm]","base_len_vae"]

            # [lam_tau0_est,r"$\tau_0=\frac{dx_{decay}m_{est}}{p_{est}}~[ns]$","t0_est"]
            # [dir_acos_K,"Kaon Acos momentum to beam","K_Acos"]
            
            ]:
                
                # plt.hist(base_len_vae,bins=30)
    # plt.xlabel("VAE base length [cm]")
    # plt.ylabel("Freq")
    # plt.yscale("log")
    # plt.tight_layout();plt.savefig(PLOTSDIR+"base_len_vae")
    # plt.clf()
                print(help[1])
                maxval=help[0][actual][True]
                mine=np.array(help[0][actual][True])
                (n, bins, patches)=plt.hist(mine[mine<np.inf], bins=25, label=f"{actual} after cuts")
                plt.clf()
                plt.hist(help[0][actual][True]+help[0][actual][False], bins=bins.tolist(),label=f"all {actual}")
                # plt.hist(help[0][actual][True], bins=list(bins), label=f"{actual} after cuts")
                # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
                # if help[2]=="lam_decay_len":
                # plt.xlim(right=np.max(help[0][actual][True]))

                # if help[2]=="t0_est":
                #     plt.axvline(x = np.log(0.2631), color = 'r', label = r'$\tau_{0\Lambda}~[ns]$')
                #     # plt.xlim(right=np.max(help[0][actual][True]))
                # if len(help[0][actual][True])>0:
                #     plt.xlim(right=np.max(help[0][actual][True]))
                plt.xlabel(help[1])
                plt.ylabel("Freq")
                #plt.yscale("log")
                # plt.legend()
                plt.tight_layout();plt.savefig(PLOTSDIR+help[2]+"_"+str(actual))
                plt.clf()



    for help in [
        # [dist_to_hip,"Extra mu dist from hip [cm]","hip_mip_dist"],
        # [dist_to_mich,"Extra mu dist from mich [cm]","mip_mich_dist"],
        # [kaon_len,"Kaon Len [cm]","Kaon_Len"],
        # [HM_acc_mu, r"$\mu$ HM scores", "mu_HM"],
        # [HM_acc_K, r"K HM scores", "K_HM"],
        # [HM_acc_mich, r"Michel HM scores", "michel_HM"],
        # [HM_acc_pi, r"$\Lambda$ Pi HM scores", "lam_HM_pi"],
        # [HM_acc_prot, r"$\Lambda$ Proton HM scores", "lam_HM_prot"],
        [ProtPi_dist_disc_total,r"$\Lambda$ p-$\pi$ proj. dist [cm]","lam_prot_pi_dist_disc","Max HIP-MIP Sep."],
        [lam_decay_len_disc_total,r"$\Lambda$ decay len [cm]","lam_decay_len_disc","Min Decay Len"],
        # [lam_dir_acos,"Lambda Acos momentum to beam","lam_Acos"],
        # [lam_momentum,"Lambda Momentum [MeV/c]","lam_mom"],
        # [lam_true_momentum,"Geant4 Lambda Momentum [MeV/c]","lam_mom_g4"],
        [base_len_vae_disc_total,"Impact Parameter [cm]","base_len_vae_disc","Impact Parameter"],
        [base_len_vae_disc_perp,"Perp. Impact Parameter [cm]","base_len_vae_perp","VAE max new"],
        [base_len_vae_disc_parr,"Planar Impact Parameter [cm]","base_len_vae_parr","VAE max new"],
        # [lam_mass_disc_total,r"$M_{\Lambda_{candidate}} [MeV]$","lam_mass_disc","lam_mass"]

        # [lam_tau0_est,r"$\tau_0=\frac{dx_{decay}m_{est}}{p_{est}}~[ns]$","t0_est"]
        # [dir_acos_K,"Kaon Acos momentum to beam","K_Acos"]
        
        ]:
            
            
            # for actual in [True,False]:
                
                # plt.hist(base_len_vae,bins=30)
    # plt.xlabel("VAE base length [cm]")
    # plt.ylabel("Freq")
    # plt.yscale("log")
    # plt.tight_layout();plt.savefig(PLOTSDIR+"base_len_vae")
    # plt.clf()
                # print(help[1])
                # maxval=help[0][actual][True]
            # mine=np.array(help[0][True][True]+)

            ##############
            # mine=np.array(help[0][True][True]+help[0][True][False]+help[0][False][True]+help[0][False][False])
            # (n, bins, patches)=plt.hist(mine[mine<10], bins=25)
            # plt.clf()
            # for actual in [False,True]:
            #     plt.hist(help[0][actual][True]+help[0][actual][False], bins=bins,label=f"all {actual}")
            #     # alpha=1
            #     # if actual==False:
            #     #     alpha=.5
            # count=2
            # for actual in [False,True]:
            #     # print(help[0][actual][True])
            #     plt.hist(help[0][actual][True], bins=list(bins), label=f"{actual} after other cuts",facecolor='none',edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][count])
            #     count+=1
            # plt.axvline(x=testcuts[help[3]], color='red', linestyle='--', linewidth=2, label=f"cut value={testcuts[help[3]]}")

            #################################


            mine=np.array(help[0][True][True]+help[0][True][False]+help[0][False][True]+help[0][False][False])
            # if help[3]=="lam_mass":
                # (n, bins, patches)=plt.hist(mine[mine<=1500], bins=50)
                # plt.clf()

                # for actual in [False,True]:
                #     # print(help[0][actual][True])
                #     plt.hist(help[0][actual][True], bins=list(bins), label=rf"all {actual}: $\mu$= "+ str(round_to_2(np.mean(help[0][actual][True])))+ r", $\sigma$= "+ str(round_to_2(np.std(help[0][actual][True]))),alpha=.7)
                # plt.axvline(x = (PION_MASS+PROT_MASS), color = 'g', label = r'$M_{\pi^-}+M_{p} ['+str(round_to_2(PION_MASS+PROT_MASS))+' GeV]$',linestyle='--', linewidth=2,)
                # plt.axvline(x = LAM_MASS, color = 'r', label = r'$M_{\Lambda} ['+str(round_to_2(LAM_MASS))+' GeV]$',linestyle='--', linewidth=2,)
                # pass
                
            # else:
            (n, bins, patches)=plt.hist(mine[mine<=10], bins=50)
            plt.clf()
            for actual in [False,True]:
                # print(help[0][actual][True])
                plt.hist(help[0][actual][True], bins=list(bins), label=f"{actual} after other cuts",alpha=.7)
            plt.axvline(x=(fakecuts|lam_pass_order)[help[3]], color='red', linestyle='--', linewidth=2, label=f"cut value={(fakecuts|lam_pass_order)[help[3]]}")

                
                
                # count+=1
            # if help[2]=="mip_len":
                # plt.axvspan(testcuts["mu_len"][0], testcuts["mu_len"][1], color='red', alpha=0.3, label=r"$K^+\rightarrow \nu\mu$ band")
                # plt.axvspan(testcuts["pi_len"][0], testcuts["pi_len"][1], color='blue', alpha=0.3, label=r"$K^+\rightarrow \pi^+\pi^0$ band")
            # count=2
            for actual in [True]:
                # np.hist(help[0][actual][True]+help[0][actual][False], bins=bins,label=f"all {actual}",facecolor='none',edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],alpha=.5)

                counts, bin_edges = np.histogram(help[0][actual][True]+help[0][actual][False], bins=bins)

                # Plot histogram bars with transparent fill and visible edges
                # plt.hist(help[0][actual][True]+help[0][actual][False], bins=bins,
                #         facecolor='red',
                #         edgecolor='none',  # we'll draw our own outline
                #         alpha=0.5)

                # Overlay a border using step plot
                # We duplicate the counts to match bin_edges length for 'post' alignment
                step_edges = np.repeat(bin_edges, 2)[1:-1]
                step_heights = np.repeat(counts, 2)
                plt.plot(step_edges, step_heights,
                        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
                        label=f"all {actual}",alpha=.5)
                # count+=1



                # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
                # if help[2]=="lam_decay_len":
                # plt.xlim(right=np.max(help[0][actual][True]))

                # if help[2]=="t0_est":
                #     plt.axvline(x = np.log(0.2631), color = 'r', label = r'$\tau_{0\Lambda}~[ns]$')
                #     # plt.xlim(right=np.max(help[0][actual][True]))
                # if len(help[0][actual][True])>0:
                #     plt.xlim(right=np.max(help[0][actual][True]))

            plt.xlabel(help[1])
            plt.ylabel("Freq")
            if help[2]=="lam_decay_len_disc":
                plt.yscale("log")
            plt.legend()
            plt.tight_layout();plt.savefig(PLOTSDIR+help[2])
            plt.clf()
    #for both of these, I want to plot the ones which do pass the cuts
    # for actual in [True]:

    for tau in [#[lam_tau0_est,r"$\tau_0~[ns]$","t0_est"],
                [lam_tau0_est_fixed,r"$\tau_0~[ns]$","t0_est_fixed"],
                #[lam_mass,r"$M_{\Lambda_{candidate}} [MeV]$","lam_mass"],
                [lam_mass_fixed,r"$M_{\Lambda_{candidate}} [MeV]$","lam_mass_fixed"]
                ]:

        truth=np.array(tau[0][True][True]+tau[0][True][False])
        passed=np.array(tau[0][True][True]+tau[0][False][True])
        background=np.array(tau[0][False][True])

        tot=np.array(tau[0][True][True]+tau[0][True][False]+tau[0][False][True])

        truth=truth[truth<=1400]
        passed=passed[passed<=1400]
        background=background[background<1400]
        tot=tot[tot<1500]

        # for i in [truth,passed,background]: i=i[i<=1300]

        # bins=np.logspace(np.log10(min(big_val)), np.log10(max(big_val)), 20)
        if tau[2] in ["t0_est","t0_est_fixed"]:
            (n, bins, patches) =plt.hist(tot, bins=np.linspace(0,5,31).tolist())
        else:
            (n, bins, patches) =plt.hist(tot, bins=30)
            
        plt.clf()
        
        

        if tau[2] in ["t0_est","t0_est_fixed"]:
        # points=np.logspace(np.log10(min(big_val)), np.log10(max(big_val)), 200)
            c0,b0,_=plt.hist(truth, bins=list(bins),label=rf"Signal $\Lambda$",alpha=.5)
            plt.ylim(0, 1.2*max(c0))
            
            counts, bin_edges, _ = plt.hist(passed, bins=list(bins), label=r"Selected $\Lambda$ + Bkg.",alpha=.5)
            
            points=np.linspace(0,bins[-1],100)
            plt.hist(tau[0][False][True],bins=bins.tolist(),label=f"Background",alpha=.5)
            plt.plot(points,np.exp(-points/0.2631)/0.2631*len(passed)*(bins[1]-bins[0]),label=r"Exp. Distr.: $\tau_{0\Lambda}~[0.2631~ns]$")

            # r'$\tau_{0\Lambda}~[0.2631~ns]$'
            

            # big_val=np.array(passed)
            # bigval=big_val[(big_val>tau_min)&(big_val<tau_max)]
            
            # bins=np.logspace(np.log10(min(big_val)), np.log10(max(big_val)), 20)
            # (n, bins, patches) =plt.hist(big_val, bins=20, label=f"remaining after cuts")
            # (n, bins, patches) =plt.hist(big_val,label=f"All after cuts",bins=30)
            # plt.hist(tau[0][False][True],bins=bins,label=f"Background after cuts")
            # plt.hist(tau[0][actual][True], bins=list(bins), label=f"remaining after cuts")

            # points=np.logspace(np.log10(min(big_val)), np.log10(max(big_val)), 200)
            # points=np.linspace(0,max(tau[0][actual][True]),100)
            # plt.plot(points,np.exp(-points/0.2631)/0.2631*len(tau[0][actual][True])*(bins[1]-bins[0]),label=r" Expected $t_{decay}$ Distribution [ns]")
            # sorted_data = np.sort(big_val)
            # cdf = np.arange(1, len(big_val) + 1) / len(big_val)

            # counts, bin_edges = np.histogram(big_val, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            counts=counts[(bin_centers>tau_min)&(bin_centers<tau_max)]
            bin_centers=bin_centers[(bin_centers>tau_min)&(bin_centers<tau_max)]
            err=np.sqrt(counts)
            err[err==0]=1

    # Calculate bin centers
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


            



    # return bin_centers.tolist(), counts.tolist()
            print(bin_centers,counts)

            # popt, pcov = curve_fit(exp_con_pdf,bin_centers , counts,sigma=err, bounds=([.0000000001,0],[np.inf,np.inf]),p0=(1,.9),maxfev=1000000)
            # perr = np.sqrt(np.diag(pcov))


            # Generate fitted CDF values
            # fitted_cdf = gaussian_cdf(sorted_data, *popt)


            # plt.axvline(x = 0.2631, color = 'r', label = r'$\tau_{0\Lambda}~[0.2631~ns]$')
            # custom_entry = Line2D([0], [0], color='none', lw=3, label=r'$\tau_{0\Lambda}~[0.2631~ns]$')
            # plt.legend([custom_entry], [r'$\tau_{0\Lambda}~[0.2631~ns]$'])
            points=np.linspace(0,tau_max,100)
            # plt.plot(points,exp_con_pdf(points,popt[0],popt[1]),label=r"$t_{fit}=$"+str(round_to_2(popt[0]))+r"$\pm$"+str(round_to_2(perr[0]))+" [ns]")#*(bins[1]-bins[0])*len(big_val)/popt[1]
            # plt.xscale('log')
            # plt.ylim(0, 1.1*len(big_val)*(bins[1]-bins[0]))

        if tau[2] in ["lam_mass","lam_mass_fixed"]:
            plt.hist(truth, bins=list(bins),alpha=.5,label=r"Signal $\Lambda$"#: $M_{\Lambda}$= "
            # + str(round_to_2(np.mean(truth)))
            # +r"$\pm$"
            # + str(round_to_2(np.std(truth))),
        )
            plt.hist(passed, bins=list(bins), alpha=.5,label=r"Selected $\Lambda$ + Bkg."# Cand.: $M_{\Lambda}$= "
            # + str(round_to_2(np.mean(passed)))
            # +r"$\pm$"
            # + str(round_to_2(np.std(passed))),
        )
            
            plt.hist(tau[0][False][True],bins=bins.tolist(),label=f"Background",alpha=.5)
            plt.axvline(x = (PION_MASS+PROT_MASS), color = 'g', label = r'$M_{\pi^-}+M_{p}~['+str(round_to_2(PION_MASS+PROT_MASS))+'MeV]$',linestyle='--', linewidth=2,)
            plt.axvline(x = LAM_MASS, color = 'r', label = r'$M_{\Lambda}~['+str(round_to_2(LAM_MASS))+'MeV]$',linestyle='--', linewidth=2,)

            

            
        # plt.xscale('log')
        
        plt.xlabel(tau[1])
        plt.ylabel("Freq")
        #plt.yscale("log")
        plt.legend()
        plt.tight_layout();plt.savefig(PLOTSDIR+tau[2])
        plt.clf()


    max_mass=3*LAM_MASS/1000

    # for mass in [[lam_mass,r"$M_{\Lambda_{candidate}} [GeV]$","lam_mass"],
    #             [lam_mass_fixed,r"$M_{\Lambda_{candidate}} [GeV]$","lam_mass_fixed"]
    #             ]:
        
    
    #     # print(lam_mass.keys(),len(lam_mass[False][False]),len(lam_mass[True][False]),len(lam_mass[False][True]),len(lam_mass[True][True]))
    #     lm0=np.array(mass[actual][True]+mass[actual][False])/1000
    #     lm1=np.array(mass[actual][True])/1000

    #     mybins=np.logspace(np.log10(min(lm1)*.99),np.log10(max(lm1[lm1<max_mass])),100)
    #     (n, bins, patches) = plt.hist(
    #         lm0[lm0<max_mass],
    #         bins=mybins,
    #         # color="blue",
    #         label=rf"all {actual}: $\mu$= "
    #         + str(round_to_2(np.mean(lm0)))
    #         + r", $\sigma$= "
    #         + str(round_to_2(np.std(lm0))),
    #     )
    #     plt.hist(
    #         lm1[lm1<max_mass],
    #         bins=bins,
    #         # color="orange",
    #         label=rf"{actual} after cuts: $\mu$= "
    #         + str(round_to_2(np.mean(lm1)))
    #         + r", $\sigma$= "
    #         + str(round_to_2(np.std(lm1))),
    #     )

    #     plt.axvline(x = (PION_MASS+PROT_MASS), color = 'g', label = r'$M_{\pi^-}+M_{p} ['+str(round_to_2(PION_MASS+PROT_MASS))+' GeV]$',linestyle='--', linewidth=2,)
    #     plt.axvline(x = LAM_MASS, color = 'r', label = r'$M_{\Lambda} ['+str(round_to_2(LAM_MASS))+' GeV]$',linestyle='--', linewidth=2,)
        
    #     plt.xlabel(r"$M_{\Lambda_{candidate}} [GeV]$")
    #     plt.ylabel("Freq")
    #     plt.legend()
    #     #plt.yscale("log")
    #     # plt.xscale("log")
    #     # plt.xticks(rotation=45)
    #     # plt.ticklabel_format(style='plain', axis='x')  # Disable scientific notation
    #     plt.tight_layout();plt.savefig(PLOTSDIR+"/lam_mass_"+str(actual))
    #     plt.clf()


        # lm0=np.array(lam_mass_fixed[actual][True]+lam_mass_fixed[actual][False])/1000
        # lm1=np.array(lam_mass_fixed[actual][True])/1000
        # (n, bins, patches) = plt.hist(
        #     lm0[lm0<max_mass],
        #     bins=mybins,
        #     # color="blue",
        #     # alpha=.75,
        #     label=rf"all {actual}: $\mu$= "
        #     + str(round_to_2(np.mean(lm0)))
        #     + r", $\sigma$= "
        #     + str(round_to_2(np.std(lm0))),
        # )
        # plt.hist(
        #     lm1[lm1<max_mass],
        #     bins=bins,
        #     # color="orange",
        #     # alpha=.75,
        #     label=rf"{actual} After Cuts: $\mu$= "
        #     + str(round_to_2(np.mean(lm1)))
        #     + r", $\sigma$= "
        #     + str(round_to_2(np.std(lm1))),
        # )



        # plt.axvline(x = (PION_MASS+PROT_MASS), color = 'g', label = r'$M_{\pi^-}+M_{p} ['+str(round_to_2(PION_MASS+PROT_MASS))+' GeV]$',linestyle='--', linewidth=2,)
        # plt.axvline(x = LAM_MASS, color = 'r', label = r'$M_{\Lambda} ['+str(round_to_2(LAM_MASS))+' GeV]$',linestyle='--', linewidth=2,)

        # plt.xlabel(r"$M_{\Lambda_{candidate}} [GeV]$")
        # plt.ylabel("Freq")
        # plt.legend()
        # #plt.yscale("log")
        # # plt.xscale("log")
        # # plt.xticks(rotation=45)
        # # plt.ticklabel_format(style='plain', axis='x')  # Disable scientific notation
        # plt.tight_layout();plt.savefig(PLOTSDIR+"/lam_mass_fixed"+str(actual))
        # plt.clf()


        # def L_mass(M_KE,H_KE,angle):
        #     H_P=np.sqrt((H_KE+PROT_MASS)**2-PROT_MASS**2)
        #     M_P=np.sqrt((M_KE+PION_MASS)**2-PION_MASS**2)
        #     return np.sqrt(PROT_MASS**2 + PION_MASS**2 + 2 * (M_KE + PION_MASS) * (H_KE + PROT_MASS) - 2 * H_P*M_P*np.cos(angle))/LAM_MASS


        



        # plt.axvline(x=54, color="r", label="ideal KDAR muon")
        # plt.xlabel("Muon Len [cm]")
        # plt.ylabel("Freq")
        # plt.legend()
        # plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/Muon_Len_fixed")
        # plt.clf()


    # plt.hist(
    #     np.array(np.array(lam_mass[2])[np.array(lam_mass[2]) < 3]),
    #     bins=list(bins),
    #     label=r"signal $\Lambda$ decay pairs: $\mu$= "
    #     + str(round_to_2(np.mean(lam_mass[2])))
    #     + r", $\sigma$= "
    #     + str(round_to_2(np.std(lam_mass[2]))),
    # )
    # plt.axvline(x = 1115.6**2, color = 'r', label = 'Lambda Mass^2 ')



    # delta = np.array(momenta[0][0]) - np.array(momenta[0][1])
    # (n, bins, patches) = plt.hist(delta, label="Before parameter cuts")
    # delta_true = np.array(momenta[1][0]) - np.array(momenta[1][1])
    # # print(len(lam_mass[0]),len(lam_mass[1]))
    # plt.hist(delta_true, bins=list(bins), label=r"True $\Lambda$ decay pairs")
    # plt.legend()
    # # ##plt.yscale('log')

    # plt.xlabel(r"Lambda decay $\Delta p_T$")
    # plt.ylabel("Freq")
    # plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/ptdiff")
    # plt.clf()

    # num=1000000
    # mip_KE=np.random.uniform(0, 200, num)
    # hip_KE=np.random.uniform(0, 200, num)
    # angle = np.random.uniform(0, np.pi, num)

    # plt.hist(L_mass(mip_KE,hip_KE,angle),bins=100,density=True,label="KE MAX=200 Mev")
    # plt.xlabel(r"Completely Random $M_{\Lambda_{candidate}}/_{\Lambda}$")

    # num = 1000000
    # mip_KE = np.random.uniform(0, 500, num)
    # hip_KE = np.random.uniform(0, 500, num)
    # angle = np.random.uniform(0, np.pi, num)

    # plt.hist(L_mass(mip_KE, hip_KE, angle), bins=100, density=True,alpha=.5,label="KE MAX=500 Mev")
    # plt.xlabel(r"Completely Random $M_{\Lambda_{candidate}}/_{\Lambda}$")

    # num = 1000000
    # mip_KE = np.random.uniform(0, 1000, num)
    # hip_KE = np.random.uniform(0, 1000, num)
    # angle = np.random.uniform(0, np.pi, num)

    # plt.hist(L_mass(mip_KE, hip_KE, angle), bins=100, density=True,alpha=.5,label="KE MAX=1000 Mev")
    # plt.xlabel(r"Completely Random $M_{\Lambda_{candidate}}/_{\Lambda}$")

    # plt.ylabel("freq")
    # plt.legend()
    # plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/completely_random_lambda_mass")
    # plt.clf()


    # pt1 = momenta[0][0] / LAM_PT_MAX
    # # pt2=momenta[0][1]/pt_true

    # # print(pt_true)

    # (n, bins, patches) = plt.hist(pt1, label="Before parameter cuts", bins=100)
    # pt1_truth = momenta[1][0] / LAM_PT_MAX
    # # print(pt1_truth)
    # # pt2_truth=momenta[1][1]/pt_true
    # plt.hist(pt1_truth, label=r"True $\Lambda$ decay pairs", bins=list(bins))
    # plt.xlabel("(Hip $p_T$)/(MAX $p_T$)")
    # plt.ylabel("freq")
    # # plt.ylabel("(Mip $p_T$)/(True $p_T$)")
    # # plt.xscale('log')
    # # #plt.yscale('log')
    # plt.xlim([0, 4])
    # plt.legend()
    # plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/pt_vs_true")
    # plt.clf()


    # plt.scatter(pt1, lam_decay_len[0], label="Before parameter cuts")

    # plt.scatter(pt1_truth, lam_decay_len[1], label=r"True $\Lambda$ decay pairs")
    # plt.xlabel("(Hip $p_T$)/(True $p_T$)")
    # plt.ylabel("Lambda decay len [cm]")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.clf()





    # for key in kaons[1]:
    #     for k in kaons[1][key]:
    #         dt+=[k.decay_t/1000]
    #         ds+=[k.decay_sep*10]
    # print(decay_t_to_dist)
    # decay_t_to_dist=decay_t_to_dist[decay_t_to_dist!=np.inf]

    # print()
    # plt.plot(np.array(ds)[np.array(ds)<100],np.array(dt)[np.array(ds)<100],'o',label="truth mu/michel pairs")
    # plt.plot(np.array(ds)[np.array(ds)<100],1/1.5*np.array(ds)[np.array(ds)<100],color="r",label="dt=1/(1.5mm/us) dx")
    # # plt.axvline(x = 1115.6**2, color = 'r', label = 'Lambda Mass^2 ')
    # # plt.xlabel("(michel.t-mu.t)/|mu.last_step-michel.first_step)|*1.5/1000/10")
    # plt.ylabel("delta t [us]")
    # plt.xlabel("delta x [cm]")
    # plt.legend()
    # # plt.xscale("log")
    # plt.legend()
    # plt.tight_layout();plt.savefig("plots/"+str(FOLDER)+"/decay_t_to_dist_2")
    # plt.clf()
    # thing_to_plot=np.array(dt)/np.array(ds)*1.5
    # plt.hist(thing_to_plot[(thing_to_plot!=np.inf)])
    # plt.tight_layout();plt.savefig("plots/"+str(FOLDER)+"/decay_t_to_dist")
    # plt.xlabel("freq")
    # plt.xlabel("(michel.t-mu.t)/|mu.last_step-michel.first_step)|*1.5/1000/10")
    # plt.clf()


    # pstar=.101
    # alphabar=.691
    # a=.18
    # asym=np.arange(0, 1, .000001)
    # qT=pstar*np.sqrt(1-np.power((asym-alphabar)/a,2))

    # pt=abs(np.array(AM[0][0]))/1000
    # alpha=np.array(AM[0][2])

    # alpha_true=np.array(AM[1][2])

    # plt.scatter(alpha,pt,label="Before parameter cuts")
    # plt.plot(asym,qT,label=r"ideal $\beta=1$",color='red',linewidth=7.0,alpha=.5)
    # plt.xlim([0,1])
    # plt.ylim([0,.6])
    # # plt.
    # if OT:
    #     # print(len(lam_mass[0]),len(lam_mass[1]))
    #     plt.scatter(alpha_true,pt_true,label="True $\Lambda$ decay pairs")

    # # plt.axvline(x = 1115.6**2, color = 'r', label = 'Lambda Mass^2 ')
    # plt.xlabel("Long. Momentum Asymmetry")
    # plt.ylabel("Transverse Momentum [GeV]")
    # # plt.colorbar()
    # plt.legend()
    # plt.tight_layout();plt.savefig("plots/"+str(FOLDER)+"/AM")
    # plt.clf()
    # ########

    # # phi=np.arange(-np.pi, np.pi, .000001)
    # r=np.sqrt((alpha-alphabar)**2/a**2+(pt/pstar)**2)
    # phi=np.arctan(pt*a/pstar/(alpha-alphabar))
    # plt.scatter(phi,r,label="Before parameter cuts")
    # # plt.plot(asym,qT,label=r"ideal $\beta=1$",color='red',linewidth=7.0,alpha=.5)
    # # plt.xlim([0,1])
    # # plt.ylim([0,.6])
    # # plt.
    # if OT:
    #     r=np.sqrt((alpha_true-alphabar)**2/a**2+(pt_true/pstar)**2)
    #     phi=np.arctan(pt_true*a/pstar/(alpha_true-alphabar))
    #     # print(len(lam_mass[0]),len(lam_mass[1]))
    #     plt.scatter(phi,r,label="True $\Lambda$ decay pairs")
    # plt.legend()
    # # #plt.yscale('log')
    # plt.tight_layout();plt.savefig("plots/"+str(FOLDER)+"/AM")


    # lam_true=[]
    # lam_true_sel=[]
    # lam_False_sel=[]
    # lam_eff=[]
    # lam_pur=[]



    # plt.clf()
    # import textwrap
    # def wrap_text(text, width=15):
    #     return "\n".join(textwrap.wrap(str(text), width=width))


    plt.figure(dpi=300)
    cols = [i for i in lam_pass_order]


    columns=[wrap_text(i) for i in cols]


    plt.figure(dpi=300)
    columns = [wrap_text(i) for i in lam_pass_order]
    rows = [r"Selected $\Lambda$",r"Bkg.", r"Signal $\Lambda$","Eff.","Pur.","Eff.*Pur."] 
    data=np.zeros((len(rows),len(columns)))


    sel_lam_T=np.zeros(len(columns))
    sel_lam_T[-1]=lam_pass_failure[0][cols[-1]]
        

    sel_lam_F=np.zeros(len(columns))
    sel_lam_F[-1]=lam_pass_failure[1][cols[-1]]

    sel_lam=np.zeros(len(columns))
    sel_lam[-1]=lam_pass_failure[0][cols[-1]]+lam_pass_failure[1][cols[-1]]




    for c in range(len(columns) - 2, -1, -1):
        sel_lam_F[c]=sel_lam_F[c+1]+lam_pass_failure[1][cols[c]]
        sel_lam_T[c]=sel_lam_T[c+1]+lam_pass_failure[0][cols[c]]

        sel_lam[c]=sel_lam[c+1]+lam_pass_failure[0][cols[c]]+lam_pass_failure[1][cols[c]]

    scaling=80000/num_nu_from_file

    for r in range(len(rows)):
        for c in range(len(columns)):
            if rows[r]==r"Signal $\Lambda$":data[r][c]=round(TrueLambdas*scaling)
            if rows[r]==r"Selected $\Lambda$":data[r][c]=round(sel_lam_T[c]*scaling)
            if rows[r]==r"Bkg.":data[r][c]=round(sel_lam_F[c]*scaling)
            # print(sel_lam_T[c],sel_lam[c],TrueLambdas)
            eff=np.nan_to_num(np.divide(sel_lam_T[c],TrueLambdas), nan=0)
            pur=np.nan_to_num(np.divide(sel_lam_T[c],sel_lam[c]),nan=0)
            if rows[r]=="Eff.":data[r][c]=rsf(eff,2)
            if rows[r]=="Pur.":data[r][c]=rsf(pur,2)
            if rows[r]=="Eff.*Pur.":data[r][c]=rsf(eff*pur,2)
            
    data2=np.flip(data,axis=0)
    # Get some pastel shades for the colors 
    assert len(rows)==6
    colors=colors = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0]   # Blue
])
    # colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows))) 
    n_rows = len(data) 
    
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    
    # Initialize the vertical-offset for 
    # the line plots. 
    y_offset = np.zeros(len(columns)) 
    
    # Plot line plots and create text labels  
    # for the table 
    cell_text = [] 
    for row in range(n_rows): 
        if row in [n_rows-3,n_rows-2,n_rows-1]:
            plt.plot(index, data[row], color=colors[row],label=rows[row]) 
        y_offset = data2[row] 
        cell_text.append([x for x in y_offset]) 
    plt.legend()
    # Reverse colors and text labels to display 
    # the last value at the top. 
    # colors = colors[::-1] 
    cell_text.reverse() 
    
    # Add a table at the bottom of the axes 
    the_table = plt.table(cellText=cell_text, 
                        rowLabels=rows, 
                        rowColours=colors, 
                        colLabels=[""]+columns[:-1], 
                        loc='bottom') 
    # the_table.auto_set_font_size(False)  # Disable automatic font size adjustment
    # the_table.set_fontsize(3)    

    for (row, col), cell in the_table.get_celld().items():
        if row == 0:  # This is the header row
            cell.set_height(0.15)  # Increase height to accommodate multiline text
            # cell.set_fontsize(16)  # Optionally make the header font larger
    # Adjust layout to make room for the table: 
    plt.subplots_adjust(bottom=0.5) 
    
    # plt.ylabel("marks".format('value_increment')) 
    plt.xticks([]) 
    plt.title('Lambda Efficiency and Purity as a Function of Cut') 
    plt.yscale("log")
    
    plt.tight_layout();plt.savefig(PLOTSDIR+"/lam_eff_purity",bbox_inches='tight')
    plt.clf()



    columns=[""]+columns[:-1]


    # Flip data for vertical layout
    data_flipped = np.transpose(data)
    rows_flipped = columns
    columns_flipped = rows

    # Format cell text and force int representation
    cell_text = []
    for row in data_flipped:
        formatted_row = [
            int(val) if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()) else val
            for val in row
        ]
        cell_text.append(formatted_row)

    # Define colors
    assert len(columns_flipped) == 6
    row_colors = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0]   # Blue
    ])

    # Add table

    table = plt.table(
    cellText=cell_text,
    rowLabels=rows_flipped,
    colLabels=columns_flipped,
    cellLoc='center',
    loc='center'
)

    # Set font size and force fit
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    for cell in table.get_celld().values():
        cell.set_text_props(ha='center', va='center')

    for (row, col), cell in table.get_celld().items():
        if row != 0 and col != -1:
            # Header row or row labels
            # Data cells: increase font size here
            cell.set_text_props(fontsize=10)

    # Alternate row colors
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        if row > 0:
            # Alternate colors: light gray and white
            cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')
        cell.set_height(0.065)
        cell.set_width(0.11)

    table.scale(0.6, 1)

    plt.axis('off')
    plt.tight_layout()
    # plt.savefig(PLOTSDIR+"/lam_eff_purity",bbox_inches='tight')
    plt.savefig(PLOTSDIR+"/lam_eff_purity_table.png", bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    main()