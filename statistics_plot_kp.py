from statistics_plot_base import *

import math

def rsf(x, sig=2):
    """Rounds a number to a specified number of significant figures.
    """
    if x == 0:
        return 0.0
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

kaon_pass_order_truth={
            # "contained kaon",
            # "contained mips",
            # "kaon_too_short",
            "Contained HIP":True,
            "Connected Non-Primary MIP":True,
            "Contained MIP":True,
            "Valid Interaction":100,
            # "Contained HIP":True,
            "Initial HIP":True,
            # "Correct HIP TPC Assoc.":True,
            
            rf"Primary $K^+$":np.inf,
            
            
            # "muon child",
            # "missing required kaon",
            # "missing required michel",
            # "muon passing len",
            # rf"Valid Single $\mu/\pi^+$ Decay",

            # "MIP_CUTS":
            #     {
                # "Connected Non-Primary MIP":True,
            r"Low MIP len $\pi^0$ Tag":15,
            "Michel Child":(20,5),
            
            "MIP Child At Most 1 Michel":True,
            "Single MIP Decay":True,
            # "Bragg Peak HIP": 1,
            # r"HIP $K^+ or >0 Scatter$":True,
            # "Correct MIP TPC Assoc.":True,
            "Valid MIP Len":25 + 35*100 + 48*100**2 + 60*100**3,

            "Forward HIP":1,
            "Separable MIP":np.pi/12,
            "":True
}


kaon_pass_order_reco={
            # "contained kaon",
            # "contained mips",
            # "kaon_too_short",
            # "Contained HIP":True,
            # "Contained MIP":True,
            "Fiducialization":True,
            "Valid Interaction":True,
            rf"Primary $K^+$":True,
            "Initial HIP":True,
            # "Correct HIP TPC Assoc.":True,
            "Valid MIP Len":25 + 35*100 + 48*100**2 + 60*100**3,
            "Michel Child":True, #TODO characterize the MIP distance in x using drift time

            "Connected Non-Primary MIP":True,
            r"Low MIP len $\pi^0$ Tag":10,
            "Come to Rest":-.65, #maybe something to not cut when length too short
            
            "Kaon Len":4.7,
            "Close to Vertex":4.3,

            "MIP Child At Most 1 Michel":True,
            "No HIP Deltas":True,
            "Come to Rest MIP":-.4,
            "Single MIP Decay":True,
            "Bragg Peak HIP": 6.6666666666666,
            # "dedx chi2":100,
            # "Nothing Before the Start":True,

            # "Forward HIP":1,
            # "Separable MIP":np.pi/24,
            # "No LOW E MIP Deltas":min_len/2,
            "":True
}



from concurrent.futures import ProcessPoolExecutor

print("cpu count",os.cpu_count())
import argparse

def load_single_file(args:tuple)->tuple[str,list[PredKaonMuMich],list]:
    file0, d0, directory2 = args

    kfile = os.path.join(d0, file0)
    kfile_truth = os.path.join(directory2, file0)
    kfile_truth = kfile_truth[:-1] + "y"

    with np.load(kfile, allow_pickle=True) as kaons, open(kfile_truth, 'rb') as f2:
        particles = np.load(f2, allow_pickle=True)
        predk = kaons['PREDKAON']

    return file0, predk, particles


# @profile
def main():

    

    parser = argparse.ArgumentParser(description='Script to plot K+ stats')
    parser.add_argument('--mode', type=str, choices=["truth", "reco"], help='Reco or Truth running mode')
    parser.add_argument('--N', type=int, default=sys.maxsize, help='Number of files to run')
    parser.add_argument('--single_file',type=str, default="",help="if set, just run this file, and don't plot")

    args = parser.parse_args()

    assert args.mode

    if args.mode=="truth":
        kaon_pass_order=kaon_pass_order_truth
    else:
        kaon_pass_order=kaon_pass_order_reco

    # parser.add_argument('--dir', type=str, help='Directory of h5 files, npyfile will go in same level directory with _files replaced with _npy')

    from collections import defaultdict

    # num_nu_from_file= np.load('num_nu.npy')
    num_nu_from_file=0

    # for i in lam_pass_order
    kaon_reason_map=[defaultdict(int),defaultdict(int)]
    kaon_pass_failure=[defaultdict(int),defaultdict(int)]


    kaon_pass_failure_mu=[defaultdict(int),defaultdict(int)]

    kaon_pass_failure_pi=[defaultdict(int),defaultdict(int)]

    



    # current_type_map=defaultdict(int)


    PLOTSDIR="plots/" + str(FOLDER) + "/Kp/"+str(args.mode)+"/"


    MAXFILES=args.N

    d0=directory.replace("_analysis", f"_analysis_{args.mode}")


    files = os.listdir(d0)
    # FILES = [os.path.join(directory, f) for f in files]
    if len(files)>MAXFILES:
        files=files[:MAXFILES]
    # KAONFILE=files
    # LAMBDAFILE=FILES


    # KAONFILE=["npyfiles/2024-10-16-kaons.npy"]
    # LAMBDAFILE=["npyfiles/2024-08-16-lambdas.npy"]
    # FOLDER = "1016"


    # filepath=sys.argv[1]
    # file=os.path.basename(filepath)
    # filenum=file.split('_')[1]+'_'+file.split('_')[2].split('-')[0]

    # outfile='outloc/processed_'+filenum+'.npy'

    # if os.path.isfile(outfile):exit()
    # print("files:",filepath,file,filenum)


    # kaons = np.load("npyfiles/test_K_" + str(FOLDER) + ".npy", allow_pickle=True)
    # lambdas = np.load("npyfiles/test_lam_" + str(FOLDER) + ".npy", allow_pickle=True)

    # print(lambdas[0])

    #what it actually is/what it is predicted as/value
    # dir_acos_K = defaultdict(lambda: defaultdict(list))
    muon_len = defaultdict(lambda: defaultdict(list))
    # all_valid_muon_len
    # muon_len_adjusted= defaultdict(lambda: defaultdict(list))
    kaon_len = defaultdict(lambda: defaultdict(list))
    HM_acc_K = [[],[]]
    HM_acc_MIP=[[],[]]

    K_primary=[[],[]]
    MIP_primary=[[],[]]

    K_clustering=[[],[]]

    closest_FM= defaultdict(lambda: defaultdict(list))
    

    
    # K_mom=[[],[]]

    K_mom=[]
    mu_mom=[]
    pi_mom=[]

    Bragg_peak_disc=defaultdict(lambda: defaultdict(list))

    Bragg_peak_sigma_disc=defaultdict(lambda: defaultdict(list))

    Bragg_peak_mip=defaultdict(lambda: defaultdict(list))

    come_to_rest_dict=defaultdict(lambda: defaultdict(list))
    come_to_rest_prot=defaultdict(lambda: defaultdict(list))
    come_to_rest_pi=defaultdict(lambda: defaultdict(list))
    chi2=defaultdict(lambda: defaultdict(list))

    come_to_rest_simpler=defaultdict(lambda: defaultdict(list))


    come_to_rest_len=defaultdict(lambda: defaultdict(list[tuple]))

    Bragg_peak_len=defaultdict(lambda: defaultdict(list[tuple]))

    Bragg_peak_simple=defaultdict(lambda: defaultdict(list[tuple]))

    forward_disc=defaultdict(lambda: defaultdict(list))


    primary_kp=defaultdict(lambda: defaultdict(list))


    MCS_dir=[[0,0],[0,0]]





    

    # HM_acc_mu = defaultdict(lambda: defaultdict(list))
    # HM_acc_mich = defaultdict(lambda: defaultdict(list))
    # dist_to_hip = defaultdict(lambda: defaultdict(list))
    # dist_to_mich = defaultdict(lambda: defaultdict(list))

    MIP_gamma_cost_disc=[[],[]]
    MIP_gamma_impact_disc=[[],[]]

    # kaon



    # K_csda_over_calo=defaultdict(lambda: defaultdict(list))

    # missing_K_lens=[]
    K_dir=[[],[]]
    MIP_dir=[]

    K_dir_before=[[],[]]
    # missing_pi_lens=[]

    tracking_threshold=[[],[]]


    vertex_dz=[]

    vertex_displacement=[]

    HS_count=[[],[]]

    HS_mistakes=[[],[]]

    

    # k_extra_children_dist = defaultdict(lambda: defaultdict(list))
    # k_extra_children_angle = defaultdict(lambda: defaultdict(list))

    # mu_extra_children_dist = defaultdict(lambda: defaultdict(list))
    # mu_extra_children_angle = defaultdict(lambda: defaultdict(list))

    # lam_mass = defaultdict(lambda: defaultdict(list))
    # # momenta = [[[], []], [[], []]]
    # lam_decay_len = defaultdict(lambda: defaultdict(list))
    # lam_dir_acos = defaultdict(lambda: defaultdict(list))
    # HM_acc_prot =defaultdict(lambda: defaultdict(list))
    # HM_acc_pi = defaultdict(lambda: defaultdict(list))
    # ProtPi_dist = defaultdict(lambda: defaultdict(list))

    # mip_len=defaultdict(lambda: defaultdict(list))
    # hip_len=defaultdict(lambda: defaultdict(list))
    # vae = defaultdict(lambda: defaultdict(list))



    TrueKaons=0
    TrueKaonsmu=0
    TrueKaonspi=0
    # TrueLambdas=0
    correctlyselectedkaons=0
    # correctlyselectedlambdas=0
    selectedkaons=0
    # selectedlambdas=0


    # num_nu=0

    LOCAL_EVENT_DISPLAYS=event_display_new_path+"/Kp/"+args.mode+'/'

    assert os.path.isdir(LOCAL_EVENT_DISPLAYS), f"{LOCAL_EVENT_DISPLAYS} does not exist"


    NONLOCAL_EVENT_DISPLAYS=base_directory+FOLDER+"_files_"+args.mode
    
    assert os.path.isdir(NONLOCAL_EVENT_DISPLAYS), f"{NONLOCAL_EVENT_DISPLAYS} does not exist"



    # print(LAM_PT_MAX,"Maximum allowable pT")
    # dt=[]
    # ds=[]
    # MAXFILES=np.inf
    print("only including", MAXFILES, "files")
    filecount=0



    ###################
    if args.single_file=="":
        args_list = [(file0, d0, directory2) for file0 in files]
    else:
        # assert os.path.isfile(args.single_file)
        assert args.single_file in files
        args_list = [(args.single_file, d0, directory2)]

    results:list[tuple[str,list[PredKaonMuMich],list]] = []
    
    # with ProcessPoolExecutor() as executor:
    #     for result in tqdm(executor.map(load_single_file, args_list), total=len(args_list)):
    #         results.append(result)

    from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor

    results = []
    max_cpus = os.cpu_count()
    # with ProcessPoolExecutor(max_workers=max_cpus) as executor:
    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(load_single_file, arg) for arg in args_list]
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         results.append(future.result())


    for args0 in tqdm(args_list, total=len(args_list)):
        result = load_single_file(args0)
        results.append(result)

    for file0, predk, particles in results:
        if filecount == 0 and args.single_file=="":
            clear_html_files(LOCAL_EVENT_DISPLAYS)
        if filecount % 500 == 0:
            print("filecount", filecount, "kaon eff/pur",
                np.divide(correctlyselectedkaons, TrueKaons) if TrueKaons else 0,
                np.divide(correctlyselectedkaons, selectedkaons) if selectedkaons else 0,
                [correctlyselectedkaons, selectedkaons, TrueKaons])
        if filecount == MAXFILES:
            break
        filecount += 1

        kfile=os.path.join(d0, file0)
        kfile_truth=os.path.join(directory2, file0)
        kfile_truth=kfile_truth[:-1]+"y"

        def save_my_html(newpath,name,name2):
            # print("saving:",newpath)
            copy_and_rename_file(NONLOCAL_EVENT_DISPLAYS,kfile,newpath,name,name2,args.mode)
        K_to_pi_list=[]
        K_to_mu_list=[]

    ###################
    #########################
    # for file0 in files:
    #     # continue
    #     # print("running",kfile)

    #     kfile=os.path.join(d0, file0)
    #     kfile_truth=os.path.join(directory2, file0)
    #     kfile_truth=kfile_truth[:-1]+"y"

    #     def save_my_html(newpath,name,name2):
    #         copy_and_rename_file(NONLOCAL_EVENT_DISPLAYS,kfile,newpath,name,name2,args.mode)


    #     if filecount==0:clear_html_files(LOCAL_EVENT_DISPLAYS)
    #     if filecount%500==0:print("filecount ",filecount,"kaon eff/pur",np.divide(correctlyselectedkaons,TrueKaons),np.divide(correctlyselectedkaons,selectedkaons),[correctlyselectedkaons,selectedkaons,TrueKaons])
    #     if filecount==MAXFILES:break
    #     filecount+=1


    #     with np.load(kfile, allow_pickle=True) as kaons, open(kfile_truth, 'rb',) as f2:
    #         particles = np.load(f2, allow_pickle=True)
    #         # kaons = np.load(kfile, allow_pickle=True)
    #         predk: list[PredKaonMuMich] = kaons['PREDKAON']
    #################################
        if True:

            # num_nu+=kaons['NUMNU']

            both_there=True
            for f in [kfile,kfile_truth]:
                both_there*=os.path.exists(f)#, f"File not found: {f}"
            if not both_there:
                print("had to skip because",f"File not found: {file0,kfile,kfile_truth}")
                continue

            # particles = np.load(kfile_truth, allow_pickle=True)

            num_nu_from_file+=len(particles[1][0])
            # interactions=particles[3]
            # lambdas_true=particles[0][3122]
            
            interactions=particles[2]


            valid_ints=[]

            # interaction_pos_dict=defaultdict(list)
            for I in interactions:
                i=interactions[I]
                if is_contained(i[0][:3],margin=margin0):
                    primpdgs=[z for z in i[1].keys() if z[1]==321]
                    if len(primpdgs):
                        # TODO this is vaid if not in mPVMPR
                        if len(primpdgs)==0:
                            continue
                        
                        # assert len(primpdgs)==1,i[1].keys()
                        # print(i[1][primpdgs[0]])
                        # print()
                        kaon_list=[zz for zz in i[1][primpdgs[0]] if zz[0]==321]


                        

                        if len(kaon_list)==0:
                            print("HUH???",primpdgs,i[1][primpdgs[0]])
                            continue
                        kaon_list=kaon_list[0]
                        child_list=[zz for zz in i[1][primpdgs[0]] if zz[0] in [-13,211]]
                        
                        # for zz in i[1][primpdgs[0]]:

                            # print(zz,"zz")
                            # if zz[0] in [-13]:
                        # print(,"kaon_list")
                        valid_decays=[zz[0] for zz in i[1][primpdgs[0]] if zz[0] not in  [321]]

                        


                        increased=False
                        child_list_mu=[zz for zz in i[1][primpdgs[0]] if zz[0] in [-13]]
                        child_list_pi=[zz for zz in i[1][primpdgs[0]] if zz[0] in [211]]


                        if valid_decays in [[211],[-13]]:
                            # print(valid_decays,child_list_mu,child_list_pi)
                            kaon_ke=csda_ke_lar(min_len,KAON_MASS)
                            assert type(kaon_ke)==float
                            if valid_decays in [[211]] and child_list_pi[0][1]>200 and child_list_pi[0][1]<210 and is_contained(np.array(child_list[0][-1][-2]),margin=0) and is_contained(np.array(child_list[0][-1][-1]),margin=0) and kaon_list[1]>.95*math.sqrt(kaon_ke**2 + 2 * kaon_ke * KAON_MASS):
                                # assert 
                                # print("Got a pi")
                                valid_ints+=[[I[0],I[1]]]
                                TrueKaons+=1
                                increased=True
                                # assert 
                            if valid_decays in [[-13]] and child_list_mu[0][1]>230 and child_list_mu[0][1]<240 and is_contained(np.array(child_list[0][-1][-2]),margin=0) and is_contained(np.array(child_list[0][-1][-1]),margin=0) and kaon_list[1]>.95*math.sqrt(kaon_ke**2 + 2 * kaon_ke * KAON_MASS):
                                # assert 
                                # print("Got a mu")
                                valid_ints+=[[I[0],I[1]]]
                                TrueKaons+=1
                                increased=True
                            # raise Exception(child_list[0][-1],child_list[0][-1][-2])



                        if increased:
                            if [zz[0] for zz in i[1][primpdgs[0]]]==[211,321]:
                                K_to_pi_list+=[(I[0],I[1])]
                                
                                
                            if [zz[0] for zz in i[1][primpdgs[0]]]==[-13,321]:
                                K_to_mu_list+=[(I[0],I[1])]
                            
                            


                        # if len

                        # print("LOOK HERE",i[1][primpdgs[0]])

                        if len(child_list_mu):
                            mu_mom+=[child_list_mu[0][1]]
                            if increased:TrueKaonsmu+=1

                        if len(child_list_pi):
                            pi_mom+=[child_list_pi[0][1]]
                            if increased:TrueKaonspi+=1

                        K_mom+=[kaon_list[1]]

                        

                        if len(child_list):

                        # ent=[z for z in i[1].keys() if z[1]==321][0]
                        # start=
                            start=np.array(kaon_list[-1][-2])
                            # end
                            end=np.array(child_list[0][-1][-2])
                            # print(start,end)
                            # if 
                            missing_len=np.linalg.norm(np.array(start)-np.array(end))
                            # print(missing_len)
                        else:
                            missing_len=-1
                        # print("found K+",primpdgs)
                        # testlam=i[1][primpdgs[0]]
                        add_it=True
                        # TrueKaons+=1
                        # if I[0] in predk:
                        # missing_len=-1
                        found_something=False
                        for k in predk:
                            if k.truth_interaction_id==I[1] and k.event_number==I[0]:
                                # if len(k.truth_Kp):
                                    # p=k.truth_Kp[list(k.truth_Kp.keys())[0]]
                                
                                if k.truth_hip!=None:
                                    if k.truth_hip.pdg_code==321 and k.truth_hip.is_primary:
                                        found_something=True
                                    # break
                                if k.reason=="":
                                    add_it=False
                                    # print("found the K+ in events")
                        if add_it and increased:
                            
                            # print("completely missed a K+")

                            # print(f"found completely missing with length {missing_len}")

                            
                            
                            # try:
                            save_my_html(LOCAL_EVENT_DISPLAYS+'completely_missed',create_html_filename(I[0], kfile,extra=""),I[0])
                        tracking_threshold[found_something]+=[missing_len]
                        # else:
                            # tracking_threshold[True]+=[missing_len]



            # for key in predk:
            if True:
                for mu in predk:
                    # assert type(mu)==PredKaonMuMich

                    # raise Exception(mu.__dict__)
                    # for attr in mu.__slots__:
                    #     value = getattr(mu, attr)
                    #     print(f"{attr}: {type(value)}, size: {asizeof.asizeof(value)} bytes")

                    key=mu.event_number

                    is_true=mu.truth*is_contained(mu.truth_interaction_vertex,margin=margin0)*([key,mu.truth_interaction_id] in valid_ints)#*((key,mu.truth_interaction_id) in K_to_pi_list)
                    if is_true:
                        assert mu.truth_hip is not None
                        is_true*=(mu.truth_hip.ke>csda_ke_lar(min_len,KAON_MASS))*(mu.truth_hip.ancestor_creation_process=="primary")

                    # truth_hip=mu.truth_hip

                    if is_true:
                        assert mu.truth_hip is not None
                        K_dir_before[0]+=[angle_between(mu.hip.momentum,mu.truth_hip.momentum)]
                        K_dir_before[1]+=[mu.hip.reco_length]
                    
                    pc=mu.pass_cuts(kaon_pass_order)
                    inter =mu.reco_vertex
                    # pc*=is_contained(inter,margin=margin0)
                    

                    

                    def quick_save(dir):
                        save_my_html(LOCAL_EVENT_DISPLAYS+dir,create_html_filename(key, kfile,extra=str(mu.hip.id)),key)

                    # if mu.error!="":
                    #     print("VERY BAD ERROR",key,mu.error)
                    # print("event",key)

                    

                    
                    


                    if not mu.is_flash_matched:
                        if len([np.linalg.norm(np.array(i[0])-inter) for i in mu.fm_interactions if i[1]]):
                            closest_FM[is_true][1]+=[min([np.linalg.norm(np.array(i[0])-inter) for i in mu.fm_interactions if i[1]])]

                    # if is_true:
                        



                    if is_true:
                        # print(mu.hm_pred)
                        if len(mu.hm_pred): HM_acc_K[0] += [mu.hm_pred[mu.hip.id][HIP_HM]/(mu.hm_pred[mu.hip.id][HIP_HM]+mu.hm_pred[mu.hip.id][MIP_HM])]
                        HM_acc_K[1] += [int(mu.hip.pid in [3,4,5])]

                        K_primary[int(mu.hip.is_primary)]+=[np.linalg.norm(mu.hip.start_point-inter)]

                        overlaps=mu.match_overlaps
                        if len(overlaps)==0:
                            K_clustering[False]+=[0]
                        else:
                            K_clustering[mu.is_flash_matched]+=[overlaps[0]]

                            if overlaps[0]<.2:
                                quick_save('/error_clustering/')


                        # for m in mu.truth_pi0_gamma:
                            # if 

                        # K_mom[0]+=[np.linalg.norm(fixed_prot_mom)]
                        # K_mom[1]+=[truth_hip.p]

                        assert mu.truth_hip is not None
                        K_dir[0]+=[angle_between(mu.hip.momentum,mu.truth_hip.momentum)]
                        K_dir[1]+=[mu.hip.reco_length]
                        if len(mu.match_overlaps)>0 or type(mu.hip)==TruthParticle:
                            vertex_displacement+=[np.linalg.norm(mu.truth_interaction_vertex-inter)]
                            vertex_dz+=[mu.truth_interaction_vertex[2]-inter[2]]

                            if np.linalg.norm(mu.truth_interaction_vertex-inter)>20:
                                quick_save('/error_vertex_true_to_reco_dist/')
                        if angle_between(mu.hip.momentum,mu.truth_hip.momentum)>np.pi/2 and mu.hip.reco_length>min_len:
                            quick_save('/error_hip_direction/')

                    

                    # if not is_true and not pc:
                    #     continue


                    # if is_true:
                        # if pc:
                            # assert (key,mu.truth_interaction_id) in K_to_pi_list+K_to_mu_list,((key,mu.truth_interaction_id),K_to_pi_list+K_to_mu_list)




                    # if not is_true:
                        # print(mu.pass_failure)

                    # plen=[i.reco_length for i in p]

                    good_mip_len=False
                    good_mip_forward=False
                    good_prim=False

                    for k in mu.potential_kaons:
                        for p in k[1]:
                            mc=p[1]
                            if mc in [[],["Kaon Len"]]:
                                good_mip_len=True
                            if mc in [[],["Forward HIP"]]:
                                good_mip_forward=True
                            if mc in [[],["Close to Vertex"]]:
                                good_prim=True
                    
                    kaon_len[is_true][good_mip_len]+= [mu.hip.reco_length]

                    if mu.truth_hip is not None:

                        forward_disc[is_true][good_mip_forward]+= [angle_between(mu.truth_hip.momentum,np.array([0,0,1]))]
                        primary_kp[is_true][good_prim]+=[np.linalg.norm(mu.hip.start_point-mu.reco_vertex)]


                    

                    # unique_mip_match=(lam_mip_candidate.match_ids[0]==best_mip_match.id)*(particles[best_mip_match.match_ids[0]].id==lam_mip_candidate.id)

                    # if pc and not is_true:
                    #     if not is_contained(inter,margin=margin0):
                    #         continue

                    mpf=mu.pass_failure[0]
                    if mpf=="":
                        assert len(mu.pass_failure)==1,mu.pass_failure

                    # seen_mips=[]
                    if True:

                    # if mode

                        for pk in mu.potential_kaons:
                            for passing_mu in pk[1]:
                                if passing_mu[0].reco_length<=0: continue
                                # if passing_mu[0] in seen_mips: continue
                                truth_mip=None
                                if type(passing_mu[0])==TruthParticle:
                                    truth_mip=passing_mu[0]
                                elif type(passing_mu[0])==RecoParticle:

                                    
                                    if passing_mu[0].is_matched:
                                        if passing_mu[0].match_ids[0] in mu.decay_mip_dict:
                                            truth_mip=mu.decay_mip_dict[passing_mu[0].match_ids[0]]

                                    # max_edep=0
                                    # for i in mu.particles:
                                    #     for v in range(len(i.points)):
                                    #         if np.linalg.norm(i.points[v]-passing_mu[0].end_point)<2.5:
                                    #             max_edep=max(max_edep,i.depositions[v])
                                    # end = 
                                    # all_points = np.concatenate([p.points for p in mu.particles], axis=0)
                                    # all_deps   = np.concatenate([p.depositions for p in mu.particles], axis=0)

                                    # dists = np.linalg.norm(all_points - end, axis=1)
                                    # mask = dists < 2.5
                                valid_decay=mu.truth
                                if truth_mip is None:
                                    valid_decay=0
                                elif not pk[0].is_matched:
                                    valid_decay=0
                                elif pk[0].match_ids[0] not in mu.kaon_path:
                                    valid_decay=0
                                else:
                                    assert type(truth_mip)==TruthParticle
                                    valid_decay*=int(truth_mip.parent_id==mu.kaon_path[pk[0].match_ids[0]].id or norm3d(truth_mip.start_point-mu.kaon_path[pk[0].match_ids[0]].end_point)<min_len/2)

                                if pc and is_true and passing_mu[0].reco_length>10:
                                    # dists = np.linalg.norm(passing_mu[0].points - passing_mu[0].end_point, axis=1)
                                    # mask = dists < 10
                                    # max_edep = np.median(passing_mu[0].depositions[mask]*np.power(dists[mask],.42)) if np.any(mask) else 0
                                    max_edep,_=Bragg_Peak(passing_mu[0])
                                    if valid_decay:
                                        assert truth_mip is not None
                                        Bragg_peak_mip[angle_between(passing_mu[0].momentum,truth_mip.momentum)<np.pi/2][True]+=[max_edep] #TODO there is something up with the scaling when the cut parameter is changed 

                                    # dists = np.linalg.norm(passing_mu[0].points - passing_mu[0].start_point, axis=1)
                                    # mask = dists < 10
                                    # max_edep = np.median(passing_mu[0].depositions[mask]*np.power(dists[mask],.42)) if np.any(mask) else 0
                                    pm_copy=copy.deepcopy(passing_mu[0])
                                    pm_copy.start_point,pm_copy.end_point=pm_copy.end_point,pm_copy.start_point
                                    max_edep,_=Bragg_Peak(pm_copy)
                                    if valid_decay:
                                        assert truth_mip is not None
                                        Bragg_peak_mip[angle_between(passing_mu[0].momentum,truth_mip.momentum)>np.pi/2][True]+=[max_edep] #TODO there is something up with the scaling when the cut parameter is changed 


                                    
                               
                                # order=np.argsort(dists)
                                # deps=pk[0].depositions
                                # mask = dists < 10
                                # max_edep = np.median(pk[0].depositions[mask]*np.power(dists[mask],.42)) if np.any(mask) else 0
                                # max_edep=

                                # if len(mask)>=10:
                                    dists = np.linalg.norm(pk[0].points - pk[0].end_point, axis=1)
                                    slope, intercept = np.polyfit(dists ,pk[0].depositions, 1)
                                # else:
                                    # slope=np.nan
                                # sigma_edep=slope#np.std(pk[0].depositions[mask]*np.power(dists[mask],.42))/np.std(pk[0].depositions[mask]) if np.any(mask) else 0

                                    # max_edep = np.max(all_deps[mask]) if np.any(mask) else 0


                                    

                                
                                if pk[0].reco_length>10: Bragg_peak_disc[valid_decay][passing_mu[1] in [["Bragg Peak HIP"],[]]]+=[Bragg_Peak(pk[0])[0]] #TODO there is something up with the scaling when the cut parameter is changed 
                                # Bragg_peak_sigma_disc[valid_decay][passing_mu[1] in [["Bragg Peak HIP"],[]]]+=[sigma_edep] #TODO there is something up with the scaling when the cut parameter is changed 

                                if pk[0].reco_length>5:
                                    come_to_rest_dict[valid_decay][passing_mu[1] in [["Come to Rest"],[]]]+=[come_to_rest(pk[0])] #TODO there is something up with the scaling when the cut parameter is changed 
                                    come_to_rest_prot[valid_decay][passing_mu[1] in [["Come to Rest"],[]]]+=[come_to_rest(pk[0],PROT_MASS)] #TODO there is something up with the scaling when the cut parameter is changed 
                                    come_to_rest_pi[valid_decay][passing_mu[1] in [["Come to Rest"],[]]]+=[come_to_rest(pk[0],PION_MASS)] #TODO there is something up with the scaling when the cut parameter is changed 
                                    come_to_rest_simpler[valid_decay][passing_mu[1] in [["Come to Rest"],[]]]+=[np.clip(csda_ke_lar(pk[0].reco_length, KAON_MASS)-pk[0].calo_ke,-150,150)]

                                if (valid_decay or passing_mu[1] in [["dedx chi2"],[]]) and pk[0].reco_length>10 and np.sum([norm3d(f.start_point-pk[0].end_point)<min_len for f in mu.particles if f!=passing_mu[0]])==0:
                                    chi2[valid_decay][passing_mu[1] in [["dedx chi2"],[]]]+=[Bragg_Peak(pk[0])[1]]

                                

                                if (valid_decay or passing_mu[1] in [["Come to Rest"],[]]) and pk[0].reco_length>0:
                                    come_to_rest_len[valid_decay][passing_mu[1] in [["Come to Rest"],[]]]+=[(pk[0].reco_length,come_to_rest(pk[0],KAON_MASS))]
                                if (valid_decay or passing_mu[1] in [["Bragg Peak HIP"],[]]) and pk[0].reco_length>10:
                                    Bragg_peak_len[valid_decay][passing_mu[1] in [["Bragg Peak HIP"],[]]]+=[(pk[0].reco_length,Bragg_Peak(pk[0])[0])]


                                    # for i in range(1,10):



                                if valid_decay:
                                    assert truth_mip is not None
                                    MCS_dir[int(np.dot(truth_mip.end_point-truth_mip.start_point,passing_mu[0].end_point-passing_mu[0].start_point)>0)][int(MCS_direction_prediction(passing_mu[0]))]+=1

                                    # if passing_mu[1] in [["Come to Rest"],[]]:
                                        # seen_mips+=[passing_mu[0]]
                                                            
                                        
                                
                                # ent=ent0[0]
                                # if type(ent)==int:continue

                                if truth_mip==None:
                                    mip_truth0=False
                                else:

                                    
                                    mip_truth0=int((truth_mip.ancestor_pdg_code==321)*(truth_mip.parent_pdg_code==321)*(truth_mip.creation_process=="Decay")*(abs(truth_mip.pdg_code) in [211,13])*valid_decay)
                                
                                mip_truth1=mip_truth0*([key,mu.truth_interaction_id] in valid_ints)
                                            

                                        # for g in mu.truth_pi0_gamma:
                                        #     truth
                                    # MIP_gamma_cost_disc
                                plen=passing_mu[0].reco_length
                                # if len(passing_mu)>1:
                                    # quick_save('/error_mip_break')
                                # print(passing_mu[1])
                                muon_len[mip_truth1][passing_mu[1] in [["Valid MIP Len"],[]]] += [plen]

                                if (not mip_truth1) and passing_mu[1]==[] and mip_truth0:
                                    quick_save('/good_mip_bad_city/')
                            # print(pk[2])

                            
                        #     for non_passing_mu in pk[2]:
                        #         if non_passing_mu[0][0] in seen_mips:
                        #             continue
                        #         seen_mips+=[non_passing_mu[0][0]]
                        #         if non_passing_mu[0][0].reco_length<=0: continue
                        #         if type(non_passing_mu[0][0])==TruthParticle:
                        #             truth_mip=non_passing_mu[0][0]
                        #         elif type(non_passing_mu[0][0])==RecoParticle:

                        #             truth_mip=None
                        #             if non_passing_mu[0][0].is_matched:
                        #                 if non_passing_mu[0][0].match_ids[0] in mu.decay_mip_dict:
                        #                     truth_mip=mu.decay_mip_dict[non_passing_mu[0][0].match_ids[0]]
                        #             if mpf=="Bragg Peak HIP":
                        #                 max_edep=0
                        #                 for i in mu.particles:
                        #                     for v in range(len(i.points)):
                        #                         if np.linalg.norm(i.points[v]-non_passing_mu[0][0].end_point)<2.5:
                        #                             max_edep=max(max_edep,i.depositions[v])
                        #                 Bragg_peak_disc[truth_mip is not None][True]+=[max_edep]
                        #             if mpf=="Come to Rest":
                        #                 come_to_rest_dict[truth_mip is not None][True]+=[come_to_rest(pk[0])]
                        #         else:
                        #             raise Exception()
                                
                        #         if truth_mip==None:
                        #             mip_truth=False
                        #         else:
                        #             mip_truth=(truth_mip.ancestor_pdg_code==321)*(truth_mip.parent_pdg_code==321)*(truth_mip.creation_process=="Decay")*(abs(truth_mip.pdg_code) in [211,13])
                        #         # print(non_passing_mu[1])
                        #         plen=np.sum([i.reco_length for i in non_passing_mu[0]])
                        #         muon_len[mip_truth][non_passing_mu[1]=="Valid MIP Len"] += [plen]#TODO this has to be fixed scaling way too big 
                        #     # print("adding",(ent.ancestor_pdg_code==321)*(ent.creation_process=="Decay")*(abs(ent.pdg_code) in [211,13]),rf"Primary $K^+$" not in mu.pass_failure,ent.reco_length)
                        # # print(muon_len)

                        # print(pot_mu[1])
                        

                        # 
                    # HM_acc_mu[is_true][mpf==""] += [mu.mu_hm_acc]
                    # print("kaon event",key)
                    # if mu.true_signal:
                    #     # print("found a true_signal mu")
                    #     muon_len[2]+=[mu.mip_len_base]
                    if is_true:
                        mistake=False
                        HS_count[0]+=[len(mu.kaon_path)-1]
                        reco_HS=-1
                        for rp in mu.particles:
                            if not rp.is_matched: continue
                            if not rp.shape in [TRACK_SHP]: continue
                            
                            if rp.match_ids[0] in mu.kaon_path and rp.match_overlaps[0]>.1 and len(rp.points)>10:
                                reco_HS+=1
                                if angle_between(rp.end_point-rp.start_point,mu.kaon_path[rp.match_ids[0]].end_point-mu.kaon_path[rp.match_ids[0]].start_point)>np.pi/2:# or (rp.pid not in [KAON_PID,PROT_PID] and rp!=mu.hip):
                                    mistake=True
                            if rp.match_ids[0] in mu.decay_mip_dict and rp.match_overlaps[0]>.1 and len(rp.points)>10:
                                if angle_between(rp.end_point-rp.start_point,mu.decay_mip_dict[rp.match_ids[0]].end_point-mu.decay_mip_dict[rp.match_ids[0]].start_point)>np.pi/2 or rp.pid not in [MUON_PID,PION_PID]:
                                    mistake=True
                        HS_mistakes[mistake]+=[reco_HS]


                        HS_count[1]+=[reco_HS]

                        for mip_candidate in mu.particles:
                            if not mip_candidate.is_matched: continue
                            # truth_mip=mip_candidate
                            mip_cutoff=10

                            if mip_candidate.match_ids[0] in mu.decay_mip_dict:
                                mip_truth:TruthParticle=mu.decay_mip_dict[mip_candidate.match_ids[0]]
                                if len(mu.hm_pred)>mip_cutoff: HM_acc_MIP[0]+=[mu.hm_pred[mip_candidate.id][MIP_HM]/(mu.hm_pred[mip_candidate.id][HIP_HM]+mu.hm_pred[mip_candidate.id][MIP_HM])]
                                if mip_candidate.reco_length>mip_cutoff: HM_acc_MIP[1]+=[int(mip_candidate.pid in [2,3])]

                                if mip_candidate.reco_length>mip_cutoff: MIP_primary[int(mip_candidate.is_primary)]+=[np.linalg.norm(mip_candidate.start_point-inter)]

                                if mip_candidate.reco_length>mip_cutoff: 
                                    MIP_dir+=[angle_between(mip_candidate.momentum,mip_truth.momentum)]
                                    if angle_between(mip_candidate.momentum,mip_truth.momentum)>np.pi/2:
                                        quick_save('/error_mip_direction/')
                                if mip_truth.pdg_code==211:
                                    for g_cand in [p for p in mu.particles if is_contained(p.start_point,margin=-5) and p.shape in [MICHL_SHP,SHOWR_SHP,LOWES_SHP] and len(p.points)>=3]:
                                        adg=False
                                        g_match=None
                                        if g_cand.is_matched:
                                            if g_cand.match_ids[0] in mu.truth_pi0_gamma:
                                                adg=True
                                                g_match=mu.truth_pi0_gamma[g_cand.match_ids[0]]
                                        MIP_gamma_cost_disc[adg]+=[(cos_gamma_to_E(mip_candidate.start_point,g_cand.start_point,mip_candidate.momentum)-g_cand.reco_ke)/g_cand.reco_ke]#[np.clip(cos_gamma_to_pip(g_cand.reco_ke)-np.cos(angle_between(mip_candidate.momentum,g_cand.start_point-mip_candidate.start_point)),-2,2)]
                                        if adg:
                                            assert g_match is not None
                                            assert g_match.pdg_code==22 and g_match.ancestor_pdg_code==321
                                            # print("TESTING",cos_gamma_to_E(mip_candidate.start_point,g_cand.start_point,mip_candidate.momentum),g_cand.reco_ke)
                                            # print("testing_truth",cos_gamma_to_E(mip_truth.start_point,g_match.start_point,mip_truth.momentum),g_match.reco_ke)
                                        # if np.clip(cos_gamma_to_pip(g_cand.reco_ke)-np.cos(angle_between(mip_candidate.momentum,g_cand.start_point-mip_candidate.start_point)),-2,2)>1:
                                            # quick_save('/error_cos_gamma')
                                        MIP_gamma_impact_disc[adg]+=[impact_parameter(mip_candidate.start_point,g_cand.start_point,g_cand.momentum)]
                                    
                    
                    if pc:
                        selectedkaons+=1
                        if not is_true:
                            # assert os.path.exists(kfile)
                            quick_save('/false_found/'+mu.reason)
                            print("GOT A FALSE KAON",mu.hip.id,mu.hip.pdg_code,mu.reason,mu.truth_list)#,[(i.hip_id,i.truth,i.hip.pdg_code,i.proj_dist_from_hip<testcuts["par_child_dist max"][0],len([(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in i.k_extra_children if p.dist_to_parent<testcuts["par_child_dist max"][0] and (p.child_hm_pred in [SHOWR_HM,MICHL_HM] or p.child.reco_length>5)])==0, min(np.linalg.norm(i.hip.end_point-mu.mip.start_point),np.linalg.norm(i.hip.start_point-mu.mip.start_point))<testcuts["par_child_dist max"][0] and np.linalg.norm(i.hip.end_point-mu.mip.start_point)<np.linalg.norm(i.hip.start_point-mu.mip.start_point),i.truth_list) for i in mu.potential_kaons],key,kfile,[mu.reason,mpf])
                    if not is_true:
                        kaon_pass_failure[1][mpf]+=1
                        kaon_reason_map[1][mu.reason]+=1

                        plen=-1
                        for k in mu.potential_kaons:
                            for p in k[1]:
                                plen=p[0].reco_length
                        # assert plen>=0
                        # if plen==-1:
                        #     for non_passing_mu in pk[2]
                        # for k in self.potential_kaons:#this loop looks for appropriate mips at one of the ends of the hadronic group
                        # for p in copy.copy(k[1]):
                        #     # assert type(p)==list[Particle],(type(p),type[p[0]])
                        #     plen=np.sum([i.reco_length for i in p])
                        # print("PLEN",plen)                      
                        if plen>=40:
                            kaon_pass_failure_mu[1][mpf]+=1
                        elif plen<=40:
                        # if (key,mu.truth_interaction_id) in K_to_pi_list:
                            kaon_pass_failure_pi[1][mpf]+=1
                        else:
                            raise Exception(plen)
                                
                            #     if plen>0:
                            #         break
                            # if plen>0:
                            #     break

                        # if (key,mu.truth_interaction_id) in K_to_mu_list:
                        

                    
                    
                    if is_true:
                        
                        # print("current_type",current_type_map)
                        kaon_pass_failure[0][mpf]+=1
                        kaon_reason_map[0][mu.reason]+=1
                        # TrueKaons+=1
                        if pc:
                            quick_save('/true_found')
                            correctlyselectedkaons+=1
                        else:
                            true_reason=''
                            for pk in mu.potential_kaons:
                                for p in pk[1]:
                                    if len(p[1])!=0:
                                        true_reason=p[1]

                            print("MISSED A GOOD KAON",key,kfile,mu.hip.pdg_code,mu.truth_list, true_reason)
                            quick_save('/true_missed/'+mpf)


                        if (key,mu.truth_interaction_id) in K_to_mu_list:
                            kaon_pass_failure_mu[0][mpf]+=1
                        if (key,mu.truth_interaction_id) in K_to_pi_list:
                            kaon_pass_failure_pi[0][mpf]+=1

                        
                        # print(
                        #     "true mu from kaon:",
                        #     key,
                        #     "    ",
                        #     "mu hm:",mu.mu_hm_acc,
                        #     "mu len:",mu.mip_len_base,
                        # )
                        # if mu.mip_len_base<50: print("TRUTH WARNING KAON TOO SHORT",key)
                        # muon_len[1] += [mu.mip_len_base]
                        # HM_acc_mu[1] += [mu.mu_hm_acc]
                        # muon_len_adjusted[0]+=[mu.mip_len_base]
                        # muon_len_adjusted[1]+=[mu.mip_len_base]
                        # for mk in mu.potential_kaons:
                        #     if mk.truth:
                        #         muon_len_adjusted[1][-1]+=mk.dist_from_hip
                        # for mm in mu.potential_michels:
                        #     if mm.truth:
                        #         muon_len_adjusted[1][-1] += mm.dist_to_mich
                    # K_csda_over_calo[abs(mu.hip.pdg_code)==321][mpf==""] += [mu.hip.csda_ke/mu.hip.calo_ke]




                    if (not is_true) and mpf=="":
                        # print(mu.potential_kaons[0][1])

                        # valid_mip=None\
                        passing_mip=None
                        plen=-1
                        for k in mu.potential_kaons:
                            for p in k[1]:
                                
                                if p[1]!=[]:
                                    # raise Exception(p[1])
                                    continue
                                plen=p[0].reco_length
                                if plen>0:
                                    passing_mip=p[0]
                                    break
                            if plen>0:
                                break
                        
                        
                        # print(plen)
                        if not mu.hip.is_matched:
                            quick_save(f'/backgrounds/unmatched_hip_{plen>=40}')
                        else:
                            assert mu.truth_hip is not None
                            assert passing_mip is not None

                            # truth_mip=None
                            # for pk in 

                            
                            # pk=mu.potential_kaons[0]
                            # truth_mip=pk

                            if not is_contained(mu.truth_interaction_vertex,margin=margin0):
                                quick_save(f'/backgrounds/reco_vertex_out_of_bounds_{plen>=40}')

                            elif mu.truth_hip.ke<csda_ke_lar(min_len,KAON_MASS):
                                quick_save(f'/backgrounds/low_kaon_ke_{plen>=40}')

                            elif len(passing_mip.match_ids)==0:
                                quick_save(f'/backgrounds/what_even_is_that')

                            elif passing_mip.match_ids[0] in mu.other_mip_dict and passing_mip.match_ids[0] not in mu.decay_mip_dict:
                                quick_save(f'/backgrounds/secondary_kaon_production_good_luck_with_that')

                            elif passing_mip.match_ids[0] in mu.decay_mip_dict and mu.truth_hip.pdg_code!=321:
                                quick_save(f'/backgrounds/good_KDAR_bad_primary')

                            # elif mip_truth is None:
                            #     quick_save(f'/backgrounds/what_even_is_that')
                            


                            # elif len(mu.kaon_path)==0:
                            #     quick_save(f'/backgrounds/secondary_kaon_without_primary_{truth_hip.pdg_code}_{plen>=40}')
                            
                            
                            elif mu.truth_hip.ancestor_pdg_code==2212:
                                quick_save(f'/backgrounds/prot_anc_{plen>=40}')
                            elif mu.truth_hip.ancestor_pdg_code==-321:
                                quick_save(f'/backgrounds/antikp_anc_{plen>=40}')
                            elif mu.truth_hip.pdg_code==321 and not mu.truth_hip.is_primary and mu.truth_hip.ancestor_pdg_code==321:
                                quick_save(f'/backgrounds/non_primary_kp_w_kp_ancestor_{plen>=40}')
                            elif mu.truth_hip.pdg_code==211:
                                quick_save(f'/backgrounds/overwritten_pi_{plen>=40}')
                            elif mu.truth_hip.ancestor_pdg_code==311:
                                quick_save(f'/backgrounds/k0_conv_{plen>=40}')
                            elif mu.truth_hip.pdg_code==3222:
                                quick_save(f'/backgrounds/sigmap_{plen>=40}')
                            elif mu.truth_hip.pdg_code==3112:
                                quick_save(f'/backgrounds/sigmam_{plen>=40}')
                            

                            elif mu.truth_interaction_nu_id==-1:
                                quick_save(f'/backgrounds/cosmics_{plen>=40}')

                            elif mu.truth_hip.pdg_code==2212 and mu.truth_hip.parent_pdg_code in [3222,3112]:
                                quick_save(f'/backgrounds/proton_from_sigma_{mu.truth_hip.parent_pdg_code}_{plen>=40}')

                            elif mu.truth_hip.pdg_code==321 and mu.truth_hip.ancestor_pdg_code!=321:
                                quick_save(f'/backgrounds/non_primary_kp_non_kp_ancestor_{mu.truth_hip.ancestor_pdg_code}_{plen>=40}')


                            # elif truth_hip.pdg_code==321 and truth_hip.is_primary and closest_reco_particle_to_truth_start(hip_candidate,particles,truth_particles)==hip_candidate
                            elif mu.truth_hip.pdg_code!=321 and mu.truth_hip.is_primary:
                                quick_save(f'/backgrounds/primary_non_kp_{mu.truth_hip.pdg_code}_{plen>=40}')

                            

                            elif mu.truth_hip.pdg_code!=321 and mu.truth_hip.ancestor_pdg_code!=321 and not mu.truth_hip.is_primary:
                                quick_save(f'/backgrounds/non_primary_{mu.truth_hip.pdg_code}_w_parent_{mu.truth_hip.parent_pdg_code}_w_ancestor_{mu.truth_hip.ancestor_pdg_code}_{plen>=40}')

                            else:
                                quick_save(f'/backgrounds/unknown_{mu.reason}_{plen>=40}')
                            
                    if is_true and mpf!="":
                        assert mu.truth_hip is not None
                        if not is_contained(mu.truth_hip.points,margin=3):
                            quick_save('/missing/Kaon_Uncontained')

                        
                        else:
                            quick_save('/missing/unknown_'+mpf)

                    # print(mu.hip.end_momentum)
                    # for k in mu.potential_kaons:
                    #     dir_acos_K[is_true and k.truth][mpf==""] += [k.dir_acos]
                    #     kaon_len[is_true and k.truth][mpf==""] += [k.hip_len]
                    #     HM_acc_K[is_true and k.truth][mpf==""] += [k.k_hm_acc]
                    #     dist_to_hip[is_true and k.truth][mpf==""] += [k.dist_from_hip]
                    #     K_csda_over_calo[is_true and k.truth][mpf==""] += [np.log(k.hip.csda_ke/k.hip.calo_ke)]

                        # if k.truth:
                        #     print(
                        #         "with accompanying kaon:",
                        #         key,
                        #         "    ",
                        #         # k.hip_id,
                        #         "k diracos:",k.dir_acos,
                        #         "k hm:",k.k_hm_acc,
                        #         "k extra dist:",k.dist_from_hip,
                        #         "k len:",k.hip_len,
                        #         # k.k_extra_children,
                        #     )
                            # dir_acos_K[1] += [k.dir_acos]
                            # kaon_len[1] += [k.hip_len]
                            # HM_acc_K[1] += [k.k_hm_acc]
                            # dist_to_hip[1] += [k.dist_from_hip]

                            # for kc in k.k_extra_children:
                            #     k_extra_children[0][0] += [kc.dist_to_parent]
                            #     k_extra_children[0][1] += [kc.angle_to_parent]
                            #     if kc.truth:
                            #         k_extra_children[1][0] += [kc.dist_to_parent]
                            #         k_extra_children[1][1] += [kc.angle_to_parent]

                    # for mich in mu.potential_michels:

                    #     HM_acc_mich[is_true and mich.truth][mpf==""]+=[mich.mich_hm_acc]
                    #     dist_to_mich[is_true and mich.truth][mpf==""]+=[mich.dist_to_mich]

                    #     # mu_extra_children[0][0]+=[mich.mu_extra_children[0]]
                    #     # mu_extra_children[0][1]+=[mich.mu_extra_children[1]]

                    #     if mich.truth:
                    #         print(
                    #             "with accompanying michel:",
                    #             key,
                    #             "    ",
                    #             # mich.mich_id,
                    #             "mich extra dist:",mich.dist_to_mich,
                    #             # mich.mu_extra_children,
                    #             # decay_t_to_dist: float
                    #             "mich hm:",mich.mich_hm_acc,
                    #         )

                            # HM_acc_mich[1]+=[mich.mich_hm_acc]
                            # dist_to_mich[1]+=[mich.dist_to_mich]
                            # for mc in mich.mu_extra_children:
                            #     mu_extra_children[0][0] += [mc.dist_to_parent]
                            #     mu_extra_children[0][1] += [mc.angle_to_parent]
                            #     if mc.truth:
                            #         mu_extra_children[1][0] += [mc.dist_to_parent]
                            #         mu_extra_children[1][1] += [mc.angle_to_parent]
        #         # print("")



    print("num neutrinos",num_nu_from_file)
    print("kaon eff/pur",np.divide(correctlyselectedkaons,TrueKaons) if TrueKaons else 0,np.divide(correctlyselectedkaons,selectedkaons) if selectedkaons else 0,[correctlyselectedkaons,selectedkaons,TrueKaons])
    print("kaon reasons",kaon_reason_map)
    print("kaon pass_failure",kaon_pass_failure)

    if args.single_file!="":
        print(f"returning from {args.single_file}")
        return



    for help in [(Bragg_peak_len,"PIDA","Bragg Peak Len"),(come_to_rest_len,"CSDA Len/CALO Len -1","come_to_rest_len")]:

        xs, ys, labels = [], [], []

        for gt, preds in help[0].items():
            for pred, points in preds.items():
                # Skip false-false
                if gt in (0, False) and pred in (0, False):
                    continue

                for x, y in points:   # now points is a list of tuples
                    xs.append(float(x))
                    ys.append(float(y))
                    labels.append(f"Valid Decay:{bool(gt)}, {bool(pred)} after other cuts")

        if not xs:
            print("No valid points to plot (all were false-false).")
        else:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1
            pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1

            plt.figure(figsize=(7, 7))
            for label in sorted(set(labels)):
                mask = [l == label for l in labels]
                plt.scatter(
                    [x for x, m in zip(xs, mask) if m],
                    [y for y, m in zip(ys, mask) if m],
                    label=label,
                    s=25,
                    alpha=0.7
                )
            plt.xlim(max(.1,x_min - pad_x), x_max + pad_x)
            plt.ylim(y_min - pad_y, y_max + pad_y)

        plt.xlabel("Reco length")
        plt.ylabel(help[1])
        plt.legend(fontsize=8, markerscale=0.7)
        # plt.xscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout();plt.savefig(PLOTSDIR+help[2])


    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    plt.figure()
    plt.hist2d(K_dir[0], K_dir[1], bins=[20, 20],range=[[0, np.pi], [np.array(K_dir[1]).min(), np.array(K_dir).max()]],cmap="viridis",norm=mcolors.LogNorm())  # 20 bins for each axis
    plt.colorbar(label='Counts')
    patch = mpatches.Patch(color="blue", label=rf"mean={np.mean(K_dir[0]):.2f}, std={np.std(K_dir[0]):.2f}, %<$\pi/2$={(np.sum(np.array(K_dir[0])<np.pi/2)/len(np.array(K_dir[0]))):.2f}")
    plt.legend(handles=[patch])
    plt.xlabel('Reco HIP to True HIP Angle [Rad]')
    plt.ylabel('HIP Len')
    # plt.title('2D Histogram: Angle vs Length')
    # plt.show()
    # plt.legend()
    plt.tight_layout();plt.savefig(PLOTSDIR+"hip_dir_angle")

    
    plt.figure()
    plt.hist2d(K_dir_before[0], K_dir_before[1], bins=[20, 20],range=[[0, np.pi], [np.array(K_dir_before[1]).min(), np.array(K_dir_before).max()]],cmap="viridis",norm=mcolors.LogNorm())  # 20 bins for each axis
    plt.colorbar(label='Counts')
    
    patch = mpatches.Patch(color="blue", label=rf"mean={np.mean(K_dir_before[0]):.2f}, std={np.std(K_dir_before[0]):.2f}, %<$\pi/2$={(np.sum(np.array(K_dir_before[0])<np.pi/2)/len(np.array(K_dir_before[0]))):.2f}")
    plt.legend(handles=[patch])
    plt.xlabel('Reco HIP to True HIP Angle [Rad]')
    plt.ylabel('HIP Len')
    # plt.legend()
    # plt.title('2D Histogram: Angle vs Length')
    # plt.show()
    plt.tight_layout();plt.savefig(PLOTSDIR+"hip_dir_angle_before")


    (n, bins, patches)=plt.hist(K_mom+pi_mom+mu_mom, bins=50)
    plt.clf()
    plt.hist(K_mom,label=r"$K^+$ in $K^+$ decays",bins=bins.tolist(),alpha=0.3)
    plt.hist(pi_mom,label=r"$\pi^+$ in $K^+$ decays",bins=bins.tolist(),alpha=0.3)
    plt.hist(mu_mom,label=r"$\mu$ in $K^+$ decays",bins=bins.tolist(),alpha=0.3)
    plt.xlabel("Momentum [MeV]")
    plt.ylabel("Freq")
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)   
    plt.tight_layout();plt.savefig(PLOTSDIR+"truth_momenta")


    plt.clf()
    bins = np.arange(0 - 0.5, len(set(HS_count[0]+HS_count[1])) + 1.5, 1)
    plt.hist(HS_count[0],bins=bins.tolist(),label="True",alpha=.5)
    plt.hist(HS_count[1],bins=bins.tolist(),label="Reco",alpha=.5)
    # 

    # plt.hist(data, bins=bins, edgecolor='black', align='mid')
    plt.xticks(np.arange(0,len(set(HS_count[0]+HS_count[1])) + 1))  # force integer ticks
    plt.xlabel("# Hard Scatters Before Decay")
    plt.ylabel("Freq")
    plt.grid(True, linestyle="--", alpha=0.5)   
    plt.legend()
    plt.tight_layout();plt.savefig(PLOTSDIR+"HS_count")
    
    plt.clf()


    bins = np.arange(0 - 0.5, len(set(HS_mistakes[0]+HS_mistakes[1])) + 1.5, 1)
    plt.hist(HS_mistakes[0],bins=bins.tolist(),label="PID/Dir. Mistake Free",alpha=.5)
    plt.hist(HS_mistakes[1],bins=bins.tolist(),label="PID/Dir. Mistake",alpha=.5)
    # 

    # plt.hist(data, bins=bins, edgecolor='black', align='mid')
    plt.xticks(np.arange(0,len(set(HS_mistakes[0]+HS_mistakes[1])) + 1))  # force integer ticks
    plt.xlabel("# Hard Scatters Before Decay")
    plt.ylabel("Freq")
    plt.grid(True, linestyle="--", alpha=0.5)   
    plt.legend()
    plt.tight_layout();plt.savefig(PLOTSDIR+"HS_mistakes")
    
    plt.clf()
    




    for help in [
        # [dist_to_hip,"Extra mu dist from hip [cm]","hip_mip_dist"],
        # [dist_to_mich,"Extra mu dist from mich [cm]","mip_mich_dist"],
        [kaon_len,"Kaon Len [cm]","Kaon Len"],
        [primary_kp,r"Vertex to HIP dist [cm]","Close to Vertex"],

        # [forward_disc,r"$p_z$/|p|","Forward HIP"],
        # [HM_acc_mu, r"$\mu$ HM scores", "mu_HM"],
        # [HM_acc_K, r"K HM scores", "K_HM"],
        # [HM_acc_mich, r"Michel HM scores", "michel_HM"],
        # [HM_acc_pi, r"$\Lambda$ Pi HM scores", "lam_HM_pi"],
        # [HM_acc_prot, r"$\Lambda$ Proton HM scores", "lam_HM_prot"],
        # [ProtPi_dist_disc_total,r"$\Lambda$ p-pi proj. dist~[cm]","lam_prot_pi_dist_disc","lam_dist max"],
        # [lam_decay_len_disc_total,r"$\Lambda$ decay len~[cm]","lam_decay_len_disc","lam_decay_len"],
        # [lam_dir_acos,"Lambda Acos momentum to beam","lam_Acos"],
        # [lam_momentum,"Lambda Momentum [MeV/c]","lam_mom"],
        # [lam_true_momentum,"Geant4 Lambda Momentum [MeV/c]","lam_mom_g4"],
        [muon_len,"MIP len [cm]","Valid MIP Len"],
        (Bragg_peak_disc,"Mean Edep*Sqrt[R] within 10 cm [MeV*Sqrt[cm]]","Bragg Peak HIP"),
        # (Bragg_peak_sigma_disc,"dE/dx slope [MeV/cm^2]","Bragg Peak Sigma"),
        (Bragg_peak_mip,"PIDA [MeV*[cm]^0.42]","Bragg Peak MIP"),
        
        (come_to_rest_dict,"Calo KE/CSDA KE -1","Come to Rest"),

        (come_to_rest_simpler,"CSDA KE-CALO KE","Come to Rest Simpler"),
        (come_to_rest_prot,"Calo KE/CSDA KE -1","Come to Rest prot"),
        (come_to_rest_pi,"Calo KE/CSDA KE -1","Come to Rest pi"),
        (chi2,"Chi2","chi2"),
        (closest_FM,"Closest Flashmatched Interaction [cm]","closest_FM")

        # [lam_tau0_est,r"$\tau_0=\frac{dx_{decay}m_{est}}{p_{est}}~[ns]$","t0_est"]
        # [dir_acos_K,"Kaon Acos momentum to beam","K_Acos"]
        
        ]:
            # for actual in [True,False]:
                
                # plt.hist(base_len_vae,bins=30)
    # plt.xlabel("VAE base length [cm]")
    # plt.ylabel("Freq")
    # plt.yscale("log")
    # plt.tight_layout();plt.savefig("plots/" + str(FOLDER) + "/lambda/"+"base_len_vae")
    # plt.clf()
                # print(help[1])
                # maxval=help[0][actual][True]
            # mine=np.array(help[0][True][True]+)
            # if args.mode!="truth": continue
            mine=np.array(help[0][True][True]+help[0][True][False]+help[0][False][True]+help[0][False][False])

            mine=mine[mine<2000]
            # (n, bins, patches)=plt.hist(mine, bins=100)
            # if help[2]!="closest_FM":
            if help[2] in ["Come to Rest","Come to Rest prot","Come to Rest pi"]:
                mine=mine[mine<1]
                # (n, bins, patches)=plt.hist(mine, bins=50)
            elif help[2] in ["Kaon Len","Close to Vertex"]:
                mine=mine[mine<20]
            elif help[2] in ["Valid MIP Len"]:
                mine=mine[mine<70]
                # (n, bins, patches)=plt.hist(mine, bins=50)
            # elif 

            (n, bins, patches)=plt.hist(mine, bins=50)
            # else:
            #     (n, bins, patches)=plt.hist(mine[mine<2000], bins=100)
            plt.clf()
            
            for actual in [False,True]:
                # print(help[0][actual][True])
                if help[2] not in ["closest_FM"]:
                    plt.hist(help[0][actual][True], bins=list(bins), label=f"{actual} after other cuts",alpha=.7)
                else:
                    plt.hist(help[0][actual][True], bins=list(bins), label=rf"{actual} $K^+$ Decay Interactions",alpha=.7)
                # count+=1
            if help[2]=="Valid MIP Len":
                n=kaon_pass_order["Valid MIP Len"]
                x1 = n % 100
                x2 = (n // 100) % 100
                x3 = (n // 100**2) % 100
                x4 = (n // 100**3) % 100
                mu_len=[x3,x4]
                plt.axvspan(mu_len[0], mu_len[1], color='red', alpha=0.2, label=r"$K^+\rightarrow \nu\mu$ band")
                pi_len=[x1,x2]
                plt.axvspan(pi_len[0], pi_len[1], color='blue', alpha=0.2, label=r"$K^+\rightarrow \pi^+\pi^0$ band")
            # count=2
            if help[2] in ["Kaon Len","Come to Rest","Close to Vertex","closest_FM","Bragg Peak HIP","Forward HIP"] and help[2] in kaon_pass_order:
                # klen=kaon_pass_order[help[2]]
                plt.axvline(kaon_pass_order[help[2]], color='red', linestyle='--', linewidth=2, label=f"cut value={kaon_pass_order[help[2]]}",alpha=0.5)
            # if help[2] in ["Bragg Peak HIP"]:
            #     # mycut=kaon_pass_order["Bragg Peak HIP"]
            #     plt.axvline(kaon_pass_order[help[2]], color='red', alpha=0.5, label=f"cut value={kaon_pass_order[help[2]]}")
            if help[2] in ["closest_FM","Close to Vertex"]:
                plt.yscale("log")
                # plt.axvline(kaon_pass_order["Valid Interaction"], color='red', alpha=0.5, label="Max Vertex-Vertex Distance to Flashmatched Interaction")
            # if help[2] in ["Close to Vertex"]:
                # klen=kaon_pass_order[help[2]]
                # plt.yscale("log")
                # plt.axvline(kaon_pass_order[help[2]], color='red', linestyle='--', linewidth=2, label=f"cut value={kaon_pass_order[help[2]]}",alpha=0.5)

            # if help[2] in ["Forward HIP"]:
                # plt.axvline(kaon_pass_order[help[2]], color='red', alpha=0.5, label=r"Forward $\theta$ cut")

            # if help[2] in ["Come to Rest"]:
            #     # plt.axvline(kaon_pass_order["Come to Rest"], color='red', alpha=0.5, label=r"DAR Agreement Energy Cut")
            #     # plt.axvspan(kaon_pass_order["Come to Rest"][0], kaon_pass_order["Come to Rest"][1], color='red', alpha=0.5, label=r"DAR Agreement Energy Cut")
            #     plt.axvline(kaon_pass_order[help[2]], color='red', linestyle='--', linewidth=2, label=f"cut value={kaon_pass_order[help[2]]}",alpha=0.5)


            for actual in [True]:
                # plt.hist(help[0][actual][True]+help[0][actual][False], bins=bins,label=f"all {actual}",facecolor='none',edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],alpha=.5)

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
                # alpha=1
                # if actual==False:
                #     alpha=.5
            # count=2
                # for m in testcuts["mu_len"]:
                #     plt.axvline(x=m, color='red', linestyle='--', linewidth=2, label=f"cut value={m}")
                # for m in testcuts["pi_len"]:
                #     plt.axvline(x=m, color='blue', linestyle='--', linewidth=2, label=f"cut value={m}")
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
            # plt.yscale("log")
            plt.legend(frameon=True, fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout();plt.savefig(PLOTSDIR+help[2])
            plt.clf()

    if False:
        (n, bins, patches) = plt.hist(muon_len[0], bins=list(np.linspace(40,60,40)), label="all")
        plt.hist(muon_len[1], bins=list(bins), label=r"all $K^+-\mu$")
        # plt.hist(muon_len[2], bins=list(bins), label="signal events",alpha=.5)
        plt.axvline(x=54, color="r", label="ideal KDAR muon")
        plt.xlabel("Muon Len [cm]")
        plt.ylabel("Freq")
        plt.legend()
        plt.tight_layout();plt.savefig(PLOTSDIR+"Muon_Len")
        plt.clf()


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

    


    for s in [(-np.array(vertex_dz),r"Vertex Reco-Truth $\Delta$z [cm]","vertex_dz"),
            (vertex_displacement,r"Vertex Displacement [cm]","vertex_displacement")]:

        plt.hist(s[0], label=f"mean={np.mean(s[0]):.2f}, std={np.std(s[0]):.2f}",alpha=0.5)
        # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
        plt.xlabel(s[1])
        plt.ylabel("Freq")
        plt.yscale("log")
        plt.legend(frameon=True, fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout();plt.savefig(PLOTSDIR+'/'+s[2])
        plt.clf()



        for s in [[HM_acc_MIP, r"MIP HM Accuracy", "mu_HM"],
            [HM_acc_K, f"$K^+$ HM Accuracy", "K_HM"]]:

            print("running",s[2])

            rmnans0=[z for z in s[0][0] if not np.isnan(z)]
            rmnans1=[z for z in s[0][1] if not np.isnan(z)]
            # print(rmnans0,rmnans1)
            plt.hist(rmnans0, label=f"HM: mean={np.mean(rmnans0):.2f}, std={np.std(rmnans0):.2f}, %>0.5={(np.sum(np.array(rmnans0)>.5)/len(np.array(rmnans0))):.2f}",bins=np.linspace(0,1,40).tolist(),alpha=.5)
            plt.hist(rmnans1, label=f"PID: mean={np.mean(rmnans1):.2f}, std={np.std(rmnans1):.2f}, %>0.5={(np.sum(np.array(rmnans1)>.5)/len(np.array(rmnans1))):.2f}",bins=np.linspace(0,1,40).tolist(),alpha=.5)
            # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
            plt.xlabel(s[1])
            plt.ylabel("Freq")
            # if s[2] in ["vertex_dz","vertex_displacement"]:
            plt.yscale("log")
            plt.legend(frameon=True, fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout();plt.savefig(PLOTSDIR+"/"+s[2])
            plt.clf()


        for s in [
            
            [K_dir[0], r"$K^+$ $\theta$ from True Dir.", "Kp_dir"],
            [MIP_dir, r"MIP $\theta$ from True Dir.", "MIP_dir"],
            
            ]:
        
            print("running",s[2])
            #prot_primary+=[l.decaylen]
                        # pi_primary+=[l.decaylen]

                        # prot_mom[0]+=[fixed_prot_mom]
                        # prot_mom[1]+=[truth_hip.p]
                        # pi_mom[0]+=[fixed_pi_mom]
                        # pi_mom[1]+=[truth_mip.p]


                        # pi_dir+=[angle_between(l.mip.momentum,truth_mip.p)]
                        # prot_dir+=[angle_between(l.hip.momentum,truth_hip.p)]

            # print(s[0],np.isnan(s[0]))

            rmnans=[z for z in s[0] if not np.isnan(z)]

            plt.hist(s[0], label=rf"mean={np.mean(rmnans):.2f}, std={np.std(rmnans):.2f}, %<$\pi/2$={(np.sum(np.array(rmnans)<np.pi/2)/len(np.array(rmnans))):.2f}",bins=50,alpha=.5)
            # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
            plt.xlabel(s[1])
            plt.ylabel("Freq")
            # if s[2] in ["vertex_dz","vertex_displacement"]:
            plt.yscale("log")
            plt.legend(frameon=True, fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout();plt.savefig(PLOTSDIR+"/"+s[2])
            plt.clf()


        for s in [
            [MIP_gamma_cost_disc, r"$\gamma~\Delta Cos \theta$", "MIP_gamma_cos"],
            [MIP_gamma_impact_disc,r"$\gamma$ Impact Parameter", "MIP_gamma_impact"],
            [K_clustering,r"Interaction Overlap","Cluster_overlap"],
            [tracking_threshold,r"Kaon Len [cm]","tracking_threshold"],
            (K_primary,r"Dist to Vertex [cm]","K_primary"),
            (MIP_primary,r"Dist to Vertex [cm]","MIP_primary"),
            

            ]:
        
            print("running",s[2])
            #prot_primary+=[l.decaylen]
                        # pi_primary+=[l.decaylen]

                        # prot_mom[0]+=[fixed_prot_mom]
                        # prot_mom[1]+=[truth_hip.p]
                        # pi_mom[0]+=[fixed_pi_mom]
                        # pi_mom[1]+=[truth_mip.p]


                        # pi_dir+=[angle_between(l.mip.momentum,truth_mip.p)]
                        # prot_dir+=[angle_between(l.hip.momentum,truth_hip.p)]

            # print(s[0],np.isnan(s[0]))
            # for h in [False,True]
            # rmnans=[z for z in s[0] if not np.isnan(z)]
            bins=20
            (n, bins, patches) = plt.hist(s[0][0]+s[0][1], bins=bins)
            plt.clf()
            if s[2]=="MIP_gamma_impact":
                bins=np.linspace(0,20,20)
                plt.axvspan(0, kaon_pass_order[r"Low MIP len $\pi^0$ Tag"], color='red', alpha=0.2, label=r"Acceptance band")
            if s[2]=="MIP_gamma_cos":
                # dset=
                bins=np.linspace(-3,3,30)
                # plt.axvspan(0, kaon_pass_order[r"Low MIP len $\pi^0$ Tag"][2], color='red', alpha=0.2, label=r"Acceptance band")
            
            for b in [False,True]:
                if s[2]=="tracking_threshold":
                    plt.hist(s[0][b], label=f"{str(b)}: mean={np.mean(s[0][b]):.2f}, std={np.std(s[0][b]):.2f}",bins=bins.tolist(),alpha=0.5)
                elif s[2]=="Cluster_overlap":
                    plt.hist(s[0][b], label=f"Flashmatched: {str(b)}: Count={len(s[0][b])}",alpha=0.5,bins=bins.tolist())
                elif s[2] in ["K_primary","MIP_primary"]:
                    plt.hist(s[0][b], label=f"Primary: {str(b)}: Count={len(s[0][b])}",alpha=0.5,bins=bins.tolist())
                elif s[2] in ["MIP_gamma_cos","MIP_gamma_impact"]:
                    plt.hist(s[0][b], label=rf"{str(b)} Decay $\gamma$",alpha=0.5,bins=bins.tolist())

                else:
                    raise Exception()
            # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
            plt.xlabel(s[1])
            plt.ylabel("Freq")
            # if s[2] in ["vertex_dz","vertex_displacement"]:
            if s[2] not in ["MIP_gamma_cos","MIP_gamma_impact"]:
                plt.yscale("log")
            plt.legend(frameon=True, fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout();plt.savefig(PLOTSDIR+"/"+s[2])
            plt.clf()

    for actual in [True,False]:

        for help in [
            # [dist_to_hip,"Extra mu dist from hip [cm]","hip_mip_dist"],
            # [dist_to_mich,"Extra mu dist from mich [cm]","mip_mich_dist"],
            # [kaon_len,"Kaon Len [cm]","Kaon_Len"],
            
            # [HM_acc_mich, r"Michel HM scores", "michel_HM"],
            # [HM_acc_pi, r"$\Lambda$ Pi HM scores", "lam_HM_pi"],
            # [HM_acc_prot, r"$\Lambda$ Proton HM scores", "lam_HM_prot"],
            # [ProtPi_dist,r"$\Lambda$ p-pi proj. dist [cm]","lam_prot_pi_dist"],
            # [lam_decay_len,r"$\Lambda$ decay len [cm]","lam_decay_len"],
            # [lam_dir_acos,"Lambda Acos momentum to beam","lam_Acos"],
            # [dir_acos_K,"Kaon Acos momentum to beam","K_Acos"],
            # [K_csda_over_calo,r"$ln(KE_{CSDA}/KE_{Calo})$","K_csda_over_calo"]
            
            ]:
                
                if args.mode!="truth":
                    continue
                (n, bins, patches) = plt.hist(help[0][actual][True]+help[0][actual][False], bins=20,label=f"all {actual}")
                plt.hist(help[0][actual][True], bins=list(bins), label=f"{actual} after cuts")
                # plt.axvline(x = 54, color = 'r', label = 'ideal KDAR muon')
                plt.xlabel(help[1])
                plt.ylabel("Freq")
                # plt.yscale("log")
                plt.legend(frameon=True, fontsize=11)
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout();plt.savefig(PLOTSDIR+help[2]+"_"+str(actual))
                plt.clf()



    for kpf in [(kaon_pass_failure,TrueKaons,"all"),(kaon_pass_failure_pi,TrueKaonspi,"pi"),(kaon_pass_failure_mu,TrueKaonsmu,"mu")]:

        plt.figure(dpi=300)
        cols=[]
        for i in kaon_pass_order:
            cols+=[i]

        columns=[wrap_text(i) for i in cols]

        # raise Exception(columns)


        
        rows = [r"Selected $K^+$","Bkg.", r"Signal $K^+$","Eff.","Pur.","Eff.*Pur."] 
        data=np.zeros((len(rows),len(columns)))


        sel_kaon_T=np.zeros(len(columns))
        sel_kaon_T[-1]=kpf[0][0][cols[-1]]
            

        sel_kaon_F=np.zeros(len(columns))
        sel_kaon_F[-1]=kpf[0][1][cols[-1]]

        sel_kaon=np.zeros(len(columns))
        sel_kaon[-1]=kpf[0][0][cols[-1]]+kpf[0][1][cols[-1]]




        for c in range(len(columns) - 2, -1, -1):
            sel_kaon_F[c]=sel_kaon_F[c+1]+kpf[0][1][cols[c]]
            sel_kaon_T[c]=sel_kaon_T[c+1]+kpf[0][0][cols[c]]

            sel_kaon[c]=sel_kaon[c+1]+kpf[0][0][cols[c]]+kpf[0][1][cols[c]]

        scaling=80000/num_nu_from_file

        for r in range(len(rows)):
            for c in range(len(columns)):
                if rows[r]==r"Signal $K^+$":data[r][c]=int(kpf[1]*scaling)
                if rows[r]==r"Selected $K^+$":data[r][c]=int(sel_kaon_T[c]*scaling)
                if rows[r]=="Bkg.":data[r][c]=int(sel_kaon_F[c]*scaling)
                # print(sel_kaon_T[c],sel_kaon[c],Truekaonbdas)
                eff=np.nan_to_num(np.divide(sel_kaon_T[c],kpf[1]), nan=0)
                pur=np.nan_to_num(np.divide(sel_kaon_T[c],sel_kaon[c]),nan=0)
                if rows[r]=="Eff.":data[r][c]=rsf(eff,2)
                if rows[r]=="Pur.":data[r][c]=rsf(pur,2)
                if rows[r]=="Eff.*Pur.":data[r][c]=rsf(eff*pur,2)
                
        data2=np.flip(data,axis=0)
        
        # Get some pastel shades for the colors 
        # colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows))) 
        assert len(rows)==6
        colors=colors = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0]   # Blue
    ])
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
        plt.title('Kaon Efficiency and Purity as a Function of Cut') 
        plt.yscale("log")
        
        plt.tight_layout();plt.savefig(PLOTSDIR+"kaon_eff_purity_"+kpf[2],bbox_inches='tight')
        plt.clf()


        plt.figure(dpi=300, figsize=(12, 6))

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
        plt.savefig(PLOTSDIR+"kaon_eff_purity_table_"+kpf[2], bbox_inches='tight')
        plt.clf()

    def plot_confusion(MCS_dir):
        mat = np.array(MCS_dir)
        total = mat.sum()
        percentages = mat / total * 100

        fig, ax = plt.subplots()
        cax = ax.matshow(percentages, cmap="Blues")
        fig.colorbar(cax)

        # Set axis labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Wrong", "Correct"])#MCS
        ax.set_yticklabels(["Wrong", "Correct"])#NULL

        # Rotate x labels
        plt.xticks(rotation=45)

        # Annotate the cells with counts and percentages
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{mat[i, j]}\n({percentages[i, j]:.1f}%)",
                    ha="center", va="center", color="black"
                )

        # plt.title(title)
        plt.ylabel("Null prediction")
        plt.xlabel("MCS prediction")
        # plt.tight_layout()
        plt.tight_layout()
        plt.savefig(PLOTSDIR+"MCS_pred_dir", bbox_inches='tight')
        plt.clf()


    # # Example usage
    # MCS_dir = [[10, 20],
    #         [15, 55]]

    plot_confusion(MCS_dir)

if __name__ == "__main__":
    main()


