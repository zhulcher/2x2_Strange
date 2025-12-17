from statistics_plot_base import *
from statistics_plot_lam import find_root_path,decay_angle_from_invariant_mass,MIN_LAM_MOM,MAX_LAM_MOM,MIN_PROT_MOM,MIN_PI_MOM

from print_particle_record import load_cfg

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Script to plot K+/Lam Assoc. stats')
    parser.add_argument('--mode', type=str, choices=["truth", "reco"], help='Reco or Truth running mode',default="reco")
    parser.add_argument('--N', type=int, default=sys.maxsize, help='Number of files to run')

    args = parser.parse_args()

    assert args.mode



    if args.mode=="truth":
        from statistics_plot_kp import kaon_pass_order_truth as kaon_pass_order
        from statistics_plot_lam import lam_pass_order_truth as lam_pass_order
    else:
        from statistics_plot_kp import kaon_pass_order_reco as kaon_pass_order
        from statistics_plot_lam import lam_pass_order_reco as lam_pass_order


    # if "MIP_CUTS" in kaon_pass_order:
    #     for mykey in ["Valid MIP Len","Michel Child","MIP Child At Most 1 Michel",r"Low MIP len $\pi^0$ Tag","Single MIP Decay"]:
    #         kaon_pass_order["MIP_CUTS"].pop(mykey)


    def truth_interaction_id(lk):
        if lk.is_matched:
            return lk.match_ids[0]
        return None

    # for mykey in ["Min Decay Len"]:
    #     lam_pass_order.pop(mykey)


    MAXFILES=args.N
    d0=directory.replace("_analysis", f"_analysis_{args.mode}")

    files = os.listdir(d0)

    # FILES = [os.path.join(directory, f) for f in files]
    if len(files)>MAXFILES:
        files=files[:MAXFILES]

    res_assoc=np.zeros((3,5))


    res_K=np.zeros((3,3))

    res_L=np.zeros((3,3))


    LOCAL_EVENT_DISPLAYS=event_display_new_path+"/assoc_kp_lam/"+args.mode+'/'

    assert os.path.isdir(LOCAL_EVENT_DISPLAYS), f"{LOCAL_EVENT_DISPLAYS} does not exist"


    CCNCmap={0:"CC",1:"NC"}

    NONLOCAL_EVENT_DISPLAYS=base_directory+FOLDER+"_files_"+args.mode

    # print("clearing",NONLOCAL_EVENT_DISPLAYS)
    # clear_html_files(NONLOCAL_EVENT_DISPLAYS)
    # print("cleared",NONLOCAL_EVENT_DISPLAYS)
    
    assert os.path.isdir(NONLOCAL_EVENT_DISPLAYS), f"{NONLOCAL_EVENT_DISPLAYS} does not exist"
    print("only including", MAXFILES, "files")


    # num_nu=0
    filecount=0
    for file0 in files:

        file=os.path.join(d0, file0)


        def save_my_html(newpath,name,name2):
            copy_and_rename_file(NONLOCAL_EVENT_DISPLAYS,file,newpath,name,name2,args.mode)
        file_truth=os.path.join(directory2, file0)

        file_truth=file_truth[:-1]+"y"

        both_there=True
        for f in [file,file_truth]:
            both_there*=os.path.exists(f)#, f"File not found: {f}"
        if not both_there:
            print("had to skip because",f"File not found: {file0}")
            continue


        particles = np.load(file_truth, allow_pickle=True)
        interactions=particles[2]

        larcv_keys_assoc=[[],[]]
        larcv_keys_K=[[],[]]
        larcv_keys_L=[[],[]]

        found_Kp_keys=[[],[]]
        found_lam_keys=[[],[]]

        # rej_Kp_keys=[]
        # rej_lam_keys=[]





        for I in interactions:
            i=interactions[I]
            if is_contained(i[0][:3],margin=margin0):

                primpdgsK=[z for z in i[1].keys() if z[1]==321]
                primpdgsL=[z for z in i[1].keys() if z[1]==3122]

                TrueKaonspi=0
                TrueKaonsmu=0

                idx=0
                if len(primpdgsK):
                    kaon_list=[zz for zz in i[1][primpdgsK[0]] if zz[0]==321]
                    
                    kaon_list=[zz for zz in i[1][primpdgsK[idx]] if zz[0]==321]
                    for key in range(len(primpdgsK)):
                        kaon_list=[zz for zz in i[1][primpdgsK[key]] if zz[0]==321]
                        if len(kaon_list):
                            idx=key
                            break
                    assert len(kaon_list)
                    kaon_list=kaon_list[0]
                    valid_decays=[zz[0] for zz in i[1][primpdgsK[idx]] if zz[0] not in  [321]]
                    child_list_mu=[zz for zz in i[1][primpdgsK[idx]] if zz[0] in [-13]]
                    child_list_pi=[zz for zz in i[1][primpdgsK[idx]] if zz[0] in [211]]

                    min_kaon_ke=csda_ke_lar(min_len,KAON_MASS)
                    min_kaon_p=math.sqrt(min_kaon_ke**2 + 2 * min_kaon_ke * KAON_MASS)

                    # assert type(min_kaon_ke)==float
                    if valid_decays in [[211]] and child_list_pi[0][1]>200 and child_list_pi[0][1]<210 and kaon_list[1]>min_kaon_p: #is_contained(np.array(child_list[0][-1][-2]),margin=0) and is_contained(np.array(child_list[0][-1][-1]),margin=0)
                        TrueKaonspi=1

                    if valid_decays in [[-13]] and child_list_mu[0][1]>230 and child_list_mu[0][1]<240 and kaon_list[1]>min_kaon_p: #and is_contained(np.array(child_list[0][-1][-2]),margin=0) and is_contained(np.array(child_list[0][-1][-1]),margin=0)
                        TrueKaonsmu=1
                TrueLambdas=0

                if len(primpdgsL):
                    assert len(primpdgsL)==1
                    #print("found lambda",primpdgs)
                    testlam=i[1][primpdgsL[0]]
                    ppi=[z[-1][0] for z in testlam if z[0] in [-211, 2212]]

                    momenta=[z[1] for z in testlam if z[0] in [-211, 2212]]
                    pdgs=[z[0] for z in testlam if z[0] in [-211, 2212]]
                    if (len(ppi)==2 and 
                        norm3d(np.array(i[0][:3])-np.array(ppi[0]))>min_len):#min decay len

                        assert norm3d(np.array(ppi[0])-np.array(ppi[1]))<.0001

                        pion_mom=momenta[np.argwhere(np.array(pdgs)==-211)[0][0]]
                        proton_mom=momenta[np.argwhere(np.array(pdgs)==2212)[0][0]]

                        theta,cos_theta,lam_mom=decay_angle_from_invariant_mass(pion_mom, proton_mom, PION_MASS, PROT_MASS, LAM_MASS)


                        if (MIN_LAM_MOM<lam_mom and 
                            lam_mom<MAX_LAM_MOM and 
                            theta>np.pi/12 and 
                            pion_mom>MIN_PI_MOM and 
                            proton_mom>MIN_PROT_MOM):


                            cfg = '''
                                IOManager: {
                                    Verbosity    : 4
                                    Name         : "MainIO"
                                    IOMode       : 0
                                    InputFiles   : [%s]
                                }
                                '''% (find_root_path(file0))

                            io = load_cfg(cfg)

                            io.read_entry(I[0])
                            truthinfo = io.get_data('particle', 'mpv').as_vector()
                            nice_primaries=set([x.pdg_code() for x in truthinfo if (x.pdg_code() not in [2112,12,-12,14,-14,3122,22,111,311,-311]) and i[0][:3]==(x.position().x(),x.position().y(),x.position().z())])
                            io.finalize()

                            if len(nice_primaries)==0:
                                print("skipped one")
                                # continue

                            elif len(set([13,-13,321,-321,2212,211,-211,11,-11]).intersection(nice_primaries))==0:
                                raise Exception(nice_primaries)
                            else:

                                TrueLambdas=1



                

                if len(primpdgsK) and len(primpdgsL) and TrueKaonspi+TrueKaonsmu and TrueLambdas:

                    print(f"found true K+Lam {CCNCmap[i[-4]]} event")
                    testlam=i[1][primpdgsL[0]]
                    ppi=[z for z in testlam if z[0] in [-211, 2212]]
                    if len(ppi)==2:
                        # print("found valid lambda in truth")
                        assert i[-4] in [0,1]
                        # print(f"this is an )
                        larcv_keys_assoc[i[-4]]+=[I]
                if len(primpdgsK) and TrueKaonspi+TrueKaonsmu:
                    print(f"found true K+ {CCNCmap[i[-4]]} event")
                    testK=i[1][primpdgsK[idx]]
                    # ppi=[z for z in testK if z[0] in [-211, 2212]]
                    # if len(ppi)==2:
                        # print("found valid lambda in truth")
                    assert i[-4] in [0,1]
                    # print(f"this is an )
                    larcv_keys_K[i[-4]]+=[I]
                if len(primpdgsL) and TrueLambdas:

                    print(f"found true Lam {CCNCmap[i[-4]]} event")
                    testlam=i[1][primpdgsL[0]]
                    ppi=[z for z in testlam if z[0] in [-211, 2212]]
                    if len(ppi)==2:
                        # print("found valid lambda in truth")
                        assert i[-4] in [0,1]
                        # print(f"this is an )
                        larcv_keys_L[i[-4]]+=[I]


        # print("running",lfile)
        if filecount==MAXFILES:break
        if filecount==0:clear_html_files(LOCAL_EVENT_DISPLAYS)
        if filecount%200==0:
            print("filecount ",filecount)
            print(res_assoc,"assoc")
            print(res_K,"K")
            print(res_L,"L")
        filecount+=1
        from_file = np.load(file, allow_pickle=True)
        predk: list[PredKaonMuMich] = from_file['PREDKAON']
        predl: list[Pred_Neut] = from_file['PREDLAMBDA']

        n=kaon_pass_order["Valid MIP Len"]
        x1 = n % 100
        x2 = (n // 100) % 100
        x3 = (n // 100**2) % 100
        x4 = (n // 100**3) % 100
        for k in predk:
                # if k.reason=="":
                #     # if (entry,truth_interaction_id(k)) not in true_Kp_keys:
                #     true_Kp_keys+=[(entry,truth_interaction_id(k))]
                
                primary_valid_len_mips=len([i for i in k.particles if i.is_primary and i.pdg_code==MUON_PID and 
                                            ((28<i.reco_length and i.reco_length<32) or (50<i.reco_length and i.reco_length<58))])
                pc=k.pass_cuts((kaon_pass_order))
                pc*=is_contained(k.reco_vertex,margin=margin0)


                if pc:
                    num_mip=k.primary_particle_counts[MUON_PID]-primary_valid_len_mips
                    
                    found_Kp_keys[int(num_mip<=0)]+=[(k.event_number,k.truth_interaction_id)]
                    
                    # print(k.hip.id,k.interaction.id)
                
        # for entry in predl:
        for l in predl:
                primary_valid_len_mips=len([i for i in l.particles if i.is_primary and i.pdg_code==MUON_PID and 
                                            ((28<i.reco_length and i.reco_length<32) or (50<i.reco_length and i.reco_length<58))])
                pc=l.pass_cuts((lam_pass_order))
                pc*=is_contained(l.reco_vertex,margin=margin0)

                if pc:
                    num_mip=l.primary_particle_counts[MUON_PID]-primary_valid_len_mips
                    found_lam_keys[int(num_mip==0)]+=[(l.event_number,l.truth_interaction_id)]


        def clear(a,b,c,lar):
            assert lar in larcv_keys_assoc[a]
            # assert lar in found_lam_keys
            while lar in larcv_keys_assoc[a]:
                larcv_keys_assoc[a].remove(lar)
            while lar in larcv_keys_K[a]:
                larcv_keys_K[a].remove(lar)
            while lar in larcv_keys_L[a]:
                larcv_keys_L[a].remove(lar)



            while lar in found_Kp_keys[b]:
                found_Kp_keys[b].remove(lar)
            while lar in found_lam_keys[c]:
                found_lam_keys[c].remove(lar)
            

        def clearK(a,b,lar):
            assert lar in larcv_keys_K[a]
            while lar in larcv_keys_K[a]:
                larcv_keys_K[a].remove(lar)
            while lar in found_Kp_keys[b]:
                found_Kp_keys[b].remove(lar)
            

        def clearL(a,b,lar):
            assert lar in larcv_keys_L[a]
            while lar in larcv_keys_L[a]:
                larcv_keys_L[a].remove(lar)
            while lar in found_lam_keys[b]:
                found_lam_keys[b].remove(lar)
            
            


        
        assert len(set(found_Kp_keys[0]).intersection(set(found_Kp_keys[1])))==0
        assert len(set(found_lam_keys[0]).intersection(set(found_lam_keys[1])))==0
        assert len(set(larcv_keys_assoc[0]).intersection(set(larcv_keys_assoc[1])))==0
        assert len(set(larcv_keys_L[0]).intersection(set(larcv_keys_L[1])))==0
        assert len(set(larcv_keys_K[0]).intersection(set(larcv_keys_K[1])))==0



        for CCNC in [0,1]:
            
            for lar in list(larcv_keys_assoc[CCNC]).copy():
                if lar in found_Kp_keys[CCNC] and lar in found_lam_keys[CCNC]:
                    res_assoc[CCNC][CCNC]+=1
                    res_K[CCNC][CCNC]+=1
                    res_L[CCNC][CCNC]+=1

                    if found_Kp_keys[CCNC].count(lar)>1 or found_lam_keys[CCNC].count(lar)>1:
                        print("you messed up stupid, but its still a good one",found_Kp_keys[CCNC].count(lar),found_lam_keys[CCNC].count(lar))
                        save_my_html(LOCAL_EVENT_DISPLAYS+'overcount',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])

                    
                    print(f"found a good {CCNCmap[CCNC]}")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_found',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clear(CCNC,CCNC,CCNC,lar)

                elif lar in found_Kp_keys[1-CCNC] and lar in found_lam_keys[1-CCNC]:
                    print("CCNC mistake",CCNCmap[CCNC]+'_to_'+CCNCmap[1-CCNC])
                    res_assoc[CCNC][1-CCNC]+=1
                    res_K[CCNC][1-CCNC]+=1
                    res_L[CCNC][1-CCNC]+=1
                    save_my_html(LOCAL_EVENT_DISPLAYS+'CCvsNC_'+CCNCmap[CCNC]+'_to_'+CCNCmap[1-CCNC],create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clear(CCNC,1-CCNC,1-CCNC,lar)

                elif lar in found_Kp_keys[CCNC] and lar not in found_lam_keys[CCNC]:
                    print("missed a lam")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_missed_L',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    res_assoc[CCNC][3]+=1
                    res_K[CCNC][CCNC]+=1
                    res_L[CCNC][2]+=1
                    clear(CCNC,CCNC,CCNC,lar)

                elif lar not in found_Kp_keys[CCNC] and lar in found_lam_keys[CCNC]:
                    print("missed a kp")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_missed_K',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    res_assoc[CCNC][2]+=1
                    res_K[CCNC][2]+=1
                    res_L[CCNC][CCNC]+=1
                    clear(CCNC,CCNC,CCNC,lar)

                elif lar not in found_Kp_keys[CCNC] and lar not in found_lam_keys[CCNC]:
                    print("missed both")
                    res_assoc[CCNC][4]+=1
                    res_K[CCNC][2]+=1
                    res_L[CCNC][2]+=1
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_missed_both',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clear(CCNC,CCNC,CCNC,lar)

            for lar in list(larcv_keys_K[CCNC]).copy():
                if lar in found_Kp_keys[CCNC]:
                    res_K[CCNC][CCNC]+=1
                    print(f"found a good {CCNCmap[CCNC]} K+")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_found_Kp',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clearK(CCNC,CCNC,lar)
                elif lar in found_Kp_keys[1-CCNC]:
                    print("CCNC mistake",CCNCmap[CCNC]+'_to_'+CCNCmap[1-CCNC]+ "Kp")
                    res_K[CCNC][1-CCNC]+=1
                    save_my_html(LOCAL_EVENT_DISPLAYS+'CCvsNC_'+CCNCmap[CCNC]+'_to_'+CCNCmap[1-CCNC]+"_Kp",create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clearK(CCNC,1-CCNC,lar)
                elif lar not in found_Kp_keys[CCNC]:
                    print("missed a kp")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_missed_K_Kp',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    # res_assoc[CCNC][2]+=1
                    res_K[CCNC][2]+=1
                    # res_L[CCNC][CCNC]+=1
                    clearK(CCNC,CCNC,lar)

            for lar in list(larcv_keys_L[CCNC]).copy():
                if lar in found_lam_keys[CCNC]:
                    res_L[CCNC][CCNC]+=1
                    print(f"found a good {CCNCmap[CCNC]} lam")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_found_lam',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clearL(CCNC,CCNC,lar)
                elif lar in found_lam_keys[1-CCNC]:
                    print("CCNC mistake",CCNCmap[CCNC]+'_to_'+CCNCmap[1-CCNC]+ "lam")
                    res_L[CCNC][1-CCNC]+=1
                    save_my_html(LOCAL_EVENT_DISPLAYS+'CCvsNC_'+CCNCmap[CCNC]+'_to_'+CCNCmap[1-CCNC]+"_lam",create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    clearL(CCNC,1-CCNC,lar)
                elif lar not in found_lam_keys[CCNC]:
                    print("missed a lam")
                    save_my_html(LOCAL_EVENT_DISPLAYS+'true_missed_L_lam',create_html_filename(lar[0], file,extra=str(lar[1])),lar[0])
                    # res_assoc[CCNC][2]+=1
                    res_L[CCNC][2]+=1
                    # res_L[CCNC][CCNC]+=1
                    clearL(CCNC,CCNC,lar)

        remaining_keys=[[],[]]
        for CCNC in [0,1]:
            assert len(larcv_keys_assoc[CCNC])==0,(larcv_keys_assoc,found_Kp_keys,found_lam_keys)
            assert len(larcv_keys_L[CCNC])==0,(larcv_keys_L,found_Kp_keys,found_lam_keys)
            assert len(larcv_keys_K[CCNC])==0,(larcv_keys_K,found_Kp_keys,found_lam_keys)

            remaining_keys[CCNC]=list(set(found_Kp_keys[CCNC]).union(set(found_lam_keys[CCNC])))
            if len(remaining_keys[CCNC]):
                print(f"you found some extra {CCNCmap[CCNC]}",remaining_keys[CCNC])
                for rk in remaining_keys[CCNC]:
                    if rk in found_Kp_keys[CCNC] and rk in found_lam_keys[CCNC]:
                        print("recording that now")
                        # print("found a false one")#,[[pms[i][0].reco_length,is_contained(pms[i][0].points,margin=0),pms[i][1],pms[i][0].pdg_code] for i in pms])
                        save_my_html(LOCAL_EVENT_DISPLAYS+'false_assoc_found',create_html_filename(rk[0], file,extra=str(rk[1])),rk[0])
                        res_assoc[2][CCNC]+=1
                        res_K[2][CCNC]+=1
                        res_L[2][CCNC]+=1
                    elif rk in found_Kp_keys[CCNC]:
                        print("recording that now")
                        # print("found a false one")#,[[pms[i][0].reco_length,is_contained(pms[i][0].points,margin=0),pms[i][1],pms[i][0].pdg_code] for i in pms])
                        save_my_html(LOCAL_EVENT_DISPLAYS+'false_Kp_found',create_html_filename(rk[0], file,extra=str(rk[1])),rk[0])
                        # res_assoc[2][CCNC]+=1
                        res_K[2][CCNC]+=1
                    elif rk in found_lam_keys[CCNC]:
                        print("recording that now")
                        # print("found a false one")#,[[pms[i][0].reco_length,is_contained(pms[i][0].points,margin=0),pms[i][1],pms[i][0].pdg_code] for i in pms])
                        save_my_html(LOCAL_EVENT_DISPLAYS+'false_lam_found',create_html_filename(rk[0], file,extra=str(rk[1])),rk[0])
                        # res_assoc[2][CCNC]+=1
                        res_L[2][CCNC]+=1
                    else:
                        raise Exception("How?")

    np.savez_compressed("assoc_out", ASSOC=res_assoc,LAM=res_L,KP=res_K)
