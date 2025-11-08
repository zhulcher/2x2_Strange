from statistics_plot_base import *
from collections import defaultdict


margin0=[[15,15],[15,15],[10,60]]


latextoplot={
        130:r'$K^0_L$',
        310:r'$K^0_S$',
        311:r'$K^0$',
        -311:r'$\bar{K^0}$',
        321:r'$K^+$',
        -321:r'$K^-$',
        3122:r'$\Lambda^0$',
        3222:r'$\Sigma^+$',
        3212:r'$\Sigma^0$',
        3112:r'$\Sigma^-$',
        3322:r'$\Xi^0$',
        3312:r'$\Xi^-$',
        3334:r'$\Omega^-$',
        221:r'$\eta$',
        331:r'$\eta^{\prime}$',
        # 431:r'$D^+_s$',
        333:r'$\phi(1020)$',
        # 433:r'$D_s^{*+}$',
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
        # -411:r'$D^-$',
        # 421:r'$D^0$',
        # -421:r'$\bar{D^0}$',
        # 2112:r'$n$',
}



MAXFILES=np.inf


files = os.listdir(directory2)
FILES = [os.path.join(directory2, f) for f in files if os.path.isfile(os.path.join(directory2, f))]
if len(FILES)>MAXFILES:
    FILES=FILES[:MAXFILES]

SAVEDIR="plots/" + str(FOLDER) + "/larcv_truth/"






# # print(LAM_PT_MAX,"Maximum allowable pT")
# # dt=[]
# # ds=[]
# # MAXFILES=np.inf
# print("only including", MAXFILES, "files")
# # filecount=0
# # for kfile in KAONFILE:
# #     # continue
# #     # print("running",kfile)
# #     if filecount==0:clear_html_files(event_display_new_path+"/Kp")
# #     if filecount%500==0:print("filecount ",filecount)
# #     if filecount==MAXFILES:break
# #     filecount+=1

# #     kaons = np.load(kfile, allow_pickle=True)
# #     predk: dict[int, list[PredKaonMuMich]] = kaons[0]

# #     for key in predk:
# #         for mu in predk[key]:
# #             if mu.error!="":
# #                 print("VERY BAD ERROR",key,mu.error)
# #                 copy_and_rename_file(kfile,event_display_new_path+'/Kp/error',create_html_filename(key, kfile,extra=str(mu.mip_id)),key)
# #             # print("event",key)

# #             pc=mu.pass_cuts((testcuts))


# #             muon_len[mu.truth][mu.pass_failure==""] += [mu.mip_len_base]
# #             HM_acc_mu[mu.truth][mu.pass_failure==""] += [mu.mu_hm_acc]
# #             # print("kaon event",key)
# #             # if mu.true_signal:
# #             #     # print("found a true_signal mu")
# #             #     muon_len[2]+=[mu.mip_len_base]
            
            
# #             if pc:
# #                 selectedkaons+=1
# #                 if not mu.truth:
# #                     # assert os.path.exists(kfile)
# #                     copy_and_rename_file(kfile,event_display_new_path+'/Kp/false_found/'+mu.reason,create_html_filename(key, kfile,extra=str(mu.mip_id)),key)
# #                     print("GOT A FALSE KAON",mu.mip_id,mu.mip.pdg_code,mu.mip.orig_id,[(i.hip_id,i.truth,i.hip.pdg_code,i.proj_dist_from_hip<testcuts["par_child_dist max"][0],len([(p.dist_to_parent,p.angle_to_parent,p.proj_dist_to_parent) for p in i.k_extra_children if p.dist_to_parent<testcuts["par_child_dist max"][0] and (p.child_hm_pred in [SHOWR_HM,MICHL_HM] or p.child.reco_length>5)])==0, min(np.linalg.norm(i.hip.end_point-mu.mip.start_point),np.linalg.norm(i.hip.start_point-mu.mip.start_point))<testcuts["par_child_dist max"][0] and np.linalg.norm(i.hip.end_point-mu.mip.start_point)<np.linalg.norm(i.hip.start_point-mu.mip.start_point),i.truth_list) for i in mu.potential_kaons],key,kfile,[mu.reason,mu.pass_failure])
# #             if not mu.truth:
# #                 kaon_pass_failure[1][mu.pass_failure]+=1
# #                 kaon_reason_map[1][mu.reason]+=1
            
# #             if mu.truth:
# #                 kaon_pass_failure[0][mu.pass_failure]+=1
# #                 kaon_reason_map[0][mu.reason]+=1
# #                 TrueKaons+=1
# #                 if pc:
# #                     copy_and_rename_file(kfile,event_display_new_path+'/Kp/true_found',create_html_filename(key, kfile,extra=str(mu.mip_id)),key)
# #                     correctlyselectedkaons+=1
# #                 else:
# #                     print("MISSED A GOOD KAON",key,kfile,mu.mip.orig_id)
# #                     copy_and_rename_file(kfile,event_display_new_path+'/Kp/true_missed/'+mu.pass_failure,create_html_filename(key, kfile,extra=str(mu.mip_id)),key)
# #                 print(
# #                     "true mu from kaon:",
# #                     key,
# #                     "    ",
# #                     "mu hm:",mu.mu_hm_acc,
# #                     "mu len:",mu.mip_len_base,
# #                 )
# #                 if mu.mip_len_base<50: print("TRUTH WARNING KAON TOO SHORT",key)
# #                 # muon_len[1] += [mu.mip_len_base]
# #                 # HM_acc_mu[1] += [mu.mu_hm_acc]
# #                 # muon_len_adjusted[0]+=[mu.mip_len_base]
# #                 # muon_len_adjusted[1]+=[mu.mip_len_base]
# #                 # for mk in mu.potential_kaons:
# #                 #     if mk.truth:
# #                 #         muon_len_adjusted[1][-1]+=mk.dist_from_hip
# #                 # for mm in mu.potential_michels:
# #                 #     if mm.truth:
# #                 #         muon_len_adjusted[1][-1] += mm.dist_to_mich

# #             for k in mu.potential_kaons:
# #                 dir_acos_K[mu.truth and k.truth][mu.pass_failure==""] += [k.dir_acos]
# #                 kaon_len[mu.truth and k.truth][mu.pass_failure==""] += [k.hip_len]
# #                 HM_acc_K[mu.truth and k.truth][mu.pass_failure==""] += [k.k_hm_acc]
# #                 dist_to_hip[mu.truth and k.truth][mu.pass_failure==""] += [k.dist_from_hip]

# #                 if k.truth:
# #                     print(
# #                         "with accompanying kaon:",
# #                         key,
# #                         "    ",
# #                         # k.hip_id,
# #                         "k diracos:",k.dir_acos,
# #                         "k hm:",k.k_hm_acc,
# #                         "k extra dist:",k.dist_from_hip,
# #                         "k len:",k.hip_len,
# #                         # k.k_extra_children,
# #                     )
# #                     # dir_acos_K[1] += [k.dir_acos]
# #                     # kaon_len[1] += [k.hip_len]
# #                     # HM_acc_K[1] += [k.k_hm_acc]
# #                     # dist_to_hip[1] += [k.dist_from_hip]

# #                     # for kc in k.k_extra_children:
# #                     #     k_extra_children[0][0] += [kc.dist_to_parent]
# #                     #     k_extra_children[0][1] += [kc.angle_to_parent]
# #                     #     if kc.truth:
# #                     #         k_extra_children[1][0] += [kc.dist_to_parent]
# #                     #         k_extra_children[1][1] += [kc.angle_to_parent]

# #             for mich in mu.potential_michels:

# #                 HM_acc_mich[mu.truth and mich.truth][mu.pass_failure==""]+=[mich.mich_hm_acc]
# #                 dist_to_mich[mu.truth and mich.truth][mu.pass_failure==""]+=[mich.dist_to_mich]

# #                 # mu_extra_children[0][0]+=[mich.mu_extra_children[0]]
# #                 # mu_extra_children[0][1]+=[mich.mu_extra_children[1]]

# #                 if mich.truth:
# #                     print(
# #                         "with accompanying michel:",
# #                         key,
# #                         "    ",
# #                         # mich.mich_id,
# #                         "mich extra dist:",mich.dist_to_mich,
# #                         # mich.mu_extra_children,
# #                         # decay_t_to_dist: float
# #                         "mich hm:",mich.mich_hm_acc,
# #                     )

# #                     # HM_acc_mich[1]+=[mich.mich_hm_acc]
# #                     # dist_to_mich[1]+=[mich.dist_to_mich]
# #                     # for mc in mich.mu_extra_children:
# #                     #     mu_extra_children[0][0] += [mc.dist_to_parent]
# #                     #     mu_extra_children[0][1] += [mc.angle_to_parent]
# #                     #     if mc.truth:
# #                     #         mu_extra_children[1][0] += [mc.dist_to_parent]
# #                     #         mu_extra_children[1][1] += [mc.angle_to_parent]
# #         # print("")

# num_nu=0
# filecount=0

indiv_total=defaultdict(list)
# lam_total=[]
# k_total=[]
# k0s_total=[]

indiv_fiducial=defaultdict(list)
# lam_fiducial=[]
# k_fiducial=[]
# k0s_fiducial=[]

other=defaultdict(int)

# k_energy=[]

# indiv_=defaultdict(list)
# lam_primary=[]
# k_primary=[]

indiv_decay=defaultdict(list)
# lam_decay=[]
# k_decay=[]
# k0s_decay=[]

K0S_MASS=497.611

from collections import Counter

# lam_cont=[]
# k_cont=[]

neuty=[]

pdg_dict=defaultdict(list)

filecount=0
int_type=[]
neutE=[]

neutE_K=[]

neutE_L=[]

neutE_assoc=[]

neutE_assoc_CC=[]

neutE_assoc_NC=[]



int_len=0

lam_kp=[0,0,0]
# sig_kp=[0,0,0]
#[[x.p(),pos,[],x.creation_process(),True]]

total=defaultdict(list)
fiducial=defaultdict(list)
lam_valid=defaultdict(list)
K_valid=defaultdict(list)

contained=[]
uncontained=[]

# other_lam=0

# other_kp=0



overall=[[r'primary $K^+$',[321],[[-13],[211],[-13,-11],[211,-11]],KAON_MASS],[r'primary $\Lambda$',[3122],[[-211,2212]],LAM_MASS],[r'primary $K^0$',[310,311],[[211,-211]],K0S_MASS]]


for lfile in FILES:
    # print("running",lfile)
    if filecount==MAXFILES:break
    # if filecount==0:clear_html_files(event_display_new_path+"/lambda")
    if filecount%200==0:print("filecount ",filecount)
    filecount+=1
    particles = np.load(lfile, allow_pickle=True)
    # lambdas=particles[0][3122]
    # kaons=particles[0][321]
    # k0s=particles[0][310]+particles[0][311]
    pdgs=particles[0]
    neutE+=particles[1][0]
    neuty+=[i[1] for i in particles[1][1]]
    interactions=particles[2]
    for pdg in pdgs:
        pdg_dict[pdg]+=pdgs[pdg]
    int_len+=len(interactions)

    for I in interactions:
        i=interactions[I]

        prim_pgs=[z[1] for z in i[1]]
        # for prim in i[1]:

        # myl=[np.isclose(np.linalg.norm(np.array(n)-np.array(I)),0) for n in particles[2][1]]
        # assert I in particles[2][1],(I,particles[2][1])
        # assert np.sum(myl)==1
        # spot=np.argwhere(myl)[0][0]
        # print(spot,particles[2][0])

        # neutE_init=particles[2][0][spot]

        # raise Exception(i)
        # entry=-100
        # for n in range(len(particles[1][1])):
        #     # print(np.array(particles[2][1][n])-np.array(I))
        #     if np.isclose(np.linalg.norm(np.array(particles[1][1][n])-np.array(i[0])),0,rtol=1e-02, atol=1e-02):
        #         entry=n
        # if entry==-100:
        #     raise Exception(particles[1][1],I,i)
        
        # Einit=particles[1][0][entry]

        nuE=i[-1]

        counts=Counter(prim_pgs)

        if counts[3122]>1: raise Exception(prim_pgs)
        if counts[321]>1: raise Exception(prim_pgs)
        if 3122 in prim_pgs:
            neutE_L+=[nuE]
        if 321 in prim_pgs:
            neutE_K+=[nuE]
        if 3122 in prim_pgs and 321 in prim_pgs:
            neutE_assoc+=[nuE]

            if i[2]==0:
                neutE_assoc_CC+=[nuE]
            elif i[2]==1:
                neutE_assoc_NC+=[nuE]
            else:
                raise Exception(i[2],i)
            
            # raise Exception(interactions)
            # raise Exception(i)
            # for s in [[1,NC_total,NC_fiducial,NC_lam_valid,NC_K_valid],[0,CC_total,CC_fiducial,CC_lam_valid,CC_K_valid]]

            CCNC=i[2]
            if CCNC in [0,1]:
                lam_kp[CCNC]+=1
                total[CCNC]+=[nuE]
                if is_contained(i[0][:3],margin=margin0):
                    fiducial[CCNC]+=[nuE]
                    # print(i[0][321])
                    # print(i[0][3122])
                    lam_key=[z for z in i[1] if z[1]==3122]
                    assert len(lam_key)==1
                    lam_key=lam_key[0]
                    # print(lam_key,i[1][lam_key])
                    l=set([z[0] for z in i[1][lam_key]])-set([3122])

                    # l=i[0][3122]
                    if l==set([-211, 2212]):
                        lam_valid[CCNC]+=[nuE]
                        k_key=[z for z in i[1] if z[1]==321]
                        assert len(k_key)==1
                        k_key=k_key[0]
                        k=set([z[0] for z in i[1][k_key]])-set([321])
                        # k=i[0][321]
                        if set(k)==set([-13]) or set(k)==set([211]) or set(k)==set([-13,-11]) or set(k)==set([211,-11]):
                            K_valid[CCNC]+=[nuE]
                            print(i[1][k_key])
            else:
                lam_kp[2]+=1
                
            
    # for i in interactions.values():
    #     if 3212 in i and 321 in i:
    #         # raise Exception(interactions)
    #         if 123456789 in i:
    #             sig_kp[1]+=1
    #         elif -123456789 in i:
    #             sig_kp[0]+=1
    #         else:
    #             sig_kp[2]+=1

        
        
    # for l in lambdas:
        
        for prim in i[1]:
            for inter in overall:
                # print(prim,inter[1])
                if prim[1] in inter[1]:
                    # print("got here")
                # if prim[1]==3122:
                    l=i[1][prim]

                    # print(l)
                    # print(l)
                    pl=[z[1] for z in l if z[0] in inter[1]]
                    # assert len(pl)==1
                    p=pl[0]
                    KE=np.sqrt(p**2+inter[3]**2)-inter[3]
                    indiv_total[inter[0]]+=[KE]
                    # if is_contained(l[1][:3]):
                    # int_type+=[l[5]]

                    lset=set([z[0] for z in l])-set(inter[1])

                    if is_contained(i[0][:3],margin=0):

                        
                        
                        # if np.any([lset==set(z) for z in inter[2]]):


                        if set(inter[1])=={321}:
                            
                            # if Counter([z[0] for z in l])==Counter([-13,321]) or Counter([z[0] for z in l])==Counter([211,321]):
                            mip_start=[z[-1][0] for z in l if z[0] in [-13,211]]
                            mip_end=[z[-1][1] for z in l if z[0] in [-13,211]]
                            if len(mip_start)>0 and is_contained(mip_start[0],margin=1) and is_contained(mip_end[0],margin=1):
                                # print([l])
                                contained+=[i[0][:3]]
                            elif -11 not in lset:
                                print(mip_start,mip_end,[is_contained(z,margin=1) for z in mip_start],[is_contained(z,margin=1) for z in mip_end],l)
                                uncontained+=[i[0][:3]]
                            # if lset =={211,-11} or lset =={-13,-11} or lset=={211, -11, -211} or lset =={-11}:
                            #     contained+=[i[0][:3]]
                            # elif lset =={211} or lset =={-13} or lset=={211, -211} or lset==set([]):
                            #     uncontained+=[i[0][:3]]
                            # else:
                            #     raise Exception(lset)




                    if is_contained(i[0][:3],margin=margin0):

                        # is_contained(interactions[hip_candidate.interaction_id].reco_vertex,margin=margin0)
                        indiv_fiducial[inter[0]]+=[KE]
                        # print(l[2])
                        # print(k[2])
                        if np.any([lset==set(z) for z in inter[2]]):
                            indiv_decay[inter[0]]+=[KE]

                            # if inter[1][0]==321: print("found one", lset)


                            # if set(inter[1])=={321}:
                            #     if lset =={211,-11} or lset =={-13,-11} or lset=={211, -11, -211} or lset =={-11}:
                            #         contained+=[i[0][:3]]
                            #     elif lset =={211} or lset =={-13} or lset=={211, -211} or lset==set([]):
                            #         uncontained+=[i[0][:3]]
                            #     else:
                            #         raise Exception(lset)

                                    # raise Exception("found one",lset)
                            # if l[-1]:
                            #     lam_cont+=[KE]
                        else:
                            if inter[1][0]==321: print("other", lset)
                            if len(lset):print(inter[0],lset)
                            # if len(lset)==2:
                                # raise Exception(inter[0],lset,[set(z) for z in inter[2]])
                            other[inter[0]]+=1
            

            # if prim[1]==321:
            #     k=i[1][prim]
            #     # print(k)
            #     pl=[z[1] for z in k if z[0]==321]
            #     assert len(pl)==1
            #     p=pl[0]
            #     KE=np.sqrt(p**2+KAON_MASS**2)-KAON_MASS
            #     k_total+=[KE]
            #     if is_contained(k[0][:3],margin=margin0):
            #         k_fiducial+=[KE]
            #         # if KE>40:
            #         #     k_energy+=[KE]
                        
            #             # print(k[2])

            #             # if k[-1]:
            #             #     k_cont+=[KE]
            #             # print(k[2])
            #         # print(k[2])
            #         kset=set([z[0] for z in k])-set([321])
            #         if kset==set([-13]) or kset==set([211]):

            #             k_decay+=[KE]
            #         else:
            #             # if k[2] not in [[211, -211, 211],[-11]]:
            #             #     print("other kp", k[2])
            #             print("kp",kset)
            #             other_kp+=1
            #         # else:

            # if prim[1] in [310,311]:
            #     k=i[1][prim]
            #     # print(k)
            #     pl=[z[1] for z in k if z[0] in [310,311]]
            #     assert len(pl)==1
            #     p=pl[0]
            #     KE=np.sqrt(p**2+K0S_MASS**2)-K0S_MASS
            #     k0s_total+=[KE]
            #     if is_contained(i[0][:3],margin=margin0):
            #         k0s_fiducial+=[KE]
            #         # if KE>40:
            #             # k_energy+=[KE]
                        
            #             # print(k[2])

            #             # if k[-1]:
            #             #     k_cont+=[KE]
                    
            #         kset=set([z[0] for z in k])-set([310,311])
                    
            #         if kset==set([-211, 211]):
            #             k0s_decay+=[KE]
            #         else:
            #             print("k0s",kset)
    # print()#


    print("indiv","CCNC",[(len(total[c]),len(fiducial[c]),len(lam_valid[c]),len(K_valid[c])) for c in [0,1]],len(neutE),len(neutE_K),len(neutE_L),len(neutE_assoc))
    # print([(len(indiv_total[c[0]]),len(indiv_fiducial[c[0]]),len(indiv_decay[c[0]]),(np.quantile(indiv_total[c[0]],[.25,.5,.75,.9,.95,.99]) if len(indiv_total[c[0]]) else None)) for c in overall])
    # print()
        # print(l)

scaling=80000/len(neutE)
for k in indiv_total:
    print(k+" branching",np.divide(len(indiv_decay[k]),(other[k]+len(indiv_decay[k]))))
# print("kp branching", np.divide(len(k_decay),(other_kp+len(k_decay))))
# print("lam branching", np.divide(len(lam_decay),(other_lam+len(lam_decay))))

# num = 42  # example number
# np.save('num_nu.npy',len(neutE))
print("scaling",len(neutE),scaling)
cutoff=2000

for c in [[r'primary $K^+$','with single MIP decay: '],[r'primary $\Lambda$',r'with p$\pi^-$ decay: '],[r'primary $K^0$',r'with $\pi^+\pi^-$ decay: ']]:
# for c in [["kp",1],["",0]]:
#     bins=None
#     for stat in [[total,r'Assoc. $\Lambda K^+$: '],
#                  [fiducial,r'with fiducial vertex: '],
#                  [lam_valid,r'with p$\pi^-$ decay: '],
#                  [K_valid,r'with single MIP decay: ']]:
#         if bins is None: bins=30
#         (n, bins, patches)=plt.hist(stat[0][c[1]],label=stat[1]+f'{round(len(stat[0][c[1]])* scaling)}',bins=bins,weights=np.ones_like(stat[0][c[1]]) * scaling)
    tot=np.array(indiv_total[c[0]])
    fid=np.array(indiv_fiducial[c[0]])
    dec=np.array(indiv_decay[c[0]])



    if len(tot[tot<cutoff]): (n, bins, patches)=plt.hist(tot[tot<cutoff],label=c[0]+f': {round(len(tot)* scaling)}',bins=30,weights=np.ones_like(tot[tot<cutoff]) * scaling)
    if len(fid[fid<cutoff]): plt.hist(fid[fid<cutoff],label=f'with fiducial vertex: {round(len(fid)* scaling)}',bins=bins,weights=np.ones_like(fid[fid<cutoff]) * scaling)
    if len(dec[dec<cutoff]): plt.hist(dec[dec<cutoff],label=c[1]+f': {round(len(dec)* scaling)}',bins=bins,weights=np.ones_like(dec[dec<cutoff]) * scaling)

# plt.hist(lam_cont,label=f'with children containment: {round(len(lam_cont)}',bins=bins)

    plt.legend()
    #plt.yscale('log')
    plt.xlabel("KE [MeV]")
    plt.ylabel(r"Freq scaled to 800K $\nu$")
    plt.tight_layout();plt.savefig(SAVEDIR+c[0])
    plt.clf()


print(len(contained),len(uncontained))

for c in [["NC",1],["CC",0]]:
    bins=None
    for stat in [[total,r'Assoc. $\Lambda K^+$: '],
                 [fiducial,r'with fiducial vertex: '],
                 [lam_valid,r'with p$\pi^-$ decay: '],
                 [K_valid,r'with single MIP decay: ']]:
        if bins is None: bins=30
        (n, bins, patches)=plt.hist(stat[0][c[1]],label=stat[1]+f'{round(len(stat[0][c[1]])* scaling)}',bins=bins,weights=np.ones_like(stat[0][c[1]]) * scaling)
        # plt.hist(fiducial[c[1]],label= {round(len(c[2])* scaling)}',bins=bins,weights=np.ones_like(c[2]) * scaling)
        # plt.hist(lam_valid[c[1]],label=rf'with p$\pi^-$ decay: {round(len(c[3])* scaling)}',bins=bins,weights=np.ones_like(c[3]) * scaling)
        # plt.hist(K_valid[c[1]],label=f'with single MIP decay: {round(len(c[4])* scaling)}',bins=bins,weights=np.ones_like(c[4]) * scaling)

    # plt.hist(lam_cont,label=f'with children containment: {round(len(lam_cont)}',bins=bins)

    plt.legend()
    #plt.yscale('log')
    plt.xlabel("Neutrino E [GeV]")
    plt.ylabel(r"Freq scaled to 800K $\nu$")
    plt.tight_layout();plt.savefig(SAVEDIR+c[0])
    plt.clf()


neutE=np.array(neutE)
# neutE=neutE[neutE<10]
(n, bins, patches)=plt.hist(neutE,bins=50,weights=np.ones_like(neutE) * scaling,label=rf"All $\nu$: {round(len(neutE)*scaling)}",alpha=.5)
(n, bins, patches)=plt.hist(neutE_L,weights=np.ones_like(neutE_L) * scaling,alpha=.5,label=rf"$\nu\rightarrow\Lambda+X$: {round(len(neutE_L)*scaling)}",bins=bins)
(n, bins, patches)=plt.hist(neutE_K,weights=np.ones_like(neutE_K) * scaling,alpha=.5,label=rf"$\nu\rightarrow K^++X$: {round(len(neutE_K)*scaling)}",bins=bins)
(n, bins, patches)=plt.hist(neutE_assoc,weights=np.ones_like(neutE_assoc) * scaling,alpha=.5,label=rf"All $\nu\rightarrow K^+\Lambda+X$: {round(len(neutE_assoc)*scaling)}",bins=bins)
# (n, bins, patches)=plt.hist(neutE_assoc_CC,weights=np.ones_like(neutE_assoc_CC) * scaling,alpha=.5,label=r"CC $\nu\rightarrow K^+\Lambda+X$",bins=bins)
(n, bins, patches)=plt.hist(neutE_assoc_NC,weights=np.ones_like(neutE_assoc_NC) * scaling,label=rf"NC $\nu\rightarrow K^+\Lambda+X$: {round(len(neutE_assoc_NC)*scaling)}",bins=bins)
#plt.yscale('log')
plt.xlabel(r"$E_{\nu}$ [GeV]")
plt.ylabel("Freq")
plt.axvline(x = .790, color = 'b', label = '790 MeV/c (|dS|=1)')
plt.axvline(x = 1.250, color = 'r', label = '1250 MeV/c (|dS|=0)')
# plt.axvline(x = 1.500, color = 'r', label = '1500 MeV/c (|dS|=1)')
plt.legend()
plt.yscale("log")
plt.tight_layout();plt.savefig(SAVEDIR+"neutrinos")
plt.clf()

plt.hist(neuty,bins=100,weights=np.ones_like(neuty) * scaling)
#plt.yscale('log')
# plt.xlabel(r"$E_{\nu}$ [MeV]")
# plt.ylabel("Freq")
# plt.axvline(x = .790, color = 'r', label = '790 MeV/c (|dS|=1)')
# plt.axvline(x = 1.250, color = 'r', label = '1250 MeV/c (|dS|=0)')
# plt.axvline(x = 1.500, color = 'r', label = '1500 MeV/c (|dS|=1)')
# plt.legend()
plt.tight_layout();plt.savefig(SAVEDIR+"neut_y")
plt.clf()



# print([(i,len(pdg_dict[i])) for i in pdg_dict])

# plt.bar([latextoplot[i] for i in pdgstoplot],np.array(sorted(hist2plot))/maxnum/nfiles*scalefac)


lengths = {latextoplot[key]: len(value)* scaling for key, value in pdg_dict.items() if key in latextoplot}

# Sort keys by increasing list length
sorted_keys = sorted(lengths, key=lengths.get)
sorted_lengths = [lengths[key] for key in sorted_keys]


def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], str(round(y[i])), ha='center',
         fontsize=15,
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
# Plot the bar chart
# plt.figure(figsize=(8, 5))
plt.bar(sorted_keys, sorted_lengths, edgecolor='black')
    # plt.xticks(rotation = 45)
plt.yscale('log')
# plt.rcParams['text.usetex'] = True
plt.ylabel(r"Freq scaled to 800K $\nu$")
plt.xlabel("Particle")

add_labels(sorted_keys,sorted_lengths)
# plt.legend()
plt.grid(axis='y', which='both',linestyle='--', alpha=0.7)
plt.tight_layout();plt.savefig(SAVEDIR+"pdgs")
plt.clf()


print("LAM_KP",list(np.array(lam_kp)*.64*.84*scaling),"scaling",scaling,len(neutE),int_len)

xlo=61.94
xhi=358.49

ylo=-181.86
yhi=134.96

zlo=-894.9505
zhi=894.9505

conx=np.abs([x[0] for x in contained+uncontained])
cony=[x[1] for x in contained+uncontained]
conz=[x[2] for x in contained+uncontained]

print(min(conx),max(conx),min(cony),max(cony),min(conz),max(conz))

from itertools import product
from tqdm import tqdm

def optimize_containment_from_lists(contained_vertices, uncontained_vertices, 
                                    z_bounds, w_bounds, 
                                    z_steps=10, w_steps=10):
    """
    contained_vertices: Nx3 array
    uncontained_vertices: Mx3 array
    z_bounds: (zmin_min, zmin_max, zmax_min, zmax_max)
    w_bounds: (w_min, w_max)
    Returns: best (dzmin, dzmax, w) and the figure of merit value
    """

    # union of all vertices to get bounding box
    all_vertices = np.vstack([contained_vertices, uncontained_vertices])
    xlo, ylo, zlo = np.min(all_vertices, axis=0)
    xhi, yhi, zhi = np.max(all_vertices, axis=0)

    # define grids
    dzmin_grid = np.linspace(z_bounds[0], z_bounds[1], z_steps)
    dzmax_grid = np.linspace(z_bounds[2], z_bounds[3], z_steps)
    w_grid = np.linspace(w_bounds[0], w_bounds[1], w_steps)

    best_fom = -np.inf
    best_params = None

    # loop over all parameter combinations
    total = len(dzmin_grid) * len(dzmax_grid) * len(w_grid)

    for dzmin, dzmax, w in tqdm(product(dzmin_grid, dzmax_grid, w_grid), total=total, desc="Optimizing"):

        # masks for contained events inside fiducial
        contained_mask = (
            (contained_vertices[:,2] >= zlo+dzmin) & (contained_vertices[:,2] <= zhi-dzmax) &
            (abs(contained_vertices[:,0]) >= xlo + w) & (abs(contained_vertices[:,0]) <= xhi - w) &
            (contained_vertices[:,1] >= ylo + w) & (contained_vertices[:,1] <= yhi - w)
        )
        n_contained_in_fiducial = np.sum(contained_mask)

        # masks for all events inside fiducial
        all_mask = (
            (all_vertices[:,2] >= zlo+dzmin) & (all_vertices[:,2] <= zhi-dzmax) &
            (abs(all_vertices[:,0]) >= xlo + w) & (abs(all_vertices[:,0]) <= xhi - w) &
            (all_vertices[:,1] >= ylo + w) & (all_vertices[:,1] <= yhi - w)
        )
        n_fiducial = np.sum(all_mask)




        if n_fiducial > 0:
            fom = n_contained_in_fiducial**2/n_fiducial
            if fom > best_fom:
                best_fom = fom
                best_params = (dzmin, dzmax, w)

    return best_params, best_fom


z_bounds = (0, zhi/4, 0, zhi/4)
w_bounds = (0, (xhi - xlo)/4)

print("nominal cuts:",z_bounds,w_bounds)

params, fom = optimize_containment_from_lists(
    np.array(contained),
    np.array(uncontained),
    z_bounds=z_bounds,
    w_bounds=w_bounds,
    z_steps=int(z_bounds[1]),
    w_steps=int(w_bounds[1])
)
print("Best parameters:", params)
print("Best figure of merit:", fom)