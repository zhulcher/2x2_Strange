DIR=/sdf/group/neutrino/zhulcher/

# WORKDIR=/sdf/data/neutrino/dcarber/larcv_files/NuMI_nu_cosmics/v09_89_01_01p01/
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_files/



# WORKDIR=/sdf/data/neutrino/dcarber/NuMI_nu_spine/v09_89_01p03


# WORKDIR=/sdf/data/neutrino/icarus/spine/prod/numi_nu_corsika_mix/file_list.txt
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files_reco/


# SRC_DIR="/sdf/data/neutrino/icarus/spine/prod/numi_nu_corsika_mix/output_spine"

# WORKDIR=/sdf/data/neutrino/zhulcher/grappa_inter_update_250/file_list.txt
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_250_files_reco/
# SRC_DIR="/sdf/data/neutrino/zhulcher/grappa_inter_update_250/output_spine"

WORKDIR=/sdf/data/neutrino/zhulcher/grappa_inter_very_large_250/full_file_list_clean.txt
OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_very_large_250_files_reco/
SRC_DIR="/sdf/data/neutrino/zhulcher/grappa_inter_very_large_250/output_spine"



rm -f error/*
rm -f output/*


# directory="/sdf/data/neutrino/dcarber/NuMI_nu_spine/v09_89_01p03"

if false; then

    
    DEST_DIR="$OUTDIR"

    for file in "$SRC_DIR"/*.h5; do
        filename=$(basename "$file")                            # e.g., foo_spine.h5
        base="${filename%.h5}"                                   # remove .h5
        base="${base//_spine/}"                                   # remove _spine
        target_dir="$DEST_DIR/$base"
        target_link="$target_dir/analysis_reco.h5"

        if [ ! -e "$target_link" ]; then
            mkdir -p "$target_dir"
            ln -s "$file" "$target_link"
            echo "Created symlink: $target_link -> $file"
        else
            echo "Skipping existing: $target_link"
        fi

        # if [ ! -f "$target_dir/analysis_HM_reco.h5" ]; then
        #     echo "Error: $target_dir/analysis_HM_reco.h5 does not exist." >&2
        #     #   exit 1
        # fi
    done

fi




# done




# Loop over each file matching the pattern
count=0
for file in "$WORKDIR"*; 
do
    
    echo "Reading file: $file"
    
    
    # Loop over each line inside the file
    while IFS= read -r line; 
    do
        ((count++))   
        if (( count < 30500 )); then
            continue
        fi
        # echo "Processing: $line"
        # Do something with $line, such as copying, analyzing, etc.
    

    # find "$WORKDIR" -type f -name "*.root" | while read -r file; do
        # Get the basename of each .root file
        basename_file=$(basename "$line" .root)
        # echo "help"
        # echo "$basename_file"

        mkdir -p $OUTDIR

        i=$OUTDIR/$basename_file


        file="${OUTDIR/_files/_analysis}"/npyfiles/$basename_file.npz

        if [[ ! -f "$file" ]]; then
            # rm $i/*
            echo "$file does not exist : $count"
            # if (( count% 3 == 0 )); then
            #     sbatch --partition=milano submit_job_larcvHM_reco.sbatch $DIR $i #$line
            # elif (( count% 3 == 1 )); then
            #     sbatch --partition=roma submit_job_larcvHM_reco.sbatch $DIR $i #$line
            # else
            #     sbatch --partition=ampere submit_job_larcvHM_reco.sbatch $DIR $i #$line
            # fi
            # -----------------------------------------
            # Determine least-loaded partition
            # -----------------------------------------
            #ampere milano roma turing
            partitions=(milano)
            declare -A load

            for p in "${partitions[@]}"; do
                load[$p]=$(squeue -u zhulcher -h --partition=$p -t pending,running -r | wc -l)
            done

            # Find the partition with minimum load
            best_partition="${partitions[0]}"
            best_value="${load[$best_partition]}"

            for p in "${partitions[@]}"; do
                if (( load[$p] < best_value )); then
                    best_partition="$p"
                    best_value="${load[$p]}"
                fi
            done

            echo "Submitting to least-loaded partition: $best_partition  (jobs = $best_value)"
            sbatch --partition="$best_partition" submit_job_larcvHM_reco.sbatch "$DIR" "$i"
            
        else 
            echo "$file found : $count"
            # sbatch submit_job_larcvHM_reco.sbatch $DIR $i #$line
        # continue
        fi
        
        # break

        mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`
        until (( $mine < 1000 ));
        do
            echo "still too many jobs there......"
            sleep 15
            mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`

        done

        # if [ -f "$i/analysis_HM_both.h5" ]; then
        #     echo "$i/analysis_HM_both.h5 exists."
        # else
        # echo "submit_job_larcvHM.sbatch $DIR $i $line"


        # file="$i.npy"



        

        



        
# # python3 -c "
# # import numpy as np
# # import sys

# # try:
# #     np.load('$file',allow_pickle=True)
# #     print('$file is valid')
# # except Exception as e:
# #     print(f'Invalid .npy file: {e}')
# #     sys.exit(1)
# # " || sbatch submit_job_larcvHM.sbatch $DIR $i $line
# #         fi

        
        
        
    done < "$file"
done

# # for i in "${arr[@]}"
# # do
# # #    echo "$i"
# # #    mkdir -p $i
# # #    if [ ! -e "$i/output_0_0000-edepsim.root" ]; then
# # #       cp zach/stage0/$i/output_0_0000-edepsim.root $i
# # #    fi
# # #    if [ ! -e "$i/output_0_0000-larcv.root" ]; then
# # #       cp zach/stage0/$i/output_0_0000-larcv.root $i
# # #    fi
# # #    python3 utils/add_HIPMIP.py $i/output_0_0000-larcv.root $i/output_0_0000-larcv_HM.root
# # done



# #   python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/test_io_both.cfg -s $i/output_0_0000-larcv.root -o $i/output_0_0000-analysis_both.h5
# #   python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/hdf5writer.cfg -s $i/output_0_0000-larcv_HM.root -o $i/output_0_0000-analysis_HM_both.h5

# # file_names=$(find "$WORKDIR" -type f -exec basename {} \;)

# # # Check for duplicates using sort and uniq
# # duplicates=$(echo "$file_names" | sort | uniq -d)

# # if [ -z "$duplicates" ]; then
# #     echo "All file names are unique."
# # else
# #     echo "Duplicate file names found:"
# #     echo "$duplicates"
# # fi