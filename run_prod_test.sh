DIR=/sdf/group/neutrino/zhulcher/

# WORKDIR=/sdf/data/neutrino/dcarber/larcv_files/NuMI_nu_cosmics/v09_89_01_01p01/
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_files/



# WORKDIR=/sdf/data/neutrino/icarus/spine/prod/numi_nu_corsika_mix/file_list.txt
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_larcv_truth/

WORKDIR=/sdf/group/neutrino/zhulcher/2x2_Strange/prod_test/from_franky.txt
OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/prod_test_larcv_truth/


rm -f error/*
rm -f output/*


# directory="/sdf/data/neutrino/dcarber/NuMI_nu_spine/v09_89_01p03"

# Loop over each file matching the pattern
for file in "$WORKDIR"*; do
    echo "Reading file: $file"
    
    # Loop over each line inside the file
    while IFS= read -r line; do
        # echo "Processing: $line"
        # Do something with $line, such as copying, analyzing, etc.
    

    # find "$WORKDIR" -type f -name "*.root" | while read -r file; do
        # Get the basename of each .root file
        basename_file=$(basename "$line" .root)
        # echo "help"
        # echo "$basename_file"

        mkdir -p $OUTDIR

        i=$OUTDIR/$basename_file
        
        # break

        

        # if [ -f "$i/analysis_HM_both.h5" ]; then
        #     echo "$i/analysis_HM_both.h5 exists."
        # else
        # echo "submit_job_larcvHM.sbatch $DIR $i $line"


        # file="$i.npy"



        file=$OUTDIR/$basename_file.npy

        if [[ ! -f "$file" ]]; then
            # rm $i/*
            echo "$file does not exist"
            sbatch submit_job_larcvHM_larcv.sbatch $DIR $i $line
        else 
        echo "$file found"
        continue
        fi



        mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`
        until (( $mine < 900 ));
        do
            echo "still too many jobs there......"
            sleep 15
            mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`

        done


        
# python3 -c "
# import numpy as np
# import sys

# try:
#     np.load('$file',allow_pickle=True)
#     print('$file is valid')
# except Exception as e:
#     print(f'Invalid .npy file: {e}')
#     sys.exit(1)
# " || sbatch submit_job_larcvHM.sbatch $DIR $i $line
#         fi

        
        
        
    done < "$file"
done

# for i in "${arr[@]}"
# do
# #    echo "$i"
# #    mkdir -p $i
# #    if [ ! -e "$i/output_0_0000-edepsim.root" ]; then
# #       cp zach/stage0/$i/output_0_0000-edepsim.root $i
# #    fi
# #    if [ ! -e "$i/output_0_0000-larcv.root" ]; then
# #       cp zach/stage0/$i/output_0_0000-larcv.root $i
# #    fi
# #    python3 utils/add_HIPMIP.py $i/output_0_0000-larcv.root $i/output_0_0000-larcv_HM.root
# done



#   python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/test_io_both.cfg -s $i/output_0_0000-larcv.root -o $i/output_0_0000-analysis_both.h5
#   python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/hdf5writer.cfg -s $i/output_0_0000-larcv_HM.root -o $i/output_0_0000-analysis_HM_both.h5

# file_names=$(find "$WORKDIR" -type f -exec basename {} \;)

# # Check for duplicates using sort and uniq
# duplicates=$(echo "$file_names" | sort | uniq -d)

# if [ -z "$duplicates" ]; then
#     echo "All file names are unique."
# else
#     echo "Duplicate file names found:"
#     echo "$duplicates"
# fi

# WORKDIR=/sdf/data/neutrino/dcarber/larcv_files/NuMI_nu_cosmics/v09_89_01_01p01/
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_files/



# WORKDIR=/sdf/data/neutrino/dcarber/NuMI_nu_spine/v09_89_01p03


OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/prod_test_files_truth/


rm -f error/*
rm -f output/*


# directory="/sdf/data/neutrino/dcarber/NuMI_nu_spine/v09_89_01p03"

# Loop over each file matching the pattern
for file in "$WORKDIR"*; do
    echo "Reading file: $file"
    
    # Loop over each line inside the file
    while IFS= read -r line; do
        # echo "Processing: $line"
        # Do something with $line, such as copying, analyzing, etc.
    

    # find "$WORKDIR" -type f -name "*.root" | while read -r file; do
        # Get the basename of each .root file
        basename_file=$(basename "$line" .root)
        # echo "help"
        # echo "$basename_file"

        mkdir -p $OUTDIR/$basename_file

        i=$OUTDIR/$basename_file
        
        # break

        

        # if [ -f "$i/analysis_HM_both.h5" ]; then
        #     echo "$i/analysis_HM_both.h5 exists."
        # else
        # echo "submit_job_larcvHM.sbatch $DIR $i $line"


        # file="$i.npy"



        file="${OUTDIR/_files/_analysis}"/npyfiles/$basename_file.npz

        if [[ ! -f "$file" ]]; then
            # rm $i/*
            echo "$file does not exist"
            sbatch submit_job_larcvHM_truth.sbatch $DIR $i $line
        else 
        echo "$file found"
        continue
        fi


        mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`
        until (( $mine < 900 ));
        do
            echo "still too many jobs there......"
            sleep 15
            mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`

        done

        # exit 0


        
# python3 -c "
# import numpy as np
# import sys

# try:
#     np.load('$file',allow_pickle=True)
#     print('$file is valid')
# except Exception as e:
#     print(f'Invalid .npy file: {e}')
#     sys.exit(1)
# " || sbatch submit_job_larcvHM.sbatch $DIR $i $line
#         fi

        
        
        
    done < "$file"
done

# for i in "${arr[@]}"
# do
# #    echo "$i"
# #    mkdir -p $i
# #    if [ ! -e "$i/output_0_0000-edepsim.root" ]; then
# #       cp zach/stage0/$i/output_0_0000-edepsim.root $i
# #    fi
# #    if [ ! -e "$i/output_0_0000-larcv.root" ]; then
# #       cp zach/stage0/$i/output_0_0000-larcv.root $i
# #    fi
# #    python3 utils/add_HIPMIP.py $i/output_0_0000-larcv.root $i/output_0_0000-larcv_HM.root
# done



#   python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/test_io_both.cfg -s $i/output_0_0000-larcv.root -o $i/output_0_0000-analysis_both.h5
#   python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/hdf5writer.cfg -s $i/output_0_0000-larcv_HM.root -o $i/output_0_0000-analysis_HM_both.h5

# file_names=$(find "$WORKDIR" -type f -exec basename {} \;)

# # Check for duplicates using sort and uniq
# duplicates=$(echo "$file_names" | sort | uniq -d)

# if [ -z "$duplicates" ]; then
#     echo "All file names are unique."
# else
#     echo "Duplicate file names found:"
#     echo "$duplicates"
# fi