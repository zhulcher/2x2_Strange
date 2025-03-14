DIR=/sdf/group/neutrino/zhulcher/

# WORKDIR=/sdf/data/neutrino/dcarber/larcv_files/NuMI_nu_cosmics/v09_89_01_01p01/
# OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber_files/


rm -f error/*
rm -f output/*

WORKDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky/
OUTDIR=/sdf/data/neutrino/zhulcher/BNBNUMI/simple_franky_files/


# DIR=/home



# python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/hdf5writer.cfg -s $DIR/2x2_Strange/utils/output_HM-larcv.root -o $DIR/2x2_Strange/utils/output_HM.h5
# python3 $DIR/spine/bin/run.py -c $DIR/spine/config/test_io.cfg -s $DIR/2x2_Strange/utils/MiniRun5_1E19_RHC.flow.0000001.larcv.root -o $DIR/2x2_Strange/utils/output_normal.h5


find "$WORKDIR" -type f -name "*.root" | while read -r file; do
    # Get the basename of each .root file
    basename_file=$(basename "$file" .root)
    # echo "help"
    echo "$basename_file"

    mkdir -p $OUTDIR/$basename_file

    i=$OUTDIR/$basename_file
    
    # break

    mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`
    until (( $mine < 400 ));
    do
        echo "still too many jobs there......"
        sleep 15
        mine=`squeue -u zhulcher  -h -t pending,running -r | wc -l`

    done

    # if [ -f "$i/analysis_HM_both.h5" ]; then
    #     echo "$i/analysis_HM_both.h5 exists."
    # else
    sbatch submit_job_larcvHM.sbatch $DIR $i $file
    # fi

    # if [ -f "$i/larcv_HM.root" ]; then
    #     echo "$i/larcv_HM.root exists."
    # else
    #     python3 $DIR/2x2_Strange/utils/add_HIPMIP.py $file $i/larcv_HM.root
    # fi

    # if [ -f "$i/analysis_both.h5" ]; then
    #     echo "$i/analysis_both.h5 exists."
    # else
    #     python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/test_io_both.cfg -s $file -o $i/analysis_both.h5
    # fi

    # if [ -f "$i/analysis_HM_both.h5" ]; then
    #     echo "$i/analysis_HM_both.h5 exists."
    # else
    #     python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/hdf5writer.cfg -s $i/larcv_HM.root -o $i/analysis_HM_both.h5
    # fi
    
    
    
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