#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=output/output-%j.txt
#SBATCH --error=error/error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=0:10:00
#SBATCH --account=neutrino:icarus-ml
#SBATCH --qos=preemptable


export PYTHONUNBUFFERED=TRUE
DIR=$1 
i=$2
# file=$3

echo $SLURMD_NODENAME 
echo $f
echo $i ##SBATCH --exclude=sdfrome064,sdfrome061

# SPINE_LOC=/sdf/data/neutrino/icarus/spine/prod/numi_nu_corsika_mix/output_spine/

IMAGE=/sdf/group/neutrino/images/develop.sif

# if [ -f "$i/larcv_HM.root" ]; then
#     echo "$i/larcv_HM.root exists."
# else
#     singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  $IMAGE bash -c "python3 $DIR/2x2_Strange/utils/add_HIPMIP.py $file $i/larcv_HM.root"
# fi

# if [ -f "$i/analysis_both.h5" ]; then
#     echo "$i/analysis_both.h5 exists."
# else
#     ln -s fileB.html fileA.html
#     singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  $IMAGE bash -c "python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/test_io_both.cfg -s $file -o $i/analysis_both.h5"


    # # FILE=SPINE_LOC/ basename /a/b/c/filename.root .root
    # base=$(basename $file .root)
    # FILE="$SPINE_LOC/$base"_spine.h5

    # if [ ! -f "$FILE" ]; then
    # echo "Error: File '$FILE' does not exist." >&2
    # exit 1
    # fi
    # ln -s $FILE $i/analysis_both.h5
# fi
#source $DIR/spine_prod/configure.sh; source /sdf/data/neutrino/software/OpT0Finder/configure.sh;

# if [ -f "$i/analysis_HM_both.h5" ]; then
#     echo "$i/analysis_HM_both.h5 exists."
# else
#     singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  $IMAGE bash -c "python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/hdf5writer.cfg -s $i/larcv_HM.root -o $i/analysis_HM_both.h5"
# fi

singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  /sdf/group/neutrino/images/develop.sif bash -c "python3 -m cProfile $DIR/2x2_Strange/run_analysis.py --mode reco --dir $i --compare_truth True"