#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=output/output-%j.txt
#SBATCH --error=error/error-%j.txt
#SBATCH --partition=roma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6g
#SBATCH --time=1:00:00
#SBATCH --account=neutrino:icarus-ml
#SBATCH --qos=preemptable


export PYTHONUNBUFFERED=TRUE
DIR=$1 
i=$2
file=$3

echo $SLURMD_NODENAME 
echo $f
echo $i ##SBATCH --exclude=sdfrome064,sdfrome061

# if [ -f "$i/larcv_HM.root" ]; then
#     echo "$i/larcv_HM.root exists."
# else
#     singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  /sdf/group/neutrino/images/larcv2_ub20.04-cuda12.1-pytorch2.2.1-larndsim.sif bash -c "python3 $DIR/2x2_Strange/utils/add_HIPMIP.py $file $i/larcv_HM.root"
# fi

# if [ -f "$i/analysis_both.h5" ]; then
#     echo "$i/analysis_both.h5 exists."
# else
#     singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  /sdf/group/neutrino/images/larcv2_ub20.04-cuda12.1-pytorch2.2.1-larndsim.sif bash -c "python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/test_io_both.cfg -s $file -o $i/analysis_both.h5"
# fi

# if [ -f "$i/analysis_HM_both.h5" ]; then
#     echo "$i/analysis_HM_both.h5 exists."
# else
#     singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  /sdf/group/neutrino/images/larcv2_ub20.04-cuda12.1-pytorch2.2.1-larndsim.sif bash -c "python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/configs/hdf5writer.cfg -s $i/larcv_HM.root -o $i/analysis_HM_both.h5"
# fi

singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  /sdf/group/neutrino/images/develop.sif bash -c "python3 $DIR/2x2_Strange/print_particle_record.py $i/larcv_HM.root $file"


# singularity exec --bind /sdf/group/neutrino/zhulcher,/sdf/data/neutrino/  /sdf/group/neutrino/images/larcv2_ub20.04-cuda12.1-pytorch2.2.1-larndsim.sif python3 /sdf/group/neutrino/zhulcher/4fun/conway99/conwayfinal.py $f $N
#srun -n 1 
# sleep 5                                 # wait for slurm to get the job status into its database
# sacct --format=JobID,Submit,Start,End,State,Partition,ReqTRES%30,CPUTime,MaxRSS,NodeList%30 --units=M -j $SLURM_JOBID