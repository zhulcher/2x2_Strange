DIR=~/Documents/Github
DIR=/home
python3 $DIR/spine/bin/run.py -c $DIR/2x2_Strange/hdf5writer.cfg -s $DIR/2x2_Strange/utils/output_HM-larcv.root -o $DIR/2x2_Strange/utils/output_HM.h5
python3 $DIR/spine/bin/run.py -c $DIR/spine/config/test_io.cfg -s $DIR/2x2_Strange/utils/MiniRun5_1E19_RHC.flow.0000001.larcv.root -o $DIR/2x2_Strange/utils/output_normal.h5