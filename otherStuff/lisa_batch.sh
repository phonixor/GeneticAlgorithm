#PBS -lnodes=1 -lwalltime=1:00:00
# reserve the corers and time, not that this should be related to the settings...
# why are they comments??? i dunno i just hope they work :)
#
echo "this file should be used on the lisa.sara.nl clusther"
echo "it should be called like:"
echo "qsub lisa_batch.sh model name itterations"
echo ""
#
echo "$# parameters"
#echo "$@"
echo "model: $1"
echo "name: $2"
echo "itterations: $3"
# load the proper module
module load matlab
# launch matlab
matlab /nodesktop /r addpath(genpath('./'));experiment($1,$2,$3);
