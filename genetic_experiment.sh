# shell for the job:
#PBS -S /bin/bash
# job requires at most 62 hours, 0 minutes
#     and 0 seconds wallclock time and uses one 8-core node:
#PBS -lwalltime=62:00:00 -lnodes=1
#
# load the required modules
module load python/2.7.2
# cd to the directory where the program is to be called:
cd $HOME/stage/python/
# get the nr of cpu cores
ncores=`cat /proc/cpuinfo | grep bogomips | wc -l`
# run the program
echo starting working on problem $PBS_ARRAYID
(( nprocs = ncores - 1 ))
echo $nprocs
for (( i=1; i<=nprocs; i++ )) ; do
	# very important the & at the end, so that they all run in parallel instead of sequential...
	echo nr o :$i
	python genetic_experiment.py -n $PBS_ARRAYID-$i $HOME/stage/python/models/experiment_longrun $HOME/stage/python/models/4geneWithVarProduction.model &
done
wait
echo ended working on problem $PBS_ARRAYID
#
# notes:
# call this with qsub -t 1-100 genetic_experiment.sh
#  an advantage of using the '-t' flag is that you can kill all the jobs in one command. 
#  The jobnumbers will be for example 34445-1 34445-2 ... 34445-100. 
#  The command 'qdel 34445' will remove all these jobs.
