# shell for the job:
#PBS -S /bin/bash
# job requires at most 10 hours, 0 minutes
#     and 0 seconds wallclock time and uses one 8-core node:
#PBS -lwalltime=7:00:00 -lnodes=1:cores8
#
# load the required modules
module load python/2.7.2
# cd to the directory where the program is to be called:
cd $HOME/stage/python/
# run the program
echo starting working on problem $PBS_ARRAYID
for i in `seq 7` ; do
	# very important the & at the end, so that they all run in parallel instead of sequential...
	python experiment.py -n $PBS_ARRAYID-$i $HOME/stage/python/models/test3 $HOME/stage/python/models/4geneWithVarProduction.model &
done
wait # is this needed?
python nothing.py # dummy process to fool the schedular?
echo ended working on problem $PBS_ARRAYID
#
# notes:
# call this with qsub -t 1-100 experiment.sh
#  an advantage of using the '-t' flag is that you can kill all the jobs in one command. 
#  The jobnumbers will be for example 34445-1 34445-2 ... 34445-100. 
#  The command 'qdel 34445' will remove all these jobs.
