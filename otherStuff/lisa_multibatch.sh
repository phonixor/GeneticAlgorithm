#!/bin/bash
#
# multi batch launches lisabatch.txt
#
# change the settings bellow 
model=$HOME/models/3geneWithVarProduction.model
name=exp1
itterations=5
%
for i in {1..4}
do
	# launch the lisa batch
	echo calling:
	echo $HOME/stage/lisa_batch.sh $model ${name}_$i $itterations
	echo $HOME/stage/lisa_batch.sh $model ${name}_$i $itterations | qsub
done
#wait 
# dont think i have to wait since everything is send to the scedular anyways...
