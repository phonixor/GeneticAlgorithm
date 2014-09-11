#!/bin/bash
#
modeldir=$HOME/stage/python/models
cd $modeldir

for i in {1..100}
do
	echo cp $modeldir/test$i/optimized.model $modeldir/optimized$i.model
	
	cp $modeldir/test$i/optimized.model $modeldir/optimized$i.model
	cp $modeldir/test$i/log.txt $modeldir/log$i.txt
done
#wait
# dont think i have to wait since everything is send to the scedular anyways...