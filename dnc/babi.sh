#!/bin/bash

declare -a iters=( 150000 )
declare -a input=( "en-10k" )

for task in ${input[@]}
do
	for iter in ${iters[@]}
	do
		date
		python3 train.py --input ${task} --output "${task}_${iter}"  --iterations ${iter}
		date
		python3 test.py "${task}_${iter}" ${task} ${iter} > checkpoints/${task}_${iter}/test_results.txt
		date
	done
done
