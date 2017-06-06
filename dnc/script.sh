#!/bin/bash

declare -a iters=( 2500 5000 7500 10000 )
declare -a input=( "en-10k" "family" )

for task in ${input[@]}
do
	for iter in ${iters[@]}
	do
		python3 train.py --input ${task} --output "${task}_${iter}"  --iterations ${iter}
		python3 test.py "${task}_${iter}" ${task} > checkpoints/${task}_${iter}/test_results.txt
	done
done
