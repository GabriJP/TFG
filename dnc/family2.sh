#!/bin/bash

iters=600000
task="family"

for inc in {3..3}
do
	mkdir -p checkpoints/${task}_${iters}_${inc}
	date > checkpoints/${task}_${iters}_${inc}/log.txt
	python3 train.py --input ${task} --output "${task}_${iters}_${inc}"  --iterations ${iters} >> checkpoints/${task}_${iters}_${inc}/log.txt
	date >> checkpoints/${task}_${iters}_${inc}/log.txt
	python3 test.py "${task}_${iters}_${inc}" ${task} ${iters} > checkpoints/${task}_${iters}_${inc}/test_results.txt
	date >> checkpoints/${task}_${iters}_${inc}/log.txt
done
