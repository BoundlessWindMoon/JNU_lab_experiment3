#!/bin/bash
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=dcu:1

#module purge
#module load compiler/dtk/24.04

make clean
make -j TEST=y 2>&1 | grep 'error'
testcase_1="64 256 14 14 256 3 3 1 1 1 1"
testcase_2="256 192 14 14 192 3 3 1 1 1 1"
testcase_3="16 256 26 26 512 3 3 1 1 1 1"
testcase_4="32 256 14 14 256 3 3 1 1 1 1"
testcase_5="2 1280 16 16 1280 3 3 1 1 1 1"
testcase_6="2 960 64 64 32 3 3 1 1 1 1"

./conv2dfp16demo $testcase_1
./conv2dfp16demo $testcase_2
./conv2dfp16demo $testcase_3
./conv2dfp16demo $testcase_4
./conv2dfp16demo $testcase_5
./conv2dfp16demo $testcase_6