#!/bin/bash

salloc --nodes $1 -n $2 --gpus-per-task=1 --ntasks-per-node=4 --qos interactive --time 04:00:00 --constraint gpu -G $2 --account m4055_g
