#!/bin/sh

export LANG=en_US.UTF8
export LD_LIBRARY_PATH=/mnt/embedded_root/lib:$LD_LIBRARY_PATH

#cd /mnt/acc
#nohup ./$1_host.exe 1 ./data $1_dup.xclbin &> mmul_host.log
echo "to start $1_host.exe"
nohup ./$1_host.exe
