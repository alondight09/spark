#!/bin/sh

export TARGET=logistic
PWD=${CURDIR}

compile_arm:
	make -C arm

compile_fpga:
	make -C fpga

setup:
	for id in 4 5 ; do \
		ssh computer$$id 'cd ${PWD}; scp ../accService/arm/accService_host/pkg/arm/zc706/bin/accService_host.exe root@fpga:/mnt/acc'; \
		ssh computer$$id 'cd ${PWD}; scp ../accKernel/${TARGET}/${TARGET}_lpp/pkg/arm/zc706/bin/${TARGET}_lpp.xclbin root@fpga:/mnt/acc'; \
		#ssh computer$$id 'cd ${PWD}; scp start.sh root@fpga:/mnt/acc'; \
		#ssh computer$$id 'cd ${PWD}; scp stop.sh root@fpga:/mnt/acc'; \
	done

start:
	for id in 4 5 ; do \
		ssh computer$$id 'ssh root@fpga "cd /mnt/acc; sh stop.sh accService"'; \
		ssh computer$$id 'ssh root@fpga "cd /mnt/acc; sh start.sh accService"' & \
	done
