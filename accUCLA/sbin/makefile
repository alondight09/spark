#!/bin/sh

export TARGET=logistic
PWD=${CURDIR}
namenode_dir=${HADOOP_HOME}/hdfs_data/namenode
datanode_dir=/scratch/shared/hdfs_data/datanode
slave_list= \
computer0 \
computer1 \
computer4 \
computer5 \
computer6 \
computer7 

data_list= \
../benchmark/logistic/data

compile_service:
	make -C ../accService/arm

compile_fpga:
	make -C ../accKernel/${TARGET}

setup:
	for slave in ${slave_list}; do \
		ssh $$slave 'cd ${PWD}; scp ../accService/arm/accService_host/pkg/arm/zc706/bin/accService_host.exe root@fpga:/mnt/acc'; \
		ssh $$slave 'cd ${PWD}; scp ../accKernel/${TARGET}/${TARGET}_lpp/pkg/arm/zc706/bin/${TARGET}_lpp.xclbin root@fpga:/mnt/acc'; \
		#ssh computer$$id 'cd ${PWD}; scp start.sh root@fpga:/mnt/acc'; \
		#ssh computer$$id 'cd ${PWD}; scp stop.sh root@fpga:/mnt/acc'; \
	done

start:
	for slave in ${slave_list}; do \
		ssh $$slave 'ssh root@fpga "cd /mnt/acc; sh stop.sh accService"'; \
		ssh $$slave 'ssh root@fpga "cd /mnt/acc; sh start.sh accService"'& \
	done

init:
	cd ${HADOOP_HOME}; sh sbin/stop-dfs.sh; rm -f etc/hadoop/slaves;
	cd ${SPARK_HOME}; sh sbin/stop-all.sh;
	for slave in ${slave_list}; do \
		echo "$$slave" >> ${HADOOP_HOME}/etc/hadoop/slaves; \
	done 
	rm -rf ${namenode_dir}/* 
	for slave in ${slave_list}; do \
		echo "creating hdfs in local disk on $$slave"; \
		ssh $$slave 'rm -rf ${datanode_dir}; mkdir -p ${datanode_dir}'; \
	done 
	hdfs namenode -format 
	cd ${HADOOP_HOME}; sh sbin/start-dfs.sh;
	hdfs dfs -mkdir -p /user/bjxiao;
	for data_path in ${data_list}; do \
		hdfs dfs -put $$data_path; \
	done
	cd ${SPARK_HOME}; cp ${HADOOP_HOME}/etc/hadoop/slaves conf/slaves; sh sbin/start-all.sh
