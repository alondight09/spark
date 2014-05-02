
USER_PATH=hdfs://farmer:9000/user/bjxiao/data
COMMON_CLASS=${SPARK_HOME}/conf:${SPARK_HOME}/assembly/target/scala-2.10/spark-assembly-0.9.0-incubating-hadoop2.2.0.jar
MASTER_NAME=spark://farmer.cs.ucla.edu:7078
#JAVA_OPTION = -Xmx500m
#TARGET=JavaHdfsLR
TARGET=PartitionLR
#TARGET=SocketLR
#TARGET=UserLR
#ARGUMENTS=${USER_PATH}/lr_two.txt 10
#ARGUMENTS=data/lr_data.txt 1
#ARGUMENTS=${USER_PATH}/lr_data.txt 10
#ARGUMENTS=${USER_PATH}/train_data.txt 10 10 784 ${USER_PATH}/test_data.txt
ARGUMENTS=${USER_PATH}/train_data.txt 10 10 784 1 ${USER_PATH}/test_data.txt
#ARGUMENTS=${USER_PATH}/train_data.txt 10
#ARGUMENTS=${USER_PATH}/train_few.txt 10
#TARGET=TestPartitionLR
#ARGUMENTS=${USER_PATH}/lr_simple.txt 10
#TARGET=TestPartition
#ARGUMENTS=test_partition.txt

compile:
	mvn package

test:
	java ${JAVA_OPTION} -cp target/simple-project-1.0.jar:${COMMON_CLASS} -Djava.library.path= ${TARGET} ${MASTER_NAME} ${ARGUMENTS}
