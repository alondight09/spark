
USER_PATH=/user/bjxiao
COMMON_CLASS=${SPARK_HOME}/conf:${SPARK_HOME}/assembly/target/scala-2.10/spark-assembly-0.9.0-incubating-hadoop2.2.0.jar
#TARGET=JavaHdfsLR
#TARGET=PartitionLR
#TARGET=SocketLR
TARGET=UserLR
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 hdfs://farmer:9000${USER_PATH}/lr_two.txt 10
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 data/lr_data.txt 1
ARGUMENTS=spark://farmer.cs.ucla.edu:7078 hdfs://farmer:9000${USER_PATH}/lr_data.txt 10
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 hdfs://farmer:9000${USER_PATH}/train_data.txt 10 hdfs://farmer:9000${USER_PATH}/test_data.txt
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 hdfs://farmer:9000${USER_PATH}/train_data.txt 10
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 hdfs://farmer:9000${USER_PATH}/train_few.txt 10
#TARGET=TestPartitionLR
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 hdfs://farmer:9000${USER_PATH}/lr_simple.txt 10
#TARGET=TestPartition
#ARGUMENTS=spark://farmer.cs.ucla.edu:7078 test_partition.txt

compile:
	mvn package

test:
	java -cp target/simple-project-1.0.jar:${COMMON_CLASS} -Djava.library.path= ${TARGET} ${ARGUMENTS}
