#ifndef HEADER_H
#define HEADER_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <fcntl.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <string>
#include <string.h>
#include <time.h>

using namespace std;


#define NOF_FPGA 8
#define NOF_SLAVE 8
#define CPU_PER_SLAVE 2

#define FPGA_IDLE 0
#define FPGA_HAS_BIT 1
#define FPGA_LOST 2

#define ACC_IDLE 0
#define ACC_BUSY 1
#define ACC_LOST 2

#define ACC_EXE_TIME 1

#define PORT_TO_SCHEDULER 9988 
#define PORT_TO_ACC_RQST  9989 
#define PORT_TO_ACC_DONE  9990
#endif
