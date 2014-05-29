
#define FPGA_DEVICE
#define SOCKET 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <CL/opencl.h>
#include <math.h>
#include <unistd.h>
#ifndef FPGA_DEVICE
#include "logistic_cl.h"
#endif

#define GROUP_SIZE		8
#define CHUNK_SIZE 		512

#define LABEL_SIZE		10
#define FEATURE_SIZE	784
#define PARTITION_SIZE	20000

#include <sys/time.h>
#include <time.h>

typedef struct timespec timespec;
timespec diff(timespec start, timespec end)
{   
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

timespec sum(timespec t1, timespec t2) {
  timespec temp;
  if (t1.tv_nsec + t2.tv_nsec >= 1000000000) {
    temp.tv_sec = t1.tv_sec + t2.tv_sec + 1;
    temp.tv_nsec = t1.tv_nsec + t2.tv_nsec - 1000000000;
  } else {
    temp.tv_sec = t1.tv_sec + t2.tv_sec;
    temp.tv_nsec = t1.tv_nsec + t2.tv_nsec;
  }
  return temp;
}

void printTimeSpec(timespec t) {
  printf("elapsed time: %d.%09d\n", (int)t.tv_sec, (int)t.tv_nsec);
}

timespec tic( )
{
    timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    return start_time;
}

void toc( timespec& start_time )
{
    timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    printTimeSpec( diff( start_time, current_time ) );
    start_time = current_time;
}

int load_file_to_memory(const char *filename, char **result) { 

	int size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) 
	{ 
		*result = NULL;
		return -1; // -1 means file opening fail 
	} 
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f)) 
	{ 
		free(*result);
		return -2; // -2 means file reading fail 
	} 
	fclose(f);
	(*result)[size] = 0;

	return size;
}

void computeGradient(float* weights, float* data, float* gradient, int L, int D, int n)
{
    int i, j, k;
    for( k = 0; k < n; k++ )
    {
        for( i = 0; i < L; i++ )
        {
            float dot = 0.;
            for( j = 0; j < D; j++ )
            {
                dot += weights[i*D+j]*data[k*(D+L)+j+L];
            }
            float coeff = (1. / (1. + exp(-data[k*(D+L)+i]*dot )) - 1.)*data[k*(D+L)+i];
            for( j = 0; j < D; j++ )
            {
                gradient[i*D+j] +=  coeff*data[k*(D+L)+j+L];
            }
        }
    }
}

struct cl_package
{
    cl_context context;
    cl_command_queue commandQueue;
    cl_kernel kernel;
    cl_mem d_gradient;
    cl_mem d_weights;
    cl_mem d_data;
};

void computeGradientByFPGA(float* weights, float* data, float* gradient, int L, int D, int n, struct cl_package clPackage)
{
	cl_command_queue clCommandQue = clPackage.commandQueue;
	cl_context context = clPackage.context;
	cl_kernel clKernel = clPackage.kernel;
	cl_mem d_gradient = clPackage.d_gradient;
	cl_mem d_weights = clPackage.d_weights;
	cl_mem d_data = clPackage.d_data;
    cl_int status;

    printf("kernel starts\n");
    timespec timer1 = tic( );

	float* gradient_local = (float*)malloc(L*D*GROUP_SIZE*sizeof(float));
	memset(gradient_local, 0.f, L*D*GROUP_SIZE*sizeof(float));

    /*
	status = clSetKernelArg(clKernel, 0, sizeof(int), (void *)&L);
	status |= clSetKernelArg(clKernel, 1, sizeof(int), (void *)&D);
	status |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&d_weights);
	status |= clSetKernelArg(clKernel, 4, sizeof(cl_mem), (void *)&d_data);
	status |= clSetKernelArg(clKernel, 5, sizeof(cl_mem), (void *)&d_gradient);
    */
	status = clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_weights);
	status |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_data);
	status |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&d_gradient);

	if (status != CL_SUCCESS)
		printf("clSetKernelArg error(%d)\n", status);

    size_t work_size[1]={GROUP_SIZE};
    size_t group_size[1]={1};

    int k;
	int n_chk;
    status = clEnqueueWriteBuffer(clCommandQue, d_gradient, CL_FALSE, 0, L*D*GROUP_SIZE*sizeof(float), gradient_local, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(clCommandQue, d_weights, CL_FALSE, 0, L*D*sizeof(float), weights, 0, NULL, NULL);

    printf("kernel prepared --- ");
    toc( timer1 );

    for( k = 0; k < n; k += CHUNK_SIZE ) // n: data samples (500)
    {
		if (k+CHUNK_SIZE < n)
			n_chk = CHUNK_SIZE;
		else 
			n_chk = n - k;

        int reps = n_chk/GROUP_SIZE;
        status = clSetKernelArg(clKernel, 0, sizeof(int), (void *)&reps);
		//status = clSetKernelArg(clKernel, 2, sizeof(int), (void *)&n_chk);

        status = clEnqueueWriteBuffer(clCommandQue, d_data, CL_TRUE, 0, (D+L)*n_chk*sizeof(float), data+k*(D+L), 0, NULL, NULL);

        status = clEnqueueNDRangeKernel(clCommandQue, clKernel, 1, NULL, work_size, group_size, 0, NULL, NULL);

        if (status != CL_SUCCESS)
            printf("clEnqueueNDRangeKernel error(%d)\n", status);

    }
    status = clEnqueueReadBuffer(clCommandQue, d_gradient, CL_TRUE, 0, L*D*GROUP_SIZE*sizeof(float), gradient_local, 0, NULL, NULL);

    printf("kernel done --- ");
    toc( timer1 );

    if (status != CL_SUCCESS)
        printf("clEnqueueReadBuffer error(%d)\n", status);

	int gid = 0;
	for (gid = 0; gid < GROUP_SIZE; gid++) {
		for (k = 0; k < L*D; k++) {
			gradient[k] += gradient_local[gid*L*D + k];
		}
	}
    
    printf("CPU reduction done --- ");
    toc( timer1 );

	free(gradient_local);
}

struct cl_package initFPGA( const char* xclbin, const char* kernel_name )
{
	/*****************************************/
	/* Initialize OpenCL */
	/*****************************************/

	// Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);

	//printf("Found %d platforms support OpenCL, return code %d.\n", numPlatforms, status);
 
    // Allocate enough space for each platform
    cl_platform_id *platforms = (cl_platform_id*)malloc( numPlatforms*sizeof(cl_platform_id));
 
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS)
		printf("clGetPlatformIDs error(%d)\n", status);
	
	// Retrieve the number of devices
    cl_uint numDevices = 0;
#ifndef FPGA_DEVICE
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
#else
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
#endif
	printf("Found %d devices support OpenCL.\n", numDevices);

    // Allocate enough space for each device
    cl_device_id *devices = (cl_device_id*)malloc( numDevices*sizeof(cl_device_id));

    // Fill in the devices 
#ifndef FPGA_DEVICE
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
#else
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
#endif
	
	if (status != CL_SUCCESS)
		printf("clGetDeviceIDs error(%d)\n", status);

    // Create a context and associate it with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	if (status != CL_SUCCESS)
		printf("clCreateContext error(%d)\n", status);


	//Create a command-queue
	cl_command_queue clCommandQue = clCreateCommandQueue(context, devices[0], 0, &status);

	if (status != CL_SUCCESS)
		printf("clCreateCommandQueue error(%d)\n", status);

	// 6. Load and build OpenCL kernel
	
#ifndef FPGA_DEVICE
	// Create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&logistic_cl, NULL, &status);
	if (status != 0)
		printf("clCreateProgramWithSource error(%d)\n", status);

    // Build (compile) the program for the device
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
#else
	// Load binary from disk
	unsigned char *kernelbinary;
	printf("loading %s\n", xclbin);
	int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (n_i < 0) {
		printf("ERROR: failed to load kernel from xclbin: %s\n", xclbin);
		exit(1);
	}
	size_t n_bit = n_i;

	// Create the compute program from offline
	cl_program program = clCreateProgramWithBinary(context, 1, &devices[0], &n_bit,
			(const unsigned char **) &kernelbinary, NULL, &status);
	if ((!program) || (status != CL_SUCCESS)) {
		printf("Error: Failed to create compute program from binary %d!\n", status);
		exit(1);
	}

	// Build the program executable
	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#endif

	if (status != 0) {
		char errmsg[2048];
		size_t sizemsg = 0;

		status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 2048*sizeof(char), errmsg, &sizemsg);

		printf("clBuildProgram error(%d)\n", status);
		printf("Compilation messages: \n %s", errmsg);
	}

	cl_kernel clKernel = clCreateKernel(program, kernel_name, &status);
	if (status != CL_SUCCESS)
		printf("clCreateKernel error(%d)\n", status);

	// TODO: parameterize the size of buffers
	cl_mem d_gradient = clCreateBuffer(context, CL_MEM_READ_WRITE, FEATURE_SIZE*LABEL_SIZE*GROUP_SIZE*sizeof(float), NULL, &status);
	if (status != CL_SUCCESS)
		printf("d_gradient clCreateBuffer error(%d)\n", status);

	cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, FEATURE_SIZE*LABEL_SIZE*sizeof(float), NULL, &status);
	if (status != CL_SUCCESS)
		printf("d_weights clCreateBuffer error(%d)\n", status);

	cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_ONLY, (FEATURE_SIZE+LABEL_SIZE)*CHUNK_SIZE*sizeof(float), NULL, &status);
	if (status != CL_SUCCESS)
		printf("d_data clCreateBuffer error(%d)\n", status);

    struct cl_package result;
    result.context = context;
    result.kernel = clKernel;
    result.commandQueue = clCommandQue;
    result.d_gradient = d_gradient;
    result.d_weights = d_weights;
    result.d_data = d_data;

    return result;
}

int setupSocket( int port )
{
	// socket 
	int err = 0;
	int listenfd = 0;
	socklen_t buf_size = 32*1024*1024;
	socklen_t size = sizeof(buf_size);
	struct sockaddr_in serv_addr; 

	listenfd = socket(AF_INET, SOCK_STREAM, 0);

	err = setsockopt(listenfd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));
	err = setsockopt(listenfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));
	err = getsockopt(listenfd, SOL_SOCKET, SO_SNDBUF, &buf_size, &size);
	printf("socket send buffer size: %d\n", buf_size);

	memset(&serv_addr, '0', sizeof(serv_addr));
	
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(port); 

	bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 

	listen(listenfd, 10); 

    printf("socket listening port is ready\n");

    return listenfd;
}

int acceptSocket( int listenfd )
{
    printf("waiting for host to connect\n");
    int connfd = accept(listenfd, (struct sockaddr*)NULL, NULL); 
    
    if( connfd < 0 )
    {
        printf("ERROR on accept\n");
        exit(1);
    }
    printf("host connects, start transfering data.\n");
    return connfd;
}

int recv_large_array( int connfd, char* data, size_t nbyte_exp )
{
    size_t nbyte = 0;
	size_t packet_size = 256*1024;
    unsigned int start;
    for( start = 0; start < nbyte_exp; start += packet_size )
    {
        if( start + packet_size > nbyte_exp ) packet_size = nbyte_exp - start;
        nbyte += recv(connfd, data + start, packet_size, MSG_WAITALL);
    }
    return nbyte;
}

int main(int argc, char** argv) {

    printf("************* Welcome to UCLA FPGA agent! **********\n");
#if SOCKET
    int listenfd = setupSocket( 5000 );
#endif
    struct cl_package clPackage = initFPGA("logistic_lpp.xclbin", "logistic");

    float* weights = (float*)malloc(LABEL_SIZE*FEATURE_SIZE*sizeof(float));
    float* data = (float*)malloc(PARTITION_SIZE*(FEATURE_SIZE+LABEL_SIZE)*sizeof(float));
    float* gradient = (float*)malloc(LABEL_SIZE*FEATURE_SIZE*sizeof(float));

#if SOCKET
    while (1) 
#endif
    {
#if SOCKET
        int connfd = acceptSocket(listenfd);
#endif
        printf("************* Got a new task! *************\n");
        //fread(A, n*n, sizeof(float), fin);
        //fread(B, n*n, sizeof(float), fin);
        size_t nbyte;
        int L;
        int D;
        int n;
        int cached;
#if SOCKET
        timespec timer1 = tic( );
        nbyte = recv(connfd, &L, sizeof(L), MSG_WAITALL);
        nbyte = recv(connfd, &D, sizeof(D), MSG_WAITALL);
        nbyte = recv(connfd, &n, sizeof(n), MSG_WAITALL);
        nbyte = recv(connfd, &cached, sizeof(cached), MSG_WAITALL);
        printf("parameter recieved --- ");
        toc( timer1 );
#else
        L = LABEL_SIZE;
        D = FEATURE_SIZE;
		if (argc > 1)
			n = atoi(argv[1]);
		else
			n = 100;
#endif

        printf("# of labels: %d\n", L);
        printf("# of features: %d\n", D);
        printf("# of data points: %d\n", n);
        
        if(L>1024 || D>1024 || n > 65536)
        {
            printf("ERROR: too large data size!\n");
            return 1;
        }
        memset((void*)gradient, 0, L*D*sizeof(float));

        int i;
#if SOCKET
        nbyte = recv_large_array(connfd, (char*)weights, L*D*sizeof(float));
        printf("received weights for %d bytes --- ", nbyte);
        toc( timer1 );
#if SHOW_DATA
        printf("the first 10 elements are:\n");
        for( i = 0; i < 10; i++ ) printf("%f\n", weights[i]);
#endif
        if( cached == 0 )
        {
            //nbyte = recv(connfd, data, n*(D+L)*sizeof(float), MSG_WAITALL);
            nbyte = recv_large_array( connfd, (char*)data, n*(D+L)*sizeof(float) );
            printf("received training data for %d bytes --- ", nbyte);
            toc( timer1 );
        }
#if SHOW_DATA
        printf("the first 10 elements are:\n");
        for( i = 0; i < 10; i++ ) printf("%f\n", data[i]);
#endif
#else
        for( i = 0; i < D; i++ ) weights[i]=0.;
        FILE* pFile = fopen("train_data.txt","r");
        for( i = 0; i < n*(D+L); i++ ) fscanf(pFile, "%f", data+i);
        fclose(pFile);
#endif
        
        printf("FPGA computation...\n");
        //computeGradient(weights,data,gradient,L,D,n);
        computeGradientByFPGA(weights,data,gradient,L,D,n,clPackage);
        printf("FPGA computation finished! --- ");
        toc( timer1 );

#if SOCKET
        //for( int i = 0; i < 10; i++ ) gradient[i]=i;
        nbyte = send(connfd, gradient, L*D*sizeof(float), 0);
        //int temp = tr.tv_usec;
        //nbyte = send(connfd, &temp, sizeof(int), 0);
        printf("sent gradient for %d bytes --- ", nbyte);
        toc( timer1 );
        printf("************* Task finished! *************\n");
#else
        for( i = 0; i < 10; i++ ) printf("%f\n",gradient[i]);
#endif


#if SOCKET
        close(connfd);
#endif
    }

    free(weights);
    free(data);
    free(gradient);
	clReleaseMemObject(clPackage.d_weights);
	clReleaseMemObject(clPackage.d_data);
	clReleaseMemObject(clPackage.d_gradient);

	return 0;
}
