

#include "../header.h"

struct acc_request_packet {
	int acc_type;
	int task_priority;
};

void packet_to_int(struct acc_request_packet arp, int pckt_arp[2]){
	pckt_arp[0] = arp.acc_type;
	pckt_arp[1] = arp.task_priority;
}

int main(int argc, char**argv)
{
	srand(time(NULL));
	int sockfd,n;
	struct sockaddr_in serv_addr;
	struct hostent * server;
	char sendline[1000];

	if (argc != 2)
	{
		printf("usage:  udpcli <IP address>\n");
		exit(1);
	}


	while (fgets(sendline, 10000,stdin) != NULL)
	{
		// set up a udp socket
    int portno = PORT_TO_ACC_RQST;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        cout << "ERROR opening socket" << endl;
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr)) < 0){
			cout << "ERROR in connecting " << endl;
			exit(0);
		}

		// prepare packets
		struct acc_request_packet arp;
		arp.acc_type = rand()%2;
		arp.task_priority = rand()%4;

		int pckt_arp[2];
		packet_to_int(arp, pckt_arp);
		printf ("WRITE: type %d prrty %d\n", arp.acc_type, arp.task_priority);

		int n = write(sockfd, pckt_arp, sizeof(pckt_arp));
		if (n < 0) {
			cout << "ERROR in sending packets" << endl;
			exit(0);
		}

		int pckt_response[3];
		n = read(sockfd, pckt_response, sizeof(pckt_response));
		if (n < 0)
			cout << "ERROR in reading packets" << endl;
		cout << "READ: type: " << pckt_response[0];
		cout << " use acc? "  << pckt_response[1];
		cout << " wait time: " << pckt_response[2];
		cout << endl;
	}
}
