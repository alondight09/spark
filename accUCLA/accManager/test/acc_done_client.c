
#include "../header.h"

int main(int argc, char**argv)
{
	srand(time(NULL));
	int sockfd,n;

	if (argc != 2)
	{
		printf("usage:  udpcli <IP address>\n");
		exit(1);
	}

	struct sockaddr_in serv_addr;
	struct hostent * server;

	while (1)
	{
		int x;
		cout << "type the acc type you want to free: " << endl;
		scanf("%d", &x);
		printf ("free acc: type %d", x);
		// set up a udp socket
		int portno = PORT_TO_ACC_DONE;
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


		int pckt_adp[1];
		pckt_adp[0] = x;
		int n = write(sockfd, pckt_adp, sizeof(pckt_adp));
		if (n < 0) {
			cout << "ERROR in sending packets" << endl;
			exit(0);
		}
	}
}
