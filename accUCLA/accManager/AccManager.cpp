#include "AccManager.h"

// ************************
// helper functions
// ************************

string int2string(int i) {
	std::stringstream tmp;
	tmp << i;
	return tmp.str();
}
void pckt_to_arp(int pckt[2], AccRqstPckt & arp) {
	arp.acc_type = pckt[0];
	arp.task_priority = pckt[1];
}

void pckt_to_adp(int pckt[1], AccDonePckt & adp) {
	adp.acc_type = pckt[1];
	//adp.fpga_ip.resize(15);
	//int i;
	//for(i = 0; i < 15; i++) {
	//	if (s_arp[i+2] == '\0') break;;
	//	adp.fpga_ip[i] = s_arp[i+2];
	//}
	//adp.fpga_ip = adp.fpga_ip.substr(0,i);
}
void arpn_to_pckt(AccRspns & arpn, int pckt[3]) {
	pckt[0] = arpn.acc_type;
	pckt[1] = arpn.use_acc;
	pckt[2] = arpn.wait_time;
}

string get_string_ip( struct inaddr * addr) {
	char cpu_ip[20];
	inet_ntop(AF_INET, addr, cpu_ip, sizeof(cpu_ip));
	//cout << " ip: " << cpu_ip;
	string s_ip(cpu_ip);
	return s_ip;
}
// ************************
// EOF helper functions
// ************************

#define FPGA_IP_BASE 16
#define CPU_IP_BASE 2

AccManager::AccManager() {
	// initializing FPGAs
	for(int i = 0; i < NOF_FPGA; i++) {
		//string ip_prefix = "10.0.0.";
		//string ip_surfix = int2string(FPGA_IP_BASE+i);
		//string ip = ip_prefix+ip_surfix;
		//FPGA * f = new FPGA(i, FPGA_IDLE, ip_prefix + ip_surfix);
		FPGA * f = new FPGA(i, FPGA_IDLE);
		this->add_fpga(f);
	}
	for(int i = 0; i < NOF_FPGA; i++) {
		string ip_prefix = "10.0.0.";
		string ip_surfix = int2string(CPU_IP_BASE+i);
		string ip = ip_prefix+ip_surfix;
		cpu_ip_to_fpga_id[ip] = i;
	}
}


void AccManager::add_fpga(FPGA* f) {
	m_fpga_list.push_back(f);
	m_id_2_fpga[f->get_id()] = f;
}

int AccManager::get_idle_fpga_id() {
	for(list<FPGA*>::iterator it = m_fpga_list.begin(); it != m_fpga_list.end(); it++) {
		if ((*it)->get_status() == FPGA_IDLE)
			return (*it)->get_id();
	}
	return -1;
}

void AccManager::init_socket() {
	//	listen_to_port(socketfd_scheduler, PORT_TO_SCHEDULER);
	listen_to_port(sockfd_acc_rqst, PORT_TO_ACC_RQST);
	listen_to_port(sockfd_acc_done, PORT_TO_ACC_DONE);
}

void AccManager::listen_to_port(int & sockfd, int port) {
	sockfd = 0;
	socklen_t buf_size = 0;
	socklen_t size = sizeof(buf_size);
	struct sockaddr_in serv_addr; 

	//sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) {
		cout << "ERROR opening socket" << endl;
		exit(0);
	}

	// set to non-blocking read
	int flags = fcntl(sockfd, F_GETFL);
	flags |= O_NONBLOCK;
	fcntl(sockfd, F_SETFL, flags);

	bzero((char*) &serv_addr, sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(port); 

	if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0 ) {
		cout << "ERROR on binding" << endl;
		exit(0);
	}

	listen(sockfd, NOF_SLAVE*CPU_PER_SLAVE);

	cout << "udp socket is listening port: " << port << endl;
}

/*
	 void AccManager::connect_to_slave(int & sockfd) {
	 struct sockaddr_in clnt_addr;
	 socklen_t clilen = sizeof(clnt_addr); 
	 for(int i = 0; i < NOF_SLAVE; i++) {
	 int newsockfd = accept(sockfd, (struct sockaddr *) &clnt_addr, &clilen);
	 if (newsockfd < 0) {
	 cout << "ERROR on accepting from slaves" << endl;
	 exit(1);
	 } else {
	 char ip[20];
	 inet_ntop(AF_INET, (struct inaddr*)&clnt_addr.sin_addr, ip, sizeof(ip));
	 cout << "Accept slave from ip " << ip << " port: " << ntohs(clnt_addr.sin_port) << endl; 
	 }
	 connected_sockfd[i] = newsockfd;
	 }
	 }
	 */
void AccManager::add_to_acc_rqst_list(AccRqst ar) {
	int priority = ar.arp.task_priority;
	map<int, list<AccRqst> >::iterator it = acc_rqst_list.find(priority);
	if (it != acc_rqst_list.end()) {
		list<AccRqst> oldlist = it->second;
		oldlist.push_back(ar);
		acc_rqst_list[priority] = oldlist;
	} else {
		list<AccRqst> newlist;
		newlist.push_back(ar);
		acc_rqst_list[priority] = newlist;
	}
	return;
}

void AccManager::run() {

	while(1) {

		acc_rqst_list.clear();

		// step 0: retrieve all udp packets from spark map tasks
		while(1) {
			struct sockaddr_in clnt_addr;
			socklen_t clilen = sizeof(clnt_addr); 

			int newsockfd = accept(sockfd_acc_rqst, (struct sockaddr *)&clnt_addr, &clilen);
			if (newsockfd < 0)
				break;
			else {
				int pckt_arp[2];
				while( read(newsockfd, pckt_arp, sizeof(pckt_arp)) < 0) ;
				AccRqstPckt arp;
				pckt_to_arp(pckt_arp, arp);

				AccRqst ar;
				ar.arp = arp;
				ar.sockfd = newsockfd;
				ar.addr = clnt_addr;
				
				add_to_acc_rqst_list(ar);
			}
		}

#if SLOW
		cout << "--Acc_Request--" << endl;
#endif
		for(map<int, list<AccRqst> >::reverse_iterator it = acc_rqst_list.rbegin();
				it != acc_rqst_list.rend(); it++) {
			cout << "request  priority: " << it->first << endl;
			list<AccRqst> list_ar = it->second;
			for(list<AccRqst>::iterator lar_it = list_ar.begin();
					lar_it!= list_ar.end(); lar_it++) {
				cout << "    acc type " << (*lar_it).arp.acc_type;
				cout << endl;
			}
		}
#if SLOW
		cout << "--EOF Acc_Request--" << endl << endl;

		// step 1: process all received packets
		cout << "--Acc_Response--" << endl;
#endif
		for(map<int, list<AccRqst> >::reverse_iterator it = acc_rqst_list.rbegin();
				it != acc_rqst_list.rend(); it++) {
			cout << "response  priority: " << it->first << endl;
			list<AccRqst> list_ar = it->second;
			for(list<AccRqst>::iterator lar_it = list_ar.begin();
					lar_it!= list_ar.end(); lar_it++) {

				// process a single request 
				AccRqst ar = (*lar_it);
				cout << "    acc type: " << ar.arp.acc_type;

				AccRspns arpn;
				arpn.acc_type = ar.arp.acc_type;
				// default value
				arpn.use_acc = 2;
				arpn.wait_time = 0;

				// find its local fpga
				string cpu_ip = get_string_ip((struct inaddr *)&ar.addr.sin_addr);
				cout << "; ip: " << cpu_ip;

				if (cpu_ip_to_fpga_id.find(cpu_ip) == cpu_ip_to_fpga_id.end()) {
					cout << "\nERROR: cannot find FPGAs attached to this cpu ip" << endl;
				} else {
					int fpga_id = cpu_ip_to_fpga_id[cpu_ip];
					FPGA* fpga = get_fpga_from_id(fpga_id);
					assert(fpga);

					if (fpga->get_status() == FPGA_IDLE) {
						// program
						Accelerator * new_acc = new Accelerator(ar.arp.acc_type, fpga);
						add_acc(fpga, new_acc);

						new_acc->set_status(ACC_BUSY);
						fpga->set_status(FPGA_HAS_BIT);

						arpn.use_acc = 1;
						arpn.wait_time = 0; 

					} else if (fpga->get_status() == FPGA_HAS_BIT ) {
						Accelerator * cur_acc = fpga->get_acc();
						if(cur_acc->get_status() == ACC_IDLE &&
								cur_acc->get_type() == ar.arp.acc_type) {
							// use acc directly
							arpn.use_acc = 1;
							arpn.wait_time = 0; 

							cur_acc -> set_status(ACC_BUSY);
						} else if ( cur_acc->get_status() == ACC_IDLE &&
								cur_acc->get_type () != ar.arp.acc_type) {
							// re-program
							delete_acc(fpga);
							Accelerator * new_acc = new Accelerator(ar.arp.acc_type, fpga);
							add_acc(fpga, new_acc);

							new_acc->set_status(ACC_BUSY);
							fpga->set_status(FPGA_HAS_BIT);

							arpn.use_acc = 1;
							arpn.wait_time = 0; 
						} else if (cur_acc->get_status() == ACC_BUSY) {
							arpn.use_acc = 2;
							//arpn.wait_time = ACC_EXE_TIME;
							arpn.wait_time = 0;
						} else {
							cout << "ERROR: unknow status" << endl;
						}
					} else {
						cout << "ERROR: FPGA LOST?!" << endl;
					}
				}
				// send back response
				send_acc_request_response(ar, arpn);
                cout << "; socketfd: " << ar.sockfd;
				cout << "; use_acc: " << arpn.use_acc << "; wait_time: " << arpn.wait_time << endl;
			}
		}
#if SLOW
		cout << "--EOF Acc_Response--" << endl << endl;

		// step 2: process acc done packets
		cout << "--Mark_Acc_Done--" << endl;
#endif
		while(1) {
			struct sockaddr_in clnt_addr;
			socklen_t clilen = sizeof(clnt_addr); 

			int newsockfd = accept(sockfd_acc_done, (struct sockaddr *)&clnt_addr, &clilen);
			if (newsockfd < 0)
				break;
			else {
				int pckt_adp[1];
				while( read(newsockfd, pckt_adp, sizeof(pckt_adp)) < 0) ;
				AccDonePckt adp;
				pckt_to_adp(pckt_adp, adp);
				adp.addr = clnt_addr;

				process_acc_done(adp);	
			}
		}
#if SLOW
		cout << "--EOF Mark_Acc_Done--" << endl << endl;

		cout << "--WAIT 5 Seconds--" << endl << endl;;
		sleep(5);
#endif
	}
}

FPGA* AccManager::find_idle_fpga_from_list() {
	for(list<FPGA*>::iterator it = m_fpga_list.begin();
			it != m_fpga_list.end(); it++) {
		if((*it)->get_status() == FPGA_IDLE)
			return (*it);
	}
	return NULL;
}

void AccManager::send_acc_request_response(AccRqst ar, AccRspns arpn) {
	int arpn_pckt[3];
	arpn_to_pckt(arpn, arpn_pckt);
	write(ar.sockfd, arpn_pckt, sizeof(arpn_pckt));
	//char ip[20];
	//inet_ntop(AF_INET, (struct inaddr*)&(ar.addr.sin_addr), ip, sizeof(ip));
	//cout << " ip: " << ip;
	//cout << " port: " << ntohs(ar.addr.sin_port);
	//cout << " mesg: " << s_arpn << endl;
}

/*
	 bool compare_ip(string str1, string str2){
	 cout << "compare" << endl;
	 cout << str1 << "." << endl;
	 cout << str2 << "." << endl;
	 cout << str1.size() << ":" << str2.size() << endl;;

	 int size1 = str1.size();
	 int size2 = str2.size();
	 bool tmp = true;
	 for(int i = 0; i < (size1<size2?size1:size2); i++){
	 if (str1[i] == '\0' && str2[i] == '\0')
	 break;
	 if (str1[i] != str2[i]) {
	 tmp = false;
	 break;
	 }
	 }
	 cout << "cmp result: " << tmp << endl;
	 return tmp;
	 }
	 */

void AccManager::process_acc_done(AccDonePckt adp){
	bool found = false;
	string cpu_ip = get_string_ip((struct inaddr *)&adp.addr.sin_addr);
	if (cpu_ip_to_fpga_id.find(cpu_ip) == cpu_ip_to_fpga_id.end()) {
		cout << "ERROR: cannot find FPGAs attached to this cpu" << endl;
	} else {
		int fpga_id = cpu_ip_to_fpga_id[cpu_ip];
		FPGA* fpga = get_fpga_from_id(fpga_id);
		assert(fpga);

		found = true;
		fpga->get_acc()->set_status(ACC_IDLE);

		cout << "    set ip: " << cpu_ip << " 's fpga acc to ACC_IDLE" << endl;
	}

	if (!found) {
		cout << "ERROR: get an unknown acc done request:" << endl; 
		cout << "acc_type "<< adp.acc_type <<  " cpu_ip: " << cpu_ip << endl;
		cout << "Ignoring the request" << endl << endl;
		//exit(0);
	}

}
