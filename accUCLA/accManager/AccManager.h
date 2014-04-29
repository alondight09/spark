#ifndef ACCMANAGER_H
#define ACCMANAGER_H

#include "header.h"
#include "Accelerator.h"
#include "FPGA.h" 


typedef struct {
	int acc_type;
	int task_priority;
} AccRqstPckt;

typedef struct {
	AccRqstPckt arp;
	struct sockaddr_in addr;
	int sockfd;
} AccRqst;

typedef struct {
	int acc_type;
	struct sockaddr_in addr;
} AccDonePckt;

typedef struct {
	int acc_type;
	int use_acc;
//	string fpga_ip;
	int wait_time; // in terms of seconds
} AccRspns;

class AccManager {
	public:
		AccManager();

		void add_fpga(FPGA* f);
		void add_acc(FPGA* f, Accelerator* acc ) { f->add_acc(acc); }
		void delete_acc(FPGA* f) { f->delete_acc(); }

		int get_idle_fpga_id();
		FPGA* get_fpga_from_id (int id) { 
			if (m_id_2_fpga.find(id) == m_id_2_fpga.end())
				return NULL;
			else
				return m_id_2_fpga[id]; 
		}

		void init_socket();
		void run();

	
		FPGA* find_idle_fpga_from_list();
		void add_to_acc_rqst_list(AccRqst ar);
		void send_acc_request_response(AccRqst ar, AccRspns arpn);
		void process_acc_done(AccDonePckt adp);


	private:
		list<FPGA*> m_fpga_list;
		map<int, FPGA*> m_id_2_fpga;
		map<string, int> cpu_ip_to_fpga_id;

		map<int, list<AccRqst> > acc_rqst_list;

		int sockfd_scheduler;
		int sockfd_acc_rqst;
		int sockfd_acc_done;

		int connected_sockfd[NOF_SLAVE];

		void listen_to_port(int & sockfd, int port);
	
};
#endif
