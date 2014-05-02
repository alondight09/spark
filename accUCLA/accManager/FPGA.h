#ifndef FPGA_H
#define FPGA_H

#include "header.h"

//class Accelerator;
class FPGA {
	public:
		FPGA(int id, int status, string ip) {
			m_id = id;
			m_status = status;
			m_ip = ip;
			m_acc = NULL;
		}
		FPGA(int id, int status) {
			m_id = id;
			m_status = status;
			m_acc = NULL;
		}
		int get_id() { return m_id; }
		int get_status() { return m_status; }
		string get_ip() { return m_ip; }
		Accelerator* get_acc() { return m_acc; }

		void set_status(int status) { m_status = status; }

		void add_acc(Accelerator* acc) { 
			m_acc = acc; 
			m_status = FPGA_HAS_BIT;
		}

		void delete_acc() {
			delete m_acc;
			m_acc = NULL;
			m_status = FPGA_IDLE;
		}


	private:
		int m_id;
		int m_status;
		string m_ip;

		//list<Accelerator*> m_acc_list;
		Accelerator* m_acc;

};
#endif
