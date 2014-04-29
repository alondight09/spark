#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include "header.h"


class FPGA;

class Accelerator{
	public:
		//Accelerator(int id, int status, int type, FPGA* f) { 
		//	m_id = id; 
		//	m_status = status;
		//  m_type = type;
		//  m_fpga = f;
		//} 
		Accelerator(int status, int type, FPGA* f) { 
			//m_id = -1;
			m_status = status;
			m_type = type;
			m_fpga = f;
		} 
		Accelerator(int type, FPGA* f) { 
			//m_id = -1;
			m_status = ACC_IDLE;
			m_type = type;
			m_fpga = f;
		} 


		int get_status() { return m_status; }
		int get_type() { return m_type; }
		FPGA* get_fpga() { return m_fpga; }

		void set_status(int status) { m_status = status; }

	private:
		//int m_id;
		int m_status;
		int m_type;
		FPGA* m_fpga;
};

#endif
