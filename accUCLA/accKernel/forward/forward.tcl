# *******************************************************************************
# Vendor: Xilinx 
# Associated Filename: run_picasso_clc.tcl
# Purpose: Commands to construct the OpenCL C matrix multiply example
# Device: Zynq 
# Revision History: July 1, 2013 - initial release
#                   November 5, 2013 - update to match Picasso User Guide
#                                                 
# *******************************************************************************
# Copyright (C) 2013 XILINX, Inc. 
# 
# This file contains confidential and proprietary information of Xilinx, Inc. and 
# is protected under U.S. and international copyright and other intellectual 
# property laws.
# 
# DISCLAIMER
# This disclaimer is not a license and does not grant any rights to the materials 
# distributed herewith. Except as otherwise provided in a valid license issued to 
# you by Xilinx, and to the maximum extent permitted by applicable law: 
# (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX 
# HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether 
# in contract or tort, including negligence, or under any other theory of 
# liability) for any loss or damage of any kind or nature related to, arising under 
# or in connection with these materials, including for any direct, or any indirect, 
# special, incidental, or consequential loss or damage (including loss of data, 
# profits, goodwill, or any type of loss or damage suffered as a result of any 
# action brought by a third party) even if such damage or loss was reasonably 
# foreseeable or Xilinx had been advised of the possibility of the same.
# 
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in any 
# application requiring fail-safe performance, such as life-support or safety 
# devices or systems, Class III medical devices, nuclear facilities, applications 
# related to the deployment of airbags, or any other applications that could lead 
# to death, personal injury, or severe property or environmental damage 
# (individually and collectively, "Critical Applications"). Customer assumes the 
# sole risk and liability of any use of Xilinx products in Critical Applications, 
# subject only to applicable laws and regulations governing limitations on product 
# liability. 
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT 
# ALL TIMES.

#*******************************************************************************

set TOP [file rootname [info script]]

# Create Project 
create_project -name ${TOP}_lpp -platform zc706-linux-uart -force

# Add host code
add_host_src -filename "../../accService/arm/accService.c" -type source

# Create a kernel for the matrix multiplication
create_kernel -id ${TOP} -type clc
add_kernel_src -id ${TOP} -filename "${TOP}.cl"

# Create a Xilinx OpenCL Kernel Binary Container
create_xclbin ${TOP}_lpp

# Select the execution target of the kernel
map_add_kernel_instance -N 1 -xclbin ${TOP}_lpp -id ${TOP} -target fpga0:OCL_REGION_0

# Compile the host code
compile_host -arch arm -cflags "-g -Wall -D FPGA_DEVICE"

# Build the system
build_system

# Package SD Card Image
#build_sdimage

exit
