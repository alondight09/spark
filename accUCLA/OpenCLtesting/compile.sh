#!/bin/bash

if [[ $OPENCL_ROOT == "" ]]; then
	OPENCL_ROOT=/opt/AMDAPP
fi	

src=$1
dst=`echo $src | cut -d '.' -f1`

echo "./cl_trans.sh ${dst}.cl"
./cl_trans.sh ${dst}.cl

echo "gcc -o $dst $src -I$OPENCL_ROOT/include -L$OPENCL_ROOT/lib -lOpenCL"
gcc -o $dst $src -I$OPENCL_ROOT/include -L$OPENCL_ROOT/lib/x86_64 -lOpenCL -lm -D TEST

