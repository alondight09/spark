
TARGET=accService

curr_dir=$(shell pwd)
src_dir=$(TARGET)_host/pkg/arm/zc706/bin

all: $(src_dir)/$(TARGET)_host.exe

$(src_dir)/$(TARGET)_host.exe: $(TARGET).c
	rm -rf ${TARGET}_host
	source /space/setenv_farm.sh; picasso ${TARGET}.tcl

