#makefile

CC=g++
TARGET=acc_manager 

ALLSRC=AccManager.cpp main.cpp 
ALLOBJ=$(ALLSRC:.cpp=.o)

INCLUDEDIR=-I.
CPPFLAGS=$(INCLUDEDIR) 

LDFLAGS=


all: $(TARGET)

$(TARGET) : $(ALLOBJ)
	$(CC) -g $(CPPFLAGS) $(ALLOBJ) -o $@      

.cpp.o:
	$(CC) $(CPPFLAGS) -c $< -o $@ 


	
.PHONY: clean
clean:
	rm -f *.o $(TARGET)



