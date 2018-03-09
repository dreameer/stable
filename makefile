INCLUDE = $(shell pkg-config --cflags opencv)
LIBS    = $(shell pkg-config --libs   opencv)

all:*.cpp 
	g++ -Wall -I $(INCLUDE) *.cpp  $(LIBS) 

