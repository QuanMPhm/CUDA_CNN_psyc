SHELL=/bin/bash
CC=gcc
CFLAGS=-std=gnu11 -g -ggdb
LDFLAGS=-lz -lm
OBJS=../psyc.o ../utils.o ../convolutional.o ../recurrent.o ../lstm.o ../mnist.o

include ../avx.mk



default: all

mnist_demo: $(OBJS) mnist_demo.o
	$(CC) -o ../../bin/mnist_demo $(OBJS) mnist_demo.o $(LDFLAGS)
conv_demo: $(OBJS) conv_demo.o
	$(CC) -o ../../bin/conv_demo $(OBJS) conv_demo.o $(LDFLAGS)
rnn_demo: $(OBJS) rnn_demo.o
	$(CC) -o ../../bin/rnn_demo $(OBJS) rnn_demo.o $(LDFLAGS)
lstm_demo: $(OBJS) lstm_demo.o
	$(CC) -o ../../bin/lstm_demo $(OBJS) lstm_demo.o $(LDFLAGS)
char_rnn: $(OBJS) char_rnn.o
	$(CC) -o ../../bin/char_rnn $(OBJS) char_rnn.o $(LDFLAGS)
all: mnist_demo conv_demo rnn_demo lstm_demo char_rnn

clear_tests:
	rm -f $(OBJS)
	rm -f ./*.o
        
