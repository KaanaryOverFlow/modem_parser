#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/sendfile.h>

#include "utils.h"
#include "memory.h"

#define FILE_NAME "modem.bin"

typedef struct {
	char name[12];
	unsigned int offset_in_file;
	unsigned int load_addr;
	unsigned int size;
	unsigned int crc;
	unsigned int entry_index;
} header;

void print_header(header h) {
	note("section detected : [%s] [%x] [%x] ", h.name, h.offset_in_file, h.size);
}

void swap_Endians(unsigned int value)
{
	unsigned int backup = value;
	char *pc = (char *)&value;
	FOR(i, 4)
		pc[3 - i] = (backup * 8 * i) & 0xff;
}


struct {
	int fd;	
} *ed;

void setup() {
	ed = mmap(NULL, 4096 * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (ed == MAP_FAILED) die("inital map of setup");
	ed->fd = open(FILE_NAME, O_RDONLY);
	if (ed->fd < 0) die("modem.bin couldn't opened");
}

void default_app() {
	in();
	header H;

	system("rm -rf extracted; mkdir extracted");
	FOR(i, 6) {
		lseek(ed->fd, i * 32, SEEK_SET);
		
		read(ed->fd, &H, sizeof(H));
		swap_Endians(H.offset_in_file);
		swap_Endians(H.size);
		print_header(H);

		

		lseek(ed->fd, H.offset_in_file, SEEK_SET);
		chdir("extracted");
		int fd = open(H.name, O_RDWR | O_CREAT, 0644);
		if (fd < 0) die("file couldn't opened");
		sendfile(fd, ed->fd, NULL, H.size);
		close(fd);
		chdir("..");

	}

	out();
}

void sec_main(char *param) {
	if (!param) {
		default_app();
	} else {
		die("invalid parameter");
	}

}

#ifdef SHARED_LIBRARY
void __attribute__ ((constructor)) _setup(void) {
	in();
	setup();
	default_app();
	out();
}

#else

int main(int argc, char *argv[], char *envp[]) {
	in();
	setup();
	sec_main(argv[1]);
	out();
	return 0;
}
#endif
