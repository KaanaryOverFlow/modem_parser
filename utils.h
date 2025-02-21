#include <stdarg.h>


void note(const char *fmt, ...);
void info(const char *fmt, ...);
void warn(const char *fmt, ...);
void write_line(const char *fmt, ...);
void die(const char *fmt);
void hexdump(const void* data, size_t size);
void pin_cpu(int num);
void print_progress_bar(size_t progress, size_t total);
void create_http_request(char *ip, unsigned int port, void *data, size_t type);

#define pd(val) note("%s : %d", #val, val)
#define pu(val) note("%s : %u", #val, val)
#define px(val) note("%s : 0x%x", #val, val)
#define pf(val) note("%s : %f", #val, val)
#define plu(val) note("%s : %lu", #val, val)
#define pld(val) note("%s : %ld", #val, val)
#define plx(val) note("%s : 0x%lx", #val, val)
#define plf(val) note("%s : %lf", #val, val)
#define ps(val) note("%s : %s", #val, val)

#define DEBUG note("Does anybody copy? it is %s @ %lu", __func__, __LINE__)


#define in() {note("Calling => %s",__func__);}
#define out() {note("Returing <= %s",__func__);}
#define FOR(iterator, number) for (size_t iterator = 0; iterator < number; iterator++)

#define HTTP(a, b, c, d) create_http_request(a, b, c, d)
