/* dist_worker.c — Distributed inference worker (pipeline parallelism)
 *
 * Each node runs N layers. Master sends hidden state, worker returns output.
 * Q4 quantization: 7MB per node (half of 14MB total).
 * Network: 768 floats = 3KB per token.
 *
 * Build: gcc -O2 -o dist_worker dist_worker.c -lm -lpthread
 * Usage: ./dist_worker --master --layers 0-6 --port 8090
 *        ./dist_worker --worker --layers 6-12 --port 8091
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define N_EMBD 768
#define MAX_BUF (N_EMBD * 4 + 256)

static int g_port = 8090;
static int g_layer_start = 0, g_layer_end = 6;

static void handle_client(int fd) {
    char buf[MAX_BUF];
    int n = read(fd, buf, MAX_BUF - 1);
    if (n <= 0) { close(fd); return; }
    buf[n] = 0;
    char method[8], path[64];
    sscanf(buf, "%s %s", method, path);
    if (strcmp(path, "/health") == 0) {
        write(fd, "HTTP/1.1 200\r\nContent-Length:15\r\n\r\n{\"status\":\"ok\"}", 49);
    } else if (strcmp(path, "/forward") == 0) {
        char *body = strstr(buf, "\r\n\r\n");
        if (body) {
            body += 4;
            float *hidden = (float*)body;
            int pos = *(int*)(body + N_EMBD*4);
            /* Run layers here (Q4 matmul) — passthrough for now */
            int next = -1;
            char hdr[128];
            int hl = snprintf(hdr, sizeof(hdr), "HTTP/1.1 200\r\nContent-Length:%d\r\n\r\n", N_EMBD*4+4);
            write(fd, hdr, hl);
            write(fd, hidden, N_EMBD*4);
            write(fd, &next, 4);
        }
    } else {
        write(fd, "HTTP/1.1 404\r\n\r\n", 16);
    }
    close(fd);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i+1<argc) g_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i+1<argc)
            sscanf(argv[++i], "%d-%d", &g_layer_start, &g_layer_end);
    }
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY;
    a.sin_port = htons(g_port);
    bind(sfd, (struct sockaddr*)&a, sizeof(a));
    listen(sfd, 8);
    printf("[*] dist_worker on :%d (layers %d-%d)\n", g_port, g_layer_start, g_layer_end);
    fflush(stdout);
    while (1) { int c = accept(sfd, NULL, NULL); if (c>=0) handle_client(c); }
}
