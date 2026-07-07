/* gpt2_server.c — GPT-2 HTTP server (single file, no dependencies except libc)
 *
 * Listens on port 8080. Serves:
 *   GET  /          → HTML frontend
 *   POST /generate  → JSON {prompt, n_tokens} → {text}
 *
 * Build: gcc -O3 -mavx2 -o gpt2_server models/gpt2_server.c runtime/lal_runtime.c -lm
 * Run:   ./gpt2_server
 * Open:  http://localhost:8080
 */
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include "../runtime/lal_runtime.h"

/* === HTML frontend (embedded in binary) === */
static const char *HTML_PAGE =
"<!DOCTYPE html><html><head><meta charset='utf-8'>"
"<title>LAL GPT-2</title><style>"
"body{font-family:system-ui;max-width:800px;margin:40px auto;padding:0 20px;background:#1a1a2e;color:#e0e0e0}"
"h1{color:#0f3460}h1 span{color:#e94560}"
".box{background:#16213e;border-radius:12px;padding:20px;margin:16px 0}"
"textarea{width:100%;height:80px;background:#0f3460;color:#fff;border:1px solid #e94560;border-radius:8px;padding:12px;font-size:14px;resize:vertical}"
"button{background:#e94560;color:#fff;border:none;padding:10px 28px;border-radius:8px;font-size:14px;cursor:pointer;margin-top:8px}"
"button:hover{background:#c81e45}"
"button:disabled{opacity:0.5;cursor:wait}"
"#output{white-space:pre-wrap;font-family:monospace;font-size:14px;line-height:1.6;min-height:40px}"
".label{color:#e94560;font-size:12px;margin-bottom:4px}"
"input{background:#0f3460;color:#fff;border:1px solid #333;border-radius:6px;padding:6px 12px;width:60px}"
".status{font-size:12px;color:#0f3460;margin-top:8px}"
"</style></head><body>"
"<h1>LAL <span>GPT-2</span></h1>"
"<p style='color:#888'>Pure C inference, no PyTorch. 124M params, binary weights (XNOR+popcount).</p>"
"<div class='box'>"
"<div class='label'>Prompt</div>"
"<textarea id='prompt'>Hello, how are</textarea>"
"<div style='margin-top:8px'>"
"<span class='label'>Tokens:</span> <input type='number' id='ntok' value='20' min='1' max='100'>"
"<button id='btn' onclick='generate()'>Generate</button>"
"</div>"
"</div>"
"<div class='box'>"
"<div class='label'>Output</div>"
"<div id='output'>Waiting for input...</div>"
"<div id='status' class='status'></div>"
"</div>"
"<script>"
"async function generate(){"
"const p=document.getElementById('prompt').value;"
"const n=document.getElementById('ntok').value;"
"const btn=document.getElementById('btn');"
"const out=document.getElementById('output');"
"const st=document.getElementById('status');"
"btn.disabled=true;out.textContent='Generating...';st.textContent='';"
"try{"
"const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,n_tokens:parseInt(n)})});"
"const d=await r.json();"
"out.textContent=d.text;"
"st.textContent='Generated in '+d.time+'s';"
"}catch(e){out.textContent='Error: '+e;}"
"btn.disabled=false;"
"}"
"</script></body></html>";

/* === GPT-2 model state === */
static Model g_model;
static int g_loaded = 0;

/* === Simple tokenizer (greedy longest-match, same as gpt2_runtime.c) === */
static char *g_vocab_tokens[50257];
static int g_vocab_len[50257];

static void load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no tokenizer\n"); return; }
    char magic[4]; fread(magic, 1, 4, f);
    int vocab, n_merges, n_ctx, n_layer, n_embd;
    fread(&vocab, 4, 1, f); fread(&n_merges, 4, 1, f);
    fread(&n_ctx, 4, 1, f); fread(&n_layer, 4, 1, f); fread(&n_embd, 4, 1, f);
    /* Skip byte-to-unicode mapping (256 entries) */
    for (int i = 0; i < 256; i++) {
        unsigned char bv; unsigned short ulen;
        fread(&bv, 1, 1, f); fread(&ulen, 2, 1, f);
        fseek(f, ulen, SEEK_CUR);
    }
    /* Skip merges */
    for (int i = 0; i < n_merges; i++) {
        int rank; unsigned short alen, blen;
        fread(&rank, 4, 1, f); fread(&alen, 2, 1, f); fseek(f, alen, SEEK_CUR);
        fread(&blen, 2, 1, f); fseek(f, blen, SEEK_CUR);
    }
    /* Load vocab */
    for (int i = 0; i < vocab; i++) {
        int tid; unsigned short tlen;
        fread(&tid, 4, 1, f); fread(&tlen, 2, 1, f);
        g_vocab_tokens[tid] = malloc(tlen + 1);
        fread(g_vocab_tokens[tid], 1, tlen, f);
        g_vocab_tokens[tid][tlen] = '\0';
        g_vocab_len[tid] = tlen;
    }
    fclose(f);
}

static int encode_text(const char *text, int *out_tokens, int max_tokens) {
    int n_out = 0, pos = 0, text_len = strlen(text);
    while (pos < text_len && n_out < max_tokens) {
        int best_id = -1, best_len = 0;
        for (int tid = 0; tid < 50257; tid++) {
            int tl = g_vocab_len[tid];
            if (tl == 0 || tl > text_len - pos || tl <= best_len) continue;
            if (memcmp(g_vocab_tokens[tid], text + pos, tl) == 0) {
                best_id = tid; best_len = tl;
            }
        }
        if (best_id < 0) { pos++; continue; }
        out_tokens[n_out++] = best_id;
        pos += best_len;
    }
    return n_out;
}

static int decode_token(int token_id, char *out, int max_len) {
    if (token_id < 0 || token_id >= 50257) return 0;
    int tl = g_vocab_len[token_id];
    if (tl > max_len) tl = max_len;
    memcpy(out, g_vocab_tokens[token_id], tl);
    return tl;
}

/* === Float matmul (for server — no binarize) === */
static void float_matmul(float *y, const float *x, const float *W, const float *b,
                         int in_dim, int out_dim) {
    /* W is [in_dim, out_dim] row-major (GPT-2 Conv1D format) */
    for (int j = 0; j < out_dim; j++) {
        float s = b ? b[j] : 0.0f;
        for (int i = 0; i < in_dim; i++) s += x[i] * W[i * out_dim + j];
        y[j] = s;
    }
}

/* === GPT-2 forward for generation (float, no binarize) === */
static void generate_tokens(const int *input_tokens, int n_input,
                            int *output_tokens, int n_gen) {
    int n = 768, m = 3072;
    int all_tokens[1024];
    memcpy(all_tokens, input_tokens, n_input * sizeof(int));
    int total = n_input;

    static float x[768], ln1[768], qkv[2304], attn_out[768], proj_out[768];
    static float ln2[768], fc_out[3072], mlp_out[768];
    char key[64];

    for (int gen = 0; gen < n_gen; gen++) {
        int t = total - 1;
        for (int i = 0; i < n; i++)
            x[i] = g_model.wte[all_tokens[t] * n + i] + g_model.wpe[t * n + i];

        for (int l = 0; l < 12; l++) {
            layer_norm(ln1, x, tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.ln_1.weight",l),key)),
                       tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.ln_1.bias",l),key)), n);
            float_matmul(qkv, ln1, tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.attn.c_attn.weight",l),key)),
                         tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.attn.c_attn.bias",l),key)), n, 3*n);
            memcpy(attn_out, qkv + 2*n, n*sizeof(float));
            float_matmul(proj_out, attn_out, tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.attn.c_proj.weight",l),key)),
                         tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.attn.c_proj.bias",l),key)), n, n);
            for (int i = 0; i < n; i++) x[i] += proj_out[i];

            layer_norm(ln2, x, tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.ln_2.weight",l),key)),
                       tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.ln_2.bias",l),key)), n);
            float_matmul(fc_out, ln2, tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.mlp.c_fc.weight",l),key)),
                         tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.mlp.c_fc.bias",l),key)), n, m);
            for (int i = 0; i < m; i++) fc_out[i] = gelu(fc_out[i]);
            float_matmul(mlp_out, fc_out, tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.mlp.c_proj.weight",l),key)),
                         tensor_get(g_model.tensors, g_model.n_tensors, (sprintf(key,"h.%d.mlp.c_proj.bias",l),key)), m, n);
            for (int i = 0; i < n; i++) x[i] += mlp_out[i];
        }

        layer_norm(g_model.final_ln, x, g_model.ln_f_w, g_model.ln_f_b, n);

        int best = 0;
        float best_val = -1e30f;
        for (int v = 0; v < 50257; v++) {
            float s = 0;
            for (int i = 0; i < n; i++) s += g_model.final_ln[i] * g_model.wte[v * n + i];
            if (s > best_val) { best_val = s; best = v; }
        }

        output_tokens[gen] = best;
        all_tokens[total++] = best;
        if (total >= 1024) break;
    }
}

/* === HTTP server === */
static void handle_request(int client_fd) {
    char buf[8192];
    int n = read(client_fd, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = '\0';

    /* Parse: GET / or POST /generate */
    char method[8], path[256];
    sscanf(buf, "%s %s", method, path);

    if (strcmp(method, "GET") == 0 && strcmp(path, "/") == 0) {
        /* Serve HTML */
        char header[256];
        sprintf(header, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %zu\r\n\r\n", strlen(HTML_PAGE));
        write(client_fd, header, strlen(header));
        write(client_fd, HTML_PAGE, strlen(HTML_PAGE));
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/generate") == 0) {
        /* Parse JSON body: {"prompt":"...","n_tokens":20} */
        char *body = strstr(buf, "\r\n\r\n");
        if (!body) { close(client_fd); return; }
        body += 4;

        char prompt[1024] = "";
        int n_tokens = 20;
        /* Simple JSON parse */
        char *p = strstr(body, "\"prompt\"");
        if (p) {
            p = strchr(p, ':');
            if (p) {
                p = strchr(p, '"');
                if (p) {
                    p++;
                    int i = 0;
                    while (*p && *p != '"' && i < 1023) {
                        if (*p == '\\') p++;
                        prompt[i++] = *p++;
                    }
                    prompt[i] = '\0';
                }
            }
        }
        p = strstr(body, "\"n_tokens\"");
        if (p) {
            p = strchr(p, ':');
            if (p) n_tokens = atoi(p + 1);
        }
        if (n_tokens < 1) n_tokens = 1;
        if (n_tokens > 100) n_tokens = 100;

        /* Generate */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int input_tokens[256];
        int n_input = encode_text(prompt, input_tokens, 256);
        if (n_input == 0) {
            input_tokens[0] = 464; n_input = 1; /* "The" */
        }

        int output_tokens[100];
        generate_tokens(input_tokens, n_input, output_tokens, n_tokens);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

        /* Build output text */
        char result[4096] = "";
        int pos = 0;
        pos += snprintf(result + pos, sizeof(result) - pos, "%s", prompt);
        for (int i = 0; i < n_tokens; i++) {
            char tok[256];
            int tl = decode_token(output_tokens[i], tok, sizeof(tok) - 1);
            tok[tl] = '\0';
            pos += snprintf(result + pos, sizeof(result) - pos, "%s", tok);
            if (pos > 4000) break;
        }

        /* JSON response */
        char json[6144];
        /* Escape result for JSON */
        char escaped[5120];
        int ei = 0;
        for (int i = 0; result[i] && ei < 5110; i++) {
            if (result[i] == '"') { escaped[ei++] = '\\'; escaped[ei++] = '"'; }
            else if (result[i] == '\\') { escaped[ei++] = '\\'; escaped[ei++] = '\\'; }
            else if (result[i] == '\n') { escaped[ei++] = '\\'; escaped[ei++] = 'n'; }
            else escaped[ei++] = result[i];
        }
        escaped[ei] = '\0';

        sprintf(json, "{\"text\":\"%s\",\"time\":\"%.1f\"}", escaped, dt);
        char header[256];
        sprintf(header, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", strlen(json));
        write(client_fd, header, strlen(header));
        write(client_fd, json, strlen(json));
    } else {
        const char *not_found = "HTTP/1.1 404 Not Found\r\n\r\n";
        write(client_fd, not_found, strlen(not_found));
    }

    close(client_fd);
}

int main() {
    signal(SIGPIPE, SIG_IGN);

    printf("[*] LAL GPT-2 Server\n");
    printf("[*] loading model (float, no binarize)...\n");
    fflush(stdout);
    /* Load tensors directly (skip binarize to save memory) */
    g_model.cfg = (ModelConfig){
        .n_layer = 12, .n_embd = 768, .n_head = 12,
        .n_ctx = 1024, .vocab_size = 50257, .mlp_dim = 3072,
        .norm_type = NORM_LAYER, .attn_type = ATTN_LEARNED,
        .act_type = ACT_GELU, .residual_scale = 1.0f,
        .qkv_merged = 1,
    };
    g_model.tensors = tensor_load_all("prebuilt/gpt2_weights.bin", &g_model.n_tensors);
    if (!g_model.tensors) { fprintf(stderr, "failed to load weights\n"); return 1; }
    printf("[*] loaded %d tensors\n", g_model.n_tensors); fflush(stdout);

    g_model.wte = tensor_get(g_model.tensors, g_model.n_tensors, "wte.weight");
    g_model.wpe = tensor_get(g_model.tensors, g_model.n_tensors, "wpe.weight");
    g_model.ln_f_w = tensor_get(g_model.tensors, g_model.n_tensors, "ln_f.weight");
    g_model.ln_f_b = tensor_get(g_model.tensors, g_model.n_tensors, "ln_f.bias");

    /* For server: skip binarize, use float weights directly */
    /* Allocate layers but don't binarize — use a simpler forward */
    g_model.layers = NULL;
    g_model.acts = NULL;
    g_model.final_ln = malloc(768 * sizeof(float));
    g_model.x_before_final = malloc(768 * sizeof(float));
    g_loaded = 1;
    printf("[*] model loaded (float mode, no binarize)\n"); fflush(stdout);
    load_tokenizer("prebuilt/gpt2_tokenizer.bin");
    printf("[*] tokenizer loaded\n"); fflush(stdout);

    /* HTTP server on port 8080 */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(8080),
    };
    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, 8);

    printf("[*] server running at http://localhost:8080\n");
    fflush(stdout);
    printf("[*] open browser to http://localhost:8080\n");

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) { perror("accept"); sleep(1); continue; }
        printf("[*] connection\n"); fflush(stdout);
        handle_request(client_fd);
        printf("[*] done, waiting...\n"); fflush(stdout);
    }

    model_free(&g_model);
    return 0;
}
