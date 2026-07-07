/* build_corpus.c — tokenize a text corpus with the server's GPT-2 BPE
 * tokenizer and emit a C header (train_data.h) for STE fine-tuning.
 *
 * Why a separate tool: models/gpt2.c's encode() is a hand-written
 * if-else over 10 hardcoded sentences. To scale STE training to hundreds
 * of diverse sentences we need real BPE tokenization. This reuses the
 * server's tokenizer binary + greedy longest-match encode_text algorithm
 * (already validated end-to-end via the inference server).
 *
 * Build: gcc -O2 -o scripts/build_corpus scripts/build_corpus.c -lm
 * Run:   ./scripts/build_corpus prebuilt/gpt2_tokenizer.bin corpus.txt > train_data.h
 *
 * corpus.txt format: one sentence per line (8-32 tokens worth).
 * Output format (consumed by models/gpt2.c):
 *   static const int MAX_TOK = 32;
 *   static const struct { int ids[32]; int n; } train_data[] = {
 *     { {464, 3139, ...}, 7 },
 *     ...
 *   };
 *   #define N_TRAIN (sizeof(train_data)/sizeof(train_data[0]))
 *
 * Lines that tokenize to > 32 tokens are skipped with a stderr warning.
 * Empty lines and lines starting with '#' are skipped.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define VOCAB_SIZE 50257
#define HASH_CAPACITY 524288u
#define MAX_TOK 32

typedef struct { uint32_t token_id; uint32_t len_hash; } HashEntry;
static HashEntry *g_hash;
static char  *g_vocab_tokens[VOCAB_SIZE];
static int    g_vocab_len[VOCAB_SIZE];

static uint32_t hash_bytes(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static void hash_insert(const char *s, int len, int token_id) {
    uint32_t h = hash_bytes(s, len);
    uint32_t key = ((uint32_t)len << 16) | (h & 0xFFFF);
    uint32_t idx = h % HASH_CAPACITY;
    for (;;) {
        if (g_hash[idx].token_id == 0xFFFFFFFF) {
            g_hash[idx].token_id = (uint32_t)token_id;
            g_hash[idx].len_hash = key;
            return;
        }
        idx = (idx + 1) % HASH_CAPACITY;
    }
}

static int hash_lookup(const char *s, int len) {
    uint32_t h = hash_bytes(s, len);
    uint32_t key = ((uint32_t)len << 16) | (h & 0xFFFF);
    uint32_t idx = h % HASH_CAPACITY;
    for (;;) {
        if (g_hash[idx].token_id == 0xFFFFFFFF) return -1;
        if (g_hash[idx].len_hash == key) {
            int tid = (int)g_hash[idx].token_id;
            if (g_vocab_len[tid] == len && memcmp(g_vocab_tokens[tid], s, len) == 0)
                return tid;
        }
        idx = (idx + 1) % HASH_CAPACITY;
    }
}

static void load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[!] no tokenizer at %s\n", path); exit(1); }
    char magic[4]; fread(magic, 1, 4, f);
    int vocab, n_merges, n_ctx, n_layer, n_embd;
    fread(&vocab, 4, 1, f); fread(&n_merges, 4, 1, f);
    fread(&n_ctx, 4, 1, f); fread(&n_layer, 4, 1, f); fread(&n_embd, 4, 1, f);
    for (int i = 0; i < 256; i++) {
        unsigned char bv; unsigned short ulen;
        fread(&bv, 1, 1, f); fread(&ulen, 2, 1, f);
        fseek(f, ulen, SEEK_CUR);
    }
    for (int i = 0; i < n_merges; i++) {
        int rank; unsigned short alen, blen;
        fread(&rank, 4, 1, f); fread(&alen, 2, 1, f); fseek(f, alen, SEEK_CUR);
        fread(&blen, 2, 1, f); fseek(f, blen, SEEK_CUR);
    }
    for (int i = 0; i < VOCAB_SIZE; i++) { g_vocab_tokens[i] = NULL; g_vocab_len[i] = 0; }
    for (int i = 0; i < vocab; i++) {
        int tid; unsigned short tlen;
        fread(&tid, 4, 1, f); fread(&tlen, 2, 1, f);
        g_vocab_tokens[tid] = malloc(tlen + 1);
        fread(g_vocab_tokens[tid], 1, tlen, f);
        g_vocab_tokens[tid][tlen] = '\0';
        g_vocab_len[tid] = tlen;
    }
    fclose(f);
    g_hash = malloc(HASH_CAPACITY * sizeof(HashEntry));
    memset(g_hash, 0xFF, HASH_CAPACITY * sizeof(HashEntry));
    for (int i = 0; i < VOCAB_SIZE; i++)
        if (g_vocab_tokens[i]) hash_insert(g_vocab_tokens[i], g_vocab_len[i], i);
}

/* Greedy longest-match tokenization (same as server's encode_text). */
static int encode_text(const char *text, int *out, int max_out) {
    int pos = 0, n = 0;
    int textlen = (int)strlen(text);
    while (pos < textlen && n < max_out) {
        int best_len = 0, best_id = -1;
        int max_len = textlen - pos;
        if (max_len > 32) max_len = 32;  /* GPT-2 max token byte length */
        for (int len = max_len; len >= 1; len--) {
            int tid = hash_lookup(text + pos, len);
            if (tid >= 0) { best_len = len; best_id = tid; break; }
        }
        if (best_len == 0) { pos++; continue; }  /* skip unknown byte */
        out[n++] = best_id;
        pos += best_len;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s tokenizer.bin corpus.txt > train_data.h\n", argv[0]);
        return 1;
    }
    load_tokenizer(argv[1]);
    FILE *cf = fopen(argv[2], "r");
    if (!cf) { fprintf(stderr, "[!] no corpus at %s\n", argv[2]); return 1; }

    /* Header */
    printf("/* Auto-generated by scripts/build_corpus.c — do not edit.\n");
    printf(" * Pre-tokenized training corpus for STE fine-tuning (models/gpt2.c).\n");
    printf(" * Source: %s  Tokenizer: %s\n", argv[2], argv[1]);
    printf(" * Each entry: { int ids[32]; int n; }  (n = token count, <= 32)\n");
    printf(" * Consume in gpt2.c:\n");
    printf(" *   #include \"train_data.h\"\n");
    printf(" *   for (i=0; i<N_TRAIN; i++) train_one(&train_data[i]);\n");
    printf(" */\n");
    printf("#ifndef TRAIN_DATA_H\n#define TRAIN_DATA_H\n\n");
    printf("#define MAX_TOK %d\n", MAX_TOK);
    printf("static const struct { int ids[%d]; int n; } train_data[] = {\n", MAX_TOK);

    char line[1024];
    int kept = 0, skipped = 0;
    while (fgets(line, sizeof(line), cf)) {
        /* strip trailing newline */
        int L = (int)strlen(line);
        while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';
        if (L == 0) continue;
        if (line[0] == '#') continue;
        int toks[MAX_TOK];
        int n = encode_text(line, toks, MAX_TOK);
        if (n < 4) { fprintf(stderr, "[skip] too short (%d tok): %.60s\n", n, line); skipped++; continue; }
        if (n > MAX_TOK) { fprintf(stderr, "[skip] too long (%d tok): %.60s\n", n, line); skipped++; continue; }
        printf("  { {");
        for (int i = 0; i < n; i++) printf("%s%d", i ? "," : " ", toks[i]);
        printf("}, %d },", n);
        /* comment with first 40 chars of source for readability */
        printf("  /* ");
        for (int i = 0; i < 40 && line[i]; i++) {
            char c = line[i]; if (c == '*' || c == '/') c = ' '; printf("%c", c);
        }
        printf(" */\n");
        kept++;
    }
    fclose(cf);
    printf("};\n");
    printf("#define N_TRAIN (sizeof(train_data)/sizeof(train_data[0]))\n");
    printf("#endif /* TRAIN_DATA_H */\n");
    fprintf(stderr, "[*] %d sentences kept, %d skipped → stdout\n", kept, skipped);
    return 0;
}
