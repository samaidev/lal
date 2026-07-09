/* deep_quality_test.c — Multi-layer cumulative quality test
 *
 * Simulates GPT-2 inference through 12 layers:
 *   x → LayerNorm → Q8 matmul → GELU → LayerNorm → Q8 matmul → residual
 *   × 12 layers + final LM head
 *
 * Measures how quality degrades cumulatively vs float reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define DIM 768
#define N_LAYERS 12
#define VOCAB 50257
#define N_PROMPTS 8

/* ---- LayerNorm ---- */
static void layer_norm(float *out, const float *x, const float *g, const float *b,
                       int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = (x[i] - mean) * inv * g[i] + b[i];
}

/* ---- GELU ---- */
static float gelu_f(float x) { return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }
static void gelu(float *x, int n) { for (int i=0;i<n;i++) x[i]=gelu_f(x[i]); }

/* ---- Float matmul (column-major W) ---- */
static void matmul_f(float *y, const float *W, const float *x, int ind, int outd) {
    for (int j=0;j<outd;j++) {
        float s=0;
        for (int i=0;i<ind;i++) s+=W[(size_t)i*outd+j]*x[i];
        y[j]=s;
    }
}

/* ---- Q8 per-row matmul (quantize on the fly, like LAL server) ---- */
static void matmul_q8(float *y, const float *W, const float *x, int ind, int outd) {
    for (int j=0;j<outd;j++) {
        /* Per-row quantization */
        float mx=0;
        for (int i=0;i<ind;i++) mx=fmaxf(mx,fabsf(W[(size_t)i*outd+j]));
        float sc=mx/127.0f; if(sc<1e-8f)sc=1e-8f;
        float inv=1.0f/sc;
        int32_t dot=0;
        for (int i=0;i<ind;i++) {
            int q=(int)lroundf(W[(size_t)i*outd+j]*inv);
            if(q>127)q=127; if(q<-127)q=-127;
            dot += q*(int)lroundf(x[i]/sc);  /* quantize x with same scale */
        }
        y[j]=(float)dot*sc*sc;  /* scale_x * scale_w */
    }
}

/* Actually LAL quantizes x separately with its own scale. Let me do it properly. */
static void matmul_q8_proper(float *y, const float *W, const float *x, int ind, int outd) {
    /* Quantize x */
    float x_max=0;
    for (int i=0;i<ind;i++) x_max=fmaxf(x_max,fabsf(x[i]));
    float x_sc=x_max/127.0f; if(x_sc<1e-8f)x_sc=1e-8f;
    float x_inv=1.0f/x_sc;

    for (int j=0;j<outd;j++) {
        /* Per-row weight quantization */
        float w_max=0;
        for (int i=0;i<ind;i++) w_max=fmaxf(w_max,fabsf(W[(size_t)i*outd+j]));
        float w_sc=w_max/127.0f; if(w_sc<1e-8f)w_sc=1e-8f;
        float w_inv=1.0f/w_sc;
        int32_t dot=0, w_sum=0;
        for (int i=0;i<ind;i++) {
            int wq=(int)lroundf(W[(size_t)i*outd+j]*w_inv);
            if(wq>127)wq=127; if(wq<-127)wq=-127;
            w_sum += wq;
            int xq=(int)lroundf(x[i]*x_inv)+128; /* uint8 offset */
            if(xq>255)xq=255; if(xq<0)xq=0;
            dot += xq * wq;
        }
        dot -= 128 * w_sum; /* zero-point removal */
        y[j] = (float)dot * x_sc * w_sc;
    }
}

/* ---- Softmax ---- */
static void softmax(float *x, int n) {
    float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m) m=x[i];
    float s=0; for(int i=0;i<n;i++) { x[i]=expf(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=s;
}

/* ---- Cosine similarity ---- */
static float cos_sim(const float *a, const float *b, int n) {
    float dot=0,na=0,nb=0;
    for(int i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return dot/(sqrtf(na)*sqrtf(nb)+1e-12f);
}

/* ---- Correlation ---- */
static float corr(const float *a, const float *b, int n) {
    float ma=0,mb=0; for(int i=0;i<n;i++){ma+=a[i];mb+=b[i];} ma/=n;mb/=n;
    float num=0,da=0,db=0;
    for(int i=0;i<n;i++){num+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)*(a[i]-ma);db+=(b[i]-mb)*(b[i]-mb);}
    return num/sqrtf(da*db+1e-12f);
}

/* ---- Random normal (Box-Muller) ---- */
static float randn(void) {
    float u1=rand()/(float)RAND_MAX, u2=rand()/(float)RAND_MAX;
    if(u1<1e-10f)u1=1e-10f;
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2);
}

int main(void) {
    srand(42);
    printf("=== Deep Quality Test: %d-layer transformer (GPT-2-like) ===\n", N_LAYERS);
    printf("dim=%d, vocab=%d, prompts=%d\n\n", DIM, VOCAB, N_PROMPTS);

    /* Allocate: 12 layers × (c_fc [dim,4*dim] + c_proj [4*dim,dim]) + LN params */
    int hd = 4*DIM; /* hidden dim = 3072 */
    float **W_fc = malloc(N_LAYERS*sizeof(float*));
    float **W_proj = malloc(N_LAYERS*sizeof(float*));
    float **ln1_g = malloc(N_LAYERS*sizeof(float*));
    float **ln1_b = malloc(N_LAYERS*sizeof(float*));
    float **ln2_g = malloc(N_LAYERS*sizeof(float*));
    float **ln2_b = malloc(N_LAYERS*sizeof(float*));

    /* GPT-2 weight init: N(0, 0.02/sqrt(dim)) for fc, 1/sqrt(dim) for proj */
    for (int l=0;l<N_LAYERS;l++) {
        W_fc[l] = malloc((size_t)DIM*hd*sizeof(float));
        W_proj[l] = malloc((size_t)hd*DIM*sizeof(float));
        ln1_g[l] = malloc(DIM*sizeof(float));
        ln1_b[l] = malloc(DIM*sizeof(float));
        ln2_g[l] = malloc(DIM*sizeof(float));
        ln2_b[l] = malloc(DIM*sizeof(float));
        /* Init weights */
        float s1 = 0.02f/sqrtf(DIM);
        for (size_t i=0;i<(size_t)DIM*hd;i++) W_fc[l][i] = randn()*s1;
        float s2 = 1.0f/sqrtf(hd);
        for (size_t i=0;i<(size_t)hd*DIM;i++) W_proj[l][i] = randn()*s2;
        for (int i=0;i<DIM;i++) { ln1_g[l][i]=1; ln1_b[l][i]=0; ln2_g[l][i]=1; ln2_b[l][i]=0; }
    }

    /* LM head: [dim, vocab] — only test first 1000 for speed */
    int test_vocab = 1000;
    float *W_lm = malloc((size_t)DIM*test_vocab*sizeof(float));
    float s3 = 0.02f/sqrtf(DIM);
    for (size_t i=0;i<(size_t)DIM*test_vocab;i++) W_lm[i]=randn()*s3;

    /* Buffers */
    float *x_f = malloc(DIM*sizeof(float));
    float *x_q = malloc(DIM*sizeof(float));
    float *h_f = malloc(hd*sizeof(float));
    float *h_q = malloc(hd*sizeof(float));
    float *tmp = malloc(DIM*sizeof(float));
    float *logits_f = malloc(test_vocab*sizeof(float));
    float *logits_q = malloc(test_vocab*sizeof(float));
    float *ln_out_f = malloc(DIM*sizeof(float));
    float *ln_out_q = malloc(DIM*sizeof(float));
    float *fc_out_f = malloc(DIM*sizeof(float));
    float *fc_out_q = malloc(DIM*sizeof(float));
    float *ln_f = malloc(DIM*sizeof(float));
    float *ln_q = malloc(DIM*sizeof(float));
    float *fg = calloc(DIM, sizeof(float));
    float *fb = calloc(DIM, sizeof(float));
    for(int i=0;i<DIM;i++) fg[i]=1;

    printf("%-6s %-8s %-10s %-10s %-10s %-10s\n",
           "Layer", "Prompt", "CosSim", "Corr", "MaxRErr", "MeanRErr");
    printf("------ -------- ---------- ---------- ---------- ----------\n");

    float total_cos=0, total_corr=0;
    int total_count=0;

    for (int p=0;p<N_PROMPTS;p++) {
        /* Init prompt: random normalized vector (like LayerNorm output) */
        for (int i=0;i<DIM;i++) x_f[i]=randn(); x_f[0]=(float)(p+1)*2.7f;
        memcpy(x_q, x_f, DIM*sizeof(float));

        float cos_at_0 = cos_sim(x_f, x_q, DIM);

        for (int l=0;l<N_LAYERS;l++) {
            /* Float: residual = x + proj(gelu(fc(ln1(x)))) */
            layer_norm(ln_out_f, x_f, ln1_g[l], ln1_b[l], DIM);
            matmul_f(fc_out_f, W_fc[l], ln_out_f, DIM, DIM);
            gelu(fc_out_f, DIM);
            for (int i=0;i<DIM;i++) x_f[i] += fc_out_f[i];

            /* Q8 path */
            layer_norm(ln_out_q, x_q, ln1_g[l], ln1_b[l], DIM);
            matmul_q8_proper(fc_out_q, W_fc[l], ln_out_q, DIM, DIM);
            gelu(fc_out_q, DIM);
            for (int i=0;i<DIM;i++) x_q[i] += fc_out_q[i];

            /* Metrics */
            float cs = cos_sim(x_f, x_q, DIM);
            float cr = corr(x_f, x_q, DIM);
            float maxe=0, meane=0;
            for (int i=0;i<DIM;i++) {
                float r=fabsf(x_f[i]); if(r<1e-6f)r=1e-6f;
                float e=fabsf(x_f[i]-x_q[i])/r;
                if(e>maxe)maxe=e; meane+=e;
            }
            meane/=DIM;

            if (l==0 || l==N_LAYERS-1 || l==N_LAYERS/2) {
                printf("%-6d %-8d %-10.6f %-10.6f %-10.4f %-10.4f\n",
                       l+1, p+1, cs, cr, maxe, meane);
            }

            total_cos += cs; total_corr += cr; total_count++;
        }

        /* LM head: compare logits */
        layer_norm(ln_f, x_f, fg, fb, DIM);
        layer_norm(ln_q, x_q, fg, fb, DIM);
        matmul_f(logits_f, W_lm, ln_f, DIM, test_vocab);
        matmul_q8_proper(logits_q, W_lm, ln_q, DIM, test_vocab);
        softmax(logits_f, test_vocab);
        softmax(logits_q, test_vocab);
        float logit_cs = cos_sim(logits_f, logits_q, test_vocab);
        /* Top-1 agreement */
        int top_f=0, top_q=0;
        for(int i=1;i<test_vocab;i++){
            if(logits_f[i]>logits_f[top_f])top_f=i;
            if(logits_q[i]>logits_q[top_q])top_q=i;
        }
        int agree = (top_f==top_q);
        /* Top-5 overlap */
        int idx_f[5],idx_q[5]; float cf[5],cq[5];
        for(int k=0;k<5;k++){cf[k]=-1e9;cq[k]=-1e9;idx_f[k]=0;idx_q[k]=0;}
        for(int i=0;i<test_vocab;i++){
            for(int k=0;k<5;k++){
                if(logits_f[i]>cf[k]){
                    for(int j=4;j>k;j--){cf[j]=cf[j-1];idx_f[j]=idx_f[j-1];}
                    cf[k]=logits_f[i];idx_f[k]=i;break;
                }
            }
            for(int k=0;k<5;k++){
                if(logits_q[i]>cq[k]){
                    for(int j=4;j>k;j--){cq[j]=cq[j-1];idx_q[j]=idx_q[j-1];}
                    cq[k]=logits_q[i];idx_q[k]=i;break;
                }
            }
        }
        int top5_overlap=0;
        for(int i=0;i<5;i++) for(int j=0;j<5;j++) if(idx_f[i]==idx_q[j]) top5_overlap++;

        printf("  LM head: cos_sim=%.6f, top1=%s, top5_overlap=%d/5\n",
               logit_cs, agree?"MATCH":"DIFF", top5_overlap);
        printf("\n");
    }

    printf("====== Summary ======\n");
    printf("Avg cosine similarity across all layers:  %.6f\n", total_cos/total_count);
    printf("Avg correlation across all layers:        %.6f\n", total_corr/total_count);

    free(W_lm);free(x_f);free(x_q);free(h_f);free(h_q);free(tmp);
    free(logits_f);free(logits_q);free(ln_out_f);free(ln_out_q);
    free(fc_out_f);free(fc_out_q);free(ln_f);free(ln_q);free(fg);free(fb);
    for(int l=0;l<N_LAYERS;l++){
        free(W_fc[l]);free(W_proj[l]);free(ln1_g[l]);free(ln1_b[l]);
        free(ln2_g[l]);free(ln2_b[l]);
    }
    free(W_fc);free(W_proj);free(ln1_g);free(ln1_b);free(ln2_g);free(ln2_b);
    return 0;
}