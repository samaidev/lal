/* deep_q4_vs_q8.c — Multi-layer cumulative quality: Q4 vs Q8 vs Float
 *
 * Simulates GPT-2 inference through 12 layers for both Q4 and Q8,
 * comparing cumulative degradation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define DIM 768
#define N_LAYERS 12
#define VOCAB_TEST 1000
#define N_PROMPTS 8

static void layer_norm(float *out, const float *x, const float *g, const float *b, int n) {
    float mean=0,var=0;
    for(int i=0;i<n;i++) mean+=x[i]; mean/=n;
    for(int i=0;i<n;i++) var+=(x[i]-mean)*(x[i]-mean); var/=n;
    float inv=1.0f/sqrtf(var+1e-5f);
    for(int i=0;i<n;i++) out[i]=(x[i]-mean)*inv*g[i]+b[i];
}

static float gelu_f(float x){return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));}
static void gelu_v(float *x,int n){for(int i=0;i<n;i++) x[i]=gelu_f(x[i]);}

static void matmul_f(float *y, const float *W, const float *x, int ind, int outd) {
    for(int j=0;j<outd;j++){float s=0;for(int i=0;i<ind;i++) s+=W[(size_t)i*outd+j]*x[i]; y[j]=s;}
}

/* Q8 per-row matmul (LAL style: per-row scale, uint8 x, int8 w) */
static void matmul_q8(float *y, const float *W, const float *x, int ind, int outd) {
    float x_max=0; for(int i=0;i<ind;i++) x_max=fmaxf(x_max,fabsf(x[i]));
    float x_sc=x_max/127.0f; if(x_sc<1e-8f)x_sc=1e-8f;
    for(int j=0;j<outd;j++){
        float w_max=0; for(int i=0;i<ind;i++) w_max=fmaxf(w_max,fabsf(W[(size_t)i*outd+j]));
        float w_sc=w_max/127.0f; if(w_sc<1e-8f)w_sc=1e-8f;
        int32_t dot=0,w_sum=0;
        for(int i=0;i<ind;i++){
            int wq=(int)lroundf(W[(size_t)i*outd+j]/w_sc);
            if(wq>127)wq=127;if(wq<-127)wq=-127;
            w_sum+=wq;
            int xq=(int)lroundf(x[i]/x_sc)+128;
            if(xq>255)xq=255;if(xq<0)xq=0;
            dot+=xq*wq;
        }
        dot-=128*w_sum;
        y[j]=(float)dot*x_sc*w_sc;
    }
}

/* Q4 per-row matmul */
static void matmul_q4(float *y, const float *W, const float *x, int ind, int outd) {
    float x_max=0; for(int i=0;i<ind;i++) x_max=fmaxf(x_max,fabsf(x[i]));
    float x_sc=x_max/127.0f; if(x_sc<1e-8f)x_sc=1e-8f;
    for(int j=0;j<outd;j++){
        float w_max=0; for(int i=0;i<ind;i++) w_max=fmaxf(w_max,fabsf(W[(size_t)i*outd+j]));
        float w_sc=w_max/7.0f; if(w_sc<1e-8f)w_sc=1e-8f;
        int32_t dot=0,w_sum=0;
        for(int i=0;i<ind;i++){
            int wq=(int)lroundf(W[(size_t)i*outd+j]/w_sc);
            if(wq>7)wq=7;if(wq<-7)wq=-7;
            w_sum+=wq;
            int xq=(int)lroundf(x[i]/x_sc)+128;
            if(xq>255)xq=255;if(xq<0)xq=0;
            dot+=xq*wq;
        }
        dot-=128*w_sum;
        y[j]=(float)dot*x_sc*w_sc;
    }
}

static void softmax(float *x,int n){
    float m=x[0];for(int i=1;i<n;i++)if(x[i]>m)m=x[i];
    float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    for(int i=0;i<n;i++)x[i]/=s;
}

static float cos_sim(const float *a,const float *b,int n){
    float d=0,na=0,nb=0;
    for(int i=0;i<n;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return d/(sqrtf(na)*sqrtf(nb)+1e-12f);
}

static float corr_f(const float *a,const float *b,int n){
    float ma=0,mb=0;for(int i=0;i<n;i++){ma+=a[i];mb+=b[i];}ma/=n;mb/=n;
    float num=0,da=0,db=0;
    for(int i=0;i<n;i++){num+=(a[i]-ma)*(b[i]-mb);da+=(a[i]-ma)*(a[i]-ma);db+=(b[i]-mb)*(b[i]-mb);}
    return num/sqrtf(da*db+1e-12f);
}

static float randn(void){
    float u1=rand()/(float)RAND_MAX,u2=rand()/(float)RAND_MAX;
    if(u1<1e-10f)u1=1e-10f;
    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2);
}

int main(void){
    srand(42);
    printf("=== Deep Quality: Q4 vs Q8 vs Float — %d-layer GPT-2 ===\n\n", N_LAYERS);

    /* Allocate 12 layers of DIMxDIM weights */
    float **W = malloc(N_LAYERS*sizeof(float*));
    float **ln_g = malloc(N_LAYERS*sizeof(float*));
    float **ln_b = malloc(N_LAYERS*sizeof(float*));
    float s = 0.02f/sqrtf(DIM);
    for(int l=0;l<N_LAYERS;l++){
        W[l]=malloc((size_t)DIM*DIM*sizeof(float));
        ln_g[l]=malloc(DIM*sizeof(float));
        ln_b[l]=malloc(DIM*sizeof(float));
        for(size_t i=0;i<(size_t)DIM*DIM;i++) W[l][i]=randn()*s;
        for(int i=0;i<DIM;i++){ln_g[l][i]=1;ln_b[l][i]=0;}
    }

    /* LM head [DIM, VOCAB_TEST] */
    float *W_lm = malloc((size_t)DIM*VOCAB_TEST*sizeof(float));
    float s2=0.02f/sqrtf(DIM);
    for(size_t i=0;i<(size_t)DIM*VOCAB_TEST;i++) W_lm[i]=randn()*s2;

    /* Buffers */
    float *x_f=malloc(DIM*sizeof(float)),*x_q8=malloc(DIM*sizeof(float)),*x_q4=malloc(DIM*sizeof(float));
    float *ln_f=malloc(DIM*sizeof(float)),*ln_q8=malloc(DIM*sizeof(float)),*ln_q4=malloc(DIM*sizeof(float));
    float *fc_f=malloc(DIM*sizeof(float)),*fc_q8=malloc(DIM*sizeof(float)),*fc_q4=malloc(DIM*sizeof(float));
    float *lg_f=malloc(VOCAB_TEST*sizeof(float)),*lg_q8=malloc(VOCAB_TEST*sizeof(float)),*lg_q4=malloc(VOCAB_TEST*sizeof(float));
    float *fg=calloc(DIM,sizeof(float)),*fb=calloc(DIM,sizeof(float));
    for(int i=0;i<DIM;i++)fg[i]=1;

    int q8_top1_match=0, q4_top1_match=0;
    int q8_top5_total=0, q4_top5_total=0;
    float q8_logit_cos=0, q4_logit_cos=0;
    float q8_layer_cos=0, q4_layer_cos=0;
    float q8_layer_maxre=0, q4_layer_maxre=0;
    float q8_layer_meane=0, q4_layer_meane=0;
    int layer_count=0;

    printf("%-6s %-6s | %-8s %-8s | %-10s %-10s | %-10s %-10s | %-6s %-6s\n",
           "Layer","Prompt","Q8 Cos","Q4 Cos","Q8 Corr","Q4 Corr","Q8 MaxRE","Q4 MaxRE","Q8 T1","Q4 T1");
    printf("------ ------ | -------- -------- | ---------- ---------- | ---------- ---------- | ------ ------\n");

    for(int p=0;p<N_PROMPTS;p++){
        for(int i=0;i<DIM;i++) x_f[i]=randn(); x_f[0]=(float)(p+1)*2.7f;
        memcpy(x_q8,x_f,DIM*sizeof(float));
        memcpy(x_q4,x_f,DIM*sizeof(float));

        for(int l=0;l<N_LAYERS;l++){
            /* Float */
            layer_norm(ln_f,x_f,ln_g[l],ln_b[l],DIM);
            matmul_f(fc_f,W[l],ln_f,DIM,DIM);
            gelu_v(fc_f,DIM);
            for(int i=0;i<DIM;i++) x_f[i]+=fc_f[i];

            /* Q8 */
            layer_norm(ln_q8,x_q8,ln_g[l],ln_b[l],DIM);
            matmul_q8(fc_q8,W[l],ln_q8,DIM,DIM);
            gelu_v(fc_q8,DIM);
            for(int i=0;i<DIM;i++) x_q8[i]+=fc_q8[i];

            /* Q4 */
            layer_norm(ln_q4,x_q4,ln_g[l],ln_b[l],DIM);
            matmul_q4(fc_q4,W[l],ln_q4,DIM,DIM);
            gelu_v(fc_q4,DIM);
            for(int i=0;i<DIM;i++) x_q4[i]+=fc_q4[i];

            float cs8=cos_sim(x_f,x_q8,DIM), cs4=cos_sim(x_f,x_q4,DIM);
            float cr8=corr_f(x_f,x_q8,DIM), cr4=corr_f(x_f,x_q4,DIM);
            float mre8=0,mre4=0,me8=0,me4=0;
            for(int i=0;i<DIM;i++){
                float r=fabsf(x_f[i]);if(r<1e-6f)r=1e-6f;
                float e8=fabsf(x_f[i]-x_q8[i])/r, e4=fabsf(x_f[i]-x_q4[i])/r;
                if(e8>mre8)mre8=e8; if(e4>mre4)mre4=e4;
                me8+=e8; me4+=e4;
            }
            me8/=DIM; me4/=DIM;

            q8_layer_cos+=cs8; q4_layer_cos+=cs4; layer_count++;
            if(mre8>q8_layer_maxre)q8_layer_maxre=mre8;
            if(mre4>q4_layer_maxre)q4_layer_maxre=mre4;
            q8_layer_meane+=me8; q4_layer_meane+=me4;

            if(l==0||l==N_LAYERS/2-1||l==N_LAYERS-1){
                printf("%-6d %-6d | %-8.6f %-8.6f | %-10.6f %-10.6f | %-10.4f %-10.4f |\n",
                       l+1,p+1,cs8,cs4,cr8,cr4,mre8,mre4);
            }
        }

        /* LM head */
        layer_norm(ln_f,x_f,fg,fb,DIM);
        layer_norm(ln_q8,x_q8,fg,fb,DIM);
        layer_norm(ln_q4,x_q4,fg,fb,DIM);
        matmul_f(lg_f,W_lm,ln_f,DIM,VOCAB_TEST);
        matmul_q8(lg_q8,W_lm,ln_q8,DIM,VOCAB_TEST);
        matmul_q4(lg_q4,W_lm,ln_q4,DIM,VOCAB_TEST);
        softmax(lg_f,VOCAB_TEST);
        softmax(lg_q8,VOCAB_TEST);
        softmax(lg_q4,VOCAB_TEST);

        q8_logit_cos+=cos_sim(lg_f,lg_q8,VOCAB_TEST);
        q4_logit_cos+=cos_sim(lg_f,lg_q4,VOCAB_TEST);

        /* Top-1 */
        int tf=0,tq8=0,tq4=0;
        for(int i=1;i<VOCAB_TEST;i++){
            if(lg_f[i]>lg_f[tf])tf=i;
            if(lg_q8[i]>lg_q8[tq8])tq8=i;
            if(lg_q4[i]>lg_q4[tq4])tq4=i;
        }
        if(tf==tq8)q8_top1_match++;
        if(tf==tq4)q4_top1_match++;
        printf("  LM head: Q8 cos=%.6f, Q4 cos=%.6f, Q8 top1=%s, Q4 top1=%s\n",
               cos_sim(lg_f,lg_q8,VOCAB_TEST),cos_sim(lg_f,lg_q4,VOCAB_TEST),
               tf==tq8?"MATCH":"DIFF", tf==tq4?"MATCH":"DIFF");

        /* Top-5 overlap */
        int idx_f[5],idx_q8[5],idx_q4[5];
        float cf[5],cq8[5],cq4[5];
        for(int k=0;k<5;k++){cf[k]=cq8[k]=cq4[5]=-1e9;idx_f[k]=idx_q8[k]=idx_q4[k]=0;}
        /* Float top-5 */
        for(int k=0;k<5;k++){cf[k]=-1e9;for(int i=0;i<VOCAB_TEST;i++){int ins=0;if(i!=idx_f[0]&&i!=idx_f[1]&&i!=idx_f[2]&&i!=idx_f[3]&&i!=idx_f[4])ins=1;{
            float mn=1e9;int mi=0;for(int kk=0;kk<k;kk++)if(cf[kk]<mn){mn=cf[kk];mi=kk;}
            if(lg_f[i]>mn){cf[mi]=lg_f[i];idx_f[mi]=i;}
        }}}
        /* Simpler approach */
        {int used[5]={0};
        for(int k=0;k<5;k++){float best=-1e9;int bi=0;
            for(int i=0;i<VOCAB_TEST;i++){int skip=0;for(int kk=0;kk<k;kk++)if(idx_f[kk]==i)skip=1;if(!skip&&lg_f[i]>best){best=lg_f[i];bi=i;}}
            idx_f[k]=bi;cf[k]=best;}
        for(int k=0;k<5;k++){float best=-1e9;int bi=0;
            for(int i=0;i<VOCAB_TEST;i++){int skip=0;for(int kk=0;k<k;kk++)if(idx_q8[kk]==i)skip=1;if(!skip&&lg_q8[i]>best){best=lg_q8[i];bi=i;}}
            idx_q8[k]=bi;cq8[k]=best;}
        for(int k=0;k<5;k++){float best=-1e9;int bi=0;
            for(int i=0;i<VOCAB_TEST;i++){int skip=0;for(int kk=0;k<k;kk++)if(idx_q4[kk]==i)skip=1;if(!skip&&lg_q4[i]>best){best=lg_q4[i];bi=i;}}
            idx_q4[k]=bi;cq4[k]=best;}
        }
        int o8=0,o4=0;
        for(int i=0;i<5;i++)for(int j=0;j<5;j++){if(idx_f[i]==idx_q8[j])o8++;if(idx_f[i]==idx_q4[j])o4++;}
        q8_top5_total+=o8; q4_top5_total+=o4;
        printf("  Top-5: Q8 overlap=%d/5, Q4 overlap=%d/5\n\n",o8,o4);
    }

    printf("\n====================== FINAL SUMMARY ======================\n\n");
    printf("%-35s %-12s %-12s %-12s\n","Metric","Float→Q8","Float→Q4","Q8 vs Q4");
    printf("%-35s %-12s %-12s %-12s\n","-----","---------","---------","--------");
    printf("%-35s %-12.6f %-12.6f %-12.6f\n","Avg layer cosine similarity",
           q8_layer_cos/layer_count, q4_layer_cos/layer_count, q8_layer_cos/layer_count - q4_layer_cos/layer_count);
    printf("%-35s %-12.4f %-12.4f\n","Avg layer mean relative error",
           q8_layer_meane/layer_count, q4_layer_meane/layer_count);
    printf("%-35s %-12.4f %-12.4f\n","Max layer relative error (any layer)",
           q8_layer_maxre, q4_layer_maxre);
    printf("%-35s %-12.6f %-12.6f\n","Avg LM head logit cosine similarity",
           q8_logit_cos/N_PROMPTS, q4_logit_cos/N_PROMPTS);
    printf("%-35s %-12d/%-9d %-12d/%d\n","Top-1 prediction match",
           q8_top1_match, N_PROMPTS, q4_top1_match, N_PROMPTS);
    printf("%-35s %-12.1f/5      %-12.1f/5\n","Avg Top-5 overlap",
           (float)q8_top5_total/N_PROMPTS, (float)q4_top5_total/N_PROMPTS);

    /* Speed from previous benchmarks */
    printf("\n====================== SPEED-QUALITY TRADEOFF ======================\n\n");
    printf("%-12s %-8s %-10s %-10s %-10s %-12s\n","Method","Bits","Speed(ms)","vs Float","Quality","Memory");
    printf("%-12s %-8s %-10s %-10s %-10s %-12s\n","------","----","---------","---------","-------","------");
    printf("%-12s %-8d %-10.3f %-10s %-10s %-12s\n","Float32",32,0.153,"1.0x","1.000000","2304 KB");
    printf("%-12s %-8d %-10.3f %-10.2fx %-10.6f %-12s\n","Q8 (LAL)",8,0.018,"8.5x","0.999936","579 KB (4x)");
    printf("%-12s %-8d %-10.3f %-10.2fx %-10.6f %-12s\n","Q4",4,0.026,"5.7x","0.977330","291 KB (8x)");

    free(W_lm);free(x_f);free(x_q8);free(x_q4);
    free(ln_f);free(ln_q8);free(ln_q4);
    free(fc_f);free(fc_q8);free(fc_q4);
    free(lg_f);free(lg_q8);free(lg_q4);free(fg);free(fb);
    for(int l=0;l<N_LAYERS;l++){free(W[l]);free(ln_g[l]);free(ln_b[l]);}
    free(W);free(ln_g);free(ln_b);
    return 0;
}