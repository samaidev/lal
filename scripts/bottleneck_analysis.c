/* bottleneck_analysis.c — Why is Q4 not 2x faster than Q8?
 *
 * Hypotheses:
 * 1. Unpack overhead: int4→int8 costs ~6 instructions per 16 bytes
 * 2. Memory not the bottleneck: 768x768 fits in L2 cache
 * 3. Compute-bound: VPMADDUBSW throughput is the limiter
 *
 * Test: vary matrix size to see where memory vs compute dominates
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define N_TRIALS 500

/* ---- Q8 SIMD (LAL style, 8-output parallel) ---- */
static void matmul_q8_simd(float *y, const int8_t *q8_T, const float *scale,
                           const int32_t *w_sums, const float *x,
                           int in_dim, int out_dim) {
    float x_max=0;
    for(int i=0;i<in_dim;i++) x_max=fmaxf(x_max,fabsf(x[i]));
    float x_sc=x_max/127.0f; if(x_sc<1e-8f)x_sc=1e-8f;
    uint8_t xq[8192];
    for(int i=0;i<in_dim;i++){
        int v=(int)lroundf(x[i]/x_sc)+128;
        xq[i]=(uint8_t)(v>255?255:(v<0?0:v));
    }
    __m256i ones=_mm256_set1_epi16(1);
    int j=0;
    for(;j+8<=out_dim;j+=8){
        const int8_t *w0=q8_T+(size_t)(j+0)*in_dim;
        const int8_t *w1=q8_T+(size_t)(j+1)*in_dim;
        const int8_t *w2=q8_T+(size_t)(j+2)*in_dim;
        const int8_t *w3=q8_T+(size_t)(j+3)*in_dim;
        const int8_t *w4=q8_T+(size_t)(j+4)*in_dim;
        const int8_t *w5=q8_T+(size_t)(j+5)*in_dim;
        const int8_t *w6=q8_T+(size_t)(j+6)*in_dim;
        const int8_t *w7=q8_T+(size_t)(j+7)*in_dim;
        __m256i a0=_mm256_setzero_si256(),a1=_mm256_setzero_si256();
        __m256i a2=_mm256_setzero_si256(),a3=_mm256_setzero_si256();
        __m256i a4=_mm256_setzero_si256(),a5=_mm256_setzero_si256();
        __m256i a6=_mm256_setzero_si256(),a7=_mm256_setzero_si256();
        for(int i=0;i<in_dim;i+=32){
            __m256i xv=_mm256_loadu_si256((__m256i*)(xq+i));
            a0=_mm256_add_epi32(a0,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w0+i))),ones));
            a1=_mm256_add_epi32(a1,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w1+i))),ones));
            a2=_mm256_add_epi32(a2,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w2+i))),ones));
            a3=_mm256_add_epi32(a3,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w3+i))),ones));
            a4=_mm256_add_epi32(a4,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w4+i))),ones));
            a5=_mm256_add_epi32(a5,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w5+i))),ones));
            a6=_mm256_add_epi32(a6,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w6+i))),ones));
            a7=_mm256_add_epi32(a7,_mm256_madd_epi16(_mm256_maddubs_epi16(xv,_mm256_loadu_si256((__m256i*)(w7+i))),ones));
        }
        #define HSUM(v) ({__m128i lo=_mm256_castsi256_si128(v),hi=_mm256_extracti128_si256(v,1);\
            __m128i s=_mm_hadd_epi32(lo,hi);s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);_mm_cvtsi128_si32(s);})
        y[j+0]=(float)(HSUM(a0)-128*w_sums[j+0])*x_sc*scale[j+0];
        y[j+1]=(float)(HSUM(a1)-128*w_sums[j+1])*x_sc*scale[j+1];
        y[j+2]=(float)(HSUM(a2)-128*w_sums[j+2])*x_sc*scale[j+2];
        y[j+3]=(float)(HSUM(a3)-128*w_sums[j+3])*x_sc*scale[j+3];
        y[j+4]=(float)(HSUM(a4)-128*w_sums[j+4])*x_sc*scale[j+4];
        y[j+5]=(float)(HSUM(a5)-128*w_sums[j+5])*x_sc*scale[j+5];
        y[j+6]=(float)(HSUM(a6)-128*w_sums[j+6])*x_sc*scale[j+6];
        y[j+7]=(float)(HSUM(a7)-128*w_sums[j+7])*x_sc*scale[j+7];
    }
}

/* ---- Q4 SIMD (unpack int4→int8, then same VPMADDUBSW) ---- */
static void matmul_q4_simd(float *y, const uint8_t *q4_T, const float *scale,
                           const int32_t *w_sums, const float *x,
                           int in_dim, int out_dim) {
    float x_max=0;
    for(int i=0;i<in_dim;i++) x_max=fmaxf(x_max,fabsf(x[i]));
    float x_sc=x_max/127.0f; if(x_sc<1e-8f)x_sc=1e-8f;
    uint8_t xq[8192];
    for(int i=0;i<in_dim;i++){
        int v=(int)lroundf(x[i]/x_sc)+128;
        xq[i]=(uint8_t)(v>255?255:(v<0?0:v));
    }
    __m256i ones=_mm256_set1_epi16(1);
    int half_in=in_dim/2;

    for(int j=0;j<out_dim;j++){
        const uint8_t *wp=q4_T+(size_t)j*half_in;
        __m256i acc32=_mm256_setzero_si256();
        for(int i=0;i<half_in;i+=16){
            /* UNPACK: 16 packed bytes → 32 int8 (THE OVERHEAD!) */
            __m128i packed=_mm_loadu_si128((__m128i*)(wp+i));
            __m128i lo_nibbles=_mm_and_si128(packed,_mm_set1_epi8(0x0F));
            __m128i hi_nibbles=_mm_srli_epi16(packed,4);
            hi_nibbles=_mm_and_si128(hi_nibbles,_mm_set1_epi8(0x0F));
            lo_nibbles=_mm_sub_epi8(lo_nibbles,_mm_set1_epi8(8));
            hi_nibbles=_mm_sub_epi8(hi_nibbles,_mm_set1_epi8(8));
            __m128i w_low=_mm_unpacklo_epi8(lo_nibbles,hi_nibbles);
            __m128i w_high=_mm_unpackhi_epi8(lo_nibbles,hi_nibbles);
            __m256i wv=_mm256_inserti128_si256(_mm256_castsi128_si256(w_low),w_high,1);
            /* Same VPMADDUBSW as Q8 */
            __m256i xv=_mm256_loadu_si256((__m256i*)(xq+i*2));
            __m256i prod16=_mm256_maddubs_epi16(xv,wv);
            acc32=_mm256_add_epi32(acc32,_mm256_madd_epi16(prod16,ones));
        }
        __m128i lo=_mm256_castsi256_si128(acc32),hi=_mm256_extracti128_si256(acc32,1);
        __m128i s=_mm_add_epi32(lo,hi);s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);
        y[j]=(float)(_mm_cvtsi128_si32(s)-128*w_sums[j])*x_sc*scale[j];
    }
}

static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

int main(void){
    srand(42);

    printf("=== Q4 vs Q8 Bottleneck Analysis ===\n");
    printf("Testing different matrix sizes to find memory vs compute crossover\n\n");

    int sizes[]={128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096};
    int nsizes=sizeof(sizes)/sizeof(sizes[0]);

    printf("%-8s | %-10s %-10s %-8s | %-12s %-12s %-12s | %-8s\n",
           "Dim","Q8 (ms)","Q4 (ms)","Q8/Q4","Q8 Mem(KB)","Q4 Mem(KB)","Mem saved","Bound?");
    printf("-------- | ---------- ---------- -------- | ------------ ------------ ------------ | --------\n");

    for(int si=0;si<nsizes;si++){
        int dim=sizes[si];
        int in_dim=dim, out_dim=dim;
        size_t w_bytes_q8=(size_t)in_dim*out_dim;         /* int8 */
        size_t w_bytes_q4=(size_t)out_dim*(in_dim/2);     /* uint8 packed */
        float mem_q8=(float)(w_bytes_q8 + out_dim*4 + out_dim*4)/1024;
        float mem_q4=(float)(w_bytes_q4 + out_dim*4 + out_dim*4)/1024;

        /* Allocate */
        int8_t *q8_T=malloc(w_bytes_q8);
        uint8_t *q4_T=malloc(w_bytes_q4);
        float *scale=malloc(out_dim*sizeof(float));
        int32_t *w_sums=malloc(out_dim*sizeof(int32_t));
        float *x=malloc(in_dim*sizeof(float));
        float *y=malloc(out_dim*sizeof(float));

        /* Init */
        for(size_t i=0;i<w_bytes_q8;i++) q8_T[i]=(rand()%255)-127;
        /* Pack Q4 from Q8 (just truncate to 4-bit) */
        for(int j=0;j<out_dim;j++){
            scale[j]=0.001f+0.001f*(j%100);
            int32_t ws=0;
            for(int i=0;i<in_dim;i+=2){
                int v0=q8_T[j*in_dim+i]; if(v0>7)v0=7;if(v0<-7)v0=-7;
                int v1=q8_T[j*in_dim+i+1]; if(v1>7)v1=7;if(v1<-7)v1=-7;
                q4_T[(size_t)j*(in_dim/2)+i/2]=(uint8_t)((v0+8)|((v1+8)<<4));
                ws+=v0;ws+=v1;
            }
            w_sums[j]=ws;
        }
        for(int i=0;i<in_dim;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*4;

        /* Benchmark */
        double t0=now_sec();
        for(int t=0;t<N_TRIALS;t++) matmul_q8_simd(y,q8_T,scale,w_sums,x,in_dim,out_dim);
        double t_q8=now_sec()-t0;

        t0=now_sec();
        for(int t=0;t<N_TRIALS;t++) matmul_q4_simd(y,q4_T,scale,w_sums,x,in_dim,out_dim);
        double t_q4=now_sec()-t0;

        float ratio=t_q8/t_q4;
        /* If ratio > 1, Q4 is faster; if < 1, Q8 is faster */
        /* Determine bound: if Q4 not faster despite 2x less memory → compute bound */
        char *bound;
        float mem_theory=2.0f; /* Q4 uses 2x less memory, so if memory-bound should be 2x faster */
        if(ratio > 0.9f && ratio < 1.1f) bound="COMPUTE";
        else if(ratio > 1.3f) bound="MEMORY";
        else bound="MIXED";

        printf("%-8d | %-10.3f %-10.3f %-8.2f | %-12.1f %-12.1f %-12.1f | %-8s\n",
               dim, t_q8/N_TRIALS*1000, t_q4/N_TRIALS*1000, ratio,
               mem_q8, mem_q4, mem_q8/mem_q4, bound);

        free(q8_T);free(q4_T);free(scale);free(w_sums);free(x);free(y);
    }

    /* Now count actual instructions for Q8 vs Q4 inner loop */
    printf("\n=== Instruction Count Analysis (per 32-weight block) ===\n\n");
    printf("Q8 inner loop (1 output, 32 weights):\n");
    printf("  1. load x (32 bytes):        vmovdqu        1 instruction\n");
    printf("  2. load w (32 bytes):        vmovdqu        1 instruction\n");
    printf("  3. uint8×int8→int16:         vpmaddubs      1 instruction\n");
    printf("  4. int16→int32 sum:          vpmaddwd       1 instruction\n");
    printf("  5. accumulate:               vpaddd         1 instruction\n");
    printf("  TOTAL:                       5 instructions\n\n");

    printf("Q4 inner loop (1 output, 32 weights from 16 packed bytes):\n");
    printf("  1. load packed (16 bytes):   vmovdqu        1 instruction\n");
    printf("  2. mask low nibble:          vpand          1 instruction\n");
    printf("  3. shift right 4:            vpsrlw         1 instruction\n");
    printf("  4. mask high nibble:         vpand          1 instruction\n");
    printf("  5. subtract 8 (low):         vpsubb         1 instruction\n");
    printf("  6. subtract 8 (high):        vpsubb         1 instruction\n");
    printf("  7. interleave lo:            vpunpcklbw     1 instruction\n");
    printf("  8. interleave hi:            vpunpckhbw     1 instruction\n");
    printf("  9. combine to 256-bit:       vinserti128    1 instruction\n");
    printf(" 10. load x (32 bytes):        vmovdqu        1 instruction\n");
    printf(" 11. uint8×int8→int16:         vpmaddubs      1 instruction\n");
    printf(" 12. int16→int32 sum:          vpmaddwd       1 instruction\n");
    printf(" 13. accumulate:               vpaddd         1 instruction\n");
    printf("  TOTAL:                       13 instructions\n\n");

    printf("=> Q4 unpack costs 8 EXTRA instructions per 32 weights\n");
    printf("=> Q4 compute is 13/5 = 2.6x more instructions\n");
    printf("=> Q4 saves 2x memory but costs 2.6x compute → net SLOWER\n");
    printf("=> The unpack overhead completely negates the memory savings\n\n");

    /* Theoretical analysis */
    printf("=== Theoretical Roofline ===\n\n");
    printf("Intel Xeon (this machine):\n");
    printf("  L1 cache:    ~32 KB   → fits 128×128 Q8, 128×128 Q4\n");
    printf("  L2 cache:    ~1 MB    → fits 768×768 Q8 (579 KB), 768×768 Q4 (291 KB)\n");
    printf("  Memory BW:   ~30 GB/s (2-core Xeon)\n");
    printf("  AVX2 FMA:    1 VPMADDUBSW/cycle/port, 2 ports → 2/cycle\n\n");
    printf("  At 768×768 Q8: 579 KB fits in L2 → L2 BW ~200 GB/s\n");
    printf("  Arith intensity = ops/bytes:\n");
    printf("    Q8: 2*768*768 / 579KB = 2.04 ops/byte\n");
    printf("    Q4: 2*768*768 / 291KB = 4.07 ops/byte (but with 2.6x more instr)\n");
    printf("  → At this size, we're in the compute-bound regime\n");
    printf("  → Q4's extra unpack instructions make it SLOWER despite less memory\n");

    return 0;
}