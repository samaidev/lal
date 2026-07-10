#ifndef LAL_TOKENIZER_H
#define LAL_TOKENIZER_H
/*
 * lal_tokenizer.h - GPT-2 BPE byte-level decode helper.
 *
 * HuggingFace tokenizers use a byte-to-unicode mapping where invisible
 * bytes (space, newline, tab) are represented as special Unicode chars:
 *   Ġ (U+0120, UTF-8: 0xC4 0xA0) = space (0x20)
 *   Ċ (U+010A, UTF-8: 0xC4 0x8A) = newline (0x0A)
 *   Ď (U+010E, UTF-8: 0xC4 0x8E) = carriage return (0x0D)
 *   ĥ (U+0125, UTF-8: 0xC4 0xA5) = tab (0x09)
 *
 * This header provides lal_decode_bpe_token() which converts these back
 * to actual bytes. CJK and other printable Unicode pass through unchanged.
 */

#include <string.h>

/* Decode a single BPE vocab string to actual text.
 * vocab_str: the raw string from tokenizer.json (may contain Ġ, Ċ, etc.)
 * out:       output buffer
 * maxlen:    size of out buffer
 */
static inline void lal_decode_bpe_token(const char *vocab_str,
                                        char *out, int maxlen) {
    int o = 0;
    for (int i = 0; vocab_str[i] && o < maxlen - 4; i++) {
        unsigned char c = (unsigned char)vocab_str[i];
        /* 2-byte UTF-8 sequences in the 0xC4 0x80-0xBF range (Ġ, Ċ, etc.) */
        if (c == 0xC4 && (unsigned char)vocab_str[i+1] == 0xA0) { out[o++] = ' ';  i++; }
        else if (c == 0xC4 && (unsigned char)vocab_str[i+1] == 0x8A) { out[o++] = '\n'; i++; }
        else if (c == 0xC4 && (unsigned char)vocab_str[i+1] == 0x8E) { out[o++] = '\r'; i++; }
        else if (c == 0xC4 && (unsigned char)vocab_str[i+1] == 0xA5) { out[o++] = '\t'; i++; }
        else out[o++] = c;
    }
    out[o] = 0;
}

#endif /* LAL_TOKENIZER_H */
