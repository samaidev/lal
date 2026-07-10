#ifndef LAL_TOKENIZER_H
#define LAL_TOKENIZER_H
/*
 * lal_tokenizer.h - GPT-2 BPE byte-level decode/encode helpers.
 *
 * HuggingFace tokenizers use a byte-to-unicode mapping where invisible
 * bytes (control chars, space, newline, etc.) are represented as Unicode
 * chars in the U+0100 - U+0143 range. The full mapping covers 68 "missing"
 * bytes: 0-32, 127-160, 173.
 *
 * Common special chars:
 *   Ġ (U+0120, UTF-8: 0xC4 0xA0) = space (0x20)
 *   Ċ (U+010A, UTF-8: 0xC4 0x8A) = newline (0x0A)
 *   Ď (U+010E, UTF-8: 0xC4 0x8E) = carriage return (0x0D)
 *   ĥ (U+0125, UTF-8: 0xC4 0xA5) = tab (0x09)
 *   Ă (U+0102, UTF-8: 0xC4 0x82) = 0x02 (STX)
 *   ė (U+0117, UTF-8: 0xC4 0x97) = 0x17 (ETB)
 *
 * This header provides:
 *   lal_decode_bpe_token()  - vocab string -> raw bytes
 *   lal_encode_byte_to_bpe() - raw byte -> unicode codepoint (for encoder)
 *   lal_bpe_byte_for_cp()    - inverse: unicode codepoint -> raw byte
 *
 * CJK and other printable Unicode (U+0100..U+10FFFF beyond U+0143) pass
 * through as their UTF-8 bytes (the model emits them directly).
 */

#include <string.h>

/* Decode one UTF-8 sequence starting at s into *cp.
 * Returns number of bytes consumed (1-4). On invalid byte, returns 1 and
 * stores the byte as-is. */
static inline int lal_utf8_decode(const char *s, unsigned int *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && s[1]) {
        *cp = ((unsigned int)(c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && s[1] && s[2]) {
        *cp = ((unsigned int)(c & 0x0F) << 12)
            | (((unsigned char)s[1] & 0x3F) << 6)
            | ((unsigned char)s[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && s[1] && s[2] && s[3]) {
        *cp = ((unsigned int)(c & 0x07) << 18)
            | (((unsigned char)s[1] & 0x3F) << 12)
            | (((unsigned char)s[2] & 0x3F) << 6)
            | ((unsigned char)s[3] & 0x3F);
        return 4;
    }
    *cp = c; return 1;
}

/* GPT-2 bytes_to_unicode() inverse:
 *   - Printable ASCII 0x21-0x7E  -> byte = codepoint
 *   - Latin-1 printable 0xA1-0xAC, 0xAE-0xFF -> byte = codepoint
 *   - Special range U+0100-U+0143 -> the 68 "missing" bytes (0-32, 127-160, 173)
 *   - Else: pass codepoint through (CJK etc.) — caller should keep UTF-8 bytes
 *
 * Returns: 0 on success and fills *out_byte.
 *          1 if codepoint is "passthrough" (caller should emit UTF-8 bytes).
 */
static inline int lal_bpe_byte_for_cp(unsigned int cp, unsigned char *out_byte) {
    if (cp >= 0x21 && cp <= 0x7E) { *out_byte = (unsigned char)cp; return 0; }
    if (cp >= 0xA1 && cp <= 0xAC) { *out_byte = (unsigned char)cp; return 0; }
    if (cp >= 0xAE && cp <= 0xFF) { *out_byte = (unsigned char)cp; return 0; }
    if (cp >= 0x100 && cp <= 0x143) {
        unsigned int idx = cp - 0x100;
        if (idx < 33)       { *out_byte = (unsigned char)idx;        return 0; }
        else if (idx < 67)  { *out_byte = (unsigned char)(idx + 94); return 0; }
        else                { *out_byte = 173;                       return 0; }
    }
    return 1; /* passthrough */
}

/* Forward mapping: byte -> unicode codepoint (for encoder). */
static inline unsigned int lal_bpe_cp_for_byte(unsigned char b) {
    if ((b >= 0x21 && b <= 0x7E) ||
        (b >= 0xA1 && b <= 0xAC) ||
        (b >= 0xAE && b <= 0xFF)) return b;
    /* Missing byte: find its index in [0-32, 127-160, 173] */
    if (b <= 32)  return 0x100 + b;
    if (b == 173) return 0x143;
    if (b >= 127 && b <= 160) return 0x100 + 33 + (b - 127);
    return b; /* shouldn't happen */
}

/* Encode a unicode codepoint as UTF-8. Returns # bytes written (1-4). */
static inline int lal_utf8_encode(unsigned int cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Decode a single BPE vocab string to actual bytes.
 * vocab_str: the raw string from tokenizer.json (may contain Ġ, Ċ, etc.
 *            as UTF-8 bytes; also any Unicode for CJK / emoji).
 * out:       output buffer (raw bytes, NOT null-terminated unicode)
 * maxlen:    size of out buffer
 *
 * For each UTF-8 codepoint in vocab_str:
 *   - If it maps to a single byte (printable ASCII, Latin-1, or special
 *     U+0100-U+0143), emit that single byte.
 *   - Else (CJK, emoji, etc.), emit the original UTF-8 bytes verbatim.
 */
static inline void lal_decode_bpe_token(const char *vocab_str,
                                        char *out, int maxlen) {
    int o = 0;
    int i = 0;
    while (vocab_str[i] && o < maxlen - 4) {
        unsigned int cp;
        int adv = lal_utf8_decode(vocab_str + i, &cp);
        unsigned char b;
        if (lal_bpe_byte_for_cp(cp, &b) == 0) {
            out[o++] = (char)b;
        } else {
            /* Passthrough: emit original UTF-8 bytes */
            for (int k = 0; k < adv && o < maxlen - 1; k++)
                out[o++] = vocab_str[i + k];
        }
        i += adv;
    }
    out[o] = 0;
}

#endif /* LAL_TOKENIZER_H */
