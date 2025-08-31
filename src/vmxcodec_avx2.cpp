/*
* MIT License
*
* Copyright (c) 2025 Open Media Transport Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
*/
#include "pch.h"
#include "vmxcodec_x86.h"
#include "vmxcodec_avx2.h"
#include <fstream>

/// X86 Compilation Notes:
/// This file requires AVX2. 
/// To ensure broad compatibility it should be the only one compiled with the AVX2 flag, with all others with SSE4.2
#if defined(AVX2)

#define GETBITSB_BMI(data, numBits, n) { \
			data.BitsLeft -= numBits; \
			n = _bextr_u64(data.TempRead,data.BitsLeft, numBits); \
	}

#define Get2MagSignV256(input) { \
	__m256i b = _mm256_adds_epi16(input, input);  \
	__m256i c = _mm256_srai_epi16(input, 15);  \
	input = _mm256_xor_si256(b, c); \
}

#define Get2MagSignPlusOneV256(input) { \
	__m256i b = _mm256_adds_epi16(input, input);  \
	__m256i c = _mm256_srai_epi16(input, 15);  \
	input = _mm256_xor_si256(b, c); \
	input = _mm256_add_epi16(input, _mm256_set1_epi16(1)); \
}

//===========================
//IDCT Tables for 256bit SIMD
//===========================
__declspec(align(32)) const short one_corr_256[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
__declspec(align(32)) const short round_inv_row_256[16] = { IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0 };
__declspec(align(32)) const short round_inv_col_256[16] = { IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL };
__declspec(align(32)) const short round_inv_corr_256[16] = { IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR };

__declspec(align(32)) const short round_inv_row_256_10[16] = { IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0 };
__declspec(align(32)) const short round_inv_col_256_10[16] = { IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10 };
__declspec(align(32)) const short round_inv_corr_256_10[16] = { IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10 };

__declspec(align(32)) const short tg_1_16_256[16] = { 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036 }; // tg * (2<<16) + 0.5
__declspec(align(32)) const short tg_2_16_256[16] = { 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146 }; // tg * (2<<16) + 0.5
__declspec(align(32)) const short tg_3_16_256[16] = { -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746 }; // tg * (2<<16) + 0.5
__declspec(align(32)) const short cos_4_16_256[16] = { -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195 };// cos * (2<<16) + 0.5

__declspec(align(32)) const short tab_i_04_256[] = {
	16384, 21407, 16384, 8867,
	16384, -8867, 16384, -21407,
	16384, 21407, 16384, 8867,
	16384, -8867, 16384, -21407, 

	16384, 8867, -16384, -21407, 
	-16384, 21407, 16384, -8867, 
	16384, 8867, -16384, -21407, 
	-16384, 21407, 16384, -8867, 

	22725, 19266, 19266, -4520, 
	12873, -22725, 4520, -12873, 
	22725, 19266, 19266, -4520, 
	12873, -22725, 4520, -12873, 

	12873, 4520, -22725, -12873, 
	4520, 19266, 19266, -22725, 
	12873, 4520, -22725, -12873, 
	4520, 19266, 19266, -22725 }; 

__declspec(align(32)) const short tab_i_17_256[] = {
	22725, 29692, 22725, 12299,
	22725, -12299, 22725, -29692,
	22725, 29692, 22725, 12299,
	22725, -12299, 22725, -29692,

	22725, 12299, -22725, -29692,
	-22725, 29692, 22725, -12299,
	22725, 12299, -22725, -29692,
	-22725, 29692, 22725, -12299,

	31521, 26722, 26722, -6270,
	17855, -31521, 6270, -17855,
	31521, 26722, 26722, -6270,
	17855, -31521, 6270, -17855,

	17855, 6270, -31521, -17855,
	6270, 26722, 26722, -31521,
	17855, 6270, -31521, -17855,
	6270, 26722, 26722, -31521 }; 

__declspec(align(32)) const short tab_i_26_256[] = {
	21407, 27969, 21407, 11585,
	21407, -11585, 21407, -27969, 
	21407, 27969, 21407, 11585,
	21407, -11585, 21407, -27969, 

	21407, 11585, -21407, -27969, 
	-21407, 27969, 21407, -11585, 
	21407, 11585, -21407, -27969, 
	-21407, 27969, 21407, -11585, 

	29692, 25172, 25172, -5906,	
	16819, -29692, 5906, -16819, 
	29692, 25172, 25172, -5906,	
	16819, -29692, 5906, -16819, 

		16819, 5906, -29692, -16819, 
	5906, 25172, 25172, -29692, 
	16819, 5906, -29692, -16819, 
	5906, 25172, 25172, -29692 }; 

__declspec(align(32)) const short tab_i_35_256[] = {
	19266, 25172, 19266, 10426,
	19266, -10426, 19266, -25172,
	19266, 25172, 19266, 10426,
	19266, -10426, 19266, -25172,

	19266, 10426, -19266, -25172,
	-19266, 25172, 19266, -10426,
	19266, 10426, -19266, -25172,
	-19266, 25172, 19266, -10426,

	26722, 22654, 22654, -5315,
	15137, -26722, 5315, -15137, 
	26722, 22654, 22654, -5315, 
	15137, -26722, 5315, -15137,

	15137, 5315, -26722, -15137,
	5315, 22654, 22654, -26722,
15137, 5315, -26722, -15137,
5315, 22654, 22654, -26722 }; 

//===========================
//FDCT Tables for 256bit SIMD
//===========================

__declspec(align(32)) const unsigned short ftab1_256[] = {
	16384, 16384, 22725, 19266, 56669, 44129, 42811, 52663,16384, 16384, 22725, 19266, 56669, 44129, 42811, 52663,
	16384, 16384, 12873, 4520, 21407, 8867, 19266, 61016,16384, 16384, 12873, 4520, 21407, 8867, 19266, 61016,
	16384, 49152, 12873, 42811, 21407, 56669, 19266, 42811,16384, 49152, 12873, 42811, 21407, 56669, 19266, 42811,
	49152, 16384, 4520, 19266, 8867, 44129, 4520, 52663,49152, 16384, 4520, 19266, 8867, 44129, 4520, 52663
};

__declspec(align(32)) const unsigned short ftab2_256[] = {
	22725, 22725, 31521, 26722, 53237, 35844, 34015, 47681,22725, 22725, 31521, 26722, 53237, 35844, 34015, 47681,
	22725, 22725, 17855, 6270, 29692, 12299, 26722, 59266,22725, 22725, 17855, 6270, 29692, 12299, 26722, 59266,
	22725, 42811, 17855, 34015, 29692, 53237, 26722, 34015,22725, 42811, 17855, 34015, 29692, 53237, 26722, 34015,
	42811, 22725, 6270, 26722, 12299, 35844, 6270, 47681,42811, 22725, 6270, 26722, 12299, 35844, 6270, 47681
};

__declspec(align(32)) const unsigned short ftab3_256[] = {
	21407, 21407, 29692, 25172, 53951, 37567, 35844, 48717,21407, 21407, 29692, 25172, 53951, 37567, 35844, 48717,
	21407, 21407, 16819, 5906, 27969, 11585, 25172, 59630,21407, 21407, 16819, 5906, 27969, 11585, 25172, 59630,
	21407, 44129, 16819, 35844, 27969, 53951, 25172, 35844,21407, 44129, 16819, 35844, 27969, 53951, 25172, 35844,
	44129, 21407, 5906, 25172, 11585, 37567, 5906, 48717,44129, 21407, 5906, 25172, 11585, 37567, 5906, 48717
};

__declspec(align(32)) const unsigned short ftab4_256[] = {
	19266, 19266, 26722, 22654, 55110, 40364, 38814, 50399,19266, 19266, 26722, 22654, 55110, 40364, 38814, 50399,
	19266, 19266, 15137, 5315, 25172, 10426, 22654, 60221,19266, 19266, 15137, 5315, 25172, 10426, 22654, 60221,
	19266, 46270, 15137, 38814, 25172, 55110, 22654, 38814,19266, 46270, 15137, 38814, 25172, 55110, 22654, 38814,
	46270, 19266, 5315, 22654, 10426, 40364, 5315, 50399,46270, 19266, 5315, 22654, 10426, 40364, 5315, 50399
};

inline void GetIntFrom2MagSignMinus1V_256(__m256i* input)
{
	__m256i one = _mm256_set1_epi16(1);
	*input = _mm256_sub_epi16(*input, one);
	__m256i mask = _mm256_and_si256(*input, one);
	__m256i x = _mm256_adds_epi16(*input, mask);
	__m256i a = _mm256_srli_epi16(x, 1);
	__m256i b = _mm256_mullo_epi16(x, mask);
	*input = _mm256_subs_epi16(a, b);
}

void VMX_ZIG_INVQUANTIZE_IDCT_8X8_256(short* srca, short* srcb, unsigned short* matrix, BYTE* dst, int stride, short addVal) {

	//load 8x8x16x2
	__m256i a0 = _mm256_loadu2_m128i((__m128i*) & srcb[0], (__m128i*) & srca[0]);
	__m256i a1 = _mm256_loadu2_m128i((__m128i*) & srcb[8], (__m128i*) & srca[8]);
	__m256i a2 = _mm256_loadu2_m128i((__m128i*) & srcb[16], (__m128i*) & srca[16]);
	__m256i a3 = _mm256_loadu2_m128i((__m128i*) & srcb[24], (__m128i*) & srca[24]);
	__m256i a4 = _mm256_loadu2_m128i((__m128i*) & srcb[32], (__m128i*) & srca[32]);
	__m256i a5 = _mm256_loadu2_m128i((__m128i*) & srcb[40], (__m128i*) & srca[40]);
	__m256i a6 = _mm256_loadu2_m128i((__m128i*) & srcb[48], (__m128i*) & srca[48]);
	__m256i a7 = _mm256_loadu2_m128i((__m128i*) & srcb[56], (__m128i*) & srca[56]);

	//Inverse Zig Zag ~47 instructions. Around 40% faster than a loop
	__m256i v0 = _mm256_shuffle_epi8(a0, _mm256_set_epi8(7, 6, 15, 14, 9, 8, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0, 7, 6, 15, 14, 9, 8, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0)); //0,1,5,6,2,4,7,3
	__m256i v1 = _mm256_shuffle_epi8(a1, _mm256_set_epi8(7, 6, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 5, 4, 7, 6, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 5, 4)); //10,8,12,13,14,15,9,11
	__m256i v3 = _mm256_shuffle_epi8(a3, _mm256_set_epi8(9, 8, 7, 6, 13, 12, 3, 2, 11, 10, 5, 4, 15, 14, 1, 0, 9, 8, 7, 6, 13, 12, 3, 2, 11, 10, 5, 4, 15, 14, 1, 0)); //24,31,26,29,25,30,27,28

	//a0 0, 1, 5, 6, 14, 15, 27, 28, 
	a0 = _mm256_blend_epi16(v0, v1, 0x30); //-,0,0,0,14,15,-,-
	a0 = _mm256_blend_epi16(a0, v3, 0xC0); //-,-,-,-,-,-,27,28

	__m256i v2 = _mm256_shuffle_epi8(a2, _mm256_set_epi8(5, 4, 13, 12, 9, 8, 1, 0, 3, 2, 15, 14, 7, 6, 11, 10, 5, 4, 13, 12, 9, 8, 1, 0, 3, 2, 15, 14, 7, 6, 11, 10)); //21,19,23,17,16,20,22,18
	__m256i v5 = _mm256_shuffle_epi8(a5, _mm256_set_epi8(7, 6, 3, 2, 11, 10, 13, 12, 15, 14, 5, 4, 9, 8, 1, 0, 7, 6, 3, 2, 11, 10, 13, 12, 15, 14, 5, 4, 9, 8, 1, 0)); //40,44,42,47,46,45,41,43

	//a2 3, 8, 12, 17, 25, 30, 41, 43,
	a2 = _mm256_srli_si256(v0, 14); //3
	a2 = _mm256_blend_epi16(a2, v3, 0x30); //25,30
	a2 = _mm256_blend_epi16(a2, v1, 0x6); //8,12
	a2 = _mm256_blend_epi16(a2, v2, 0x8); //-,-,-,17,-,-,-,-
	a2 = _mm256_blend_epi16(a2, v5, 0xC0); //-,-,-,-,-,-,41,43

	//v3 = _mm_shuffle_epi8(a3, _mm_set_epi8(-1, -1, 11, 10, 5, 4, 15, 14, 1, 0, -1, -1, -1, -1, -1, -1)); //-,-,-,24,31,26,29,-
	v3 = _mm256_slli_si256(v3, 6);
	__m256i v4 = _mm256_shuffle_epi8(a4, _mm256_set_epi8(13, 12, 3, 2, 5, 4, 15, 14, 1, 0, 11, 10, 9, 8, 7, 6, 13, 12, 3, 2, 5, 4, 15, 14, 1, 0, 11, 10, 9, 8, 7, 6)); //35,36,37,32,39,34,33,38

	//a1 2, 4, 7, 13, 16, 26, 29, 42,
	v0 = _mm256_srli_si256(v0, 8); //2,4,7,3,0,0,0,0
	a1 = _mm256_blend_epi16(v0, v1, 0x8); //13
	a1 = _mm256_blend_epi16(a1, v2, 0x10); //-,-,-,-,16,-,-,-
	a1 = _mm256_blend_epi16(a1, v3, 0x60); //-,-,-,-,-,26,29,-

	__m256i v6 = _mm256_shuffle_epi8(a6, _mm256_set_epi8(13, 12, 9, 8, -1, -1, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, 13, 12, 9, 8, -1, -1, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1)); //-,-,48,49,50,-,52,54
	__m256i v7 = _mm256_shuffle_epi8(a7, _mm256_set_epi8(15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 7, 6, 1, 0, 9, 8, 15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 7, 6, 1, 0, 9, 8)); //60,56,59,61,57,58,62,63

	//a4 10, 19, 23, 32, 39, 45, 52, 54, 
	a4 = _mm256_blend_epi16(v1, v6, 0xC0); //10 + 52,54
	a4 = _mm256_blend_epi16(a4, v2, 0x6); //-,19,23,-,-,-,-,-
	a4 = _mm256_blend_epi16(a4, v4, 0x18); //32,39
	a4 = _mm256_blend_epi16(a4, v5, 0x20); //45

	__m256i x6 = _mm256_shuffle_epi8(a6, _mm256_set_epi8(11, 10, 15, 14, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 10, 15, 14, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)); //-,-,-,-,-,51,55,53

	//a3 9, 11, 18, 24, 31, 40, 44, 53, 
	a3 = _mm256_srli_si256(v1, 12); //9,11,-,-,-,-,-,-
	a3 = _mm256_blend_epi16(a3, v3, 0x18); //-,-,-,24,31,-,-,-
	a3 = _mm256_blend_epi16(a3, x6, 0x80); //53

	//a7 35, 36, 48, 49, 57, 58, 62, 63 
	a7 = _mm256_blend_epi16(v7, v4, 0x3); //-,-,-,-,57,58,62,63 + //35,36
	a7 = _mm256_blend_epi16(a7, v6, 0xC); //48,49

	//a6 21, 34, 37, 47, 50, 56, 59, 61,
	a6 = _mm256_slli_si256(v7, 8); //-,-,-,-,56,59,61
	a6 = _mm256_blend_epi16(a6, v4, 0x4); //-,-,37,-,-,-,-,-
	a6 = _mm256_blend_epi16(a6, v6, 0x10); //50
	a6 = _mm256_blend_epi16(a6, v5, 0x8); //47
	a6 = _mm256_blend_epi16(a6, v2, 0x1); //21

	v4 = _mm256_srli_si256(v4, 8); //-,34,33,38,-,-,-,-

	//a5 20, 22, 33, 38, 46, 51, 55, 60,
	v2 = _mm256_srli_si256(v2, 10);
	a5 = _mm256_slli_si256(v7, 14); //-,-,-,-,-,-,60
	a5 = _mm256_blend_epi16(a5, v4, 0xC); //33,38
	a5 = _mm256_blend_epi16(a5, v2, 0x3); //20,22
	a5 = _mm256_blend_epi16(a5, v5, 0x10); //46
	a5 = _mm256_blend_epi16(a5, x6, 0x60); //51,55

	a6 = _mm256_blend_epi16(a6, v4, 0x2); //34
	a3 = _mm256_blend_epi16(a3, v2, 0x4); //18

	v5 = _mm256_slli_si256(v5, 10); //-,-,-,-,-,40,44,42
	a3 = _mm256_blend_epi16(a3, v5, 0x60); //40,44
	a1 = _mm256_blend_epi16(a1, v5, 0x80); //42


	//load quant
	__m256i c0 = _mm256_load_si256((__m256i*) & matrix[0]);
	__m256i c1 = _mm256_load_si256((__m256i*) & matrix[16]);
	__m256i c2 = _mm256_load_si256((__m256i*) & matrix[32]);
	__m256i c3 = _mm256_load_si256((__m256i*) & matrix[48]);
	__m256i c4 = _mm256_load_si256((__m256i*) & matrix[64]);
	__m256i c5 = _mm256_load_si256((__m256i*) & matrix[80]);
	__m256i c6 = _mm256_load_si256((__m256i*) & matrix[96]);
	__m256i c7 = _mm256_load_si256((__m256i*) & matrix[112]);

	//multiply
	a0 = _mm256_mullo_epi16(a0, c0);
	a1 = _mm256_mullo_epi16(a1, c1);
	a2 = _mm256_mullo_epi16(a2, c2);
	a3 = _mm256_mullo_epi16(a3, c3);
	a4 = _mm256_mullo_epi16(a4, c4);
	a5 = _mm256_mullo_epi16(a5, c5);
	a6 = _mm256_mullo_epi16(a6, c6);
	a7 = _mm256_mullo_epi16(a7, c7);

	//shift
	a0 = _mm256_srai_epi16(a0, 4);
	a1 = _mm256_srai_epi16(a1, 4);
	a2 = _mm256_srai_epi16(a2, 4);
	a3 = _mm256_srai_epi16(a3, 4);
	a4 = _mm256_srai_epi16(a4, 4);
	a5 = _mm256_srai_epi16(a5, 4);
	a6 = _mm256_srai_epi16(a6, 4);
	a7 = _mm256_srai_epi16(a7, 4);

	// /////////////////////////////
	// //////////Row 1 And row 3
	// ////////////////////////////
	__m256i r_xmm0 = a0;
	__m256i r_xmm4 = a2;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	__m256i r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_04_256[0]));
	__m256i r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_04_256[32]));
	__m256i r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_04_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_04_256[48]));
	__m256i r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	__m256i r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_26_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	__m256i r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_26_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_26_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_26_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row0 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row2 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 5 And row 7
	// ////////////////////////////
	r_xmm0 = a4;
	r_xmm4 = a6;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_04_256[0]));
	r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_04_256[32]));
	r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_04_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_04_256[48]));
	r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_26_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_26_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_26_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_26_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row4 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row6 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 4 And row 2
	// ////////////////////////////
	r_xmm0 = a3;
	r_xmm4 = a1;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_35_256[0]));
	r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_35_256[32]));
	r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_35_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_35_256[48]));
	r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_17_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_17_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_17_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_17_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row3 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row1 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 6 And row 8
	// ////////////////////////////
	r_xmm0 = a5;
	r_xmm4 = a7;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_35_256[0]));
	r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_35_256[32]));
	r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_35_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_35_256[48]));
	r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_17_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_17_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_17_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_17_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row5 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row7 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	r_xmm1 = _mm256_loadu_si256((__m256i*) tg_3_16_256);
	r_xmm2 = row5;
	r_xmm3 = row3;
	r_xmm0 = _mm256_mulhi_epi16(row5, r_xmm1);
	r_xmm1 = _mm256_mulhi_epi16(r_xmm1, r_xmm3);
	r_xmm5 = _mm256_loadu_si256((__m256i*) tg_1_16_256);
	r_xmm6 = row7;
	r_xmm4 = _mm256_mulhi_epi16(row7, r_xmm5);
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm2);
	r_xmm5 = _mm256_mulhi_epi16(r_xmm5, row1);
	r_xmm1 = _mm256_adds_epi16(r_xmm1, r_xmm3);
	r_xmm7 = row6;
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm3);
	r_xmm3 = _mm256_loadu_si256((__m256i*) tg_2_16_256);
	r_xmm2 = _mm256_subs_epi16(r_xmm2, r_xmm1);
	r_xmm7 = _mm256_mulhi_epi16(r_xmm7, r_xmm3);
	r_xmm1 = r_xmm0;
	r_xmm3 = _mm256_mulhi_epi16(r_xmm3, row2);
	r_xmm5 = _mm256_subs_epi16(r_xmm5, r_xmm6);
	r_xmm4 = _mm256_adds_epi16(r_xmm4, row1);
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm4);
	r_xmm0 = _mm256_adds_epi16(r_xmm0, *((__m256i*) one_corr_256));
	r_xmm4 = _mm256_subs_epi16(r_xmm4, r_xmm1);
	r_xmm6 = r_xmm5;
	r_xmm5 = _mm256_subs_epi16(r_xmm5, r_xmm2);
	r_xmm5 = _mm256_adds_epi16(r_xmm5, *((__m256i*) one_corr_256));
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm2);
	__m256i temp7 = r_xmm0;
	r_xmm1 = r_xmm4;
	r_xmm0 = _mm256_load_si256((__m256i*) cos_4_16_256);
	r_xmm4 = _mm256_adds_epi16(r_xmm4, r_xmm5);
	r_xmm2 = _mm256_load_si256((__m256i*) cos_4_16_256);
	r_xmm2 = _mm256_mulhi_epi16(r_xmm2, r_xmm4);
	__m256i temp3 = r_xmm6;
	r_xmm1 = _mm256_subs_epi16(r_xmm1, r_xmm5);
	r_xmm7 = _mm256_adds_epi16(r_xmm7, row2);
	r_xmm3 = _mm256_subs_epi16(r_xmm3, row6);
	r_xmm6 = row0;
	r_xmm0 = _mm256_mulhi_epi16(r_xmm0, r_xmm1);
	r_xmm5 = row4;
	r_xmm5 = _mm256_adds_epi16(r_xmm5, r_xmm6);
	r_xmm6 = _mm256_subs_epi16(r_xmm6, row4);
	r_xmm4 = _mm256_adds_epi16(r_xmm4, r_xmm2);
	r_xmm4 = _mm256_or_si256(r_xmm4, *((__m256i*) one_corr_256));
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm1);
	r_xmm0 = _mm256_or_si256(r_xmm0, *((__m256i*) one_corr_256));
	r_xmm2 = r_xmm5;
	r_xmm5 = _mm256_adds_epi16(r_xmm5, r_xmm7);
	r_xmm1 = r_xmm6;
	r_xmm5 = _mm256_adds_epi16(r_xmm5, *((__m256i*) round_inv_col_256));
	r_xmm2 = _mm256_subs_epi16(r_xmm2, r_xmm7);
	r_xmm7 = temp7;
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm3);
	r_xmm6 = _mm256_adds_epi16(r_xmm6, *((__m256i*) round_inv_col_256));
	r_xmm7 = _mm256_adds_epi16(r_xmm7, r_xmm5);
	r_xmm7 = _mm256_srai_epi16(r_xmm7, SHIFT_INV_COL);
	r_xmm1 = _mm256_subs_epi16(r_xmm1, r_xmm3);
	r_xmm1 = _mm256_adds_epi16(r_xmm1, *((__m256i*) round_inv_corr_256));
	r_xmm3 = r_xmm6;
	r_xmm2 = _mm256_adds_epi16(r_xmm2, *((__m256i*) round_inv_corr_256));
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm4);

	__m256i vadd = _mm256_set1_epi16(addVal);
	r_xmm7 = _mm256_adds_epi16(r_xmm7, vadd);

	r_xmm6 = _mm256_srai_epi16(r_xmm6, SHIFT_INV_COL);

	r_xmm6 = _mm256_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm256_packus_epi16(r_xmm7, r_xmm6);

	r_xmm7 = r_xmm1;
	r_xmm1 = _mm256_adds_epi16(r_xmm1, r_xmm0);

	r_xmm1 = _mm256_srai_epi16(r_xmm1, SHIFT_INV_COL);

	BYTE itemp[32];
	_mm256_storeu_si256((__m256i*) (&itemp[0]), r_xmm6);
	memcpy(dst, &itemp[0], 8);
	memcpy(dst + 8, &itemp[16], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	memcpy(dst + 8, &itemp[24], 8);
	dst += stride;

	r_xmm6 = temp3;
	r_xmm7 = _mm256_subs_epi16(r_xmm7, r_xmm0);
	r_xmm7 = _mm256_srai_epi16(r_xmm7, SHIFT_INV_COL);

	r_xmm1 = _mm256_adds_epi16(r_xmm1, vadd);

	r_xmm5 = _mm256_subs_epi16(r_xmm5, temp7);
	r_xmm5 = _mm256_srai_epi16(r_xmm5, SHIFT_INV_COL);

	r_xmm5 = _mm256_adds_epi16(r_xmm5, vadd);

	r_xmm3 = _mm256_subs_epi16(r_xmm3, r_xmm4);
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm2);
	r_xmm2 = _mm256_subs_epi16(r_xmm2, temp3);
	r_xmm6 = _mm256_srai_epi16(r_xmm6, SHIFT_INV_COL);
	r_xmm2 = _mm256_srai_epi16(r_xmm2, SHIFT_INV_COL);

	r_xmm6 = _mm256_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm256_packus_epi16(r_xmm1, r_xmm6);

	r_xmm3 = _mm256_srai_epi16(r_xmm3, SHIFT_INV_COL);

	r_xmm2 = _mm256_adds_epi16(r_xmm2, vadd);

	r_xmm7 = _mm256_adds_epi16(r_xmm7, vadd);

	r_xmm7 = _mm256_packus_epi16(r_xmm2, r_xmm7);

	_mm256_storeu_si256((__m256i*) (&itemp[0]), r_xmm6);
	memcpy(dst, &itemp[0], 8);
	memcpy(dst + 8, &itemp[16], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	memcpy(dst + 8, &itemp[24], 8);
	dst += stride;

	_mm256_storeu_si256((__m256i*) (&itemp[0]), r_xmm7);
	memcpy(dst, &itemp[0], 8);
	memcpy(dst + 8, &itemp[16], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	memcpy(dst + 8, &itemp[24], 8);
	dst += stride;

	r_xmm3 = _mm256_adds_epi16(r_xmm3, vadd);
	r_xmm3 = _mm256_packus_epi16(r_xmm3, r_xmm5);

	_mm256_storeu_si256((__m256i*) (&itemp[0]), r_xmm3);
	memcpy(dst, &itemp[0], 8);
	memcpy(dst + 8, &itemp[16], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	memcpy(dst + 8, &itemp[24], 8);
	dst += stride;

}


void VMX_ZIG_INVQUANTIZE_IDCT_8X8_256_16(short* srca, short* srcb, unsigned short* matrix, BYTE* dst, int stride, short addVal) {

	//load 8x8x16x2
	__m256i a0 = _mm256_loadu2_m128i((__m128i*) & srcb[0], (__m128i*) & srca[0]);
	__m256i a1 = _mm256_loadu2_m128i((__m128i*) & srcb[8], (__m128i*) & srca[8]);
	__m256i a2 = _mm256_loadu2_m128i((__m128i*) & srcb[16], (__m128i*) & srca[16]);
	__m256i a3 = _mm256_loadu2_m128i((__m128i*) & srcb[24], (__m128i*) & srca[24]);
	__m256i a4 = _mm256_loadu2_m128i((__m128i*) & srcb[32], (__m128i*) & srca[32]);
	__m256i a5 = _mm256_loadu2_m128i((__m128i*) & srcb[40], (__m128i*) & srca[40]);
	__m256i a6 = _mm256_loadu2_m128i((__m128i*) & srcb[48], (__m128i*) & srca[48]);
	__m256i a7 = _mm256_loadu2_m128i((__m128i*) & srcb[56], (__m128i*) & srca[56]);

	//Inverse Zig Zag ~47 instructions. Around 40% faster than a loop
	__m256i v0 = _mm256_shuffle_epi8(a0, _mm256_set_epi8(7, 6, 15, 14, 9, 8, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0, 7, 6, 15, 14, 9, 8, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0)); //0,1,5,6,2,4,7,3
	__m256i v1 = _mm256_shuffle_epi8(a1, _mm256_set_epi8(7, 6, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 5, 4, 7, 6, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 5, 4)); //10,8,12,13,14,15,9,11
	__m256i v3 = _mm256_shuffle_epi8(a3, _mm256_set_epi8(9, 8, 7, 6, 13, 12, 3, 2, 11, 10, 5, 4, 15, 14, 1, 0, 9, 8, 7, 6, 13, 12, 3, 2, 11, 10, 5, 4, 15, 14, 1, 0)); //24,31,26,29,25,30,27,28

	//a0 0, 1, 5, 6, 14, 15, 27, 28, 
	a0 = _mm256_blend_epi16(v0, v1, 0x30); //-,0,0,0,14,15,-,-
	a0 = _mm256_blend_epi16(a0, v3, 0xC0); //-,-,-,-,-,-,27,28

	__m256i v2 = _mm256_shuffle_epi8(a2, _mm256_set_epi8(5, 4, 13, 12, 9, 8, 1, 0, 3, 2, 15, 14, 7, 6, 11, 10, 5, 4, 13, 12, 9, 8, 1, 0, 3, 2, 15, 14, 7, 6, 11, 10)); //21,19,23,17,16,20,22,18
	__m256i v5 = _mm256_shuffle_epi8(a5, _mm256_set_epi8(7, 6, 3, 2, 11, 10, 13, 12, 15, 14, 5, 4, 9, 8, 1, 0, 7, 6, 3, 2, 11, 10, 13, 12, 15, 14, 5, 4, 9, 8, 1, 0)); //40,44,42,47,46,45,41,43

	//a2 3, 8, 12, 17, 25, 30, 41, 43,
	a2 = _mm256_srli_si256(v0, 14); //3
	a2 = _mm256_blend_epi16(a2, v3, 0x30); //25,30
	a2 = _mm256_blend_epi16(a2, v1, 0x6); //8,12
	a2 = _mm256_blend_epi16(a2, v2, 0x8); //-,-,-,17,-,-,-,-
	a2 = _mm256_blend_epi16(a2, v5, 0xC0); //-,-,-,-,-,-,41,43

	//v3 = _mm_shuffle_epi8(a3, _mm_set_epi8(-1, -1, 11, 10, 5, 4, 15, 14, 1, 0, -1, -1, -1, -1, -1, -1)); //-,-,-,24,31,26,29,-
	v3 = _mm256_slli_si256(v3, 6);
	__m256i v4 = _mm256_shuffle_epi8(a4, _mm256_set_epi8(13, 12, 3, 2, 5, 4, 15, 14, 1, 0, 11, 10, 9, 8, 7, 6, 13, 12, 3, 2, 5, 4, 15, 14, 1, 0, 11, 10, 9, 8, 7, 6)); //35,36,37,32,39,34,33,38

	//a1 2, 4, 7, 13, 16, 26, 29, 42,
	v0 = _mm256_srli_si256(v0, 8); //2,4,7,3,0,0,0,0
	a1 = _mm256_blend_epi16(v0, v1, 0x8); //13
	a1 = _mm256_blend_epi16(a1, v2, 0x10); //-,-,-,-,16,-,-,-
	a1 = _mm256_blend_epi16(a1, v3, 0x60); //-,-,-,-,-,26,29,-

	__m256i v6 = _mm256_shuffle_epi8(a6, _mm256_set_epi8(13, 12, 9, 8, -1, -1, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, 13, 12, 9, 8, -1, -1, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1)); //-,-,48,49,50,-,52,54
	__m256i v7 = _mm256_shuffle_epi8(a7, _mm256_set_epi8(15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 7, 6, 1, 0, 9, 8, 15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 7, 6, 1, 0, 9, 8)); //60,56,59,61,57,58,62,63

	//a4 10, 19, 23, 32, 39, 45, 52, 54, 
	a4 = _mm256_blend_epi16(v1, v6, 0xC0); //10 + 52,54
	a4 = _mm256_blend_epi16(a4, v2, 0x6); //-,19,23,-,-,-,-,-
	a4 = _mm256_blend_epi16(a4, v4, 0x18); //32,39
	a4 = _mm256_blend_epi16(a4, v5, 0x20); //45

	__m256i x6 = _mm256_shuffle_epi8(a6, _mm256_set_epi8(11, 10, 15, 14, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 10, 15, 14, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)); //-,-,-,-,-,51,55,53

	//a3 9, 11, 18, 24, 31, 40, 44, 53, 
	a3 = _mm256_srli_si256(v1, 12); //9,11,-,-,-,-,-,-
	a3 = _mm256_blend_epi16(a3, v3, 0x18); //-,-,-,24,31,-,-,-
	a3 = _mm256_blend_epi16(a3, x6, 0x80); //53

	//a7 35, 36, 48, 49, 57, 58, 62, 63 
	a7 = _mm256_blend_epi16(v7, v4, 0x3); //-,-,-,-,57,58,62,63 + //35,36
	a7 = _mm256_blend_epi16(a7, v6, 0xC); //48,49

	//a6 21, 34, 37, 47, 50, 56, 59, 61,
	a6 = _mm256_slli_si256(v7, 8); //-,-,-,-,56,59,61
	a6 = _mm256_blend_epi16(a6, v4, 0x4); //-,-,37,-,-,-,-,-
	a6 = _mm256_blend_epi16(a6, v6, 0x10); //50
	a6 = _mm256_blend_epi16(a6, v5, 0x8); //47
	a6 = _mm256_blend_epi16(a6, v2, 0x1); //21

	v4 = _mm256_srli_si256(v4, 8); //-,34,33,38,-,-,-,-

	//a5 20, 22, 33, 38, 46, 51, 55, 60,
	v2 = _mm256_srli_si256(v2, 10);
	a5 = _mm256_slli_si256(v7, 14); //-,-,-,-,-,-,60
	a5 = _mm256_blend_epi16(a5, v4, 0xC); //33,38
	a5 = _mm256_blend_epi16(a5, v2, 0x3); //20,22
	a5 = _mm256_blend_epi16(a5, v5, 0x10); //46
	a5 = _mm256_blend_epi16(a5, x6, 0x60); //51,55

	a6 = _mm256_blend_epi16(a6, v4, 0x2); //34
	a3 = _mm256_blend_epi16(a3, v2, 0x4); //18

	v5 = _mm256_slli_si256(v5, 10); //-,-,-,-,-,40,44,42
	a3 = _mm256_blend_epi16(a3, v5, 0x60); //40,44
	a1 = _mm256_blend_epi16(a1, v5, 0x80); //42


	//load quant
	__m256i c0 = _mm256_load_si256((__m256i*) & matrix[0]);
	__m256i c1 = _mm256_load_si256((__m256i*) & matrix[16]);
	__m256i c2 = _mm256_load_si256((__m256i*) & matrix[32]);
	__m256i c3 = _mm256_load_si256((__m256i*) & matrix[48]);
	__m256i c4 = _mm256_load_si256((__m256i*) & matrix[64]);
	__m256i c5 = _mm256_load_si256((__m256i*) & matrix[80]);
	__m256i c6 = _mm256_load_si256((__m256i*) & matrix[96]);
	__m256i c7 = _mm256_load_si256((__m256i*) & matrix[112]);

	//multiply
	a0 = _mm256_mullo_epi16(a0, c0);
	a1 = _mm256_mullo_epi16(a1, c1);
	a2 = _mm256_mullo_epi16(a2, c2);
	a3 = _mm256_mullo_epi16(a3, c3);
	a4 = _mm256_mullo_epi16(a4, c4);
	a5 = _mm256_mullo_epi16(a5, c5);
	a6 = _mm256_mullo_epi16(a6, c6);
	a7 = _mm256_mullo_epi16(a7, c7);

	//__m256i mmax = _mm256_set1_epi16(4095);
	//__m256i mmin = _mm256_set1_epi16(-4095);
	//a0 = _mm256_min_epi16(a0, mmax);
	//a0 = _mm256_max_epi16(a0, mmin);

	//a1 = _mm256_min_epi16(a1, mmax);
	//a1 = _mm256_max_epi16(a1, mmin);

	//a2 = _mm256_min_epi16(a2, mmax);
	//a2 = _mm256_max_epi16(a2, mmin);

	//a3 = _mm256_min_epi16(a3, mmax);
	//a3 = _mm256_max_epi16(a3, mmin);

	//a4 = _mm256_min_epi16(a4, mmax);
	//a4 = _mm256_max_epi16(a4, mmin);

	//a5 = _mm256_min_epi16(a5, mmax);
	//a5 = _mm256_max_epi16(a5, mmin);

	//a6 = _mm256_min_epi16(a6, mmax);
	//a6 = _mm256_max_epi16(a6, mmin);

	//a7 = _mm256_min_epi16(a7, mmax);
	//a7 = _mm256_max_epi16(a7, mmin);

	//shift
	a0 = _mm256_srai_epi16(a0, 2);
	a1 = _mm256_srai_epi16(a1, 2);
	a2 = _mm256_srai_epi16(a2, 2);
	a3 = _mm256_srai_epi16(a3, 2);
	a4 = _mm256_srai_epi16(a4, 2);
	a5 = _mm256_srai_epi16(a5, 2);
	a6 = _mm256_srai_epi16(a6, 2);
	a7 = _mm256_srai_epi16(a7, 2);


	// /////////////////////////////
	// //////////Row 1 And row 3
	// ////////////////////////////
	__m256i r_xmm0 = a0;
	__m256i r_xmm4 = a2;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	__m256i r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_04_256[0]));
	__m256i r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_04_256[32]));
	__m256i r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_04_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_04_256[48]));
	__m256i r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	__m256i r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_26_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	__m256i r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_26_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_26_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_26_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row0 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row2 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 5 And row 7
	// ////////////////////////////
	r_xmm0 = a4;
	r_xmm4 = a6;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_04_256[0]));
	r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_04_256[32]));
	r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_04_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_04_256[48]));
	r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_26_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_26_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_26_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_26_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row4 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row6 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 4 And row 2
	// ////////////////////////////
	r_xmm0 = a3;
	r_xmm4 = a1;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_35_256[0]));
	r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_35_256[32]));
	r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_35_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_35_256[48]));
	r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_17_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_17_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_17_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_17_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row3 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row1 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 6 And row 8
	// ////////////////////////////
	r_xmm0 = a5;
	r_xmm4 = a7;
	r_xmm0 = _mm256_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm256_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm256_madd_epi16(r_xmm1, *((__m256i*) & tab_i_35_256[0]));
	r_xmm3 = _mm256_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm256_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm256_madd_epi16(r_xmm3, *((__m256i*) & tab_i_35_256[32]));
	r_xmm2 = _mm256_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm256_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm256_madd_epi16(r_xmm2, *((__m256i*) & tab_i_35_256[16]));
	r_xmm4 = _mm256_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm256_add_epi32(r_xmm1, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm256_madd_epi16(r_xmm0, *((__m256i*) & tab_i_35_256[48]));
	r_xmm5 = _mm256_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm256_madd_epi16(r_xmm5, *((__m256i*) & tab_i_17_256[0]));
	r_xmm1 = _mm256_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm256_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm256_madd_epi16(r_xmm6, *((__m256i*) & tab_i_17_256[16]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm256_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm256_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm256_madd_epi16(r_xmm7, *((__m256i*) & tab_i_17_256[32]));
	r_xmm0 = _mm256_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm256_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm256_add_epi32(r_xmm5, *((__m256i*) round_inv_row_256_10));
	r_xmm4 = _mm256_madd_epi16(r_xmm4, *((__m256i*) & tab_i_17_256[48]));
	r_xmm5 = _mm256_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm256_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm256_shuffle_epi32(r_xmm2, 0x1b);
	__m256i row5 = _mm256_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm256_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm256_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm256_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm256_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm256_shuffle_epi32(r_xmm6, 0x1b);
	__m256i row7 = _mm256_packs_epi32(r_xmm4, r_xmm6);
	r_xmm1 = _mm256_loadu_si256((__m256i*) tg_3_16_256);
	r_xmm2 = row5;
	r_xmm3 = row3;
	r_xmm0 = _mm256_mulhi_epi16(row5, r_xmm1);
	r_xmm1 = _mm256_mulhi_epi16(r_xmm1, r_xmm3);
	r_xmm5 = _mm256_loadu_si256((__m256i*) tg_1_16_256);
	r_xmm6 = row7;
	r_xmm4 = _mm256_mulhi_epi16(row7, r_xmm5);
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm2);
	r_xmm5 = _mm256_mulhi_epi16(r_xmm5, row1);
	r_xmm1 = _mm256_adds_epi16(r_xmm1, r_xmm3);
	r_xmm7 = row6;
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm3);
	r_xmm3 = _mm256_loadu_si256((__m256i*) tg_2_16_256);
	r_xmm2 = _mm256_subs_epi16(r_xmm2, r_xmm1);
	r_xmm7 = _mm256_mulhi_epi16(r_xmm7, r_xmm3);
	r_xmm1 = r_xmm0;
	r_xmm3 = _mm256_mulhi_epi16(r_xmm3, row2);
	r_xmm5 = _mm256_subs_epi16(r_xmm5, r_xmm6);
	r_xmm4 = _mm256_adds_epi16(r_xmm4, row1);
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm4);
	r_xmm0 = _mm256_adds_epi16(r_xmm0, *((__m256i*) one_corr_256));
	r_xmm4 = _mm256_subs_epi16(r_xmm4, r_xmm1);
	r_xmm6 = r_xmm5;
	r_xmm5 = _mm256_subs_epi16(r_xmm5, r_xmm2);
	r_xmm5 = _mm256_adds_epi16(r_xmm5, *((__m256i*) one_corr_256));
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm2);
	__m256i temp7 = r_xmm0;
	r_xmm1 = r_xmm4;
	r_xmm0 = _mm256_load_si256((__m256i*) cos_4_16_256);
	r_xmm4 = _mm256_adds_epi16(r_xmm4, r_xmm5);
	r_xmm2 = _mm256_load_si256((__m256i*) cos_4_16_256);
	r_xmm2 = _mm256_mulhi_epi16(r_xmm2, r_xmm4);
	__m256i temp3 = r_xmm6;
	r_xmm1 = _mm256_subs_epi16(r_xmm1, r_xmm5);
	r_xmm7 = _mm256_adds_epi16(r_xmm7, row2);
	r_xmm3 = _mm256_subs_epi16(r_xmm3, row6);
	r_xmm6 = row0;
	r_xmm0 = _mm256_mulhi_epi16(r_xmm0, r_xmm1);
	r_xmm5 = row4;
	r_xmm5 = _mm256_adds_epi16(r_xmm5, r_xmm6);
	r_xmm6 = _mm256_subs_epi16(r_xmm6, row4);
	r_xmm4 = _mm256_adds_epi16(r_xmm4, r_xmm2);
	r_xmm4 = _mm256_or_si256(r_xmm4, *((__m256i*) one_corr_256));
	r_xmm0 = _mm256_adds_epi16(r_xmm0, r_xmm1);
	r_xmm0 = _mm256_or_si256(r_xmm0, *((__m256i*) one_corr_256));
	r_xmm2 = r_xmm5;
	r_xmm5 = _mm256_adds_epi16(r_xmm5, r_xmm7);
	r_xmm1 = r_xmm6;
	r_xmm5 = _mm256_adds_epi16(r_xmm5, *((__m256i*) round_inv_col_256_10));
	r_xmm2 = _mm256_subs_epi16(r_xmm2, r_xmm7);
	r_xmm7 = temp7;
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm3);
	r_xmm6 = _mm256_adds_epi16(r_xmm6, *((__m256i*) round_inv_col_256_10));
	r_xmm7 = _mm256_adds_epi16(r_xmm7, r_xmm5);
	r_xmm7 = _mm256_srai_epi16(r_xmm7, SHIFT_INV_COL10);
	r_xmm1 = _mm256_subs_epi16(r_xmm1, r_xmm3);
	r_xmm1 = _mm256_adds_epi16(r_xmm1, *((__m256i*) round_inv_corr_256_10));
	r_xmm3 = r_xmm6;
	r_xmm2 = _mm256_adds_epi16(r_xmm2, *((__m256i*) round_inv_corr_256_10));
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm4);

	__m256i vadd = _mm256_set1_epi16(addVal);
	__m256i mmax = _mm256_set1_epi16(1023);
	__m256i mmin = _mm256_setzero_si256();

	r_xmm7 = _mm256_adds_epi16(r_xmm7, vadd);
	r_xmm7 = _mm256_min_epi16(r_xmm7, mmax);
	r_xmm7 = _mm256_max_epi16(r_xmm7, mmin);
	r_xmm7 = _mm256_slli_epi16(r_xmm7, 6);

	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm7);
	dst += stride;

	r_xmm6 = _mm256_srai_epi16(r_xmm6, SHIFT_INV_COL10);

	r_xmm6 = _mm256_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm256_min_epi16(r_xmm6, mmax);
	r_xmm6 = _mm256_max_epi16(r_xmm6, mmin);
	r_xmm6 = _mm256_slli_epi16(r_xmm6, 6);

	r_xmm7 = r_xmm1;
	r_xmm1 = _mm256_adds_epi16(r_xmm1, r_xmm0);

	r_xmm1 = _mm256_srai_epi16(r_xmm1, SHIFT_INV_COL10);

	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm6);
	dst += stride;

	r_xmm6 = temp3;
	r_xmm7 = _mm256_subs_epi16(r_xmm7, r_xmm0);
	r_xmm7 = _mm256_srai_epi16(r_xmm7, SHIFT_INV_COL10);

	r_xmm1 = _mm256_adds_epi16(r_xmm1, vadd);
	r_xmm1 = _mm256_min_epi16(r_xmm1, mmax);
	r_xmm1 = _mm256_max_epi16(r_xmm1, mmin);
	r_xmm1 = _mm256_slli_epi16(r_xmm1, 6);

	r_xmm5 = _mm256_subs_epi16(r_xmm5, temp7);
	r_xmm5 = _mm256_srai_epi16(r_xmm5, SHIFT_INV_COL10);

	r_xmm5 = _mm256_adds_epi16(r_xmm5, vadd);
	r_xmm5 = _mm256_min_epi16(r_xmm5, mmax);
	r_xmm5 = _mm256_max_epi16(r_xmm5, mmin);
	r_xmm5 = _mm256_slli_epi16(r_xmm5, 6);

	r_xmm3 = _mm256_subs_epi16(r_xmm3, r_xmm4);
	r_xmm6 = _mm256_adds_epi16(r_xmm6, r_xmm2);
	r_xmm2 = _mm256_subs_epi16(r_xmm2, temp3);
	r_xmm6 = _mm256_srai_epi16(r_xmm6, SHIFT_INV_COL10);
	r_xmm2 = _mm256_srai_epi16(r_xmm2, SHIFT_INV_COL10);

	r_xmm6 = _mm256_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm256_min_epi16(r_xmm6, mmax);
	r_xmm6 = _mm256_max_epi16(r_xmm6, mmin);
	r_xmm6 = _mm256_slli_epi16(r_xmm6, 6);

	r_xmm3 = _mm256_srai_epi16(r_xmm3, SHIFT_INV_COL10);

	r_xmm2 = _mm256_adds_epi16(r_xmm2, vadd);
	r_xmm2 = _mm256_min_epi16(r_xmm2, mmax);
	r_xmm2 = _mm256_max_epi16(r_xmm2, mmin);
	r_xmm2 = _mm256_slli_epi16(r_xmm2, 6);

	r_xmm7 = _mm256_adds_epi16(r_xmm7, vadd);
	r_xmm7 = _mm256_min_epi16(r_xmm7, mmax);
	r_xmm7 = _mm256_max_epi16(r_xmm7, mmin);
	r_xmm7 = _mm256_slli_epi16(r_xmm7, 6);

	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm1);
	dst += stride;
	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm6);
	dst += stride;

	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm2);
	dst += stride;
	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm7);
	dst += stride;

	r_xmm3 = _mm256_adds_epi16(r_xmm3, vadd);
	r_xmm3 = _mm256_min_epi16(r_xmm3, mmax);
	r_xmm3 = _mm256_max_epi16(r_xmm3, mmin);
	r_xmm3 = _mm256_slli_epi16(r_xmm3, 6);

	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm3);
	dst += stride;
	_mm256_storeu_si256((__m256i*) (&dst[0]), r_xmm5);
	dst += stride;

}

//Forward DCT8x8 + Quantize + ZigZag
void VMX_FDCT_8X8_QUANT_ZIG_256(const BYTE* src, int stride, unsigned short* matrix, short addVal, __m256i* out0, __m256i* out1, __m256i* out2, __m256i* out3, __m256i* out4, __m256i* out5, __m256i* out6, __m256i* out7) {

	// Load input
	__m128i iin0 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin1 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin2 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin3 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin4 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin5 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin6 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i iin7 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;

	__m256i in0 = _mm256_cvtepu8_epi16(iin0);
	__m256i in1 = _mm256_cvtepu8_epi16(iin1);
	__m256i in2 = _mm256_cvtepu8_epi16(iin2);
	__m256i in3 = _mm256_cvtepu8_epi16(iin3);
	__m256i in4 = _mm256_cvtepu8_epi16(iin4);
	__m256i in5 = _mm256_cvtepu8_epi16(iin5);
	__m256i in6 = _mm256_cvtepu8_epi16(iin6);
	__m256i in7 = _mm256_cvtepu8_epi16(iin7);

	__m256i vadd = _mm256_set1_epi16(addVal);
	in0 = _mm256_adds_epi16(in0, vadd);
	in1 = _mm256_adds_epi16(in1, vadd);
	in2 = _mm256_adds_epi16(in2, vadd);
	in3 = _mm256_adds_epi16(in3, vadd);
	in4 = _mm256_adds_epi16(in4, vadd);
	in5 = _mm256_adds_epi16(in5, vadd);
	in6 = _mm256_adds_epi16(in6, vadd);
	in7 = _mm256_adds_epi16(in7, vadd);


	// Load input data
	__m256i xmm0 = in0; 
	__m256i xmm2 = in2;
	__m256i xmm3 = xmm0;
	__m256i xmm4 = xmm2;
	__m256i xmm7 = in7;
	__m256i xmm5 = in5;

	// First stage
	xmm0 = _mm256_subs_epi16(xmm0, xmm7); 
	xmm7 = _mm256_adds_epi16(xmm7, xmm3);         
	xmm2 = _mm256_subs_epi16(xmm2, xmm5);        
	xmm5 = _mm256_adds_epi16(xmm5, xmm4); 

	xmm3 = in3;
	xmm4 = in4;
	__m256i xmm1 = xmm3;
	xmm3 = _mm256_subs_epi16(xmm3, xmm4);
	xmm4 = _mm256_adds_epi16(xmm4, xmm1);

	__m256i xmm6 = in6;
	xmm1 = in1;
	__m256i tmp = xmm1;
	xmm1 = _mm256_subs_epi16(xmm1, xmm6);
	xmm6 = _mm256_adds_epi16(xmm6, tmp);

	// Second stage
	__m256i tm03 = _mm256_subs_epi16(xmm7, xmm4);
	__m256i tm12 = _mm256_subs_epi16(xmm6, xmm5);
	xmm4 = _mm256_adds_epi16(xmm4, xmm4);
	xmm5 = _mm256_adds_epi16(xmm5, xmm5);

	__m256i tp03 = _mm256_adds_epi16(xmm4, tm03);
	__m256i tp12 = _mm256_adds_epi16(xmm5, tm12);

	// Shift operations
	xmm2 = _mm256_slli_epi16(xmm2, SHIFT_FRW_COL + 1);  
	xmm1 = _mm256_slli_epi16(xmm1, SHIFT_FRW_COL + 1);    
	tp03 = _mm256_slli_epi16(tp03, SHIFT_FRW_COL);        
	tp12 = _mm256_slli_epi16(tp12, SHIFT_FRW_COL);   
	tm03 = _mm256_slli_epi16(tm03, SHIFT_FRW_COL);  
	tm12 = _mm256_slli_epi16(tm12, SHIFT_FRW_COL);  
	xmm3 = _mm256_slli_epi16(xmm3, SHIFT_FRW_COL); 
	xmm0 = _mm256_slli_epi16(xmm0, SHIFT_FRW_COL);  

	// Output calculations
	in4 = _mm256_subs_epi16(tp03, tp12); 
	__m256i diff = _mm256_subs_epi16(xmm1, xmm2); 
	tp12 = _mm256_adds_epi16(tp12, tp12);
	xmm2 = _mm256_adds_epi16(xmm2, xmm2);
	in0 = _mm256_adds_epi16(tp12, in4); 

	__m256i sum = _mm256_adds_epi16(xmm2, diff); 

	// Tan2
	__m256i tan2v = _mm256_set1_epi16(FDCT_TAN2);
	__m256i tmp1 = _mm256_mulhi_epi16(tan2v, tm03);
	in6 = _mm256_subs_epi16(tmp1, tm12); 
	__m256i tmp2 = _mm256_mulhi_epi16(tan2v, tm12);
	in2 = _mm256_adds_epi16(tmp2, tm03); 

	// Sqrt2
	__m256i sqrt2v = _mm256_set1_epi16(FDCT_SQRT2);
	__m256i rounder = _mm256_set1_epi16(FDCT_ROUND1);

	__m256i tp65 = _mm256_mulhi_epi16(sum, sqrt2v);
	in2 = _mm256_or_si256(in2, rounder);
	in6 = _mm256_or_si256(in6, rounder);
	__m256i tm65 = _mm256_mulhi_epi16(diff, sqrt2v);
	tp65 = _mm256_or_si256(tp65, rounder);

	// Final calculations
	__m256i tm465 = _mm256_subs_epi16(xmm3, tm65);
	__m256i tm765 = _mm256_subs_epi16(xmm0, tp65);
	__m256i tp765 = _mm256_adds_epi16(tp65, xmm0);
	__m256i tp465 = _mm256_adds_epi16(tm65, xmm3);

	__m256i tan3v = _mm256_set1_epi16(FDCT_TAN3);
	__m256i tan1v = _mm256_set1_epi16(FDCT_TAN1);

	__m256i tmp3 = _mm256_mulhi_epi16(tm465, tan3v);
	__m256i tmp4 = _mm256_mulhi_epi16(tp465, tan1v);
	tmp3 = _mm256_adds_epi16(tmp3, tm465);

	__m256i tmp5 = _mm256_mulhi_epi16(tm765, tan3v);
	tmp5 = _mm256_adds_epi16(tmp5, tm765);
	__m256i tmp6 = _mm256_mulhi_epi16(tp765, tan1v);

	in1 = _mm256_adds_epi16(tmp4, tp765);
	in3 = _mm256_subs_epi16(tm765, tmp3);
	in5 = _mm256_adds_epi16(tm465, tmp5);
	in7 = _mm256_subs_epi16(tmp6, tp465);

	__m256i round = _mm256_set1_epi32(RND_FRW_ROW);

	/////////////////////
	//ROW 0
	/////////////////////
	xmm0 = in0;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	__m256i temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[16]));
	__m256i temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[32]));
	__m256i temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[48]));
	__m256i temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in0 = xmm0;

	/////////////////////
	//ROW 1
	/////////////////////
	xmm0 = in1;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in1 = xmm0;

	/////////////////////
	//ROW 2
	/////////////////////
	xmm0 = in2;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in2 = xmm0;

	/////////////////////
	//ROW 3
	/////////////////////
	xmm0 = in3;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in3 = xmm0;

	/////////////////////
	//ROW 4
	/////////////////////
	xmm0 = in4;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in4 = xmm0;

	/////////////////////
	//ROW 5
	/////////////////////
	xmm0 = in5;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in5 = xmm0;

	/////////////////////
	//ROW 6
	/////////////////////
	xmm0 = in6;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in6 = xmm0;

	/////////////////////
	//ROW 7
	/////////////////////
	xmm0 = in7;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011 );
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100 );
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in7 = xmm0;

	//===================
	//BEGIN QUANTIZATION. We eliminate the 4 shift back and forth by doing DCT and QUANTIZE together! 
	// This also improves the quality slightly due to less rounding errors...
	//===================

	//abs source
	__m256i b0 = _mm256_abs_epi16(in0);
	__m256i b1 = _mm256_abs_epi16(in1);
	__m256i b2 = _mm256_abs_epi16(in2);
	__m256i b3 = _mm256_abs_epi16(in3);
	__m256i b4 = _mm256_abs_epi16(in4);
	__m256i b5 = _mm256_abs_epi16(in5);
	__m256i b6 = _mm256_abs_epi16(in6);
	__m256i b7 = _mm256_abs_epi16(in7);

	//////shift 4 bits
	//b0 = _mm256_slli_epi16(b0, 4);
	//b1 = _mm256_slli_epi16(b1, 4);
	//b2 = _mm256_slli_epi16(b2, 4);
	//b3 = _mm256_slli_epi16(b3, 4);
	//b4 = _mm256_slli_epi16(b4, 4);
	//b5 = _mm256_slli_epi16(b5, 4);
	//b6 = _mm256_slli_epi16(b6, 4);
	//b7 = _mm256_slli_epi16(b7, 4);

	//load correction
	__m256i c0 = _mm256_load_si256((__m256i*) & matrix[0]);
	__m256i c1 = _mm256_load_si256((__m256i*) & matrix[16]);
	__m256i c2 = _mm256_load_si256((__m256i*) & matrix[32]);
	__m256i c3 = _mm256_load_si256((__m256i*) & matrix[48]);
	__m256i c4 = _mm256_load_si256((__m256i*) & matrix[64]);
	__m256i c5 = _mm256_load_si256((__m256i*) & matrix[80]);
	__m256i c6 = _mm256_load_si256((__m256i*) & matrix[96]);
	__m256i c7 = _mm256_load_si256((__m256i*) & matrix[112]);

	//add correction
	b0 = _mm256_add_epi16(b0, c0);
	b1 = _mm256_add_epi16(b1, c1);
	b2 = _mm256_add_epi16(b2, c2);
	b3 = _mm256_add_epi16(b3, c3);
	b4 = _mm256_add_epi16(b4, c4);
	b5 = _mm256_add_epi16(b5, c5);
	b6 = _mm256_add_epi16(b6, c6);
	b7 = _mm256_add_epi16(b7, c7);

	//load reciprocal
	c0 = _mm256_load_si256((__m256i*) & matrix[128]);
	c1 = _mm256_load_si256((__m256i*) & matrix[144]);
	c2 = _mm256_load_si256((__m256i*) & matrix[160]);
	c3 = _mm256_load_si256((__m256i*) & matrix[176]);
	c4 = _mm256_load_si256((__m256i*) & matrix[192]);
	c5 = _mm256_load_si256((__m256i*) & matrix[208]);
	c6 = _mm256_load_si256((__m256i*) & matrix[224]);
	c7 = _mm256_load_si256((__m256i*) & matrix[240]);


	//multiply reciprocal
	b0 = _mm256_mulhi_epu16(b0, c0);
	b1 = _mm256_mulhi_epu16(b1, c1);
	b2 = _mm256_mulhi_epu16(b2, c2);
	b3 = _mm256_mulhi_epu16(b3, c3);
	b4 = _mm256_mulhi_epu16(b4, c4);
	b5 = _mm256_mulhi_epu16(b5, c5);
	b6 = _mm256_mulhi_epu16(b6, c6);
	b7 = _mm256_mulhi_epu16(b7, c7);

	//load scale
	c0 = _mm256_load_si256((__m256i*) & matrix[256]);
	c1 = _mm256_load_si256((__m256i*) & matrix[272]);
	c2 = _mm256_load_si256((__m256i*) & matrix[288]);
	c3 = _mm256_load_si256((__m256i*) & matrix[304]);
	c4 = _mm256_load_si256((__m256i*) & matrix[320]);
	c5 = _mm256_load_si256((__m256i*) & matrix[336]);
	c6 = _mm256_load_si256((__m256i*) & matrix[352]);
	c7 = _mm256_load_si256((__m256i*) & matrix[368]);

	//multiply scale
	b0 = _mm256_mulhi_epu16(b0, c0);
	b1 = _mm256_mulhi_epu16(b1, c1);
	b2 = _mm256_mulhi_epu16(b2, c2);
	b3 = _mm256_mulhi_epu16(b3, c3);
	b4 = _mm256_mulhi_epu16(b4, c4);
	b5 = _mm256_mulhi_epu16(b5, c5);
	b6 = _mm256_mulhi_epu16(b6, c6);
	b7 = _mm256_mulhi_epu16(b7, c7);

	//sign
	in0 = _mm256_sign_epi16(b0, in0);
	in1 = _mm256_sign_epi16(b1, in1);
	in2 = _mm256_sign_epi16(b2, in2);
	in3 = _mm256_sign_epi16(b3, in3);
	in4 = _mm256_sign_epi16(b4, in4);
	in5 = _mm256_sign_epi16(b5, in5);
	in6 = _mm256_sign_epi16(b6, in6);
	in7 = _mm256_sign_epi16(b7, in7);

	//===================
	//BEGIN ZIG ZAG. ~38 instructions. Probably not optimal, but sure beats using a loop!
	//===================

	__m256i r0 = _mm256_unpacklo_epi16(in0, in1); 	//0	8	1	9	2	10	3	11
	__m256i r1 = _mm256_unpackhi_epi16(in0, in1); 	//4	12	5	13	6	14	7	15
	__m256i r2 = _mm256_unpacklo_epi16(in2, in3); 	// 16	24	17	25	18	26	19	27
	__m256i r3 = _mm256_unpackhi_epi16(in2, in3); 	//20	28	21	29	22	30	23	31
	__m256i r4 = _mm256_unpacklo_epi16(in4, in5); 	//32	40	33	41	34	42	35	43
	__m256i r5 = _mm256_unpackhi_epi16(in4, in5); 	//36	44	37	45	38	46	39	47
	__m256i r6 = _mm256_unpacklo_epi16(in6, in7); 	//48	56	49	57	50	58	51	59
	__m256i r7 = _mm256_unpackhi_epi16(in6, in7); 	//52	60	53	61	54	62	55	63		

	in0 = _mm256_shuffle_epi8(r0, _mm256_set_epi8(11, 10, 13, 12, 9, 8, 7, 6, -1, -1, 3, 2, 5, 4, 1, 0, 11, 10, 13, 12, 9, 8, 7, 6, -1, -1, 3, 2, 5, 4, 1, 0));
	in2 = _mm256_slli_si256(in2, 6);
	in0 = _mm256_blend_epi16(in0, in2, 0x8);
	__m256i t = _mm256_blend_epi16(r2, r4, 1);
	in1 = _mm256_blend_epi16(r0, r1, 5);
	t = _mm256_shuffle_epi8(t, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 1, 0, 3, 2, 5, 4, -1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 1, 0, 3, 2, 5, 4));
	in1 = _mm256_shuffle_epi8(in1, _mm256_set_epi8(5, 4, 1, 0, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 4, 1, 0, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
	in1 = _mm256_or_si256(in1, t);
	t = _mm256_alignr_epi8(r6, r1, 2);
	in2 = _mm256_blend_epi16(r4, r2, 96);
	in2 = _mm256_blend_epi16(in2, t, 1 + 128);
	in2 = _mm256_shuffle_epi8(in2, _mm256_set_epi8(9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0));
	t = _mm256_blend_epi16(r1, r3, 7);
	t = _mm256_blend_epi16(t, r2, 128);
	in3 = _mm256_shuffle_epi8(t, _mm256_set_epi8(3, 2, 5, 4, 11, 10, 13, 12, 9, 8, 7, 6, 1, 0, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 9, 8, 7, 6, 1, 0, 15, 14));
	t = _mm256_blend_epi16(r4, r6, 30);
	t = _mm256_blend_epi16(t, r5, 1);
	in4 = _mm256_shuffle_epi8(t, _mm256_set_epi8(1, 0, 15, 14, 9, 8, 7, 6, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 15, 14, 9, 8, 7, 6, 3, 2, 5, 4, 11, 10, 13, 12));
	t = _mm256_alignr_epi8(r6, r1, 14);
	in5 = _mm256_blend_epi16(r3, r5, 6);
	in5 = _mm256_blend_epi16(in5, t, 1 + 128);
	in5 = _mm256_shuffle_epi8(in5, _mm256_set_epi8(15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6));
	in6 = _mm256_blend_epi16(r5, r3, 128);
	t = _mm256_blend_epi16(r6, r7, 1);
	t = _mm256_shuffle_epi8(t, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 15, 14, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 15, 14, 11, 10));
	in6 = _mm256_shuffle_epi8(in6, _mm256_set_epi8(11, 10, 13, 12, 15, 14, 9, 8, 7, 6, -1, -1, -1, -1, -1, -1, 11, 10, 13, 12, 15, 14, 9, 8, 7, 6, -1, -1, -1, -1, -1, -1));
	r5 = _mm256_srli_si256(r5, 6);
	in6 = _mm256_or_si256(in6, t);
	in7 = _mm256_shuffle_epi8(r7, _mm256_set_epi8(15, 14, 11, 10, 13, 12, -1, -1, 9, 8, 7, 6, 3, 2, 5, 4, 15, 14, 11, 10, 13, 12, -1, -1, 9, 8, 7, 6, 3, 2, 5, 4));
	in7 = _mm256_blend_epi16(in7, r5, 0x10);

	*out0 = in0;
	*out1 = in1;
	*out2 = in2;
	*out3 = in3;
	*out4 = in4;
	*out5 = in5;
	*out6 = in6;
	*out7 = in7;
}


//Forward DCT8x8 + Quantize + ZigZag for 16bit input
void VMX_FDCT_8X8_QUANT_ZIG_256_16(const BYTE* src, int stride, unsigned short* matrix, short addVal, __m256i* out0, __m256i* out1, __m256i* out2, __m256i* out3, __m256i* out4, __m256i* out5, __m256i* out6, __m256i* out7) {

	// Load input
	__m256i in0 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in1 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in2 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in3 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in4 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in5 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in6 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;
	__m256i in7 = _mm256_loadu_si256((__m256i*) & src[0]);
	src += stride;

	//Shift down by 6 bits, as actual DCT range is between -512 and +512
	in0 = _mm256_srli_epi16(in0, 6);
	in1 = _mm256_srli_epi16(in1, 6);
	in2 = _mm256_srli_epi16(in2, 6);
	in3 = _mm256_srli_epi16(in3, 6);
	in4 = _mm256_srli_epi16(in4, 6);
	in5 = _mm256_srli_epi16(in5, 6);
	in6 = _mm256_srli_epi16(in6, 6);
	in7 = _mm256_srli_epi16(in7, 6);

	__m256i vadd = _mm256_set1_epi16(addVal);
	in0 = _mm256_adds_epi16(in0, vadd);
	in1 = _mm256_adds_epi16(in1, vadd);
	in2 = _mm256_adds_epi16(in2, vadd);
	in3 = _mm256_adds_epi16(in3, vadd);
	in4 = _mm256_adds_epi16(in4, vadd);
	in5 = _mm256_adds_epi16(in5, vadd);
	in6 = _mm256_adds_epi16(in6, vadd);
	in7 = _mm256_adds_epi16(in7, vadd);


	// Load input data
	__m256i xmm0 = in0;
	__m256i xmm2 = in2;
	__m256i xmm3 = xmm0;
	__m256i xmm4 = xmm2;
	__m256i xmm7 = in7;
	__m256i xmm5 = in5;

	// First stage
	xmm0 = _mm256_subs_epi16(xmm0, xmm7);
	xmm7 = _mm256_adds_epi16(xmm7, xmm3);
	xmm2 = _mm256_subs_epi16(xmm2, xmm5);
	xmm5 = _mm256_adds_epi16(xmm5, xmm4);

	xmm3 = in3;
	xmm4 = in4;
	__m256i xmm1 = xmm3;
	xmm3 = _mm256_subs_epi16(xmm3, xmm4);
	xmm4 = _mm256_adds_epi16(xmm4, xmm1);

	__m256i xmm6 = in6;
	xmm1 = in1;
	__m256i tmp = xmm1;
	xmm1 = _mm256_subs_epi16(xmm1, xmm6);
	xmm6 = _mm256_adds_epi16(xmm6, tmp);

	// Second stage
	__m256i tm03 = _mm256_subs_epi16(xmm7, xmm4);
	__m256i tm12 = _mm256_subs_epi16(xmm6, xmm5);
	xmm4 = _mm256_adds_epi16(xmm4, xmm4);
	xmm5 = _mm256_adds_epi16(xmm5, xmm5);

	__m256i tp03 = _mm256_adds_epi16(xmm4, tm03);
	__m256i tp12 = _mm256_adds_epi16(xmm5, tm12);

	// Shift operations
	xmm2 = _mm256_slli_epi16(xmm2, SHIFT_FRW_COL10 + 1);
	xmm1 = _mm256_slli_epi16(xmm1, SHIFT_FRW_COL10 + 1);
	tp03 = _mm256_slli_epi16(tp03, SHIFT_FRW_COL10);
	tp12 = _mm256_slli_epi16(tp12, SHIFT_FRW_COL10);
	tm03 = _mm256_slli_epi16(tm03, SHIFT_FRW_COL10);
	tm12 = _mm256_slli_epi16(tm12, SHIFT_FRW_COL10);
	xmm3 = _mm256_slli_epi16(xmm3, SHIFT_FRW_COL10);
	xmm0 = _mm256_slli_epi16(xmm0, SHIFT_FRW_COL10);

	// Output calculations
	in4 = _mm256_subs_epi16(tp03, tp12);
	__m256i diff = _mm256_subs_epi16(xmm1, xmm2);
	tp12 = _mm256_adds_epi16(tp12, tp12);
	xmm2 = _mm256_adds_epi16(xmm2, xmm2);
	in0 = _mm256_adds_epi16(tp12, in4);

	__m256i sum = _mm256_adds_epi16(xmm2, diff);

	// Tan2
	__m256i tan2v = _mm256_set1_epi16(FDCT_TAN2);
	__m256i tmp1 = _mm256_mulhi_epi16(tan2v, tm03);
	in6 = _mm256_subs_epi16(tmp1, tm12);
	__m256i tmp2 = _mm256_mulhi_epi16(tan2v, tm12);
	in2 = _mm256_adds_epi16(tmp2, tm03);

	// Sqrt2
	__m256i sqrt2v = _mm256_set1_epi16(FDCT_SQRT2);
	__m256i rounder = _mm256_set1_epi16(FDCT_ROUND1);

	__m256i tp65 = _mm256_mulhi_epi16(sum, sqrt2v);
	in2 = _mm256_or_si256(in2, rounder);
	in6 = _mm256_or_si256(in6, rounder);
	__m256i tm65 = _mm256_mulhi_epi16(diff, sqrt2v);
	tp65 = _mm256_or_si256(tp65, rounder);

	// Final calculations
	__m256i tm465 = _mm256_subs_epi16(xmm3, tm65);
	__m256i tm765 = _mm256_subs_epi16(xmm0, tp65);
	__m256i tp765 = _mm256_adds_epi16(tp65, xmm0);
	__m256i tp465 = _mm256_adds_epi16(tm65, xmm3);

	__m256i tan3v = _mm256_set1_epi16(FDCT_TAN3);
	__m256i tan1v = _mm256_set1_epi16(FDCT_TAN1);

	__m256i tmp3 = _mm256_mulhi_epi16(tm465, tan3v);
	__m256i tmp4 = _mm256_mulhi_epi16(tp465, tan1v);
	tmp3 = _mm256_adds_epi16(tmp3, tm465);

	__m256i tmp5 = _mm256_mulhi_epi16(tm765, tan3v);
	tmp5 = _mm256_adds_epi16(tmp5, tm765);
	__m256i tmp6 = _mm256_mulhi_epi16(tp765, tan1v);

	in1 = _mm256_adds_epi16(tmp4, tp765);
	in3 = _mm256_subs_epi16(tm765, tmp3);
	in5 = _mm256_adds_epi16(tm465, tmp5);
	in7 = _mm256_subs_epi16(tmp6, tp465);

	__m256i round = _mm256_set1_epi32(RND_FRW_ROW10);

	/////////////////////
	//ROW 0
	/////////////////////
	xmm0 = in0;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	__m256i temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[16]));
	__m256i temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[32]));
	__m256i temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[48]));
	__m256i temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in0 = xmm0;

	/////////////////////
	//ROW 1
	/////////////////////
	xmm0 = in1;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in1 = xmm0;

	/////////////////////
	//ROW 2
	/////////////////////
	xmm0 = in2;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in2 = xmm0;

	/////////////////////
	//ROW 3
	/////////////////////
	xmm0 = in3;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in3 = xmm0;

	/////////////////////
	//ROW 4
	/////////////////////
	xmm0 = in4;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab1_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab1_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in4 = xmm0;

	/////////////////////
	//ROW 5
	/////////////////////
	xmm0 = in5;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab4_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab4_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in5 = xmm0;

	/////////////////////
	//ROW 6
	/////////////////////
	xmm0 = in6;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab3_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab3_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in6 = xmm0;

	/////////////////////
	//ROW 7
	/////////////////////
	xmm0 = in7;
	xmm1 = _mm256_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm256_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm256_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm256_adds_epi16(xmm0, xmm1);
	xmm2 = _mm256_subs_epi16(xmm2, xmm1);

	xmm0 = _mm256_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm256_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[16]));
	temp2 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[32]));
	temp3 = _mm256_madd_epi16(xmm2, *((__m256i*) & ftab2_256[48]));
	temp4 = _mm256_madd_epi16(xmm0, *((__m256i*) & ftab2_256[0]));

	xmm0 = _mm256_add_epi32(temp4, temp1);
	xmm2 = _mm256_add_epi32(temp3, temp2);

	xmm0 = _mm256_add_epi32(xmm0, round);
	xmm2 = _mm256_add_epi32(xmm2, round);

	xmm0 = _mm256_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm256_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm256_packs_epi32(xmm0, xmm2);

	in7 = xmm0;

	//===================
	//BEGIN QUANTIZATION. We eliminate the 4 shift back and forth by doing DCT and QUANTIZE together! 
	// This also improves the quality slightly due to less rounding errors...
	//===================

	//abs source
	__m256i b0 = _mm256_abs_epi16(in0);
	__m256i b1 = _mm256_abs_epi16(in1);
	__m256i b2 = _mm256_abs_epi16(in2);
	__m256i b3 = _mm256_abs_epi16(in3);
	__m256i b4 = _mm256_abs_epi16(in4);
	__m256i b5 = _mm256_abs_epi16(in5);
	__m256i b6 = _mm256_abs_epi16(in6);
	__m256i b7 = _mm256_abs_epi16(in7);

	//////shift 4 bits
	//b0 = _mm256_slli_epi16(b0, 4);
	//b1 = _mm256_slli_epi16(b1, 4);
	//b2 = _mm256_slli_epi16(b2, 4);
	//b3 = _mm256_slli_epi16(b3, 4);
	//b4 = _mm256_slli_epi16(b4, 4);
	//b5 = _mm256_slli_epi16(b5, 4);
	//b6 = _mm256_slli_epi16(b6, 4);
	//b7 = _mm256_slli_epi16(b7, 4);

	//load correction
	__m256i c0 = _mm256_load_si256((__m256i*) & matrix[0]);
	__m256i c1 = _mm256_load_si256((__m256i*) & matrix[16]);
	__m256i c2 = _mm256_load_si256((__m256i*) & matrix[32]);
	__m256i c3 = _mm256_load_si256((__m256i*) & matrix[48]);
	__m256i c4 = _mm256_load_si256((__m256i*) & matrix[64]);
	__m256i c5 = _mm256_load_si256((__m256i*) & matrix[80]);
	__m256i c6 = _mm256_load_si256((__m256i*) & matrix[96]);
	__m256i c7 = _mm256_load_si256((__m256i*) & matrix[112]);

	//add correction
	b0 = _mm256_add_epi16(b0, c0);
	b1 = _mm256_add_epi16(b1, c1);
	b2 = _mm256_add_epi16(b2, c2);
	b3 = _mm256_add_epi16(b3, c3);
	b4 = _mm256_add_epi16(b4, c4);
	b5 = _mm256_add_epi16(b5, c5);
	b6 = _mm256_add_epi16(b6, c6);
	b7 = _mm256_add_epi16(b7, c7);

	//load reciprocal
	c0 = _mm256_load_si256((__m256i*) & matrix[128]);
	c1 = _mm256_load_si256((__m256i*) & matrix[144]);
	c2 = _mm256_load_si256((__m256i*) & matrix[160]);
	c3 = _mm256_load_si256((__m256i*) & matrix[176]);
	c4 = _mm256_load_si256((__m256i*) & matrix[192]);
	c5 = _mm256_load_si256((__m256i*) & matrix[208]);
	c6 = _mm256_load_si256((__m256i*) & matrix[224]);
	c7 = _mm256_load_si256((__m256i*) & matrix[240]);


	//multiply reciprocal
	b0 = _mm256_mulhi_epu16(b0, c0);
	b1 = _mm256_mulhi_epu16(b1, c1);
	b2 = _mm256_mulhi_epu16(b2, c2);
	b3 = _mm256_mulhi_epu16(b3, c3);
	b4 = _mm256_mulhi_epu16(b4, c4);
	b5 = _mm256_mulhi_epu16(b5, c5);
	b6 = _mm256_mulhi_epu16(b6, c6);
	b7 = _mm256_mulhi_epu16(b7, c7);

	//load scale
	c0 = _mm256_load_si256((__m256i*) & matrix[256]);
	c1 = _mm256_load_si256((__m256i*) & matrix[272]);
	c2 = _mm256_load_si256((__m256i*) & matrix[288]);
	c3 = _mm256_load_si256((__m256i*) & matrix[304]);
	c4 = _mm256_load_si256((__m256i*) & matrix[320]);
	c5 = _mm256_load_si256((__m256i*) & matrix[336]);
	c6 = _mm256_load_si256((__m256i*) & matrix[352]);
	c7 = _mm256_load_si256((__m256i*) & matrix[368]);

	//multiply scale
	b0 = _mm256_mulhi_epu16(b0, c0);
	b1 = _mm256_mulhi_epu16(b1, c1);
	b2 = _mm256_mulhi_epu16(b2, c2);
	b3 = _mm256_mulhi_epu16(b3, c3);
	b4 = _mm256_mulhi_epu16(b4, c4);
	b5 = _mm256_mulhi_epu16(b5, c5);
	b6 = _mm256_mulhi_epu16(b6, c6);
	b7 = _mm256_mulhi_epu16(b7, c7);

	//sign
	in0 = _mm256_sign_epi16(b0, in0);
	in1 = _mm256_sign_epi16(b1, in1);
	in2 = _mm256_sign_epi16(b2, in2);
	in3 = _mm256_sign_epi16(b3, in3);
	in4 = _mm256_sign_epi16(b4, in4);
	in5 = _mm256_sign_epi16(b5, in5);
	in6 = _mm256_sign_epi16(b6, in6);
	in7 = _mm256_sign_epi16(b7, in7);

	//===================
	//BEGIN ZIG ZAG. ~38 instructions. Probably not optimal, but sure beats using a loop!
	//===================

	__m256i r0 = _mm256_unpacklo_epi16(in0, in1); 	//0	8	1	9	2	10	3	11
	__m256i r1 = _mm256_unpackhi_epi16(in0, in1); 	//4	12	5	13	6	14	7	15
	__m256i r2 = _mm256_unpacklo_epi16(in2, in3); 	// 16	24	17	25	18	26	19	27
	__m256i r3 = _mm256_unpackhi_epi16(in2, in3); 	//20	28	21	29	22	30	23	31
	__m256i r4 = _mm256_unpacklo_epi16(in4, in5); 	//32	40	33	41	34	42	35	43
	__m256i r5 = _mm256_unpackhi_epi16(in4, in5); 	//36	44	37	45	38	46	39	47
	__m256i r6 = _mm256_unpacklo_epi16(in6, in7); 	//48	56	49	57	50	58	51	59
	__m256i r7 = _mm256_unpackhi_epi16(in6, in7); 	//52	60	53	61	54	62	55	63		

	in0 = _mm256_shuffle_epi8(r0, _mm256_set_epi8(11, 10, 13, 12, 9, 8, 7, 6, -1, -1, 3, 2, 5, 4, 1, 0, 11, 10, 13, 12, 9, 8, 7, 6, -1, -1, 3, 2, 5, 4, 1, 0));
	in2 = _mm256_slli_si256(in2, 6);
	in0 = _mm256_blend_epi16(in0, in2, 0x8);
	__m256i t = _mm256_blend_epi16(r2, r4, 1);
	in1 = _mm256_blend_epi16(r0, r1, 5);
	t = _mm256_shuffle_epi8(t, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 1, 0, 3, 2, 5, 4, -1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 1, 0, 3, 2, 5, 4));
	in1 = _mm256_shuffle_epi8(in1, _mm256_set_epi8(5, 4, 1, 0, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 4, 1, 0, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
	in1 = _mm256_or_si256(in1, t);
	t = _mm256_alignr_epi8(r6, r1, 2);
	in2 = _mm256_blend_epi16(r4, r2, 96);
	in2 = _mm256_blend_epi16(in2, t, 1 + 128);
	in2 = _mm256_shuffle_epi8(in2, _mm256_set_epi8(9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0));
	t = _mm256_blend_epi16(r1, r3, 7);
	t = _mm256_blend_epi16(t, r2, 128);
	in3 = _mm256_shuffle_epi8(t, _mm256_set_epi8(3, 2, 5, 4, 11, 10, 13, 12, 9, 8, 7, 6, 1, 0, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 9, 8, 7, 6, 1, 0, 15, 14));
	t = _mm256_blend_epi16(r4, r6, 30);
	t = _mm256_blend_epi16(t, r5, 1);
	in4 = _mm256_shuffle_epi8(t, _mm256_set_epi8(1, 0, 15, 14, 9, 8, 7, 6, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 15, 14, 9, 8, 7, 6, 3, 2, 5, 4, 11, 10, 13, 12));
	t = _mm256_alignr_epi8(r6, r1, 14);
	in5 = _mm256_blend_epi16(r3, r5, 6);
	in5 = _mm256_blend_epi16(in5, t, 1 + 128);
	in5 = _mm256_shuffle_epi8(in5, _mm256_set_epi8(15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6));
	in6 = _mm256_blend_epi16(r5, r3, 128);
	t = _mm256_blend_epi16(r6, r7, 1);
	t = _mm256_shuffle_epi8(t, _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 15, 14, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 15, 14, 11, 10));
	in6 = _mm256_shuffle_epi8(in6, _mm256_set_epi8(11, 10, 13, 12, 15, 14, 9, 8, 7, 6, -1, -1, -1, -1, -1, -1, 11, 10, 13, 12, 15, 14, 9, 8, 7, 6, -1, -1, -1, -1, -1, -1));
	r5 = _mm256_srli_si256(r5, 6);
	in6 = _mm256_or_si256(in6, t);
	in7 = _mm256_shuffle_epi8(r7, _mm256_set_epi8(15, 14, 11, 10, 13, 12, -1, -1, 9, 8, 7, 6, 3, 2, 5, 4, 15, 14, 11, 10, 13, 12, -1, -1, 9, 8, 7, 6, 3, 2, 5, 4));
	in7 = _mm256_blend_epi16(in7, r5, 0x10);

	*out0 = in0;
	*out1 = in1;
	*out2 = in2;
	*out3 = in3;
	*out4 = in4;
	*out5 = in5;
	*out6 = in6;
	*out7 = in7;
}

void VMX_EncodePlaneInternal256(VMX_INSTANCE* instance, VMX_PLANE * pPlane, VMX_SLICE_SET* s)
{
	VMX_PLANE plane = *pPlane;

	//int width = plane.Size.width;
	int height = VMX_SLICE_HEIGHT;
	short dcPred = 0;
	BYTE* src = plane.Data + s->Offset[plane.Index];
	int stride = plane.Stride;

	uint32_t numZeros = 0;
	uint64_t mIndex1 = 0;
	uint64_t mIndex2 = 0;
	uint32_t input = 0;
	uint32_t bc = 0;

	short dc1 = 0;
	short dc2 = 0;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlock = s->TempBlock;
	short* TempBlock2 = s->TempBlock2;

	__m256i zo = _mm256_setzero_si256();

	int dcshift = instance->DCShift;
	int dcround = 0;
	if (dcshift) dcround = 1 << (dcshift - 1);

	int addVal = 0;
	if (plane.Index == 0 || plane.Index == 3) addVal = -128;

	unsigned short* matrix = instance->EncodeMatrix256;

	short* pos = TempBlock;
	short* end;
	uint64_t nz;

	GolombZeroCodeLookup zeroLut;

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 16)
		{

			__m256i a0, a1, a2, a3, a4, a5, a6, a7;
			VMX_FDCT_8X8_QUANT_ZIG_256(src, stride, matrix, addVal, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
			src += 16;

			__m256i b0, b2,b4, b6;
			b0 = _mm256_packs_epi16(a0, a1);  //128A = two rows of first block 128B = two rows of second block
			b2 = _mm256_packs_epi16(a2, a3);
			b4 = _mm256_packs_epi16(a4, a5);
			b6 = _mm256_packs_epi16(a6, a7);

			b0 = _mm256_cmpeq_epi8(b0, zo); 
			b2 = _mm256_cmpeq_epi8(b2, zo);
			b4 = _mm256_cmpeq_epi8(b4, zo);
			b6 = _mm256_cmpeq_epi8(b6, zo);

			dc1 = _mm256_extract_epi16(a0, 0);
			dc1 += dcround;
			dc1 >>= dcshift;
			dc2 = _mm256_extract_epi16(a0, 8);
			dc2 += dcround;
			dc2 >>= dcshift;

			EncodeDC(dataDC, (dc1 - dcPred));
			dcPred = dc1;
			EncodeDC(dataDC, (dc2 - dcPred));
			dcPred = dc2;
			EmitBitsMax(dataDC);

			uint32_t i0 = _mm256_movemask_epi8(b0) | 0x10001; //clear first bit always, as DC is separate
			uint32_t i1 = _mm256_movemask_epi8(b2);
			uint32_t i2 = _mm256_movemask_epi8(b4);
			uint32_t i3 = _mm256_movemask_epi8(b6);
			mIndex1 = (uint64_t)((i0 & 0xFFFF));
			mIndex1 |= (uint64_t)(i1 & 0xFFFF) << 16;
			mIndex1 |= (uint64_t)(i2 & 0xFFFF) << 32;
			mIndex1 |= (uint64_t)(i3 & 0xFFFF) << 48;
			mIndex1 = ~mIndex1;

			mIndex2 = (uint64_t)((i0 >> 16));
			mIndex2 |= (uint64_t)(i1 >> 16) << 16;
			mIndex2 |= (uint64_t)(i2 >> 16) << 32;
			mIndex2 |= (uint64_t)(i3 >> 16) << 48;
			mIndex2 = ~mIndex2;

			if ((mIndex1 == 0) && (mIndex2 == 0))
			{
				numZeros += 128;
				continue;
			}

			//This is plus one as it saves an addition at the encoding stage which can be quite costly
			//This doesn't impact the zeros as they are determined exclusively from mIndex from this point onward
			Get2MagSignPlusOneV256(a0); 
			_mm256_storeu2_m128i((__m128i*) & TempBlock2[0], (__m128i*) & TempBlock[0], a0);

			if ((mIndex1 & 0x00000000FFFFFF00) || (mIndex2 & 0x00000000FFFFFF00)) {
				Get2MagSignPlusOneV256(a1);
				Get2MagSignPlusOneV256(a2);
				Get2MagSignPlusOneV256(a3);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[8], (__m128i*) & TempBlock[8], a1);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[16], (__m128i*) & TempBlock[16], a2);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[24], (__m128i*) & TempBlock[24], a3);
			}
			if ((mIndex1 & 0xFFFFFFFF00000000) || (mIndex2 & 0xFFFFFFFF00000000)) {
				Get2MagSignPlusOneV256(a4);
				Get2MagSignPlusOneV256(a5);
				Get2MagSignPlusOneV256(a6);
				Get2MagSignPlusOneV256(a7);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[32], (__m128i*) & TempBlock[32], a4);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[40], (__m128i*) & TempBlock[40], a5);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[48], (__m128i*) & TempBlock[48], a6);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[56], (__m128i*) & TempBlock[56], a7);
			}	

			pos = TempBlock;
			end = pos + 64;

			for (int z = 0; z < 2; z++)
			{
				if (mIndex1)
				{
					nz = _tzcnt_u64(mIndex1);
					numZeros += nz;
					EncodeZeros(dataAC);
					EmitBits32(dataAC);
					EncodeValue(dataAC, pos[0]);
					pos++;
					mIndex1 >>= nz;
					mIndex1 >>= 1; //NB: Separate shifts necessary, as nz could be 63 and a shift of 64 won't work as any shift outside 0-63 is masked out by CPU.
					EmitBitsMax(dataAC);

					while (1)
					{
						if (mIndex1)
						{
							nz = _tzcnt_u64(mIndex1);
							EncodeZerosSmall(dataAC);
							EncodeValue(dataAC, pos[0]);
							pos++;
							mIndex1 >>= (nz + 1);
							EmitBitsMax(dataAC);
						}
						else {
							numZeros = end - pos;
							break;
						}
					}
				}
				else {
					numZeros += end - pos;
				}
				mIndex1 = mIndex2;
				pos = TempBlock2;
				end = pos + 64;
			}				
		}
		src += (7 * stride);
	}
	EncodeZeros(dataAC);
	EmitBits32(dataAC);
	FlushRemainingBits(dataAC);
	FlushRemainingBits(dataDC);
	s->DC = dataDC;
	s->AC = dataAC;
}


void VMX_EncodePlaneInternal256_16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_PLANE plane = *pPlane;

	//int width = plane.Size.width;
	int height = VMX_SLICE_HEIGHT;
	short dcPred = 0;
	BYTE* src = plane.Data + s->Offset16[plane.Index];
	int stride = plane.Stride * 2;

	uint32_t numZeros = 0;
	uint64_t mIndex1 = 0;
	uint64_t mIndex2 = 0;
	uint32_t input = 0;
	uint32_t bc = 0;

	short dc1 = 0;
	short dc2 = 0;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlock = s->TempBlock;
	short* TempBlock2 = s->TempBlock2;

	__m256i zo = _mm256_setzero_si256();

	int dcshift = instance->DCShift;
	int dcround = 0;
	if (dcshift) dcround = 1 << (dcshift - 1);

	int addVal = 0;
	if (plane.Index == 0 || plane.Index == 3) addVal = -512;

	unsigned short* matrix = instance->EncodeMatrix256;

	short* pos = TempBlock;
	short* end;
	uint64_t nz;

	GolombZeroCodeLookup zeroLut;

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 32)
		{

			__m256i a0, a1, a2, a3, a4, a5, a6, a7;
			VMX_FDCT_8X8_QUANT_ZIG_256_16(src, stride, matrix, addVal, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
			src += 32;

			__m256i b0, b2, b4, b6;
			b0 = _mm256_packs_epi16(a0, a1);  //128A = two rows of first block 128B = two rows of second block
			b2 = _mm256_packs_epi16(a2, a3);
			b4 = _mm256_packs_epi16(a4, a5);
			b6 = _mm256_packs_epi16(a6, a7);

			b0 = _mm256_cmpeq_epi8(b0, zo);
			b2 = _mm256_cmpeq_epi8(b2, zo);
			b4 = _mm256_cmpeq_epi8(b4, zo);
			b6 = _mm256_cmpeq_epi8(b6, zo);

			dc1 = _mm256_extract_epi16(a0, 0);
			dc1 += dcround;
			dc1 >>= dcshift;
			dc2 = _mm256_extract_epi16(a0, 8);
			dc2 += dcround;
			dc2 >>= dcshift;

			EncodeDC(dataDC, (dc1 - dcPred));
			dcPred = dc1;
			EncodeDC(dataDC, (dc2 - dcPred));
			dcPred = dc2;
			EmitBitsMax(dataDC);

			uint32_t i0 = _mm256_movemask_epi8(b0) | 0x10001; //clear first bit always, as DC is separate
			uint32_t i1 = _mm256_movemask_epi8(b2);
			uint32_t i2 = _mm256_movemask_epi8(b4);
			uint32_t i3 = _mm256_movemask_epi8(b6);
			mIndex1 = (uint64_t)((i0 & 0xFFFF));
			mIndex1 |= (uint64_t)(i1 & 0xFFFF) << 16;
			mIndex1 |= (uint64_t)(i2 & 0xFFFF) << 32;
			mIndex1 |= (uint64_t)(i3 & 0xFFFF) << 48;
			mIndex1 = ~mIndex1;

			mIndex2 = (uint64_t)((i0 >> 16));
			mIndex2 |= (uint64_t)(i1 >> 16) << 16;
			mIndex2 |= (uint64_t)(i2 >> 16) << 32;
			mIndex2 |= (uint64_t)(i3 >> 16) << 48;
			mIndex2 = ~mIndex2;

			if ((mIndex1 == 0) && (mIndex2 == 0))
			{
				numZeros += 128;
				continue;
			}

			//This is plus one as it saves an addition at the encoding stage which can be quite costly
			//This doesn't impact the zeros as they are determined exclusively from mIndex from this point onward
			Get2MagSignPlusOneV256(a0);
			_mm256_storeu2_m128i((__m128i*) & TempBlock2[0], (__m128i*) & TempBlock[0], a0);

			if ((mIndex1 & 0x00000000FFFFFF00) || (mIndex2 & 0x00000000FFFFFF00)) {
				Get2MagSignPlusOneV256(a1);
				Get2MagSignPlusOneV256(a2);
				Get2MagSignPlusOneV256(a3);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[8], (__m128i*) & TempBlock[8], a1);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[16], (__m128i*) & TempBlock[16], a2);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[24], (__m128i*) & TempBlock[24], a3);
			}
			if ((mIndex1 & 0xFFFFFFFF00000000) || (mIndex2 & 0xFFFFFFFF00000000)) {
				Get2MagSignPlusOneV256(a4);
				Get2MagSignPlusOneV256(a5);
				Get2MagSignPlusOneV256(a6);
				Get2MagSignPlusOneV256(a7);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[32], (__m128i*) & TempBlock[32], a4);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[40], (__m128i*) & TempBlock[40], a5);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[48], (__m128i*) & TempBlock[48], a6);
				_mm256_storeu2_m128i((__m128i*) & TempBlock2[56], (__m128i*) & TempBlock[56], a7);
			}

			pos = TempBlock;
			end = pos + 64;

			for (int z = 0; z < 2; z++)
			{
				if (mIndex1)
				{
					nz = _tzcnt_u64(mIndex1);
					numZeros += nz;
					EncodeZeros(dataAC);
					EmitBits32(dataAC);
					EncodeValue(dataAC, pos[0]);
					pos++;
					mIndex1 >>= nz;
					mIndex1 >>= 1; //NB: Separate shifts necessary, as nz could be 63 and a shift of 64 won't work as any shift outside 0-63 is masked out by CPU.
					EmitBitsMax(dataAC);

					while (1)
					{
						if (mIndex1)
						{
							nz = _tzcnt_u64(mIndex1);
							EncodeZerosSmall(dataAC);
							EncodeValue(dataAC, pos[0]);
							pos++;
							mIndex1 >>= (nz + 1);
							EmitBitsMax(dataAC);
						}
						else {
							numZeros = end - pos;
							break;
						}
					}
				}
				else {
					numZeros += end - pos;
				}
				mIndex1 = mIndex2;
				pos = TempBlock2;
				end = pos + 64;
			}
		}
		src += (7 * stride);
	}
	EncodeZeros(dataAC);
	EmitBits32(dataAC);
	FlushRemainingBits(dataAC);
	FlushRemainingBits(dataDC);
	s->DC = dataDC;
	s->AC = dataAC;
}

void VMX_DecodePlaneInternal256(VMX_INSTANCE* instance, VMX_PLANE * pPlane, VMX_SLICE_SET* s)
{
	buffer_t b = 0;
	buffer_t bc = 0;
	buffer_t val = 0;

	VMX_PLANE plane = *pPlane;

	int height = VMX_SLICE_HEIGHT;
	int shift = 0;
	if (plane.Index == 0 || plane.Index == 3)
	{
		shift = 128;
	}

	char* pDst = (char*)(plane.Data + s->Offset[plane.Index]);
	int stride = plane.Stride;
	int dcPred = 0;
	int dcshift = instance->DCShift;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlockA = s->TempBlock;
	short* TempBlockB = s->TempBlock3;
	short* CurrentBlock = TempBlockA;

	unsigned short* matrix256 = instance->DecodeMatrix256;
	unsigned short* matrix128 = instance->DecodeMatrix;
	int validCount[2] = { 0,0 };

	buffer_t termsToDecode = 0; //Unsigned to ensure corrupt data doesnt overflow negative

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 16)
		{
			CurrentBlock = TempBlockA;
			for (int z = 0; z < 2; z++)
			{
				validCount[z] = 0;
				memset(CurrentBlock, 0, 128);
				//Decode 64 values			

				if (termsToDecode < 64)
				{
					validCount[z] = 1;
				}				

				while (termsToDecode < 64)
				{
					GolombLookup l = GolombLookupLut[(dataAC.TempRead >> (dataAC.BitsLeft - 12)) & 0xFFF];
					if (l.length)
					{
						dataAC.BitsLeft -= l.length;
						CurrentBlock[termsToDecode] = l.value;
						termsToDecode += l.zeros;
					}
					else {
						bc = 0;
						GETBITB(dataAC, b);
						if (b)
						{
							GETBITB(dataAC, b);
							if (b)
							{
								termsToDecode++;
							}
							else {
								GETZEROSB(dataAC, bc);
								bc += 2;
								GETBITSB_BMI(dataAC, bc, val);
								termsToDecode += val;
								bc = 0;
							}
						}
						else {
							GETZEROSB(dataAC, bc);
							bc += 2;
							GETBITSB_BMI(dataAC, bc, val);
							CurrentBlock[termsToDecode] = GetIntFrom2MagSign((val - 1));
							termsToDecode++;
						}
					}

					RELOADBITS(dataAC);
				}
				termsToDecode -= 64;

				//Read DC
				GETBIT(dataDC, b);
				if (b)
				{
					GETBIT(dataDC, b);
				}
				else {
					GETZEROS(dataDC, bc);
					bc += 2;
					GETBITS(dataDC, bc, val);
					CurrentBlock[0] = GetIntFrom2MagSign((val - 1));
					CurrentBlock[0] <<= dcshift;
				}
				CurrentBlock[0] += dcPred;
				dcPred = CurrentBlock[0];
				CurrentBlock = TempBlockB;
			}
			if (validCount[0] == 0 && validCount[1] == 0)
			{
				_mm256_zeroupper();
				VMX_BROADCAST_DC_8X8_128(TempBlockA[0], (BYTE*)pDst, stride, shift);
				VMX_BROADCAST_DC_8X8_128(TempBlockB[0], (BYTE*)pDst + 8, stride, shift);
			}
			else if (validCount[0] == 0)
			{				
				_mm256_zeroupper();
				VMX_BROADCAST_DC_8X8_128(TempBlockA[0], (BYTE*)pDst, stride, shift);
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_128(TempBlockB, matrix128, (BYTE*)pDst + 8, stride, shift);
			}
			else if (validCount[1] == 0)
			{
				_mm256_zeroupper();
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_128(TempBlockA, matrix128, (BYTE*)pDst, stride, shift);
				VMX_BROADCAST_DC_8X8_128(TempBlockB[0], (BYTE*)pDst + 8, stride, shift);
			}
			else {
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_256(TempBlockA, TempBlockB, matrix256, (BYTE*)pDst, stride, shift);
			}
			pDst += 16;
		}
		pDst += (7 * stride);

	}

	REWINDOVERREAD(dataAC);

	FLUSHREMAININGREADBITS(dataDC);
	FLUSHREMAININGREADBITS(dataAC);
	s->AC = dataAC;
	s->DC = dataDC;
}


void VMX_DecodePlaneInternal256_16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	buffer_t b = 0;
	buffer_t bc = 0;
	buffer_t val = 0;

	VMX_PLANE plane = *pPlane;

	int height = VMX_SLICE_HEIGHT;
	int shift = 0;
	if (plane.Index == 0 || plane.Index == 3)
	{
		shift = 512;
	}

	char* pDst = (char*)(plane.Data + s->Offset16[plane.Index]);
	int stride = plane.Stride * 2;
	int dcPred = 0;
	int dcshift = instance->DCShift;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlockA = s->TempBlock;
	short* TempBlockB = s->TempBlock3;
	short* CurrentBlock = TempBlockA;

	unsigned short* matrix256 = instance->DecodeMatrix256;
	unsigned short* matrix128 = instance->DecodeMatrix;
	int validCount[2] = { 0,0 };

	buffer_t termsToDecode = 0; //Unsigned to ensure corrupt data doesnt overflow negative

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 32)
		{
			CurrentBlock = TempBlockA;
			for (int z = 0; z < 2; z++)
			{
				validCount[z] = 0;
				memset(CurrentBlock, 0, 128);
				//Decode 64 values			

				if (termsToDecode < 64)
				{
					validCount[z] = 1;
				}

				while (termsToDecode < 64)
				{
					GolombLookup l = GolombLookupLut[(dataAC.TempRead >> (dataAC.BitsLeft - 12)) & 0xFFF];
					if (l.length)
					{
						dataAC.BitsLeft -= l.length;
						CurrentBlock[termsToDecode] = l.value;
						termsToDecode += l.zeros;
					}
					else {
						bc = 0;
						GETBITB(dataAC, b);
						if (b)
						{
							GETBITB(dataAC, b);
							if (b)
							{
								termsToDecode++;
							}
							else {
								GETZEROSB(dataAC, bc);
								bc += 2;
								GETBITSB_BMI(dataAC, bc, val);
								termsToDecode += val;
								bc = 0;
							}
						}
						else {
							GETZEROSB(dataAC, bc);
							bc += 2;
							GETBITSB_BMI(dataAC, bc, val);
							CurrentBlock[termsToDecode] = GetIntFrom2MagSign((val - 1));
							termsToDecode++;
						}
					}

					RELOADBITS(dataAC);
				}
				termsToDecode -= 64;

				//Read DC
				GETBIT(dataDC, b);
				if (b)
				{
					GETBIT(dataDC, b);
				}
				else {
					GETZEROS(dataDC, bc);
					bc += 2;
					GETBITS(dataDC, bc, val);
					CurrentBlock[0] = GetIntFrom2MagSign((val - 1));
					CurrentBlock[0] <<= dcshift;
				}
				CurrentBlock[0] += dcPred;
				dcPred = CurrentBlock[0];
				CurrentBlock = TempBlockB;
			}
			if (validCount[0] == 0 && validCount[1] == 0)
			{
				_mm256_zeroupper();
				VMX_BROADCAST_DC_8X8_128_16(TempBlockA[0], (BYTE*)pDst, stride, shift);
				VMX_BROADCAST_DC_8X8_128_16(TempBlockB[0], (BYTE*)pDst + 16, stride, shift);
			}
			else if (validCount[0] == 0)
			{
				_mm256_zeroupper();
				VMX_BROADCAST_DC_8X8_128_16(TempBlockA[0], (BYTE*)pDst, stride, shift);
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_128_16(TempBlockB, matrix128, (BYTE*)pDst + 16, stride, shift);
			}
			else if (validCount[1] == 0)
			{
				_mm256_zeroupper();
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_128_16(TempBlockA, matrix128, (BYTE*)pDst, stride, shift);
				VMX_BROADCAST_DC_8X8_128_16(TempBlockB[0], (BYTE*)pDst + 16, stride, shift);
			}
			else {
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_256_16(TempBlockA, TempBlockB, matrix256, (BYTE*)pDst, stride, shift);
			}
			pDst += 32;
		}
		pDst += (7 * stride);

	}

	REWINDOVERREAD(dataAC);

	FLUSHREMAININGREADBITS(dataDC);
	FLUSHREMAININGREADBITS(dataAC);
	s->AC = dataAC;
	s->DC = dataDC;
}

#endif