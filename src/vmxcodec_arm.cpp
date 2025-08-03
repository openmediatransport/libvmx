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
#include "vmxcodec_arm.h"
#if defined(ARM64)

#define Get2MagSignV(input) { \
	__m128i b = _mm_adds_epi16(input, input);  \
	__m128i c = _mm_srai_epi16(input, 15);  \
	input = _mm_xor_si128(b, c); \
}

#define Get2MagSignPlusOneV(input) { \
	__m128i b = _mm_adds_epi16(input, input);  \
	__m128i c = _mm_srai_epi16(input, 15);  \
	input = _mm_xor_si128(b, c); \
	input = _mm_add_epi16(input, _mm_set1_epi16(1)); \
}

//===========================
//IDCT Tables for 128bit SIMD
//===========================
__declspec(align(16)) const short one_corr_128[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
__declspec(align(16)) const short round_inv_row_128[8] = { IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0, IRND_INV_ROW, 0 };
__declspec(align(16)) const short round_inv_col_128[8] = { IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL, IRND_INV_COL };
__declspec(align(16)) const short round_inv_corr_128[8] = { IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR, IRND_INV_CORR };

__declspec(align(16)) const short round_inv_row_128_10[8] = { IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0, IRND_INV_ROW10, 0 };
__declspec(align(16)) const short round_inv_col_128_10[8] = { IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10, IRND_INV_COL10 };
__declspec(align(16)) const short round_inv_corr_128_10[8] = { IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10, IRND_INV_CORR10 };

__declspec(align(16)) const short tg_1_16_128[8] = { 13036, 13036, 13036, 13036, 13036, 13036, 13036, 13036 }; // tg * (2<<16) + 0.5
__declspec(align(16)) const short tg_2_16_128[8] = { 27146, 27146, 27146, 27146, 27146, 27146, 27146, 27146 }; // tg * (2<<16) + 0.5
__declspec(align(16)) const short tg_3_16_128[8] = { -21746, -21746, -21746, -21746, -21746, -21746, -21746, -21746 }; // tg * (2<<16) + 0.5
__declspec(align(16)) const short cos_4_16_128[8] = { -19195, -19195, -19195, -19195, -19195, -19195, -19195, -19195 };// cos * (2<<16) + 0.5

__declspec(align(16)) const short tab_i_04_128[] = {
	16384, 21407, 16384, 8867,
	16384, -8867, 16384, -21407,
	16384, 8867, -16384, -21407,
	-16384, 21407, 16384, -8867,
	22725, 19266, 19266, -4520,
	12873, -22725, 4520, -12873,
	12873, 4520, -22725, -12873,
	4520, 19266, 19266, -22725 };

__declspec(align(16)) const short tab_i_17_128[] = {
	22725, 29692, 22725, 12299,
	22725, -12299, 22725, -29692,
	22725, 12299, -22725, -29692,
	-22725, 29692, 22725, -12299,
	31521, 26722, 26722, -6270,
	17855, -31521, 6270, -17855,
	17855, 6270, -31521, -17855,
	6270, 26722, 26722, -31521 };

__declspec(align(16)) const short tab_i_26_128[] = {
	21407, 27969, 21407, 11585,
	21407, -11585, 21407, -27969,
	21407, 11585, -21407, -27969,
	-21407, 27969, 21407, -11585,
	29692, 25172, 25172, -5906,
	16819, -29692, 5906, -16819,
	16819, 5906, -29692, -16819,
	5906, 25172, 25172, -29692 };

__declspec(align(16)) const short tab_i_35_128[] = {
	19266, 25172, 19266, 10426,
	19266, -10426, 19266, -25172,
	19266, 10426, -19266, -25172,
	-19266, 25172, 19266, -10426,
	26722, 22654, 22654, -5315,
	15137, -26722, 5315, -15137,
	15137, 5315, -26722, -15137,
	5315, 22654, 22654, -26722 };

//===========================
//FDCT Tables for 128bit SIMD
//===========================

__declspec(align(16)) const unsigned short ftab1_128[] = {
	16384, 16384, 22725, 19266, 56669, 44129, 42811, 52663,
	16384, 16384, 12873, 4520, 21407, 8867, 19266, 61016,
	16384, 49152, 12873, 42811, 21407, 56669, 19266, 42811,
	49152, 16384, 4520, 19266, 8867, 44129, 4520, 52663
};

__declspec(align(16)) const unsigned short ftab2_128[] = {
	22725, 22725, 31521, 26722, 53237, 35844, 34015, 47681,
	22725, 22725, 17855, 6270, 29692, 12299, 26722, 59266,
	22725, 42811, 17855, 34015, 29692, 53237, 26722, 34015,
	42811, 22725, 6270, 26722, 12299, 35844, 6270, 47681
};

__declspec(align(16)) const unsigned short ftab3_128[] = {
	21407, 21407, 29692, 25172, 53951, 37567, 35844, 48717,
	21407, 21407, 16819, 5906, 27969, 11585, 25172, 59630,
	21407, 44129, 16819, 35844, 27969, 53951, 25172, 35844,
	44129, 21407, 5906, 25172, 11585, 37567, 5906, 48717
};

__declspec(align(16)) const unsigned short ftab4_128[] = {
	19266, 19266, 26722, 22654, 55110, 40364, 38814, 50399,
	19266, 19266, 15137, 5315, 25172, 10426, 22654, 60221,
	19266, 46270, 15137, 38814, 25172, 55110, 22654, 38814,
	46270, 19266, 5315, 22654, 10426, 40364, 5315, 50399
};


//Forward DCT8x8 + Quantize + ZigZag
void VMX_FDCT_8X8_QUANT_ZIG_128(const BYTE* src, int stride, unsigned short* matrix, short addVal, __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3, __m128i* out4, __m128i* out5, __m128i* out6, __m128i* out7) {

	// Load input
	__m128i in0 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in1 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in2 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in3 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in4 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in5 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in6 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;
	__m128i in7 = _mm_loadl_epi64((__m128i*) & src[0]);
	src += stride;

	in0 = _mm_cvtepu8_epi16(in0);
	in1 = _mm_cvtepu8_epi16(in1);
	in2 = _mm_cvtepu8_epi16(in2);
	in3 = _mm_cvtepu8_epi16(in3);
	in4 = _mm_cvtepu8_epi16(in4);
	in5 = _mm_cvtepu8_epi16(in5);
	in6 = _mm_cvtepu8_epi16(in6);
	in7 = _mm_cvtepu8_epi16(in7);

	__m128i vadd = _mm_set1_epi16(addVal);
	in0 = _mm_adds_epi16(in0, vadd);
	in1 = _mm_adds_epi16(in1, vadd);
	in2 = _mm_adds_epi16(in2, vadd);
	in3 = _mm_adds_epi16(in3, vadd);
	in4 = _mm_adds_epi16(in4, vadd);
	in5 = _mm_adds_epi16(in5, vadd);
	in6 = _mm_adds_epi16(in6, vadd);
	in7 = _mm_adds_epi16(in7, vadd);

	// Load input data
	__m128i xmm0 = in0;
	__m128i xmm2 = in2;
	__m128i xmm3 = xmm0;
	__m128i xmm4 = xmm2;
	__m128i xmm7 = in7;
	__m128i xmm5 = in5;

	// First stage
	xmm0 = _mm_subs_epi16(xmm0, xmm7);
	xmm7 = _mm_adds_epi16(xmm7, xmm3);
	xmm2 = _mm_subs_epi16(xmm2, xmm5);
	xmm5 = _mm_adds_epi16(xmm5, xmm4);

	xmm3 = in3;
	xmm4 = in4;
	__m128i xmm1 = xmm3;
	xmm3 = _mm_subs_epi16(xmm3, xmm4);
	xmm4 = _mm_adds_epi16(xmm4, xmm1);

	__m128i xmm6 = in6;
	xmm1 = in1;
	__m128i tmp = xmm1;
	xmm1 = _mm_subs_epi16(xmm1, xmm6);
	xmm6 = _mm_adds_epi16(xmm6, tmp);

	// Second stage
	__m128i tm03 = _mm_subs_epi16(xmm7, xmm4);
	__m128i tm12 = _mm_subs_epi16(xmm6, xmm5);
	xmm4 = _mm_adds_epi16(xmm4, xmm4);
	xmm5 = _mm_adds_epi16(xmm5, xmm5);

	__m128i tp03 = _mm_adds_epi16(xmm4, tm03);
	__m128i tp12 = _mm_adds_epi16(xmm5, tm12);

	// Shift operations
	xmm2 = _mm_slli_epi16(xmm2, SHIFT_FRW_COL + 1);
	xmm1 = _mm_slli_epi16(xmm1, SHIFT_FRW_COL + 1);
	tp03 = _mm_slli_epi16(tp03, SHIFT_FRW_COL);
	tp12 = _mm_slli_epi16(tp12, SHIFT_FRW_COL);
	tm03 = _mm_slli_epi16(tm03, SHIFT_FRW_COL);
	tm12 = _mm_slli_epi16(tm12, SHIFT_FRW_COL);
	xmm3 = _mm_slli_epi16(xmm3, SHIFT_FRW_COL);
	xmm0 = _mm_slli_epi16(xmm0, SHIFT_FRW_COL);

	// Output calculations
	in4 = _mm_subs_epi16(tp03, tp12);
	__m128i diff = _mm_subs_epi16(xmm1, xmm2);
	tp12 = _mm_adds_epi16(tp12, tp12);
	xmm2 = _mm_adds_epi16(xmm2, xmm2);
	in0 = _mm_adds_epi16(tp12, in4);

	__m128i sum = _mm_adds_epi16(xmm2, diff);

	// Tan2
	__m128i tan2v = _mm_set1_epi16(FDCT_TAN2);
	__m128i tmp1 = _mm_mulhi_epi16(tan2v, tm03);
	in6 = _mm_subs_epi16(tmp1, tm12);
	__m128i tmp2 = _mm_mulhi_epi16(tan2v, tm12);
	in2 = _mm_adds_epi16(tmp2, tm03);

	// Sqrt2
	__m128i sqrt2v = _mm_set1_epi16(FDCT_SQRT2);
	__m128i rounder = _mm_set1_epi16(FDCT_ROUND1);

	__m128i tp65 = _mm_mulhi_epi16(sum, sqrt2v);
	in2 = _mm_or_si128(in2, rounder);
	in6 = _mm_or_si128(in6, rounder);
	__m128i tm65 = _mm_mulhi_epi16(diff, sqrt2v);
	tp65 = _mm_or_si128(tp65, rounder);

	// Final calculations
	__m128i tm465 = _mm_subs_epi16(xmm3, tm65);
	__m128i tm765 = _mm_subs_epi16(xmm0, tp65);
	__m128i tp765 = _mm_adds_epi16(tp65, xmm0);
	__m128i tp465 = _mm_adds_epi16(tm65, xmm3);

	__m128i tan3v = _mm_set1_epi16(FDCT_TAN3);
	__m128i tan1v = _mm_set1_epi16(FDCT_TAN1);

	__m128i tmp3 = _mm_mulhi_epi16(tm465, tan3v);
	__m128i tmp4 = _mm_mulhi_epi16(tp465, tan1v);
	tmp3 = _mm_adds_epi16(tmp3, tm465);

	__m128i tmp5 = _mm_mulhi_epi16(tm765, tan3v);
	tmp5 = _mm_adds_epi16(tmp5, tm765);
	__m128i tmp6 = _mm_mulhi_epi16(tp765, tan1v);

	in1 = _mm_adds_epi16(tmp4, tp765);
	in3 = _mm_subs_epi16(tm765, tmp3);
	in5 = _mm_adds_epi16(tm465, tmp5);
	in7 = _mm_subs_epi16(tmp6, tp465);

	__m128i round = _mm_set1_epi32(RND_FRW_ROW);
	/////////////////////
	//ROW 0
	/////////////////////
	xmm0 = in0;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	__m128i temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[8]));
	__m128i temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[16]));
	__m128i temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[24]));
	__m128i temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in0 = xmm0;

	/////////////////////
	//ROW 1
	/////////////////////
	xmm0 = in1;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in1 = xmm0;

	/////////////////////
	//ROW 2
	/////////////////////
	xmm0 = in2;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in2 = xmm0;

	/////////////////////
	//ROW 3
	/////////////////////
	xmm0 = in3;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in3 = xmm0;

	/////////////////////
	//ROW 4
	/////////////////////
	xmm0 = in4;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in4 = xmm0;

	/////////////////////
	//ROW 5
	/////////////////////
	xmm0 = in5;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in5 = xmm0;

	/////////////////////
	//ROW 6
	/////////////////////
	xmm0 = in6;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in6 = xmm0;

	/////////////////////
	//ROW 7
	/////////////////////
	xmm0 = in7;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in7 = xmm0;

	//===================
	//BEGIN QUANTIZATION. We eliminate the 4 shift back and forth by doing DCT and QUANTIZE together! 
	// This also improves the quality slightly due to less rounding errors...
	//===================

	//abs source
	__m128i b0 = _mm_abs_epi16(in0);
	__m128i b1 = _mm_abs_epi16(in1);
	__m128i b2 = _mm_abs_epi16(in2);
	__m128i b3 = _mm_abs_epi16(in3);
	__m128i b4 = _mm_abs_epi16(in4);
	__m128i b5 = _mm_abs_epi16(in5);
	__m128i b6 = _mm_abs_epi16(in6);
	__m128i b7 = _mm_abs_epi16(in7);

	//////shift 4 bits
	//b0 = _mm_slli_epi16(b0, 4);
	//b1 = _mm_slli_epi16(b1, 4);
	//b2 = _mm_slli_epi16(b2, 4);
	//b3 = _mm_slli_epi16(b3, 4);
	//b4 = _mm_slli_epi16(b4, 4);
	//b5 = _mm_slli_epi16(b5, 4);
	//b6 = _mm_slli_epi16(b6, 4);
	//b7 = _mm_slli_epi16(b7, 4);

	//load correction
	__m128i c0 = _mm_load_si128((__m128i*) & matrix[0]);
	__m128i c1 = _mm_load_si128((__m128i*) & matrix[8]);
	__m128i c2 = _mm_load_si128((__m128i*) & matrix[16]);
	__m128i c3 = _mm_load_si128((__m128i*) & matrix[24]);
	__m128i c4 = _mm_load_si128((__m128i*) & matrix[32]);
	__m128i c5 = _mm_load_si128((__m128i*) & matrix[40]);
	__m128i c6 = _mm_load_si128((__m128i*) & matrix[48]);
	__m128i c7 = _mm_load_si128((__m128i*) & matrix[56]);

	//add correction
	b0 = _mm_add_epi16(b0, c0);
	b1 = _mm_add_epi16(b1, c1);
	b2 = _mm_add_epi16(b2, c2);
	b3 = _mm_add_epi16(b3, c3);
	b4 = _mm_add_epi16(b4, c4);
	b5 = _mm_add_epi16(b5, c5);
	b6 = _mm_add_epi16(b6, c6);
	b7 = _mm_add_epi16(b7, c7);

	//load reciprocal
	c0 = _mm_load_si128((__m128i*) & matrix[64]);
	c1 = _mm_load_si128((__m128i*) & matrix[72]);
	c2 = _mm_load_si128((__m128i*) & matrix[80]);
	c3 = _mm_load_si128((__m128i*) & matrix[88]);
	c4 = _mm_load_si128((__m128i*) & matrix[96]);
	c5 = _mm_load_si128((__m128i*) & matrix[104]);
	c6 = _mm_load_si128((__m128i*) & matrix[112]);
	c7 = _mm_load_si128((__m128i*) & matrix[120]);

	//multiply reciprocal
	b0 = _mm_mulhi_epu16(b0, c0);
	b1 = _mm_mulhi_epu16(b1, c1);
	b2 = _mm_mulhi_epu16(b2, c2);
	b3 = _mm_mulhi_epu16(b3, c3);
	b4 = _mm_mulhi_epu16(b4, c4);
	b5 = _mm_mulhi_epu16(b5, c5);
	b6 = _mm_mulhi_epu16(b6, c6);
	b7 = _mm_mulhi_epu16(b7, c7);

	//load scale
	c0 = _mm_load_si128((__m128i*) & matrix[128]);
	c1 = _mm_load_si128((__m128i*) & matrix[136]);
	c2 = _mm_load_si128((__m128i*) & matrix[144]);
	c3 = _mm_load_si128((__m128i*) & matrix[152]);
	c4 = _mm_load_si128((__m128i*) & matrix[160]);
	c5 = _mm_load_si128((__m128i*) & matrix[168]);
	c6 = _mm_load_si128((__m128i*) & matrix[176]);
	c7 = _mm_load_si128((__m128i*) & matrix[184]);

	//multiply scale
	b0 = _mm_mulhi_epu16(b0, c0);
	b1 = _mm_mulhi_epu16(b1, c1);
	b2 = _mm_mulhi_epu16(b2, c2);
	b3 = _mm_mulhi_epu16(b3, c3);
	b4 = _mm_mulhi_epu16(b4, c4);
	b5 = _mm_mulhi_epu16(b5, c5);
	b6 = _mm_mulhi_epu16(b6, c6);
	b7 = _mm_mulhi_epu16(b7, c7);

	//sign
	in0 = _mm_sign_epi16(b0, in0);
	in1 = _mm_sign_epi16(b1, in1);
	in2 = _mm_sign_epi16(b2, in2);
	in3 = _mm_sign_epi16(b3, in3);
	in4 = _mm_sign_epi16(b4, in4);
	in5 = _mm_sign_epi16(b5, in5);
	in6 = _mm_sign_epi16(b6, in6);
	in7 = _mm_sign_epi16(b7, in7);

	//===================
	//BEGIN ZIG ZAG. ~38 instructions. Probably not optimal, but sure beats using a loop!
	//===================

	__m128i r0 = _mm_unpacklo_epi16(in0, in1); 	//0	8	1	9	2	10	3	11
	__m128i r1 = _mm_unpackhi_epi16(in0, in1); 	//4	12	5	13	6	14	7	15
	__m128i r2 = _mm_unpacklo_epi16(in2, in3); 	// 16	24	17	25	18	26	19	27
	__m128i r3 = _mm_unpackhi_epi16(in2, in3); 	//20	28	21	29	22	30	23	31
	__m128i r4 = _mm_unpacklo_epi16(in4, in5); 	//32	40	33	41	34	42	35	43
	__m128i r5 = _mm_unpackhi_epi16(in4, in5); 	//36	44	37	45	38	46	39	47
	__m128i r6 = _mm_unpacklo_epi16(in6, in7); 	//48	56	49	57	50	58	51	59
	__m128i r7 = _mm_unpackhi_epi16(in6, in7); 	//52	60	53	61	54	62	55	63		

	in0 = _mm_shuffle_epi8(r0, _mm_set_epi8(11, 10, 13, 12, 9, 8, 7, 6, -1, -1, 3, 2, 5, 4, 1, 0));
	in2 = _mm_slli_si128(in2, 6);
	in0 = _mm_blend_epi16(in0, in2, 0x8);
	__m128i t = _mm_blend_epi16(r2, r4, 1);
	in1 = _mm_blend_epi16(r0, r1, 5);
	t = _mm_shuffle_epi8(t, _mm_set_epi8(-1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 1, 0, 3, 2, 5, 4));
	in1 = _mm_shuffle_epi8(in1, _mm_set_epi8(5, 4, 1, 0, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
	in1 = _mm_or_si128(in1, t);
	t = _mm_alignr_epi8(r6, r1, 2);
	in2 = _mm_blend_epi16(r4, r2, 96);
	in2 = _mm_blend_epi16(in2, t, 1 + 128);
	in2 = _mm_shuffle_epi8(in2, _mm_set_epi8(9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0));
	t = _mm_blend_epi16(r1, r3, 7);
	t = _mm_blend_epi16(t, r2, 128);
	in3 = _mm_shuffle_epi8(t, _mm_set_epi8(3, 2, 5, 4, 11, 10, 13, 12, 9, 8, 7, 6, 1, 0, 15, 14));
	t = _mm_blend_epi16(r4, r6, 30);
	t = _mm_blend_epi16(t, r5, 1);
	in4 = _mm_shuffle_epi8(t, _mm_set_epi8(1, 0, 15, 14, 9, 8, 7, 6, 3, 2, 5, 4, 11, 10, 13, 12));
	t = _mm_alignr_epi8(r6, r1, 14);
	in5 = _mm_blend_epi16(r3, r5, 6);
	in5 = _mm_blend_epi16(in5, t, 1 + 128);
	in5 = _mm_shuffle_epi8(in5, _mm_set_epi8(15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6));
	in6 = _mm_blend_epi16(r5, r3, 128);
	t = _mm_blend_epi16(r6, r7, 1);
	t = _mm_shuffle_epi8(t, _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 15, 14, 11, 10));
	in6 = _mm_shuffle_epi8(in6, _mm_set_epi8(11, 10, 13, 12, 15, 14, 9, 8, 7, 6, -1, -1, -1, -1, -1, -1));
	r5 = _mm_srli_si128(r5, 6);
	in6 = _mm_or_si128(in6, t);
	in7 = _mm_shuffle_epi8(r7, _mm_set_epi8(15, 14, 11, 10, 13, 12, -1, -1, 9, 8, 7, 6, 3, 2, 5, 4));
	in7 = _mm_blend_epi16(in7, r5, 0x10);

	*out0 = in0;
	*out1 = in1;
	*out2 = in2;
	*out3 = in3;
	*out4 = in4;
	*out5 = in5;
	*out6 = in6;
	*out7 = in7;
}


//16-Bit Forward DCT8x8 + Quantize + ZigZag. Source is 16-bit unsigned values
void VMX_FDCT_8X8_QUANT_ZIG_128_16(const BYTE* src, int stride, unsigned short* matrix, short addVal, __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3, __m128i* out4, __m128i* out5, __m128i* out6, __m128i* out7) {

	// Load input
	__m128i in0 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in1 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in2 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in3 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in4 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in5 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in6 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;
	__m128i in7 = _mm_loadu_si128((__m128i*) & src[0]);
	src += stride;

	//Shift down by 6 bits, as actual DCT range is between -512 and +512
	in0 = _mm_srli_epi16(in0, 6);
	in1 = _mm_srli_epi16(in1, 6);
	in2 = _mm_srli_epi16(in2, 6);
	in3 = _mm_srli_epi16(in3, 6);
	in4 = _mm_srli_epi16(in4, 6);
	in5 = _mm_srli_epi16(in5, 6);
	in6 = _mm_srli_epi16(in6, 6);
	in7 = _mm_srli_epi16(in7, 6);

	__m128i vadd = _mm_set1_epi16(addVal);
	in0 = _mm_adds_epi16(in0, vadd);
	in1 = _mm_adds_epi16(in1, vadd);
	in2 = _mm_adds_epi16(in2, vadd);
	in3 = _mm_adds_epi16(in3, vadd);
	in4 = _mm_adds_epi16(in4, vadd);
	in5 = _mm_adds_epi16(in5, vadd);
	in6 = _mm_adds_epi16(in6, vadd);
	in7 = _mm_adds_epi16(in7, vadd);

	// Load input data
	__m128i xmm0 = in0;
	__m128i xmm2 = in2;
	__m128i xmm3 = xmm0;
	__m128i xmm4 = xmm2;
	__m128i xmm7 = in7;
	__m128i xmm5 = in5;

	// First stage
	xmm0 = _mm_subs_epi16(xmm0, xmm7);
	xmm7 = _mm_adds_epi16(xmm7, xmm3);
	xmm2 = _mm_subs_epi16(xmm2, xmm5);
	xmm5 = _mm_adds_epi16(xmm5, xmm4);

	xmm3 = in3;
	xmm4 = in4;
	__m128i xmm1 = xmm3;
	xmm3 = _mm_subs_epi16(xmm3, xmm4);
	xmm4 = _mm_adds_epi16(xmm4, xmm1);

	__m128i xmm6 = in6;
	xmm1 = in1;
	__m128i tmp = xmm1;
	xmm1 = _mm_subs_epi16(xmm1, xmm6);
	xmm6 = _mm_adds_epi16(xmm6, tmp);

	// Second stage
	__m128i tm03 = _mm_subs_epi16(xmm7, xmm4);
	__m128i tm12 = _mm_subs_epi16(xmm6, xmm5);
	xmm4 = _mm_adds_epi16(xmm4, xmm4);
	xmm5 = _mm_adds_epi16(xmm5, xmm5);

	__m128i tp03 = _mm_adds_epi16(xmm4, tm03);
	__m128i tp12 = _mm_adds_epi16(xmm5, tm12);

	// Shift operations
	xmm2 = _mm_slli_epi16(xmm2, SHIFT_FRW_COL10 + 1);
	xmm1 = _mm_slli_epi16(xmm1, SHIFT_FRW_COL10 + 1);
	tp03 = _mm_slli_epi16(tp03, SHIFT_FRW_COL10);
	tp12 = _mm_slli_epi16(tp12, SHIFT_FRW_COL10);
	tm03 = _mm_slli_epi16(tm03, SHIFT_FRW_COL10);
	tm12 = _mm_slli_epi16(tm12, SHIFT_FRW_COL10);
	xmm3 = _mm_slli_epi16(xmm3, SHIFT_FRW_COL10);
	xmm0 = _mm_slli_epi16(xmm0, SHIFT_FRW_COL10);

	// Output calculations
	in4 = _mm_subs_epi16(tp03, tp12);
	__m128i diff = _mm_subs_epi16(xmm1, xmm2);
	tp12 = _mm_adds_epi16(tp12, tp12);
	xmm2 = _mm_adds_epi16(xmm2, xmm2);
	in0 = _mm_adds_epi16(tp12, in4);

	__m128i sum = _mm_adds_epi16(xmm2, diff);

	// Tan2
	__m128i tan2v = _mm_set1_epi16(FDCT_TAN2);
	__m128i tmp1 = _mm_mulhi_epi16(tan2v, tm03);
	in6 = _mm_subs_epi16(tmp1, tm12);
	__m128i tmp2 = _mm_mulhi_epi16(tan2v, tm12);
	in2 = _mm_adds_epi16(tmp2, tm03);

	// Sqrt2
	__m128i sqrt2v = _mm_set1_epi16(FDCT_SQRT2);
	__m128i rounder = _mm_set1_epi16(FDCT_ROUND1);

	__m128i tp65 = _mm_mulhi_epi16(sum, sqrt2v);
	in2 = _mm_or_si128(in2, rounder);
	in6 = _mm_or_si128(in6, rounder);
	__m128i tm65 = _mm_mulhi_epi16(diff, sqrt2v);
	tp65 = _mm_or_si128(tp65, rounder);

	// Final calculations
	__m128i tm465 = _mm_subs_epi16(xmm3, tm65);
	__m128i tm765 = _mm_subs_epi16(xmm0, tp65);
	__m128i tp765 = _mm_adds_epi16(tp65, xmm0);
	__m128i tp465 = _mm_adds_epi16(tm65, xmm3);

	__m128i tan3v = _mm_set1_epi16(FDCT_TAN3);
	__m128i tan1v = _mm_set1_epi16(FDCT_TAN1);

	__m128i tmp3 = _mm_mulhi_epi16(tm465, tan3v);
	__m128i tmp4 = _mm_mulhi_epi16(tp465, tan1v);
	tmp3 = _mm_adds_epi16(tmp3, tm465);

	__m128i tmp5 = _mm_mulhi_epi16(tm765, tan3v);
	tmp5 = _mm_adds_epi16(tmp5, tm765);
	__m128i tmp6 = _mm_mulhi_epi16(tp765, tan1v);

	in1 = _mm_adds_epi16(tmp4, tp765);
	in3 = _mm_subs_epi16(tm765, tmp3);
	in5 = _mm_adds_epi16(tm465, tmp5);
	in7 = _mm_subs_epi16(tmp6, tp465);

	__m128i round = _mm_set1_epi32(RND_FRW_ROW10);
	/////////////////////
	//ROW 0
	/////////////////////
	xmm0 = in0;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	__m128i temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[8]));
	__m128i temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[16]));
	__m128i temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[24]));
	__m128i temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in0 = xmm0;

	/////////////////////
	//ROW 1
	/////////////////////
	xmm0 = in1;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in1 = xmm0;

	/////////////////////
	//ROW 2
	/////////////////////
	xmm0 = in2;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in2 = xmm0;

	/////////////////////
	//ROW 3
	/////////////////////
	xmm0 = in3;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in3 = xmm0;

	/////////////////////
	//ROW 4
	/////////////////////
	xmm0 = in4;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab1_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab1_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in4 = xmm0;

	/////////////////////
	//ROW 5
	/////////////////////
	xmm0 = in5;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab4_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab4_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in5 = xmm0;

	/////////////////////
	//ROW 6
	/////////////////////
	xmm0 = in6;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab3_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab3_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in6 = xmm0;

	/////////////////////
	//ROW 7
	/////////////////////
	xmm0 = in7;
	xmm1 = _mm_shufflehi_epi16(xmm0, 0b00011011);
	xmm0 = _mm_shuffle_epi32(xmm0, 0b01000100);
	xmm1 = _mm_shuffle_epi32(xmm1, 0b11101110);

	xmm2 = xmm0;
	xmm0 = _mm_adds_epi16(xmm0, xmm1);
	xmm2 = _mm_subs_epi16(xmm2, xmm1);

	xmm0 = _mm_unpacklo_epi32(xmm0, xmm2);
	xmm2 = _mm_shuffle_epi32(xmm0, 0b01001110);

	temp1 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[8]));
	temp2 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[16]));
	temp3 = _mm_madd_epi16(xmm2, *((__m128i*) & ftab2_128[24]));
	temp4 = _mm_madd_epi16(xmm0, *((__m128i*) & ftab2_128[0]));

	xmm0 = _mm_add_epi32(temp4, temp1);
	xmm2 = _mm_add_epi32(temp3, temp2);

	xmm0 = _mm_add_epi32(xmm0, round);
	xmm2 = _mm_add_epi32(xmm2, round);

	xmm0 = _mm_srai_epi32(xmm0, SHIFT_FRW_ROW10);
	xmm2 = _mm_srai_epi32(xmm2, SHIFT_FRW_ROW10);

	xmm0 = _mm_packs_epi32(xmm0, xmm2);

	in7 = xmm0;

	//===================
	//BEGIN QUANTIZATION. We eliminate the 4 shift back and forth by doing DCT and QUANTIZE together! 
	// This also improves the quality slightly due to less rounding errors...
	//===================

	//abs source
	__m128i b0 = _mm_abs_epi16(in0);
	__m128i b1 = _mm_abs_epi16(in1);
	__m128i b2 = _mm_abs_epi16(in2);
	__m128i b3 = _mm_abs_epi16(in3);
	__m128i b4 = _mm_abs_epi16(in4);
	__m128i b5 = _mm_abs_epi16(in5);
	__m128i b6 = _mm_abs_epi16(in6);
	__m128i b7 = _mm_abs_epi16(in7);

	//////shift 4 bits
	//b0 = _mm_slli_epi16(b0, 4);
	//b1 = _mm_slli_epi16(b1, 4);
	//b2 = _mm_slli_epi16(b2, 4);
	//b3 = _mm_slli_epi16(b3, 4);
	//b4 = _mm_slli_epi16(b4, 4);
	//b5 = _mm_slli_epi16(b5, 4);
	//b6 = _mm_slli_epi16(b6, 4);
	//b7 = _mm_slli_epi16(b7, 4);

	//load correction
	__m128i c0 = _mm_load_si128((__m128i*) & matrix[0]);
	__m128i c1 = _mm_load_si128((__m128i*) & matrix[8]);
	__m128i c2 = _mm_load_si128((__m128i*) & matrix[16]);
	__m128i c3 = _mm_load_si128((__m128i*) & matrix[24]);
	__m128i c4 = _mm_load_si128((__m128i*) & matrix[32]);
	__m128i c5 = _mm_load_si128((__m128i*) & matrix[40]);
	__m128i c6 = _mm_load_si128((__m128i*) & matrix[48]);
	__m128i c7 = _mm_load_si128((__m128i*) & matrix[56]);

	//add correction
	b0 = _mm_add_epi16(b0, c0);
	b1 = _mm_add_epi16(b1, c1);
	b2 = _mm_add_epi16(b2, c2);
	b3 = _mm_add_epi16(b3, c3);
	b4 = _mm_add_epi16(b4, c4);
	b5 = _mm_add_epi16(b5, c5);
	b6 = _mm_add_epi16(b6, c6);
	b7 = _mm_add_epi16(b7, c7);

	//load reciprocal
	c0 = _mm_load_si128((__m128i*) & matrix[64]);
	c1 = _mm_load_si128((__m128i*) & matrix[72]);
	c2 = _mm_load_si128((__m128i*) & matrix[80]);
	c3 = _mm_load_si128((__m128i*) & matrix[88]);
	c4 = _mm_load_si128((__m128i*) & matrix[96]);
	c5 = _mm_load_si128((__m128i*) & matrix[104]);
	c6 = _mm_load_si128((__m128i*) & matrix[112]);
	c7 = _mm_load_si128((__m128i*) & matrix[120]);

	//multiply reciprocal
	b0 = _mm_mulhi_epu16(b0, c0);
	b1 = _mm_mulhi_epu16(b1, c1);
	b2 = _mm_mulhi_epu16(b2, c2);
	b3 = _mm_mulhi_epu16(b3, c3);
	b4 = _mm_mulhi_epu16(b4, c4);
	b5 = _mm_mulhi_epu16(b5, c5);
	b6 = _mm_mulhi_epu16(b6, c6);
	b7 = _mm_mulhi_epu16(b7, c7);

	//load scale
	c0 = _mm_load_si128((__m128i*) & matrix[128]);
	c1 = _mm_load_si128((__m128i*) & matrix[136]);
	c2 = _mm_load_si128((__m128i*) & matrix[144]);
	c3 = _mm_load_si128((__m128i*) & matrix[152]);
	c4 = _mm_load_si128((__m128i*) & matrix[160]);
	c5 = _mm_load_si128((__m128i*) & matrix[168]);
	c6 = _mm_load_si128((__m128i*) & matrix[176]);
	c7 = _mm_load_si128((__m128i*) & matrix[184]);

	//multiply scale
	b0 = _mm_mulhi_epu16(b0, c0);
	b1 = _mm_mulhi_epu16(b1, c1);
	b2 = _mm_mulhi_epu16(b2, c2);
	b3 = _mm_mulhi_epu16(b3, c3);
	b4 = _mm_mulhi_epu16(b4, c4);
	b5 = _mm_mulhi_epu16(b5, c5);
	b6 = _mm_mulhi_epu16(b6, c6);
	b7 = _mm_mulhi_epu16(b7, c7);

	//sign
	in0 = _mm_sign_epi16(b0, in0);
	in1 = _mm_sign_epi16(b1, in1);
	in2 = _mm_sign_epi16(b2, in2);
	in3 = _mm_sign_epi16(b3, in3);
	in4 = _mm_sign_epi16(b4, in4);
	in5 = _mm_sign_epi16(b5, in5);
	in6 = _mm_sign_epi16(b6, in6);
	in7 = _mm_sign_epi16(b7, in7);

	//===================
	//BEGIN ZIG ZAG. ~38 instructions. Probably not optimal, but sure beats using a loop!
	//===================

	__m128i r0 = _mm_unpacklo_epi16(in0, in1); 	//0	8	1	9	2	10	3	11
	__m128i r1 = _mm_unpackhi_epi16(in0, in1); 	//4	12	5	13	6	14	7	15
	__m128i r2 = _mm_unpacklo_epi16(in2, in3); 	// 16	24	17	25	18	26	19	27
	__m128i r3 = _mm_unpackhi_epi16(in2, in3); 	//20	28	21	29	22	30	23	31
	__m128i r4 = _mm_unpacklo_epi16(in4, in5); 	//32	40	33	41	34	42	35	43
	__m128i r5 = _mm_unpackhi_epi16(in4, in5); 	//36	44	37	45	38	46	39	47
	__m128i r6 = _mm_unpacklo_epi16(in6, in7); 	//48	56	49	57	50	58	51	59
	__m128i r7 = _mm_unpackhi_epi16(in6, in7); 	//52	60	53	61	54	62	55	63		

	in0 = _mm_shuffle_epi8(r0, _mm_set_epi8(11, 10, 13, 12, 9, 8, 7, 6, -1, -1, 3, 2, 5, 4, 1, 0));
	in2 = _mm_slli_si128(in2, 6);
	in0 = _mm_blend_epi16(in0, in2, 0x8);
	__m128i t = _mm_blend_epi16(r2, r4, 1);
	in1 = _mm_blend_epi16(r0, r1, 5);
	t = _mm_shuffle_epi8(t, _mm_set_epi8(-1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 1, 0, 3, 2, 5, 4));
	in1 = _mm_shuffle_epi8(in1, _mm_set_epi8(5, 4, 1, 0, 15, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
	in1 = _mm_or_si128(in1, t);
	t = _mm_alignr_epi8(r6, r1, 2);
	in2 = _mm_blend_epi16(r4, r2, 96);
	in2 = _mm_blend_epi16(in2, t, 1 + 128);
	in2 = _mm_shuffle_epi8(in2, _mm_set_epi8(9, 8, 7, 6, 15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0));
	t = _mm_blend_epi16(r1, r3, 7);
	t = _mm_blend_epi16(t, r2, 128);
	in3 = _mm_shuffle_epi8(t, _mm_set_epi8(3, 2, 5, 4, 11, 10, 13, 12, 9, 8, 7, 6, 1, 0, 15, 14));
	t = _mm_blend_epi16(r4, r6, 30);
	t = _mm_blend_epi16(t, r5, 1);
	in4 = _mm_shuffle_epi8(t, _mm_set_epi8(1, 0, 15, 14, 9, 8, 7, 6, 3, 2, 5, 4, 11, 10, 13, 12));
	t = _mm_alignr_epi8(r6, r1, 14);
	in5 = _mm_blend_epi16(r3, r5, 6);
	in5 = _mm_blend_epi16(in5, t, 1 + 128);
	in5 = _mm_shuffle_epi8(in5, _mm_set_epi8(15, 14, 3, 2, 5, 4, 11, 10, 13, 12, 1, 0, 9, 8, 7, 6));
	in6 = _mm_blend_epi16(r5, r3, 128);
	t = _mm_blend_epi16(r6, r7, 1);
	t = _mm_shuffle_epi8(t, _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 15, 14, 11, 10));
	in6 = _mm_shuffle_epi8(in6, _mm_set_epi8(11, 10, 13, 12, 15, 14, 9, 8, 7, 6, -1, -1, -1, -1, -1, -1));
	r5 = _mm_srli_si128(r5, 6);
	in6 = _mm_or_si128(in6, t);
	in7 = _mm_shuffle_epi8(r7, _mm_set_epi8(15, 14, 11, 10, 13, 12, -1, -1, 9, 8, 7, 6, 3, 2, 5, 4));
	in7 = _mm_blend_epi16(in7, r5, 0x10);

	*out0 = in0;
	*out1 = in1;
	*out2 = in2;
	*out3 = in3;
	*out4 = in4;
	*out5 = in5;
	*out6 = in6;
	*out7 = in7;
}

void VMX_EncodePlaneInternal128(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_PLANE plane = *pPlane;

	//int width = plane.Size.width;
	int height = VMX_SLICE_HEIGHT;
	short dcPred = 0;
	short dc = 0;
	BYTE* src = plane.Data + s->Offset[plane.Index];
	int stride = plane.Stride;

	uint32_t numZeros = 0;
	uint64_t mIndex = 0;
	uint32_t input = 0;
	uint32_t bc = 0;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlock = s->TempBlock;

	__m128i zo = _mm_setzero_si128();

	int dcshift = instance->DCShift;
	int dcround = 0;
	if (dcshift) dcround = 1 << (dcshift - 1);
	int addVal = 0;
	if (plane.Index == 0 || plane.Index == 3) addVal = -128;

	unsigned short* matrix = instance->EncodeMatrix;

	uint64_t nz;
	short* pos = TempBlock;
	short* end;

	GolombZeroCodeLookup zeroLut;

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 8)
		{
			__m128i a0, a1, a2, a3, a4, a5, a6, a7;
			VMX_FDCT_8X8_QUANT_ZIG_128(src, stride, matrix, addVal, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
			src += 8;

			dc = _mm_extract_epi16(a0, 0);
			dc += dcround;
			dc >>= dcshift;

			__m128i b0, b2, b4, b6;
			b0 = _mm_packs_epi16(a0, a1);
			b2 = _mm_packs_epi16(a2, a3);
			b4 = _mm_packs_epi16(a4, a5);
			b6 = _mm_packs_epi16(a6, a7);

			b0 = _mm_cmpeq_epi8(b0, zo);
			b2 = _mm_cmpeq_epi8(b2, zo);
			b4 = _mm_cmpeq_epi8(b4, zo);
			b6 = _mm_cmpeq_epi8(b6, zo);

			mIndex = (uint64_t)(_mm_movemask_epi8(b0) | 1); //clear first bit always
			mIndex |= (uint64_t)(_mm_movemask_epi8(b2)) << 16;
			mIndex |= (uint64_t)(_mm_movemask_epi8(b4)) << 32;
			mIndex |= (uint64_t)(_mm_movemask_epi8(b6)) << 48;
			mIndex = ~mIndex;

			EncodeDC(dataDC, (dc - dcPred));
			EmitBits32(dataDC);
			dcPred = dc;

			if (mIndex == 0)
			{
				numZeros += 64;
				continue;
			}

			Get2MagSignPlusOneV(a0);
			_mm_store_si128((__m128i*) & TempBlock[0], a0);

			if ((mIndex & 0x00000000FFFFFF00)) {
				Get2MagSignPlusOneV(a1);
				Get2MagSignPlusOneV(a2);
				Get2MagSignPlusOneV(a3);
				_mm_store_si128((__m128i*) & TempBlock[8], a1);
				_mm_store_si128((__m128i*) & TempBlock[16], a2);
				_mm_store_si128((__m128i*) & TempBlock[24], a3);
			}
			if ((mIndex & 0xFFFFFFFF00000000)) {
				Get2MagSignPlusOneV(a4);
				Get2MagSignPlusOneV(a5);
				Get2MagSignPlusOneV(a6);
				Get2MagSignPlusOneV(a7);
				_mm_store_si128((__m128i*) & TempBlock[32], a4);
				_mm_store_si128((__m128i*) & TempBlock[40], a5);
				_mm_store_si128((__m128i*) & TempBlock[48], a6);
				_mm_store_si128((__m128i*) & TempBlock[56], a7);
			}

			pos = TempBlock;
			end = pos + 64;

			if (mIndex)
			{
				nz = _tzcnt_u64(mIndex);
				numZeros += nz;
				EncodeZeros(dataAC);
				EmitBits32(dataAC);
				EncodeValue(dataAC, pos[0]);
				pos++;
				mIndex >>= nz;
				mIndex >>= 1; //NB: Separate shifts necessary, as nz could be 63 and a shift of 64 won't work as any shift outside 0-63 is masked out by CPU.
				EmitBits32(dataAC);

				while (1)
				{
					if (mIndex)
					{
						nz = _tzcnt_u64(mIndex);
						EncodeZerosSmall(dataAC);
						EncodeValue(dataAC, pos[0]);
						pos++;
						mIndex >>= nz + 1;
						EmitBits32(dataAC);
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

void VMX_EncodePlaneInternal128_16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_PLANE plane = *pPlane;

	//int width = plane.Size.width;
	int height = VMX_SLICE_HEIGHT;
	short dcPred = 0;
	short dc = 0;
	BYTE* src = plane.Data + s->Offset16[plane.Index];
	int stride = plane.Stride * 2;

	uint32_t numZeros = 0;
	uint64_t mIndex = 0;
	uint32_t input = 0;
	uint32_t bc = 0;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlock = s->TempBlock;

	__m128i zo = _mm_setzero_si128();

	int dcshift = instance->DCShift;
	int dcround = 0;
	if (dcshift) dcround = 1 << (dcshift - 1);
	int addVal = 0;
	if (plane.Index == 0 || plane.Index == 3) addVal = -512;

	unsigned short* matrix = instance->EncodeMatrix;

	uint64_t nz;
	short* pos = TempBlock;
	short* end;

	GolombZeroCodeLookup zeroLut;

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 16)
		{
			__m128i a0, a1, a2, a3, a4, a5, a6, a7;
			VMX_FDCT_8X8_QUANT_ZIG_128_16(src, stride, matrix, addVal, &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);
			src += 16;

			dc = _mm_extract_epi16(a0, 0);
			dc += dcround;
			dc >>= dcshift;

			__m128i b0, b2, b4, b6;
			b0 = _mm_packs_epi16(a0, a1);
			b2 = _mm_packs_epi16(a2, a3);
			b4 = _mm_packs_epi16(a4, a5);
			b6 = _mm_packs_epi16(a6, a7);

			b0 = _mm_cmpeq_epi8(b0, zo);
			b2 = _mm_cmpeq_epi8(b2, zo);
			b4 = _mm_cmpeq_epi8(b4, zo);
			b6 = _mm_cmpeq_epi8(b6, zo);

			mIndex = (uint64_t)(_mm_movemask_epi8(b0) | 1); //clear first bit always
			mIndex |= (uint64_t)(_mm_movemask_epi8(b2)) << 16;
			mIndex |= (uint64_t)(_mm_movemask_epi8(b4)) << 32;
			mIndex |= (uint64_t)(_mm_movemask_epi8(b6)) << 48;
			mIndex = ~mIndex;

			EncodeDC(dataDC, (dc - dcPred));
			EmitBits32(dataDC);
			dcPred = dc;

			if (mIndex == 0)
			{
				numZeros += 64;
				continue;
			}

			Get2MagSignPlusOneV(a0);
			_mm_store_si128((__m128i*) & TempBlock[0], a0);

			if ((mIndex & 0x00000000FFFFFF00)) {
				Get2MagSignPlusOneV(a1);
				Get2MagSignPlusOneV(a2);
				Get2MagSignPlusOneV(a3);
				_mm_store_si128((__m128i*) & TempBlock[8], a1);
				_mm_store_si128((__m128i*) & TempBlock[16], a2);
				_mm_store_si128((__m128i*) & TempBlock[24], a3);
			}
			if ((mIndex & 0xFFFFFFFF00000000)) {
				Get2MagSignPlusOneV(a4);
				Get2MagSignPlusOneV(a5);
				Get2MagSignPlusOneV(a6);
				Get2MagSignPlusOneV(a7);
				_mm_store_si128((__m128i*) & TempBlock[32], a4);
				_mm_store_si128((__m128i*) & TempBlock[40], a5);
				_mm_store_si128((__m128i*) & TempBlock[48], a6);
				_mm_store_si128((__m128i*) & TempBlock[56], a7);
			}

			pos = TempBlock;
			end = pos + 64;

			if (mIndex)
			{
				nz = _tzcnt_u64(mIndex);
				numZeros += nz;
				EncodeZeros(dataAC);
				EmitBits32(dataAC);
				EncodeValue(dataAC, pos[0]);
				pos++;
				mIndex >>= nz;
				mIndex >>= 1; //NB: Separate shifts necessary, as nz could be 63 and a shift of 64 won't work as any shift outside 0-63 is masked out by CPU.
				EmitBits32(dataAC);

				while (1)
				{
					if (mIndex)
					{
						nz = _tzcnt_u64(mIndex);
						EncodeZerosSmall(dataAC);
						EncodeValue(dataAC, pos[0]);
						pos++;
						mIndex >>= nz + 1;
						EmitBits32(dataAC);
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

void VMX_DecodePlaneInternal128(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
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
	buffer_t termsToDecode = 0;

	int dcshift = instance->DCShift;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlock = s->TempBlock;
	short* TempBlock2 = s->TempBlock2;

	unsigned short* matrix = instance->DecodeMatrix;

	int validCount = 0;

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 8)
		{
			memset(TempBlock2, 0, 128);
			//Decode 64 values		
			validCount = 0;
			if (termsToDecode < 64)
			{
				validCount = 1;
			}
			while (termsToDecode < 64)
			{
				GolombLookup l = GolombLookupLut[(dataAC.TempRead >> (dataAC.BitsLeft - 12)) & 0xFFF];
				if (l.length)
				{
					dataAC.BitsLeft -= l.length;
					TempBlock2[termsToDecode] = l.value;
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
							GETBITSB(dataAC, bc, val);
							termsToDecode += val;
							bc = 0;
						}
					}
					else {
						GETZEROSB(dataAC, bc);
						bc += 2;
						GETBITSB(dataAC, bc, val);
						TempBlock2[termsToDecode] = GetIntFrom2MagSign((val - 1));
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
				TempBlock2[0] = GetIntFrom2MagSign((val - 1));
				TempBlock2[0] <<= dcshift;
			}

			TempBlock = TempBlock2;
			TempBlock[0] += dcPred;
			dcPred = TempBlock[0];

			if (validCount)
			{
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_128(TempBlock, matrix, (BYTE*)pDst, stride, shift);
			}
			else {
				VMX_BROADCAST_DC_8X8_128(TempBlock[0], (BYTE*)pDst, stride, shift);
			}

			pDst += 8;
		}
		pDst += (7 * stride);

	}

	REWINDOVERREAD(dataAC);

	FLUSHREMAININGREADBITS(dataDC);
	FLUSHREMAININGREADBITS(dataAC);
	s->AC = dataAC;
	s->DC = dataDC;
}

void VMX_DecodePlaneInternal128_16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
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
	buffer_t termsToDecode = 0;

	int dcshift = instance->DCShift;

	VMX_SLICE_DATA dataDC = s->DC;
	VMX_SLICE_DATA dataAC = s->AC;
	short* TempBlock = s->TempBlock;
	short* TempBlock2 = s->TempBlock2;

	unsigned short* matrix = instance->DecodeMatrix;

	int validCount = 0;

	for (int y = 0; y < height; y += 8)
	{
		for (int x = 0; x < stride; x += 16)
		{
			memset(TempBlock2, 0, 128);
			//Decode 64 values		
			validCount = 0;
			if (termsToDecode < 64)
			{
				validCount = 1;
			}
			while (termsToDecode < 64)
			{
				GolombLookup l = GolombLookupLut[(dataAC.TempRead >> (dataAC.BitsLeft - 12)) & 0xFFF];
				if (l.length)
				{
					dataAC.BitsLeft -= l.length;
					TempBlock2[termsToDecode] = l.value;
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
							GETBITSB(dataAC, bc, val);
							termsToDecode += val;
							bc = 0;
						}
					}
					else {
						GETZEROSB(dataAC, bc);
						bc += 2;
						GETBITSB(dataAC, bc, val);
						TempBlock2[termsToDecode] = GetIntFrom2MagSign((val - 1));
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
				TempBlock2[0] = GetIntFrom2MagSign((val - 1));
				TempBlock2[0] <<= dcshift;
			}

			TempBlock = TempBlock2;
			TempBlock[0] += dcPred;
			dcPred = TempBlock[0];

			validCount = 1;
			if (validCount)
			{
				VMX_ZIG_INVQUANTIZE_IDCT_8X8_128_16(TempBlock, matrix, (BYTE*)pDst, stride, shift);
			}
			else {
				VMX_BROADCAST_DC_8X8_128_16(TempBlock[0], (BYTE*)pDst, stride, shift);
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

void VMX_DecodePlanePreviewInternal(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	buffer_t b = 0;
	buffer_t bc = 0;
	buffer_t val = 0;

	VMX_PLANE plane = *pPlane;

	int stride = plane.Stride;
	int width = plane.Stride >> 3; //We need to read the correct number of DC values per row, so we use stride here
	int height = VMX_SLICE_HEIGHT >> 3;
	int shift = 0;
	if (plane.Index == 0 || plane.Index == 3)
	{
		shift = 128;
	}

	char* pDst = (char*)(plane.Data + (s->Offset[plane.Index] >> 3));
	int dcPred = 0;
	int dcshift = instance->DCShift;

	VMX_SLICE_DATA dataDC = s->DC;

	for (int y = 0; y < height; y++)
	{
		char* dst = pDst;
		for (int x = 0; x < width; x++)
		{
			short dc = 0;
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
				dc = GetIntFrom2MagSign((val - 1));
				dc <<= dcshift;
			}

			dc += dcPred;
			dcPred = dc;

			dc += 4;
			dc >>= 3;
			dc += shift;

			dst[0] = dc;
			dst += 1;
		}
		pDst += stride;
	}
	FLUSHREMAININGREADBITS(dataDC);
	s->DC = dataDC;
}

void VMX_DecodePlaneInternal(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
#if defined(AVX2)
	if (instance->avx2)
	{
		VMX_DecodePlaneInternal256(instance, pPlane, s);
	}
	else
	{
		VMX_DecodePlaneInternal128(instance, pPlane, s);
	}
#else
	VMX_DecodePlaneInternal128(instance, pPlane, s);
#endif
}

void VMX_DecodePlaneInternal16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
#if defined(AVX2)
	if (instance->avx2)
	{
		VMX_DecodePlaneInternal256_16(instance, pPlane, s);
	}
	else
	{
		VMX_DecodePlaneInternal128_16(instance, pPlane, s);
	}
#else
	VMX_DecodePlaneInternal128_16(instance, pPlane, s);
#endif
}

void VMX_EncodePlaneInternal(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
#if defined(AVX2)
	if (instance->avx2)
	{
		VMX_EncodePlaneInternal256(instance, pPlane, s);
	}
	else
	{
		VMX_EncodePlaneInternal128(instance, pPlane, s);
	}
#else
	VMX_EncodePlaneInternal128(instance, pPlane, s);
#endif
}

void VMX_EncodePlaneInternal16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
#if defined(AVX2)
	if (instance->avx2)
	{
		VMX_EncodePlaneInternal256_16(instance, pPlane, s);
	}
	else
	{
		VMX_EncodePlaneInternal128_16(instance, pPlane, s);
	}
#else
	VMX_EncodePlaneInternal128_16(instance, pPlane, s);
#endif
}

void VMX_YUY2ToPlanar(BYTE* src, int srcStride, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size)
{
	BYTE* alignedSrc;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(src, srcStride, size, &alignedSrc, &alignedStride, 64, 2);
	VMX_CopyToAlignedStrideBuffer(alignedSrc, alignedStride, src, srcStride, size, 2);

	BYTE* pSrc = alignedSrc;
	BYTE* pY = ydst;
	BYTE* pU = udst;
	BYTE* pV = vdst;

	__m128i yShuffle = _mm_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 12, 10, 8, 6, 4, 2, 0);

	for (int y = 0; y < size.height; y++)
	{
		for (int x = 0; x < (size.width * 2); x += 64)
		{
			__m128i uyvy1 = _mm_loadu_si128((__m128i*) & pSrc[x]); //Combined uy/vy pairs
			__m128i uyvy2 = _mm_loadu_si128((__m128i*) & pSrc[x + 16]);
			__m128i uyvy3 = _mm_loadu_si128((__m128i*) & pSrc[x + 32]);
			__m128i uyvy4 = _mm_loadu_si128((__m128i*) & pSrc[x + 48]);

			uyvy1 = _mm_shuffle_epi8(uyvy1, yShuffle); //yyyyyyyy uuuuvvvv
			uyvy2 = _mm_shuffle_epi8(uyvy2, yShuffle); //yyyyyyyy uuuuvvvv
			uyvy3 = _mm_shuffle_epi8(uyvy3, yShuffle); //yyyyyyyy uuuuvvvv
			uyvy4 = _mm_shuffle_epi8(uyvy4, yShuffle); //yyyyyyyy uuuuvvvv

			__m128i y1 = _mm_unpacklo_epi64(uyvy1, uyvy2);
			__m128i y2 = _mm_unpacklo_epi64(uyvy3, uyvy4);

			_mm_storeu_si128((__m128i*) & pY[0], y1);
			pY += 16;
			_mm_storeu_si128((__m128i*) & pY[0], y2);
			pY += 16;

			__m128i uv1 = _mm_unpackhi_epi32(uyvy1, uyvy2); //uuuuuuuuvvvvvvvv
			__m128i uv2 = _mm_unpackhi_epi32(uyvy3, uyvy4); //uuuuuuuuvvvvvvvv

			__m128i u = _mm_unpacklo_epi64(uv1, uv2);
			__m128i v = _mm_unpackhi_epi64(uv1, uv2);

			_mm_storeu_si128((__m128i*) & pU[0], u);
			_mm_storeu_si128((__m128i*) & pV[0], v);
			pU += 16;
			pV += 16;
		}
		ydst += ystride;
		udst += ustride;
		vdst += vstride;
		pSrc += alignedStride;
		pY = ydst;
		pU = udst;
		pV = vdst;
	}
	VMX_FreeAlignedStrideBuffer(alignedSrc, alignedStride, srcStride);
}


void VMX_AToPlanar(BYTE* src, int srcStride, BYTE* adst, int astride, VMX_SIZE size) {
	for (int y = 0; y < size.height; y++) {
		memcpy(adst, src, size.width);
		src += srcStride;
		adst += astride;
	}
}

void VMX_A16ToPlanar(BYTE* src, int srcStride, BYTE* adst, int astride, VMX_SIZE size) {
	for (int y = 0; y < size.height; y++) {
		memcpy(adst, src, size.width * 2);
		src += srcStride;
		adst += astride;
	}
}

void VMX_UYVYToPlanar(BYTE* src, int srcStride, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size)
{
	BYTE* alignedSrc;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(src, srcStride, size, &alignedSrc, &alignedStride, 64, 2);
	VMX_CopyToAlignedStrideBuffer(alignedSrc, alignedStride, src, srcStride, size, 2);

	BYTE* pSrc = alignedSrc;
	BYTE* pY = ydst;
	BYTE* pU = udst;
	BYTE* pV = vdst;

	__m128i yShuffle = _mm_set_epi8(14, 10, 6, 2, 12, 8, 4, 0, 15, 13, 11, 9, 7, 5, 3, 1);

	for (int y = 0; y < size.height; y++)
	{
		for (int x = 0; x < (size.width * 2); x += 64)
		{
			__m128i uyvy1 = _mm_loadu_si128((__m128i*) & pSrc[x]); //Combined uy/vy pairs
			__m128i uyvy2 = _mm_loadu_si128((__m128i*) & pSrc[x + 16]);
			__m128i uyvy3 = _mm_loadu_si128((__m128i*) & pSrc[x + 32]);
			__m128i uyvy4 = _mm_loadu_si128((__m128i*) & pSrc[x + 48]);

			uyvy1 = _mm_shuffle_epi8(uyvy1, yShuffle); //yyyyyyyy uuuuvvvv
			uyvy2 = _mm_shuffle_epi8(uyvy2, yShuffle); //yyyyyyyy uuuuvvvv
			uyvy3 = _mm_shuffle_epi8(uyvy3, yShuffle); //yyyyyyyy uuuuvvvv
			uyvy4 = _mm_shuffle_epi8(uyvy4, yShuffle); //yyyyyyyy uuuuvvvv

			__m128i y1 = _mm_unpacklo_epi64(uyvy1, uyvy2);
			__m128i y2 = _mm_unpacklo_epi64(uyvy3, uyvy4);

			_mm_storeu_si128((__m128i*) & pY[0], y1);
			pY += 16;
			_mm_storeu_si128((__m128i*) & pY[0], y2);
			pY += 16;

			__m128i uv1 = _mm_unpackhi_epi32(uyvy1, uyvy2); //uuuuuuuuvvvvvvvv
			__m128i uv2 = _mm_unpackhi_epi32(uyvy3, uyvy4); //uuuuuuuuvvvvvvvv

			__m128i u = _mm_unpacklo_epi64(uv1, uv2);
			__m128i v = _mm_unpackhi_epi64(uv1, uv2);

			_mm_storeu_si128((__m128i*) & pU[0], u);
			_mm_storeu_si128((__m128i*) & pV[0], v);
			pU += 16;
			pV += 16;
		}
		ydst += ystride;
		udst += ustride;
		vdst += vstride;
		pSrc += alignedStride;
		pY = ydst;
		pU = udst;
		pV = vdst;
	}
	VMX_FreeAlignedStrideBuffer(alignedSrc, alignedStride, srcStride);
}


void VMX_P216ToPlanar(BYTE* srcY, int srcStrideY, BYTE* srcUV, int srcStrideUV, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size)
{
	//Y Plane
	for (int y = 0; y < size.height; y++) {
		memcpy(ydst, srcY, size.width * 2);
		srcY += srcStrideY;
		ydst += ystride;
	}

	//UV Plane
	BYTE* alignedSrc;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(srcUV, srcStrideUV, size, &alignedSrc, &alignedStride, 32, 2);
	VMX_CopyToAlignedStrideBuffer(alignedSrc, alignedStride, srcUV, srcStrideUV, size, 2);

	BYTE* pSrc = alignedSrc;
	BYTE* pU = udst;
	BYTE* pV = vdst;

	__m128i uShuffle = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0);
	__m128i vShuffle = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 11, 10, 7, 6, 3, 2);

	for (int y = 0; y < size.height; y++)
	{
		for (int x = 0; x < (size.width * 2); x += 32)
		{
			__m128i uv1 = _mm_loadu_si128((__m128i*) & srcUV[x]); //uuvvuuvvuuvvuuvv
			__m128i uv2 = _mm_loadu_si128((__m128i*) & srcUV[x + 16]);

			__m128i u1 = _mm_shuffle_epi8(uv1, uShuffle);
			__m128i u2 = _mm_slli_si128(_mm_shuffle_epi8(uv2, uShuffle), 8);

			__m128i v1 = _mm_shuffle_epi8(uv1, vShuffle);
			__m128i v2 = _mm_slli_si128(_mm_shuffle_epi8(uv2, vShuffle), 8);

			__m128i u = _mm_or_si128(u1, u2);
			__m128i v = _mm_or_si128(v1, v2);

			_mm_storeu_si128((__m128i*) & pU[0], u);
			_mm_storeu_si128((__m128i*) & pV[0], v);

			pU += 16;
			pV += 16;
		}
		udst += ustride;
		vdst += vstride;
		srcUV += alignedStride;
		pU = udst;
		pV = vdst;
	}
	VMX_FreeAlignedStrideBuffer(alignedSrc, alignedStride, srcStrideUV);
}


void VMX_PlanarToA(BYTE* asrc, int astride, BYTE* dst, int dstStride, VMX_SIZE size) {
	for (int y = 0; y < size.height; y++) {
		memcpy(dst, asrc, size.width);
		dst += dstStride;
		asrc += astride;
	}
}

void VMX_PlanarToA16(BYTE* asrc, int astride, BYTE* dst, int dstStride, VMX_SIZE size) {
	for (int y = 0; y < size.height; y++) {
		memcpy(dst, asrc, size.width * 2);
		dst += dstStride;
		asrc += astride;
	}
}

void VMX_PlanarToUYVY(BYTE* ysrc, int ystride, BYTE* usrc, int ustride, BYTE* vsrc, int vstride, BYTE* dst, int dstStride, VMX_SIZE size)
{
	BYTE* alignedDst;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(dst, dstStride, size, &alignedDst, &alignedStride, 64, 2);

	BYTE* pDst = alignedDst;
	BYTE* pY = ysrc;
	BYTE* pU = usrc;
	BYTE* pV = vsrc;

	for (int y = 0; y < size.height; y++)
	{
		for (int x = 0; x < (size.width * 2); x += 64)
		{
			__m128i y1 = _mm_loadu_si128((__m128i*) & pY[0]);
			pY += 16;
			__m128i y2 = _mm_loadu_si128((__m128i*) & pY[0]);
			pY += 16;

			__m128i u1 = _mm_loadu_si128((__m128i*) & pU[0]);
			__m128i v1 = _mm_loadu_si128((__m128i*) & pV[0]);
			pU += 16;
			pV += 16;

			__m128i uv1 = _mm_unpacklo_epi8(u1, v1);
			__m128i uv2 = _mm_unpackhi_epi8(u1, v1);

			__m128i y3 = _mm_unpacklo_epi8(uv1, y1);
			__m128i y4 = _mm_unpackhi_epi8(uv1, y1);
			__m128i y5 = _mm_unpacklo_epi8(uv2, y2);
			__m128i y6 = _mm_unpackhi_epi8(uv2, y2);

			_mm_storeu_si128((__m128i*) & pDst[x], y3);
			_mm_storeu_si128((__m128i*) & pDst[x + 16], y4);
			_mm_storeu_si128((__m128i*) & pDst[x + 32], y5);
			_mm_storeu_si128((__m128i*) & pDst[x + 48], y6);
		}
		ysrc += ystride;
		usrc += ustride;
		vsrc += vstride;
		pY = ysrc;
		pU = usrc;
		pV = vsrc;
		pDst += alignedStride;
	}
	VMX_CopyFromAlignedStrideBufferAndFree(alignedDst, alignedStride, dst, dstStride, size, 2);
}


void VMX_PlanarToP216(BYTE* ysrc, int ystride, BYTE* usrc, int ustride, BYTE* vsrc, int vstride, BYTE* dstY, int dstStrideY, BYTE* dstUV, int dstStrideUV, VMX_SIZE size)
{
	//Y Plane
	for (int y = 0; y < size.height; y++) {
		memcpy(dstY, ysrc, size.width * 2);
		ysrc += ystride;
		dstY += dstStrideY;
	}

	//UV Plane
	BYTE* alignedDst;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(dstUV, dstStrideUV, size, &alignedDst, &alignedStride, 32, 2);

	BYTE* pDst = alignedDst;
	BYTE* pU = usrc;
	BYTE* pV = vsrc;

	for (int y = 0; y < size.height; y++)
	{
		for (int x = 0; x < (size.width * 2); x += 32)
		{
			__m128i u = _mm_loadu_si128((__m128i*) & pU[0]);
			__m128i v = _mm_loadu_si128((__m128i*) & pV[0]);

			__m128i uv1 = _mm_unpacklo_epi16(u, v);
			__m128i uv2 = _mm_unpackhi_epi16(u, v);

			_mm_storeu_si128((__m128i*) & dstUV[x], uv1);
			_mm_storeu_si128((__m128i*) & dstUV[x + 16], uv2);
			pU += 16;
			pV += 16;
		}
		usrc += ustride;
		vsrc += vstride;
		dstUV += alignedStride;
		pU = usrc;
		pV = vsrc;
	}
	VMX_CopyFromAlignedStrideBufferAndFree(alignedDst, alignedStride, dstUV, dstStrideUV, size, 2);
}

void VMX_NV12ToPlanar(BYTE* srcY, int srcStrideY, BYTE* srcUV, int srcStrideUV, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size)
{
	//Y Plane
	for (int y = 0; y < size.height; y++) {
		memcpy(ydst, srcY, size.width);
		srcY += srcStrideY;
		ydst += ystride;
	}

	BYTE* alignedSrc;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(srcUV, srcStrideUV, size, &alignedSrc, &alignedStride, 16, 1);
	VMX_CopyToAlignedStrideBuffer(alignedSrc, alignedStride, srcUV, srcStrideUV, size, 1);

	BYTE* pU = udst;
	BYTE* pV = vdst;

	BYTE* pSrc = alignedSrc;

	//UV Plane
	int height = size.height >> 1;
	int width = size.width;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x += 16) {
			__m128i uv = _mm_loadu_si128((__m128i*) & pSrc[x]); //uvuvuvuvuvuvuvuv
			uv = _mm_shuffle_epi8(uv, _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0)); //uuuuuuuuvvvvvvvv
			_mm_storel_epi64((__m128i*) & pU[0], uv);
			_mm_storel_epi64((__m128i*) & pU[ustride], uv);
			uv = _mm_srli_si128(uv, 8);
			_mm_storel_epi64((__m128i*) & pV[0], uv);
			_mm_storel_epi64((__m128i*) & pV[vstride], uv);
			pU += 8;
			pV += 8;
		}
		pSrc += alignedStride;
		udst += ustride;
		udst += ustride;
		vdst += vstride;
		vdst += vstride;
		pU = udst;
		pV = vdst;
	}

	VMX_FreeAlignedStrideBuffer(alignedSrc, alignedStride, srcStrideUV);
}


void VMX_YV12ToPlanar(BYTE* srcY, int srcStrideY, BYTE* srcU, int srcStrideU, BYTE* srcV, int srcStrideV, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size)
{
	BYTE* pY = ydst;
	BYTE* pU = udst;
	BYTE* pV = vdst;

	//Y Plane
	for (int y = 0; y < size.height; y++) {
		memcpy(pY, srcY, size.width);
		srcY += srcStrideY;
		pY += ystride;
	}

	int height = size.height >> 1;
	int width = size.width >> 1;

	//V Plane
	for (int y = 0; y < height; y++) {
		memcpy(pV, srcV, width);
		pV += vstride;
		memcpy(pV, srcV, width);
		pV += vstride;
		srcV += srcStrideV;
	}

	//U Plane
	for (int y = 0; y < height; y++) {
		memcpy(pU, srcU, width);
		pU += ustride;
		memcpy(pU, srcU, width);
		pU += ustride;
		srcU += srcStrideU;
	}
}

void VMX_PlanarToYUY2(BYTE* ysrc, int ystride, BYTE* usrc, int ustride, BYTE* vsrc, int vstride, BYTE* dst, int dstStride, VMX_SIZE size)
{
	BYTE* alignedDst;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(dst, dstStride, size, &alignedDst, &alignedStride, 64, 2);

	BYTE* pDst = alignedDst;
	BYTE* pY = ysrc;
	BYTE* pU = usrc;
	BYTE* pV = vsrc;

	for (int y = 0; y < size.height; y++)
	{
		for (int x = 0; x < (size.width * 2); x += 64)
		{
			__m128i y1 = _mm_loadu_si128((__m128i*) & pY[0]);
			pY += 16;
			__m128i y2 = _mm_loadu_si128((__m128i*) & pY[0]);
			pY += 16;

			__m128i u1 = _mm_loadu_si128((__m128i*) & pU[0]);
			__m128i v1 = _mm_loadu_si128((__m128i*) & pV[0]);
			pU += 16;
			pV += 16;

			__m128i uv1 = _mm_unpacklo_epi8(u1, v1);
			__m128i uv2 = _mm_unpackhi_epi8(u1, v1);

			__m128i y3 = _mm_unpacklo_epi8(y1, uv1);
			__m128i y4 = _mm_unpackhi_epi8(y1, uv1);
			__m128i y5 = _mm_unpacklo_epi8(y2, uv2);
			__m128i y6 = _mm_unpackhi_epi8(y2, uv2);

			_mm_storeu_si128((__m128i*) & pDst[x], y3);
			_mm_storeu_si128((__m128i*) & pDst[x + 16], y4);
			_mm_storeu_si128((__m128i*) & pDst[x + 32], y5);
			_mm_storeu_si128((__m128i*) & pDst[x + 48], y6);
		}
		ysrc += ystride;
		usrc += ustride;
		vsrc += vstride;
		pY = ysrc;
		pU = usrc;
		pV = vsrc;
		pDst += alignedStride;
	}
	VMX_CopyFromAlignedStrideBufferAndFree(alignedDst, alignedStride, dst, dstStride, size, 2);
}

inline void GetIntFrom2MagSignMinus1V_128(__m128i* input)
{
	__m128i one = _mm_set1_epi16(1);
	*input = _mm_sub_epi16(*input, one);
	__m128i mask = _mm_and_si128(*input, one);
	__m128i x = _mm_adds_epi16(*input, mask);
	__m128i a = _mm_srli_epi16(x, 1);
	__m128i b = _mm_mullo_epi16(x, mask);
	*input = _mm_subs_epi16(a, b);
}

void VMX_ZIG_INVQUANTIZE_IDCT_8X8_128(short* src, unsigned short* matrix, BYTE* dst, int stride, short addVal) {

	//load 8x8x16
	__m128i a0 = _mm_load_si128((__m128i*) & src[0]);
	__m128i a1 = _mm_load_si128((__m128i*) & src[8]);
	__m128i a2 = _mm_load_si128((__m128i*) & src[16]);
	__m128i a3 = _mm_load_si128((__m128i*) & src[24]);
	__m128i a4 = _mm_load_si128((__m128i*) & src[32]);
	__m128i a5 = _mm_load_si128((__m128i*) & src[40]);
	__m128i a6 = _mm_load_si128((__m128i*) & src[48]);
	__m128i a7 = _mm_load_si128((__m128i*) & src[56]);

	//Inverse Zig Zag ~47 instructions. Around 40% faster than a loop
	__m128i v0 = _mm_shuffle_epi8(a0, _mm_set_epi8(7, 6, 15, 14, 9, 8, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0)); //0,1,5,6,2,4,7,3
	__m128i v1 = _mm_shuffle_epi8(a1, _mm_set_epi8(7, 6, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 5, 4)); //10,8,12,13,14,15,9,11
	__m128i v3 = _mm_shuffle_epi8(a3, _mm_set_epi8(9, 8, 7, 6, 13, 12, 3, 2, 11, 10, 5, 4, 15, 14, 1, 0)); //24,31,26,29,25,30,27,28

	//a0 0, 1, 5, 6, 14, 15, 27, 28, 
	a0 = _mm_blend_epi16(v0, v1, 0x30); //-,0,0,0,14,15,-,-
	a0 = _mm_blend_epi16(a0, v3, 0xC0); //-,-,-,-,-,-,27,28

	__m128i v2 = _mm_shuffle_epi8(a2, _mm_set_epi8(5, 4, 13, 12, 9, 8, 1, 0, 3, 2, 15, 14, 7, 6, 11, 10)); //21,19,23,17,16,20,22,18
	__m128i v5 = _mm_shuffle_epi8(a5, _mm_set_epi8(7, 6, 3, 2, 11, 10, 13, 12, 15, 14, 5, 4, 9, 8, 1, 0)); //40,44,42,47,46,45,41,43

	//a2 3, 8, 12, 17, 25, 30, 41, 43,
	a2 = _mm_srli_si128(v0, 14); //3
	a2 = _mm_blend_epi16(a2, v3, 0x30); //25,30
	a2 = _mm_blend_epi16(a2, v1, 0x6); //8,12
	a2 = _mm_blend_epi16(a2, v2, 0x8); //-,-,-,17,-,-,-,-
	a2 = _mm_blend_epi16(a2, v5, 0xC0); //-,-,-,-,-,-,41,43

	//v3 = _mm_shuffle_epi8(a3, _mm_set_epi8(-1, -1, 11, 10, 5, 4, 15, 14, 1, 0, -1, -1, -1, -1, -1, -1)); //-,-,-,24,31,26,29,-
	v3 = _mm_slli_si128(v3, 6);
	__m128i v4 = _mm_shuffle_epi8(a4, _mm_set_epi8(13, 12, 3, 2, 5, 4, 15, 14, 1, 0, 11, 10, 9, 8, 7, 6)); //35,36,37,32,39,34,33,38

	//a1 2, 4, 7, 13, 16, 26, 29, 42,
	v0 = _mm_srli_si128(v0, 8); //2,4,7,3,0,0,0,0
	a1 = _mm_blend_epi16(v0, v1, 0x8); //13
	a1 = _mm_blend_epi16(a1, v2, 0x10); //-,-,-,-,16,-,-,-
	a1 = _mm_blend_epi16(a1, v3, 0x60); //-,-,-,-,-,26,29,-

	__m128i v6 = _mm_shuffle_epi8(a6, _mm_set_epi8(13, 12, 9, 8, -1, -1, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1)); //-,-,48,49,50,-,52,54
	__m128i v7 = _mm_shuffle_epi8(a7, _mm_set_epi8(15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 7, 6, 1, 0, 9, 8)); //60,56,59,61,57,58,62,63

	//a4 10, 19, 23, 32, 39, 45, 52, 54, 
	a4 = _mm_blend_epi16(v1, v6, 0xC0); //10 + 52,54
	a4 = _mm_blend_epi16(a4, v2, 0x6); //-,19,23,-,-,-,-,-
	a4 = _mm_blend_epi16(a4, v4, 0x18); //32,39
	a4 = _mm_blend_epi16(a4, v5, 0x20); //45

	__m128i x6 = _mm_shuffle_epi8(a6, _mm_set_epi8(11, 10, 15, 14, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)); //-,-,-,-,-,51,55,53

	//a3 9, 11, 18, 24, 31, 40, 44, 53, 
	a3 = _mm_srli_si128(v1, 12); //9,11,-,-,-,-,-,-
	a3 = _mm_blend_epi16(a3, v3, 0x18); //-,-,-,24,31,-,-,-
	a3 = _mm_blend_epi16(a3, x6, 0x80); //53

	//a7 35, 36, 48, 49, 57, 58, 62, 63 
	a7 = _mm_blend_epi16(v7, v4, 0x3); //-,-,-,-,57,58,62,63 + //35,36
	a7 = _mm_blend_epi16(a7, v6, 0xC); //48,49

	//a6 21, 34, 37, 47, 50, 56, 59, 61,
	a6 = _mm_slli_si128(v7, 8); //-,-,-,-,56,59,61
	a6 = _mm_blend_epi16(a6, v4, 0x4); //-,-,37,-,-,-,-,-
	a6 = _mm_blend_epi16(a6, v6, 0x10); //50
	a6 = _mm_blend_epi16(a6, v5, 0x8); //47
	a6 = _mm_blend_epi16(a6, v2, 0x1); //21

	v4 = _mm_srli_si128(v4, 8); //-,34,33,38,-,-,-,-

	//a5 20, 22, 33, 38, 46, 51, 55, 60,
	v2 = _mm_srli_si128(v2, 10);
	a5 = _mm_slli_si128(v7, 14); //-,-,-,-,-,-,60
	a5 = _mm_blend_epi16(a5, v4, 0xC); //33,38
	a5 = _mm_blend_epi16(a5, v2, 0x3); //20,22
	a5 = _mm_blend_epi16(a5, v5, 0x10); //46
	a5 = _mm_blend_epi16(a5, x6, 0x60); //51,55

	a6 = _mm_blend_epi16(a6, v4, 0x2); //34
	a3 = _mm_blend_epi16(a3, v2, 0x4); //18

	v5 = _mm_slli_si128(v5, 10); //-,-,-,-,-,40,44,42
	a3 = _mm_blend_epi16(a3, v5, 0x60); //40,44
	a1 = _mm_blend_epi16(a1, v5, 0x80); //42


	//load quant
	__m128i c0 = _mm_load_si128((__m128i*) & matrix[0]);
	__m128i c1 = _mm_load_si128((__m128i*) & matrix[8]);
	__m128i c2 = _mm_load_si128((__m128i*) & matrix[16]);
	__m128i c3 = _mm_load_si128((__m128i*) & matrix[24]);
	__m128i c4 = _mm_load_si128((__m128i*) & matrix[32]);
	__m128i c5 = _mm_load_si128((__m128i*) & matrix[40]);
	__m128i c6 = _mm_load_si128((__m128i*) & matrix[48]);
	__m128i c7 = _mm_load_si128((__m128i*) & matrix[56]);

	//multiply
	a0 = _mm_mullo_epi16(a0, c0);
	a1 = _mm_mullo_epi16(a1, c1);
	a2 = _mm_mullo_epi16(a2, c2);
	a3 = _mm_mullo_epi16(a3, c3);
	a4 = _mm_mullo_epi16(a4, c4);
	a5 = _mm_mullo_epi16(a5, c5);
	a6 = _mm_mullo_epi16(a6, c6);
	a7 = _mm_mullo_epi16(a7, c7);

	//shift
	a0 = _mm_srai_epi16(a0, 4);
	a1 = _mm_srai_epi16(a1, 4);
	a2 = _mm_srai_epi16(a2, 4);
	a3 = _mm_srai_epi16(a3, 4);
	a4 = _mm_srai_epi16(a4, 4);
	a5 = _mm_srai_epi16(a5, 4);
	a6 = _mm_srai_epi16(a6, 4);
	a7 = _mm_srai_epi16(a7, 4);

	// /////////////////////////////
	// //////////Row 1 And row 3
	// ////////////////////////////
	__m128i r_xmm0 = a0;
	__m128i r_xmm4 = a2;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	__m128i r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_04_128[0]));
	__m128i r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_04_128[16]));
	__m128i r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_04_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_04_128[24]));
	__m128i r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	__m128i r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_26_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	__m128i r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_26_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_26_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_26_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row0 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row2 = _mm_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 5 And row 7
	// ////////////////////////////
	r_xmm0 = a4;
	r_xmm4 = a6;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_04_128[0]));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_04_128[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_04_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_04_128[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_26_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_26_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_26_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_26_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row4 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row6 = _mm_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 4 And row 2
	// ////////////////////////////
	r_xmm0 = a3;
	r_xmm4 = a1;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_35_128[0]));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_35_128[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_35_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_35_128[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_17_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_17_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_17_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_17_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row3 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row1 = _mm_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 6 And row 8
	// ////////////////////////////
	r_xmm0 = a5;
	r_xmm4 = a7;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_35_128[0]));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_35_128[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_35_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_35_128[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_17_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_17_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_17_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_17_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row5 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row7 = _mm_packs_epi32(r_xmm4, r_xmm6);
	r_xmm1 = _mm_loadu_si128((__m128i*) tg_3_16_128);
	r_xmm2 = row5;
	r_xmm3 = row3;
	r_xmm0 = _mm_mulhi_epi16(row5, r_xmm1);
	r_xmm1 = _mm_mulhi_epi16(r_xmm1, r_xmm3);
	r_xmm5 = _mm_loadu_si128((__m128i*) tg_1_16_128);
	r_xmm6 = row7;
	r_xmm4 = _mm_mulhi_epi16(row7, r_xmm5);
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm2);
	r_xmm5 = _mm_mulhi_epi16(r_xmm5, row1);
	r_xmm1 = _mm_adds_epi16(r_xmm1, r_xmm3);
	r_xmm7 = row6;
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm3);
	r_xmm3 = _mm_loadu_si128((__m128i*) tg_2_16_128);
	r_xmm2 = _mm_subs_epi16(r_xmm2, r_xmm1);
	r_xmm7 = _mm_mulhi_epi16(r_xmm7, r_xmm3);
	r_xmm1 = r_xmm0;
	r_xmm3 = _mm_mulhi_epi16(r_xmm3, row2);
	r_xmm5 = _mm_subs_epi16(r_xmm5, r_xmm6);
	r_xmm4 = _mm_adds_epi16(r_xmm4, row1);
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm4);
	r_xmm0 = _mm_adds_epi16(r_xmm0, *((__m128i*) one_corr_128));
	r_xmm4 = _mm_subs_epi16(r_xmm4, r_xmm1);
	r_xmm6 = r_xmm5;
	r_xmm5 = _mm_subs_epi16(r_xmm5, r_xmm2);
	r_xmm5 = _mm_adds_epi16(r_xmm5, *((__m128i*) one_corr_128));
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm2);
	__m128i temp7 = r_xmm0;
	r_xmm1 = r_xmm4;
	r_xmm0 = _mm_load_si128((__m128i*) cos_4_16_128);
	r_xmm4 = _mm_adds_epi16(r_xmm4, r_xmm5);
	r_xmm2 = _mm_load_si128((__m128i*) cos_4_16_128);
	r_xmm2 = _mm_mulhi_epi16(r_xmm2, r_xmm4);
	__m128i temp3 = r_xmm6;
	r_xmm1 = _mm_subs_epi16(r_xmm1, r_xmm5);
	r_xmm7 = _mm_adds_epi16(r_xmm7, row2);
	r_xmm3 = _mm_subs_epi16(r_xmm3, row6);
	r_xmm6 = row0;
	r_xmm0 = _mm_mulhi_epi16(r_xmm0, r_xmm1);
	r_xmm5 = row4;
	r_xmm5 = _mm_adds_epi16(r_xmm5, r_xmm6);
	r_xmm6 = _mm_subs_epi16(r_xmm6, row4);
	r_xmm4 = _mm_adds_epi16(r_xmm4, r_xmm2);
	r_xmm4 = _mm_or_si128(r_xmm4, *((__m128i*) one_corr_128));
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm1);
	r_xmm0 = _mm_or_si128(r_xmm0, *((__m128i*) one_corr_128));
	r_xmm2 = r_xmm5;
	r_xmm5 = _mm_adds_epi16(r_xmm5, r_xmm7);
	r_xmm1 = r_xmm6;
	r_xmm5 = _mm_adds_epi16(r_xmm5, *((__m128i*) round_inv_col_128));
	r_xmm2 = _mm_subs_epi16(r_xmm2, r_xmm7);
	r_xmm7 = temp7;
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm3);
	r_xmm6 = _mm_adds_epi16(r_xmm6, *((__m128i*) round_inv_col_128));
	r_xmm7 = _mm_adds_epi16(r_xmm7, r_xmm5);
	r_xmm7 = _mm_srai_epi16(r_xmm7, SHIFT_INV_COL);
	r_xmm1 = _mm_subs_epi16(r_xmm1, r_xmm3);
	r_xmm1 = _mm_adds_epi16(r_xmm1, *((__m128i*) round_inv_corr_128));
	r_xmm3 = r_xmm6;
	r_xmm2 = _mm_adds_epi16(r_xmm2, *((__m128i*) round_inv_corr_128));
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm4);

	__m128i vadd = _mm_set1_epi16(addVal);
	r_xmm7 = _mm_adds_epi16(r_xmm7, vadd);

	r_xmm6 = _mm_srai_epi16(r_xmm6, SHIFT_INV_COL);

	r_xmm6 = _mm_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm_packus_epi16(r_xmm7, r_xmm6);

	BYTE itemp[16];
	_mm_storeu_si128((__m128i*) (&itemp[0]), r_xmm6);
	memcpy(dst, &itemp[0], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	dst += stride;

	r_xmm7 = r_xmm1;
	r_xmm1 = _mm_adds_epi16(r_xmm1, r_xmm0);

	r_xmm1 = _mm_srai_epi16(r_xmm1, SHIFT_INV_COL);
	r_xmm6 = temp3;
	r_xmm7 = _mm_subs_epi16(r_xmm7, r_xmm0);
	r_xmm7 = _mm_srai_epi16(r_xmm7, SHIFT_INV_COL);

	r_xmm1 = _mm_adds_epi16(r_xmm1, vadd);

	r_xmm5 = _mm_subs_epi16(r_xmm5, temp7);
	r_xmm5 = _mm_srai_epi16(r_xmm5, SHIFT_INV_COL);

	r_xmm5 = _mm_adds_epi16(r_xmm5, vadd);

	r_xmm3 = _mm_subs_epi16(r_xmm3, r_xmm4);
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm2);
	r_xmm2 = _mm_subs_epi16(r_xmm2, temp3);
	r_xmm6 = _mm_srai_epi16(r_xmm6, SHIFT_INV_COL);
	r_xmm2 = _mm_srai_epi16(r_xmm2, SHIFT_INV_COL);

	r_xmm6 = _mm_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm_packus_epi16(r_xmm1, r_xmm6);

	_mm_storeu_si128((__m128i*) (&itemp[0]), r_xmm6);
	memcpy(dst, &itemp[0], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	dst += stride;

	r_xmm3 = _mm_srai_epi16(r_xmm3, SHIFT_INV_COL);

	r_xmm2 = _mm_adds_epi16(r_xmm2, vadd);

	r_xmm7 = _mm_adds_epi16(r_xmm7, vadd);
	r_xmm7 = _mm_packus_epi16(r_xmm2, r_xmm7);

	_mm_storeu_si128((__m128i*) (&itemp[0]), r_xmm7);
	memcpy(dst, &itemp[0], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	dst += stride;

	r_xmm3 = _mm_adds_epi16(r_xmm3, vadd);
	r_xmm3 = _mm_packus_epi16(r_xmm3, r_xmm5);

	_mm_storeu_si128((__m128i*) (&itemp[0]), r_xmm3);
	memcpy(dst, &itemp[0], 8);
	dst += stride;
	memcpy(dst, &itemp[8], 8);
	dst += stride;

}


void VMX_ZIG_INVQUANTIZE_IDCT_8X8_128_16(short* src, unsigned short* matrix, BYTE* dst, int stride, short addVal) {

	//load 8x8x16
	__m128i a0 = _mm_load_si128((__m128i*) & src[0]);
	__m128i a1 = _mm_load_si128((__m128i*) & src[8]);
	__m128i a2 = _mm_load_si128((__m128i*) & src[16]);
	__m128i a3 = _mm_load_si128((__m128i*) & src[24]);
	__m128i a4 = _mm_load_si128((__m128i*) & src[32]);
	__m128i a5 = _mm_load_si128((__m128i*) & src[40]);
	__m128i a6 = _mm_load_si128((__m128i*) & src[48]);
	__m128i a7 = _mm_load_si128((__m128i*) & src[56]);

	//Inverse Zig Zag ~47 instructions. Around 40% faster than a loop
	__m128i v0 = _mm_shuffle_epi8(a0, _mm_set_epi8(7, 6, 15, 14, 9, 8, 5, 4, 13, 12, 11, 10, 3, 2, 1, 0)); //0,1,5,6,2,4,7,3
	__m128i v1 = _mm_shuffle_epi8(a1, _mm_set_epi8(7, 6, 3, 2, 15, 14, 13, 12, 11, 10, 9, 8, 1, 0, 5, 4)); //10,8,12,13,14,15,9,11
	__m128i v3 = _mm_shuffle_epi8(a3, _mm_set_epi8(9, 8, 7, 6, 13, 12, 3, 2, 11, 10, 5, 4, 15, 14, 1, 0)); //24,31,26,29,25,30,27,28

	//a0 0, 1, 5, 6, 14, 15, 27, 28, 
	a0 = _mm_blend_epi16(v0, v1, 0x30); //-,0,0,0,14,15,-,-
	a0 = _mm_blend_epi16(a0, v3, 0xC0); //-,-,-,-,-,-,27,28

	__m128i v2 = _mm_shuffle_epi8(a2, _mm_set_epi8(5, 4, 13, 12, 9, 8, 1, 0, 3, 2, 15, 14, 7, 6, 11, 10)); //21,19,23,17,16,20,22,18
	__m128i v5 = _mm_shuffle_epi8(a5, _mm_set_epi8(7, 6, 3, 2, 11, 10, 13, 12, 15, 14, 5, 4, 9, 8, 1, 0)); //40,44,42,47,46,45,41,43

	//a2 3, 8, 12, 17, 25, 30, 41, 43,
	a2 = _mm_srli_si128(v0, 14); //3
	a2 = _mm_blend_epi16(a2, v3, 0x30); //25,30
	a2 = _mm_blend_epi16(a2, v1, 0x6); //8,12
	a2 = _mm_blend_epi16(a2, v2, 0x8); //-,-,-,17,-,-,-,-
	a2 = _mm_blend_epi16(a2, v5, 0xC0); //-,-,-,-,-,-,41,43

	//v3 = _mm_shuffle_epi8(a3, _mm_set_epi8(-1, -1, 11, 10, 5, 4, 15, 14, 1, 0, -1, -1, -1, -1, -1, -1)); //-,-,-,24,31,26,29,-
	v3 = _mm_slli_si128(v3, 6);
	__m128i v4 = _mm_shuffle_epi8(a4, _mm_set_epi8(13, 12, 3, 2, 5, 4, 15, 14, 1, 0, 11, 10, 9, 8, 7, 6)); //35,36,37,32,39,34,33,38

	//a1 2, 4, 7, 13, 16, 26, 29, 42,
	v0 = _mm_srli_si128(v0, 8); //2,4,7,3,0,0,0,0
	a1 = _mm_blend_epi16(v0, v1, 0x8); //13
	a1 = _mm_blend_epi16(a1, v2, 0x10); //-,-,-,-,16,-,-,-
	a1 = _mm_blend_epi16(a1, v3, 0x60); //-,-,-,-,-,26,29,-

	__m128i v6 = _mm_shuffle_epi8(a6, _mm_set_epi8(13, 12, 9, 8, -1, -1, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1)); //-,-,48,49,50,-,52,54
	__m128i v7 = _mm_shuffle_epi8(a7, _mm_set_epi8(15, 14, 13, 12, 5, 4, 3, 2, 11, 10, 7, 6, 1, 0, 9, 8)); //60,56,59,61,57,58,62,63

	//a4 10, 19, 23, 32, 39, 45, 52, 54, 
	a4 = _mm_blend_epi16(v1, v6, 0xC0); //10 + 52,54
	a4 = _mm_blend_epi16(a4, v2, 0x6); //-,19,23,-,-,-,-,-
	a4 = _mm_blend_epi16(a4, v4, 0x18); //32,39
	a4 = _mm_blend_epi16(a4, v5, 0x20); //45

	__m128i x6 = _mm_shuffle_epi8(a6, _mm_set_epi8(11, 10, 15, 14, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)); //-,-,-,-,-,51,55,53

	//a3 9, 11, 18, 24, 31, 40, 44, 53, 
	a3 = _mm_srli_si128(v1, 12); //9,11,-,-,-,-,-,-
	a3 = _mm_blend_epi16(a3, v3, 0x18); //-,-,-,24,31,-,-,-
	a3 = _mm_blend_epi16(a3, x6, 0x80); //53

	//a7 35, 36, 48, 49, 57, 58, 62, 63 
	a7 = _mm_blend_epi16(v7, v4, 0x3); //-,-,-,-,57,58,62,63 + //35,36
	a7 = _mm_blend_epi16(a7, v6, 0xC); //48,49

	//a6 21, 34, 37, 47, 50, 56, 59, 61,
	a6 = _mm_slli_si128(v7, 8); //-,-,-,-,56,59,61
	a6 = _mm_blend_epi16(a6, v4, 0x4); //-,-,37,-,-,-,-,-
	a6 = _mm_blend_epi16(a6, v6, 0x10); //50
	a6 = _mm_blend_epi16(a6, v5, 0x8); //47
	a6 = _mm_blend_epi16(a6, v2, 0x1); //21

	v4 = _mm_srli_si128(v4, 8); //-,34,33,38,-,-,-,-

	//a5 20, 22, 33, 38, 46, 51, 55, 60,
	v2 = _mm_srli_si128(v2, 10);
	a5 = _mm_slli_si128(v7, 14); //-,-,-,-,-,-,60
	a5 = _mm_blend_epi16(a5, v4, 0xC); //33,38
	a5 = _mm_blend_epi16(a5, v2, 0x3); //20,22
	a5 = _mm_blend_epi16(a5, v5, 0x10); //46
	a5 = _mm_blend_epi16(a5, x6, 0x60); //51,55

	a6 = _mm_blend_epi16(a6, v4, 0x2); //34
	a3 = _mm_blend_epi16(a3, v2, 0x4); //18

	v5 = _mm_slli_si128(v5, 10); //-,-,-,-,-,40,44,42
	a3 = _mm_blend_epi16(a3, v5, 0x60); //40,44
	a1 = _mm_blend_epi16(a1, v5, 0x80); //42


	//load quant
	__m128i c0 = _mm_load_si128((__m128i*) & matrix[0]);
	__m128i c1 = _mm_load_si128((__m128i*) & matrix[8]);
	__m128i c2 = _mm_load_si128((__m128i*) & matrix[16]);
	__m128i c3 = _mm_load_si128((__m128i*) & matrix[24]);
	__m128i c4 = _mm_load_si128((__m128i*) & matrix[32]);
	__m128i c5 = _mm_load_si128((__m128i*) & matrix[40]);
	__m128i c6 = _mm_load_si128((__m128i*) & matrix[48]);
	__m128i c7 = _mm_load_si128((__m128i*) & matrix[56]);

	//multiply
	a0 = _mm_mullo_epi16(a0, c0);
	a1 = _mm_mullo_epi16(a1, c1);
	a2 = _mm_mullo_epi16(a2, c2);
	a3 = _mm_mullo_epi16(a3, c3);
	a4 = _mm_mullo_epi16(a4, c4);
	a5 = _mm_mullo_epi16(a5, c5);
	a6 = _mm_mullo_epi16(a6, c6);
	a7 = _mm_mullo_epi16(a7, c7);

	//shift
	a0 = _mm_srai_epi16(a0, 2);
	a1 = _mm_srai_epi16(a1, 2);
	a2 = _mm_srai_epi16(a2, 2);
	a3 = _mm_srai_epi16(a3, 2);
	a4 = _mm_srai_epi16(a4, 2);
	a5 = _mm_srai_epi16(a5, 2);
	a6 = _mm_srai_epi16(a6, 2);
	a7 = _mm_srai_epi16(a7, 2);

	// /////////////////////////////
	// //////////Row 1 And row 3
	// ////////////////////////////
	__m128i r_xmm0 = a0;
	__m128i r_xmm4 = a2;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	__m128i r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_04_128[0]));
	__m128i r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_04_128[16]));
	__m128i r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_04_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_04_128[24]));
	__m128i r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	__m128i r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_26_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	__m128i r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_26_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_26_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_26_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row0 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row2 = _mm_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 5 And row 7
	// ////////////////////////////
	r_xmm0 = a4;
	r_xmm4 = a6;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_04_128[0]));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_04_128[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_04_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_04_128[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_26_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_26_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_26_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_26_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row4 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row6 = _mm_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 4 And row 2
	// ////////////////////////////
	r_xmm0 = a3;
	r_xmm4 = a1;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_35_128[0]));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_35_128[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_35_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_35_128[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_17_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_17_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_17_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_17_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row3 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row1 = _mm_packs_epi32(r_xmm4, r_xmm6);
	// /////////////////////////////
	// //////////Row 6 And row 8
	// ////////////////////////////
	r_xmm0 = a5;
	r_xmm4 = a7;
	r_xmm0 = _mm_shufflelo_epi16(r_xmm0, 0xd8);
	r_xmm1 = _mm_shuffle_epi32(r_xmm0, 0);
	r_xmm1 = _mm_madd_epi16(r_xmm1, *((__m128i*) & tab_i_35_128[0]));
	r_xmm3 = _mm_shuffle_epi32(r_xmm0, 0x55);
	r_xmm0 = _mm_shufflehi_epi16(r_xmm0, 0xd8);
	r_xmm3 = _mm_madd_epi16(r_xmm3, *((__m128i*) & tab_i_35_128[16]));
	r_xmm2 = _mm_shuffle_epi32(r_xmm0, 0xaa);
	r_xmm0 = _mm_shuffle_epi32(r_xmm0, 0xff);
	r_xmm2 = _mm_madd_epi16(r_xmm2, *((__m128i*) & tab_i_35_128[8]));
	r_xmm4 = _mm_shufflehi_epi16(r_xmm4, 0xd8);
	r_xmm1 = _mm_add_epi32(r_xmm1, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_shufflelo_epi16(r_xmm4, 0xd8);
	r_xmm0 = _mm_madd_epi16(r_xmm0, *((__m128i*) & tab_i_35_128[24]));
	r_xmm5 = _mm_shuffle_epi32(r_xmm4, 0);
	r_xmm6 = _mm_shuffle_epi32(r_xmm4, 0xaa);
	r_xmm5 = _mm_madd_epi16(r_xmm5, *((__m128i*) & tab_i_17_128[0]));
	r_xmm1 = _mm_add_epi32(r_xmm1, r_xmm2);
	r_xmm2 = r_xmm1;
	r_xmm7 = _mm_shuffle_epi32(r_xmm4, 0x55);
	r_xmm6 = _mm_madd_epi16(r_xmm6, *((__m128i*) & tab_i_17_128[8]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm3);
	r_xmm4 = _mm_shuffle_epi32(r_xmm4, 0xff);
	r_xmm2 = _mm_sub_epi32(r_xmm2, r_xmm0);
	r_xmm7 = _mm_madd_epi16(r_xmm7, *((__m128i*) & tab_i_17_128[16]));
	r_xmm0 = _mm_add_epi32(r_xmm0, r_xmm1);
	r_xmm2 = _mm_srai_epi32(r_xmm2, SHIFT_INV_ROW10);
	r_xmm5 = _mm_add_epi32(r_xmm5, *((__m128i*) round_inv_row_128_10));
	r_xmm4 = _mm_madd_epi16(r_xmm4, *((__m128i*) & tab_i_17_128[24]));
	r_xmm5 = _mm_add_epi32(r_xmm5, r_xmm6);
	r_xmm6 = r_xmm5;
	r_xmm0 = _mm_srai_epi32(r_xmm0, SHIFT_INV_ROW10);
	r_xmm2 = _mm_shuffle_epi32(r_xmm2, 0x1b);
	__m128i row5 = _mm_packs_epi32(r_xmm0, r_xmm2);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm7);
	r_xmm6 = _mm_sub_epi32(r_xmm6, r_xmm4);
	r_xmm4 = _mm_add_epi32(r_xmm4, r_xmm5);
	r_xmm6 = _mm_srai_epi32(r_xmm6, SHIFT_INV_ROW10);
	r_xmm4 = _mm_srai_epi32(r_xmm4, SHIFT_INV_ROW10);
	r_xmm6 = _mm_shuffle_epi32(r_xmm6, 0x1b);
	__m128i row7 = _mm_packs_epi32(r_xmm4, r_xmm6);
	r_xmm1 = _mm_loadu_si128((__m128i*) tg_3_16_128);
	r_xmm2 = row5;
	r_xmm3 = row3;
	r_xmm0 = _mm_mulhi_epi16(row5, r_xmm1);
	r_xmm1 = _mm_mulhi_epi16(r_xmm1, r_xmm3);
	r_xmm5 = _mm_loadu_si128((__m128i*) tg_1_16_128);
	r_xmm6 = row7;
	r_xmm4 = _mm_mulhi_epi16(row7, r_xmm5);
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm2);
	r_xmm5 = _mm_mulhi_epi16(r_xmm5, row1);
	r_xmm1 = _mm_adds_epi16(r_xmm1, r_xmm3);
	r_xmm7 = row6;
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm3);
	r_xmm3 = _mm_loadu_si128((__m128i*) tg_2_16_128);
	r_xmm2 = _mm_subs_epi16(r_xmm2, r_xmm1);
	r_xmm7 = _mm_mulhi_epi16(r_xmm7, r_xmm3);
	r_xmm1 = r_xmm0;
	r_xmm3 = _mm_mulhi_epi16(r_xmm3, row2);
	r_xmm5 = _mm_subs_epi16(r_xmm5, r_xmm6);
	r_xmm4 = _mm_adds_epi16(r_xmm4, row1);
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm4);
	r_xmm0 = _mm_adds_epi16(r_xmm0, *((__m128i*) one_corr_128));
	r_xmm4 = _mm_subs_epi16(r_xmm4, r_xmm1);
	r_xmm6 = r_xmm5;
	r_xmm5 = _mm_subs_epi16(r_xmm5, r_xmm2);
	r_xmm5 = _mm_adds_epi16(r_xmm5, *((__m128i*) one_corr_128));
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm2);
	__m128i temp7 = r_xmm0;
	r_xmm1 = r_xmm4;
	r_xmm0 = _mm_load_si128((__m128i*) cos_4_16_128);
	r_xmm4 = _mm_adds_epi16(r_xmm4, r_xmm5);
	r_xmm2 = _mm_load_si128((__m128i*) cos_4_16_128);
	r_xmm2 = _mm_mulhi_epi16(r_xmm2, r_xmm4);
	__m128i temp3 = r_xmm6;
	r_xmm1 = _mm_subs_epi16(r_xmm1, r_xmm5);
	r_xmm7 = _mm_adds_epi16(r_xmm7, row2);
	r_xmm3 = _mm_subs_epi16(r_xmm3, row6);
	r_xmm6 = row0;
	r_xmm0 = _mm_mulhi_epi16(r_xmm0, r_xmm1);
	r_xmm5 = row4;
	r_xmm5 = _mm_adds_epi16(r_xmm5, r_xmm6);
	r_xmm6 = _mm_subs_epi16(r_xmm6, row4);
	r_xmm4 = _mm_adds_epi16(r_xmm4, r_xmm2);
	r_xmm4 = _mm_or_si128(r_xmm4, *((__m128i*) one_corr_128));
	r_xmm0 = _mm_adds_epi16(r_xmm0, r_xmm1);
	r_xmm0 = _mm_or_si128(r_xmm0, *((__m128i*) one_corr_128));
	r_xmm2 = r_xmm5;
	r_xmm5 = _mm_adds_epi16(r_xmm5, r_xmm7);
	r_xmm1 = r_xmm6;
	r_xmm5 = _mm_adds_epi16(r_xmm5, *((__m128i*) round_inv_col_128_10));
	r_xmm2 = _mm_subs_epi16(r_xmm2, r_xmm7);
	r_xmm7 = temp7;
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm3);
	r_xmm6 = _mm_adds_epi16(r_xmm6, *((__m128i*) round_inv_col_128_10));
	r_xmm7 = _mm_adds_epi16(r_xmm7, r_xmm5);
	r_xmm7 = _mm_srai_epi16(r_xmm7, SHIFT_INV_COL10);
	r_xmm1 = _mm_subs_epi16(r_xmm1, r_xmm3);
	r_xmm1 = _mm_adds_epi16(r_xmm1, *((__m128i*) round_inv_corr_128_10));
	r_xmm3 = r_xmm6;
	r_xmm2 = _mm_adds_epi16(r_xmm2, *((__m128i*) round_inv_corr_128_10));
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm4);

	__m128i vadd = _mm_set1_epi16(addVal);
	__m128i mmax = _mm_set1_epi16(1023);
	__m128i mmin = _mm_setzero_si128();

	r_xmm7 = _mm_adds_epi16(r_xmm7, vadd);
	r_xmm7 = _mm_min_epi16(r_xmm7, mmax);
	r_xmm7 = _mm_max_epi16(r_xmm7, mmin);
	r_xmm7 = _mm_slli_epi16(r_xmm7, 6);

	r_xmm6 = _mm_srai_epi16(r_xmm6, SHIFT_INV_COL10);

	r_xmm6 = _mm_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm_min_epi16(r_xmm6, mmax);
	r_xmm6 = _mm_max_epi16(r_xmm6, mmin);
	r_xmm6 = _mm_slli_epi16(r_xmm6, 6);

	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm7);
	dst += stride;
	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm6);
	dst += stride;

	r_xmm7 = r_xmm1;
	r_xmm1 = _mm_adds_epi16(r_xmm1, r_xmm0);

	r_xmm1 = _mm_srai_epi16(r_xmm1, SHIFT_INV_COL10);
	r_xmm6 = temp3;
	r_xmm7 = _mm_subs_epi16(r_xmm7, r_xmm0);
	r_xmm7 = _mm_srai_epi16(r_xmm7, SHIFT_INV_COL10);

	r_xmm1 = _mm_adds_epi16(r_xmm1, vadd);
	r_xmm1 = _mm_min_epi16(r_xmm1, mmax);
	r_xmm1 = _mm_max_epi16(r_xmm1, mmin);
	r_xmm1 = _mm_slli_epi16(r_xmm1, 6);

	r_xmm5 = _mm_subs_epi16(r_xmm5, temp7);
	r_xmm5 = _mm_srai_epi16(r_xmm5, SHIFT_INV_COL10);

	r_xmm5 = _mm_adds_epi16(r_xmm5, vadd);
	r_xmm5 = _mm_min_epi16(r_xmm5, mmax);
	r_xmm5 = _mm_max_epi16(r_xmm5, mmin);
	r_xmm5 = _mm_slli_epi16(r_xmm5, 6);

	r_xmm3 = _mm_subs_epi16(r_xmm3, r_xmm4);
	r_xmm6 = _mm_adds_epi16(r_xmm6, r_xmm2);
	r_xmm2 = _mm_subs_epi16(r_xmm2, temp3);
	r_xmm6 = _mm_srai_epi16(r_xmm6, SHIFT_INV_COL10);
	r_xmm2 = _mm_srai_epi16(r_xmm2, SHIFT_INV_COL10);

	r_xmm6 = _mm_adds_epi16(r_xmm6, vadd);
	r_xmm6 = _mm_min_epi16(r_xmm6, mmax);
	r_xmm6 = _mm_max_epi16(r_xmm6, mmin);
	r_xmm6 = _mm_slli_epi16(r_xmm6, 6);

	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm1);
	dst += stride;
	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm6);
	dst += stride;

	r_xmm3 = _mm_srai_epi16(r_xmm3, SHIFT_INV_COL10);

	r_xmm2 = _mm_adds_epi16(r_xmm2, vadd);
	r_xmm2 = _mm_min_epi16(r_xmm2, mmax);
	r_xmm2 = _mm_max_epi16(r_xmm2, mmin);
	r_xmm2 = _mm_slli_epi16(r_xmm2, 6);

	r_xmm7 = _mm_adds_epi16(r_xmm7, vadd);
	r_xmm7 = _mm_min_epi16(r_xmm7, mmax);
	r_xmm7 = _mm_max_epi16(r_xmm7, mmin);
	r_xmm7 = _mm_slli_epi16(r_xmm7, 6);

	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm2);
	dst += stride;
	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm7);
	dst += stride;

	r_xmm3 = _mm_adds_epi16(r_xmm3, vadd);
	r_xmm3 = _mm_min_epi16(r_xmm3, mmax);
	r_xmm3 = _mm_max_epi16(r_xmm3, mmin);
	r_xmm3 = _mm_slli_epi16(r_xmm3, 6);

	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm3);
	dst += stride;
	_mm_storeu_si128((__m128i*) (&dst[0]), r_xmm5);
	dst += stride;

}


void VMX_BROADCAST_DC_8X8_128(short src, BYTE* dst, int stride, short addVal)
{
	src += 4;
	src >>= 3;
	src += addVal;

	__m128i b = _mm_set1_epi16(src);
	__m128i a = _mm_packus_epi16(b, b);

	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storel_epi64((__m128i*) & dst[0], a);
	dst += stride;
}

void VMX_BROADCAST_DC_8X8_128_16(short src, BYTE* dst, int stride, short addVal)
{
	src >>= 1;
	src += addVal;
	__m128i a = _mm_set1_epi16(src);

	__m128i mmax = _mm_set1_epi16(1023);
	__m128i mmin = _mm_setzero_si128();
	a = _mm_min_epi16(a, mmax);
	a = _mm_max_epi16(a, mmin);
	a = _mm_slli_epi16(a, 6);

	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
	_mm_storeu_si128((__m128i*) & dst[0], a);
	dst += stride;
}

//Convert 8 16Bit pixels into 8 16Bit U or V values
static inline __m128i VMX_ConvertRGBVec(__m128i r, __m128i g, __m128i b, short mulR, short mulG, short mulB, short add)
{
	__m128i y = _mm_mullo_epi16(r, _mm_set1_epi16(mulR));
	__m128i tmp = _mm_mullo_epi16(g, _mm_set1_epi16(mulG));
	y = _mm_adds_epi16(y, tmp);
	tmp = _mm_mullo_epi16(b, _mm_set1_epi16(mulB));
	y = _mm_adds_epi16(y, tmp);
	y = _mm_adds_epi16(y, _mm_set1_epi16(128));
	y = _mm_srai_epi16(y, 8);
	y = _mm_adds_epi16(y, _mm_set1_epi16(add));
	return y;
}
//Same as above, except that the G channel result can be > 32767, so we need to upscale to 32bit for all calculations
//as there is no _mm_mullo_ for unsigned 16bit.
static inline __m128i VMX_ConvertRGBVecU(__m128i r, __m128i g, __m128i b, short mulR, short mulG, short mulB, short add)
{
	__m128i y = _mm_mullo_epi16(r, _mm_set1_epi16(mulR));

	//Convert G to 32bit
	__m128i tmp = _mm_cvtepu16_epi32(g);
	tmp = _mm_mullo_epi32(tmp, _mm_set1_epi32(mulG));
	__m128i tmp2 = _mm_srli_si128(g, 8);
	tmp2 = _mm_cvtepi16_epi32(tmp2);
	tmp2 = _mm_mullo_epi32(tmp2, _mm_set1_epi32(mulG));
	tmp = _mm_packus_epi32(tmp, tmp2);
	y = _mm_adds_epu16(y, tmp);

	tmp = _mm_mullo_epi16(b, _mm_set1_epi16(mulB));
	y = _mm_adds_epu16(y, tmp);

	y = _mm_adds_epu16(y, _mm_set1_epi16(128));
	y = _mm_srli_epi16(y, 8);
	y = _mm_adds_epu16(y, _mm_set1_epi16(add));
	return y;
}
//Create 16BPP single channel vector from 8BPP source, I.E R0R0R0R0R0R0R0R0
static inline __m128i VMX_CreateRGBVec(__m128i m1, __m128i m2, char pos)
{
	__m128i r = _mm_shuffle_epi8(m1, _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, pos + 12, -1, pos + 8, -1, pos + 4, -1, pos));
	__m128i tmp = _mm_shuffle_epi8(m2, _mm_set_epi8(-1, pos + 12, -1, pos + 8, -1, pos + 4, -1, pos, -1, -1, -1, -1, -1, -1, -1, -1));
	r = _mm_or_si128(r, tmp);
	return r;
}
//Converts a line of 16 pixels into the planar outputs in the following layout: YYYYYYYYYYYYYYYY,UUUUUUUU,VVVVVVVV,AAAAAAAAAAAAAAAA
static inline void VMX_ConvertBGRABlock(__m128i* mInput, BYTE* pY, BYTE* pU, BYTE* pV, BYTE* pA, ShortRGB cY, ShortRGB cU, ShortRGB cV)
{
	//Input = BGRA,BGRA,BGRA,BGRA x 4
	__m128i m1 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m2 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m3 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m4 = _mm_loadu_si128(mInput);

	__m128i a1 = VMX_CreateRGBVec(m1, m2, 3);
	__m128i r1 = VMX_CreateRGBVec(m1, m2, 2);
	__m128i g1 = VMX_CreateRGBVec(m1, m2, 1);
	__m128i b1 = VMX_CreateRGBVec(m1, m2, 0);

	__m128i a2 = VMX_CreateRGBVec(m3, m4, 3);
	__m128i r2 = VMX_CreateRGBVec(m3, m4, 2);
	__m128i g2 = VMX_CreateRGBVec(m3, m4, 1);
	__m128i b2 = VMX_CreateRGBVec(m3, m4, 0);

	__m128i y1 = VMX_ConvertRGBVecU(r1, g1, b1, cY.R, cY.G, cY.B, 16);
	__m128i u1 = VMX_ConvertRGBVec(r1, g1, b1, cU.R, cU.G, cU.B, 128);
	__m128i v1 = VMX_ConvertRGBVec(r1, g1, b1, cV.R, cV.G, cV.B, 128);

	__m128i y2 = VMX_ConvertRGBVecU(r2, g2, b2, cY.R, cY.G, cY.B, 16);
	__m128i u2 = VMX_ConvertRGBVec(r2, g2, b2, cU.R, cU.G, cU.B, 128);
	__m128i v2 = VMX_ConvertRGBVec(r2, g2, b2, cV.R, cV.G, cV.B, 128);

	u1 = _mm_hadd_epi16(u1, u2);
	u1 = _mm_srai_epi16(u1, 1);

	v1 = _mm_hadd_epi16(v1, v2);
	v1 = _mm_srai_epi16(v1, 1);

	y1 = _mm_packus_epi16(y1, y2);
	u1 = _mm_packus_epi16(u1, u1);
	v1 = _mm_packus_epi16(v1, v1);
	a1 = _mm_packus_epi16(a1, a2);

	_mm_storeu_si128((__m128i*) & pY[0], y1);
	_mm_storel_epi64((__m128i*) & pU[0], u1);
	_mm_storel_epi64((__m128i*) & pV[0], v1);
	_mm_storeu_si128((__m128i*) & pA[0], a1);
}

//Converts a line of 16 pixels into UYVY output
static inline int VMX_ConvertBGRXBlockUYVYConditional(__m128i* mInput, __m128i* mInputPrev, BYTE* pDst, ShortRGB cY, ShortRGB cU, ShortRGB cV)
{
	//Input = BGRA,BGRA,BGRA,BGRA x 4
	__m128i m1 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m2 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m3 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m4 = _mm_loadu_si128(mInput);

	__m128i m1p = _mm_loadu_si128(mInputPrev);
	mInputPrev++;
	__m128i m2p = _mm_loadu_si128(mInputPrev);
	mInputPrev++;
	__m128i m3p = _mm_loadu_si128(mInputPrev);
	mInputPrev++;
	__m128i m4p = _mm_loadu_si128(mInputPrev);

	__m128i cmp1 = _mm_xor_si128(m1, m1p);
	__m128i cmp2 = _mm_xor_si128(m2, m2p);
	__m128i cmp3 = _mm_xor_si128(m3, m3p);
	__m128i cmp4 = _mm_xor_si128(m4, m4p);
	cmp1 = _mm_or_si128(cmp1, cmp2);
	cmp2 = _mm_or_si128(cmp3, cmp4);
	cmp1 = _mm_or_si128(cmp1, cmp2);
	int c1 = _mm_testz_si128(cmp1, cmp1); //1 if equal
	if (c1) return 0;

	__m128i r1 = VMX_CreateRGBVec(m1, m2, 2);
	__m128i g1 = VMX_CreateRGBVec(m1, m2, 1);
	__m128i b1 = VMX_CreateRGBVec(m1, m2, 0);

	__m128i r2 = VMX_CreateRGBVec(m3, m4, 2);
	__m128i g2 = VMX_CreateRGBVec(m3, m4, 1);
	__m128i b2 = VMX_CreateRGBVec(m3, m4, 0);

	__m128i y1 = VMX_ConvertRGBVecU(r1, g1, b1, cY.R, cY.G, cY.B, 16);
	__m128i u1 = VMX_ConvertRGBVec(r1, g1, b1, cU.R, cU.G, cU.B, 128);
	__m128i v1 = VMX_ConvertRGBVec(r1, g1, b1, cV.R, cV.G, cV.B, 128);

	__m128i y2 = VMX_ConvertRGBVecU(r2, g2, b2, cY.R, cY.G, cY.B, 16);
	__m128i u2 = VMX_ConvertRGBVec(r2, g2, b2, cU.R, cU.G, cU.B, 128);
	__m128i v2 = VMX_ConvertRGBVec(r2, g2, b2, cV.R, cV.G, cV.B, 128);

	u1 = _mm_hadd_epi16(u1, u2);
	u1 = _mm_srai_epi16(u1, 1);

	v1 = _mm_hadd_epi16(v1, v2);
	v1 = _mm_srai_epi16(v1, 1);

	__m128i uv1 = _mm_unpacklo_epi16(u1, v1);
	__m128i uv2 = _mm_unpackhi_epi16(u1, v1);

	y1 = _mm_slli_si128(y1, 1);
	y2 = _mm_slli_si128(y2, 1);
	y1 = _mm_or_si128(y1, uv1);
	y2 = _mm_or_si128(y2, uv2);

	_mm_storeu_si128((__m128i*) & pDst[0], y1);
	_mm_storeu_si128((__m128i*) & pDst[16], y2);

	return 1;
}

//Converts a line of 16 pixels into UYVY output
static inline void VMX_ConvertBGRXBlockUYVY(__m128i* mInput, BYTE* pDst, ShortRGB cY, ShortRGB cU, ShortRGB cV)
{
	//Input = BGRA,BGRA,BGRA,BGRA x 4
	__m128i m1 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m2 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m3 = _mm_loadu_si128(mInput);
	mInput++;
	__m128i m4 = _mm_loadu_si128(mInput);

	__m128i r1 = VMX_CreateRGBVec(m1, m2, 2);
	__m128i g1 = VMX_CreateRGBVec(m1, m2, 1);
	__m128i b1 = VMX_CreateRGBVec(m1, m2, 0);

	__m128i r2 = VMX_CreateRGBVec(m3, m4, 2);
	__m128i g2 = VMX_CreateRGBVec(m3, m4, 1);
	__m128i b2 = VMX_CreateRGBVec(m3, m4, 0);

	__m128i y1 = VMX_ConvertRGBVecU(r1, g1, b1, cY.R, cY.G, cY.B, 16);
	__m128i u1 = VMX_ConvertRGBVec(r1, g1, b1, cU.R, cU.G, cU.B, 128);
	__m128i v1 = VMX_ConvertRGBVec(r1, g1, b1, cV.R, cV.G, cV.B, 128);

	__m128i y2 = VMX_ConvertRGBVecU(r2, g2, b2, cY.R, cY.G, cY.B, 16);
	__m128i u2 = VMX_ConvertRGBVec(r2, g2, b2, cU.R, cU.G, cU.B, 128);
	__m128i v2 = VMX_ConvertRGBVec(r2, g2, b2, cV.R, cV.G, cV.B, 128);

	u1 = _mm_hadd_epi16(u1, u2);
	u1 = _mm_srai_epi16(u1, 1);

	v1 = _mm_hadd_epi16(v1, v2);
	v1 = _mm_srai_epi16(v1, 1);

	__m128i uv1 = _mm_unpacklo_epi16(u1, v1);
	__m128i uv2 = _mm_unpackhi_epi16(u1, v1);

	y1 = _mm_slli_si128(y1, 1);
	y2 = _mm_slli_si128(y2, 1);
	y1 = _mm_or_si128(y1, uv1);
	y2 = _mm_or_si128(y2, uv2);

	_mm_storeu_si128((__m128i*) & pDst[0], y1);
	_mm_storeu_si128((__m128i*) & pDst[16], y2);
}
int VMX_BGRXToUYVYConditionalInternal(BYTE* pSrc, BYTE* pSrcPrev, int srcStride, BYTE* pDst, int iStride, VMX_SIZE sz, const ShortRGB* colorTables)
{
	__m128i* mInput = (__m128i*)pSrc;
	__m128i* mInputPrev = (__m128i*)pSrcPrev;

	int width = sz.width;
	int height = sz.height;
	int changed = 0;

	BYTE* pDstUYVY = pDst;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x += 16)
		{
			changed += VMX_ConvertBGRXBlockUYVYConditional(mInput, mInputPrev, pDstUYVY, colorTables[0], colorTables[1], colorTables[2]);
			mInput += 4;
			mInputPrev += 4;
			pDstUYVY += 32;
		}
		pSrc += srcStride;
		pSrcPrev += srcStride;
		mInput = (__m128i*)pSrc;
		mInputPrev = (__m128i*)pSrcPrev;

		pDst += iStride;
		pDstUYVY = pDst;
	}
	if (changed) return 1;
	return 0;
}
void VMX_BGRXToUYVYInternal(BYTE* pSrc, int srcStride, BYTE* pDst, int iStride, VMX_SIZE sz, const ShortRGB* colorTables)
{
	__m128i* mInput = (__m128i*)pSrc;
	int width = sz.width;
	int height = sz.height;

	BYTE* pDstUYVY = pDst;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x += 16)
		{
			VMX_ConvertBGRXBlockUYVY(mInput, pDstUYVY, colorTables[0], colorTables[1], colorTables[2]);
			mInput += 4;
			pDstUYVY += 32;
		}
		pSrc += srcStride;
		mInput = (__m128i*)pSrc;

		pDst += iStride;
		pDstUYVY = pDst;
	}
}

void VMX_BGRAToYUV4224(BYTE* pSrc, int srcStride, BYTE* pDstY, int iStrideY, BYTE* pDstU, int iStrideU, BYTE* pDstV, int iStrideV, BYTE* pDstA, int iStrideA, VMX_SIZE sz, const ShortRGB* colorTables)
{
	BYTE* alignedSrc;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(pSrc, srcStride, sz, &alignedSrc, &alignedStride, 64, 4);
	VMX_CopyToAlignedStrideBuffer(alignedSrc, alignedStride, pSrc, srcStride, sz, 4);

	BYTE* aSrc = alignedSrc;
	__m128i* mInput = (__m128i*)aSrc;
	int width = sz.width;
	int height = sz.height;

	BYTE* pY = pDstY;
	BYTE* pU = pDstU;
	BYTE* pV = pDstV;
	BYTE* pA = pDstA;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x += 16)
		{
			VMX_ConvertBGRABlock(mInput, pY, pU, pV, pA, colorTables[0], colorTables[1], colorTables[2]);
			mInput += 4;
			pY += 16;
			pU += 8;
			pV += 8;
			pA += 16;
		}
		aSrc += alignedStride;
		mInput = (__m128i*)aSrc;

		pDstY += iStrideY;
		pDstU += iStrideU;
		pDstV += iStrideV;
		pDstA += iStrideA;

		pY = pDstY;
		pU = pDstU;
		pV = pDstV;
		pA = pDstA;
	}
	VMX_FreeAlignedStrideBuffer(alignedSrc, alignedStride, srcStride);
}

void VMX_YUV4224ToBGRA(BYTE* pSrcY, int iStrideY, BYTE* pSrcU, int iStrideU, BYTE* pSrcV, int iStrideV, BYTE* pSrcA, int iStrideA, BYTE* pDst, int dstStride, VMX_SIZE sz, const short* colorTable)
{
	BYTE* alignedDst;
	int alignedStride;
	VMX_CreateAlignedStrideBuffer(pDst, dstStride, sz, &alignedDst, &alignedStride, 64, 4);

	int width = sz.width;
	int height = sz.height;

	BYTE* aDst = alignedDst;

	BYTE* pY = pSrcY;
	BYTE* pU = pSrcU;
	BYTE* pV = pSrcV;
	BYTE* pA = pSrcA;

	__m128i rounding = _mm_set1_epi16(8);

	for (int y = 0; y < height; y++)
	{
		BYTE* d = aDst;
		for (int x = 0; x < width; x += 16)
		{

			//Load 16 pixels from Y plane, 8 pixels each from U and V
			__m128i yLine = _mm_loadu_si128((__m128i*)pY);
			__m128i u = _mm_loadl_epi64((__m128i*)pU);
			__m128i v = _mm_loadl_epi64((__m128i*)pV);
			__m128i aLine = _mm_loadu_si128((__m128i*)pA);

			yLine = _mm_subs_epu8(yLine, _mm_set1_epi8(16));

			//Convert all planes to 16bit ready for multiplication
			__m128i y0 = _mm_cvtepu8_epi16(yLine);
			__m128i y1 = _mm_cvtepu8_epi16(_mm_srli_si128(yLine, 8));

			__m128i a0 = _mm_cvtepu8_epi16(aLine);
			__m128i a1 = _mm_cvtepu8_epi16(_mm_srli_si128(aLine, 8));

			u = _mm_cvtepu8_epi16(u);
			v = _mm_cvtepu8_epi16(v);
			u = _mm_sub_epi16(u, _mm_set1_epi16(128));
			v = _mm_sub_epi16(v, _mm_set1_epi16(128));

			//Duplicate U and V into pairs of the same value
			__m128i u0 = _mm_unpacklo_epi16(u, u);
			__m128i u1 = _mm_unpackhi_epi16(u, u);
			__m128i v0 = _mm_unpacklo_epi16(v, v);
			__m128i v1 = _mm_unpackhi_epi16(v, v);

			/////////////////
			//Pixel Group A
			/////////////////
			y0 = _mm_slli_epi16(y0, 6);
			y0 = _mm_mulhi_epi16(y0, _mm_set1_epi16(colorTable[0])); //19076

			v0 = _mm_slli_epi16(v0, 6);
			__m128i r = _mm_mulhi_epi16(v0, _mm_set1_epi16(colorTable[1])); //29372
			r = _mm_adds_epi16(r, y0);

			__m128i b = _mm_slli_epi16(u0, 7);
			b = _mm_mulhi_epi16(b, _mm_set1_epi16(colorTable[4])); //17304
			b = _mm_adds_epi16(b, y0);

			u0 = _mm_slli_epi16(u0, 6);
			__m128i g = _mm_mulhi_epi16(u0, _mm_set1_epi16(colorTable[2])); //3494
			__m128i tmp = _mm_mulhi_epi16(v0, _mm_set1_epi16(colorTable[3])); //8731
			g = _mm_subs_epi16(y0, g);
			g = _mm_subs_epi16(g, tmp);

			r = _mm_adds_epi16(r, rounding);
			g = _mm_adds_epi16(g, rounding);
			b = _mm_adds_epi16(b, rounding);
			r = _mm_srai_epi16(r, 4);
			g = _mm_srai_epi16(g, 4);
			b = _mm_srai_epi16(b, 4);

			//b g b g b g b g
			__m128i bg0 = _mm_unpacklo_epi16(b, g);
			__m128i bg1 = _mm_unpackhi_epi16(b, g);

			//r a r a r a r a
			__m128i ra0 = _mm_unpacklo_epi16(r, a0);
			__m128i ra1 = _mm_unpackhi_epi16(r, a0);

			//bgbgbgbgbgbgbgbg
			bg0 = _mm_packus_epi16(bg0, bg1);
			//rararararararara
			ra1 = _mm_packus_epi16(ra0, ra1);

			__m128i bgra0 = _mm_unpacklo_epi16(bg0, ra1);
			__m128i bgra1 = _mm_unpackhi_epi16(bg0, ra1);

			_mm_storeu_si128((__m128i*)d, bgra0);
			d += 16;
			_mm_storeu_si128((__m128i*)d, bgra1);
			d += 16;

			/////////////////
			//Pixel Group B
			/////////////////
			y1 = _mm_slli_epi16(y1, 6);
			y1 = _mm_mulhi_epi16(y1, _mm_set1_epi16(colorTable[0])); //19076

			v1 = _mm_slli_epi16(v1, 6);
			r = _mm_mulhi_epi16(v1, _mm_set1_epi16(colorTable[1])); //29372
			r = _mm_adds_epi16(r, y1);

			b = _mm_slli_epi16(u1, 7);
			b = _mm_mulhi_epi16(b, _mm_set1_epi16(colorTable[4])); //17304
			b = _mm_adds_epi16(b, y1);

			u1 = _mm_slli_epi16(u1, 6);
			g = _mm_mulhi_epi16(u1, _mm_set1_epi16(colorTable[2])); //3494
			tmp = _mm_mulhi_epi16(v1, _mm_set1_epi16(colorTable[3])); //8731
			g = _mm_subs_epi16(y1, g);
			g = _mm_subs_epi16(g, tmp);

			r = _mm_adds_epi16(r, rounding);
			g = _mm_adds_epi16(g, rounding);
			b = _mm_adds_epi16(b, rounding);
			r = _mm_srai_epi16(r, 4);
			g = _mm_srai_epi16(g, 4);
			b = _mm_srai_epi16(b, 4);

			//b g b g b g b g
			bg0 = _mm_unpacklo_epi16(b, g);
			bg1 = _mm_unpackhi_epi16(b, g);

			//r a r a r a r a
			ra0 = _mm_unpacklo_epi16(r, a1);
			ra1 = _mm_unpackhi_epi16(r, a1);

			//bgbgbgbgbgbgbgbg
			bg0 = _mm_packus_epi16(bg0, bg1);
			//rararararararara
			ra1 = _mm_packus_epi16(ra0, ra1);

			bgra0 = _mm_unpacklo_epi16(bg0, ra1);
			bgra1 = _mm_unpackhi_epi16(bg0, ra1);

			_mm_storeu_si128((__m128i*)d, bgra0);
			d += 16;
			_mm_storeu_si128((__m128i*)d, bgra1);
			d += 16;

			//Output = [BGRA BGRA BGRA BGRA] [BGRA BGRA BGRA BGRA]
			pY += 16;
			pU += 8;
			pV += 8;
			pA += 16;

		}
		pSrcY += iStrideY;
		pSrcU += iStrideU;
		pSrcV += iStrideV;
		pSrcA += iStrideA;
		pY = pSrcY;
		pU = pSrcU;
		pV = pSrcV;
		pA = pSrcA;
		aDst += alignedStride;
	}
	VMX_CopyFromAlignedStrideBufferAndFree(alignedDst, alignedStride, pDst, dstStride, sz, 4);
}

float VMX_CalculatePSNR_128(BYTE* pImage1, BYTE* pImage2, int stride, int bytesPerPixel, VMX_SIZE sz)
{
	if (!pImage1) return 0;
	if (!pImage2) return 0;
	if (!bytesPerPixel) return 0;
	if (stride < sz.width * bytesPerPixel) return 0;

	BYTE* p1 = pImage1;
	BYTE* p2 = pImage2;

	int64_t sums = 0;
	int pixels = sz.width * sz.height * bytesPerPixel;
	int block = 16 / bytesPerPixel;

	for (int y = 0; y < sz.height; y++) {
		for (int x = 0; x < sz.width; x += block) {

			__m128i a = _mm_loadu_si128((__m128i*)p1);
			__m128i b = _mm_loadu_si128((__m128i*)p2);

			__m128i a1 = _mm_cvtepu8_epi32(a);
			__m128i b1 = _mm_cvtepu8_epi32(b);
			a = _mm_srli_si128(a, 4);
			b = _mm_srli_si128(b, 4);
			__m128i a2 = _mm_cvtepu8_epi32(a);
			__m128i b2 = _mm_cvtepu8_epi32(b);
			a = _mm_srli_si128(a, 4);
			b = _mm_srli_si128(b, 4);
			__m128i a3 = _mm_cvtepu8_epi32(a);
			__m128i b3 = _mm_cvtepu8_epi32(b);
			a = _mm_srli_si128(a, 4);
			b = _mm_srli_si128(b, 4);
			__m128i a4 = _mm_cvtepu8_epi32(a);
			__m128i b4 = _mm_cvtepu8_epi32(b);

			a1 = _mm_sub_epi32(a1, b1);
			a1 = _mm_mullo_epi32(a1, a1);

			a2 = _mm_sub_epi32(a2, b2);
			a2 = _mm_mullo_epi32(a2, a2);

			a3 = _mm_sub_epi32(a3, b3);
			a3 = _mm_mullo_epi32(a3, a3);

			a4 = _mm_sub_epi32(a4, b4);
			a4 = _mm_mullo_epi32(a4, a4);

			a1 = _mm_add_epi32(a1, a2);
			a1 = _mm_add_epi32(a1, a3);
			a1 = _mm_add_epi32(a1, a4);

			sums += _mm_extract_epi32(a1, 0);
			sums += _mm_extract_epi32(a1, 1);
			sums += _mm_extract_epi32(a1, 2);
			sums += _mm_extract_epi32(a1, 3);

			p1 += 16;
			p2 += 16;
		}
		pImage1 += stride;
		pImage2 += stride;
		p1 = pImage1;
		p2 = pImage2;
	}

	double mse = (double)sums / (double)pixels;
	double max = 224 * 224;
	double result = 10.0 * log10(max / mse);

	return (float)result;
}


#endif