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
#pragma once
#include "vmxcodec.h"
#if defined(X64)

#include "vmxcodec_common.h"

#if defined(__GNUC__)
#define __lzcnt(t) __builtin_clz(t)
#endif
#if defined(__linux__) | defined(__APPLE__)
#include <x86intrin.h>
#define VMX_BUFFERSWAP(t) __builtin_bswap64(t)
#else
#include <intrin.h>
#define VMX_BUFFERSWAP(t) _byteswap_uint64(t)
#endif

void VMX_BROADCAST_DC_8X8_128(short src, BYTE* dst, int stride, short addVal);
void VMX_BROADCAST_DC_8X8_128_16(short src, BYTE* dst, int stride, short addVal);
void VMX_ZIG_INVQUANTIZE_IDCT_8X8_128(short* src, unsigned short* matrix, BYTE* dst, int stride, short addVal);
void VMX_ZIG_INVQUANTIZE_IDCT_8X8_128_16(short* src, unsigned short* matrix, BYTE* dst, int stride, short addVal);
void VMX_FDCT_8X8_QUANT_ZIG_128(const BYTE* src, int stride, unsigned short* matrix, short addVal, __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3, __m128i* out4, __m128i* out5, __m128i* out6, __m128i* out7);
void VMX_FDCT_8X8_QUANT_ZIG_128_16(const BYTE* src, int stride, unsigned short* matrix, short addVal, __m128i* out0, __m128i* out1, __m128i* out2, __m128i* out3, __m128i* out4, __m128i* out5, __m128i* out6, __m128i* out7);
void VMX_PlanarToUYVY(BYTE* ysrc, int ystride, BYTE* usrc, int ustride, BYTE* vsrc, int vstride, BYTE* dst, int stride, VMX_SIZE size);
void VMX_PlanarToP216(BYTE* ysrc, int ystride, BYTE* usrc, int ustride, BYTE* vsrc, int vstride, BYTE* dstY, int dstStrideY, BYTE* dstUV, int dstStrideUV, VMX_SIZE size);
void VMX_UYVYToPlanar(BYTE* src, int stride, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size);
void VMX_P216ToPlanar(BYTE* srcY, int srcStrideY, BYTE* srcUV, int srcStrideUV, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size);
void VMX_AToPlanar(BYTE* src, int srcStride, BYTE* adst, int astride, VMX_SIZE size);
void VMX_A16ToPlanar(BYTE* src, int srcStride, BYTE* adst, int astride, VMX_SIZE size);
void VMX_PlanarToA(BYTE* asrc, int astride, BYTE* dst, int dstStride, VMX_SIZE size);
void VMX_PlanarToA16(BYTE* asrc, int astride, BYTE* dst, int dstStride, VMX_SIZE size);
void VMX_PlanarToYUY2(BYTE* ysrc, int ystride, BYTE* usrc, int ustride, BYTE* vsrc, int vstride, BYTE* dst, int stride, VMX_SIZE size);
void VMX_YUY2ToPlanar(BYTE* src, int stride, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size);
void VMX_NV12ToPlanar(BYTE* srcY, int strideY, BYTE * srcUV, int strideUV, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size);
void VMX_YV12ToPlanar(BYTE* srcY, int srcStrideY, BYTE* srcU, int srcStrideU, BYTE* srcV, int srcStrideV, BYTE* ydst, int ystride, BYTE* udst, int ustride, BYTE* vdst, int vstride, VMX_SIZE size);
void VMX_EncodePlaneInternal(VMX_INSTANCE* instance, VMX_PLANE * pPlane, VMX_SLICE_SET* s);
void VMX_EncodePlaneInternal16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_YUV4224ToBGRA(BYTE* pSrcY, int iStrideY, BYTE* pSrcU, int iStrideU, BYTE* pSrcV, int iStrideV, BYTE* pSrcA, int iStrideA, BYTE* pDst, int dstStride, VMX_SIZE sz, const short * colorTable);
void VMX_BGRAToYUV4224(BYTE* pSrc, int srcStride, BYTE* pDstY, int iStrideY, BYTE* pDstU, int iStrideU, BYTE* pDstV, int iStrideV, BYTE* pDstA, int iStrideA, VMX_SIZE sz, const ShortRGB * colorTables);
void VMX_DecodePlaneInternal(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_DecodePlaneInternal16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_DecodePlanePreviewInternal(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_BGRXToUYVYInternal(BYTE* pSrc, int srcStride, BYTE* pDst, int iStride, VMX_SIZE sz, const ShortRGB* colorTables);
int VMX_BGRXToUYVYConditionalInternal(BYTE* pSrc, BYTE* pSrcPrev, int srcStride, BYTE* pDst, int iStride, VMX_SIZE sz, const ShortRGB* colorTables);
float VMX_CalculatePSNR_128(BYTE* p1, BYTE* p2, int stride, int bytesPerPixel, VMX_SIZE sz);

#endif