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
#if defined(AVX2)

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

void VMX_EncodePlaneInternal256(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_EncodePlaneInternal256_16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_DecodePlaneInternal256(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);
void VMX_DecodePlaneInternal256_16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s);

#endif