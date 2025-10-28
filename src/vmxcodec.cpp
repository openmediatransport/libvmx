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
#include "vmxcodec.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>

#if defined(X64)
#include "vmxcodec_x86.h"
#endif

#if defined(ARM64)
#include "vmxcodec_arm.h"
#endif

#include "vmxcodec_common.h"

using namespace std;

static inline void VMX_SetQualityInternal(VMX_INSTANCE* instance, int q) 
{
	int index = 0;
	for (int i = 0; i < VMX_QUALITY_COUNT; i++)
	{
		if (VMX_QUALITY[i] >= (100 - q)) {
			q = 100 - VMX_QUALITY[i];
			index = i;
			break;
		}
	}
	instance->Quality = q;
	instance->DecodeMatrix = instance->DecodeQualityPresets[index];
	instance->EncodeMatrix = instance->EncodeQualityPresets[index];
	instance->DecodeMatrix256 = instance->DecodeQualityPresets256[index];
	instance->EncodeMatrix256 = instance->EncodeQualityPresets256[index];
}

VMX_API void VMX_SetQuality(VMX_INSTANCE* instance, int q)
{
	if (!instance) return;
	if (q > VMX_MAXQ) q = VMX_MAXQ;
	if (q < instance->MinQuality) q = instance->MinQuality;
	VMX_SetQualityInternal(instance, q);
}

VMX_API int VMX_GetQuality(VMX_INSTANCE* instance)
{
	if (!instance) return 0;
	return instance->Quality;
}

VMX_API void VMX_GetEncodingParameters(VMX_INSTANCE* instance, int* frameMin, int* frameMax, int* minQuality, int* dcShift)
{
	if (!instance) return;
	*frameMin = instance->TargetBytesPerFrameMin;
	*frameMax = instance->TargetBytesPerFrameMax;
	*minQuality = instance->MinQuality;
	*dcShift = instance->DCShift;
}

VMX_API void VMX_SetEncodingParameters(VMX_INSTANCE* instance, int frameMin, int frameMax, int minQuality, int dcShift)
{
	if (!instance) return;
	instance->TargetBytesPerFrameMin = frameMin;
	instance->TargetBytesPerFrameMax = frameMax;
	instance->MinQuality = minQuality;
	instance->DCShift = dcShift;
}

void VMX_ResetData(VMX_SLICE_DATA* s)
{
	s->StreamPos = s->Stream;
	s->BitsLeft = VMX_BITSSIZE;
	s->Temp = 0;
	buffer_t* si = (buffer_t*)s->StreamPos;
	s->TempRead = VMX_BUFFERSWAP(*si);
}

void VMX_ResetStream(VMX_INSTANCE* instance)
{
	for (int i = 0; i < instance->SliceCount; i++)
	{
		VMX_ResetData(&(instance->Slices[i]->AC));
		VMX_ResetData(&(instance->Slices[i]->DC));
	}
}

VMX_API void VMX_Destroy(VMX_INSTANCE* instance)
{
	if (instance)
	{
		if (instance->Tasks)
		{
			DestroyTasks(instance->Tasks);
			instance->Tasks = NULL;
		}
		if (instance->Planes[0].Data) _mm_free(instance->Planes[0].Data);
		if (instance->Planes[1].Data) _mm_free(instance->Planes[1].Data);
		if (instance->Planes[2].Data) _mm_free(instance->Planes[2].Data);
		if (instance->Planes[3].Data) _mm_free(instance->Planes[3].Data);
		if (instance->Slices) {
			for (int i = 0; i < instance->SliceCount; i++)
			{
				if (instance->Slices[i])
				{
					if (instance->Slices[i]->AC.Stream) _mm_free(instance->Slices[i]->AC.Stream);
					if (instance->Slices[i]->DC.Stream) _mm_free(instance->Slices[i]->DC.Stream);
					delete instance->Slices[i];
				}
			}
			delete[] instance->Slices;
		}
		for (int p = 0; p < VMX_QUALITY_COUNT; p++)
		{
			_mm_free(instance->EncodeQualityPresets[p]);
			_mm_free(instance->DecodeQualityPresets[p]);
			_mm_free(instance->EncodeQualityPresets256[p]);
			_mm_free(instance->DecodeQualityPresets256[p]);
		}
		delete instance;
	}
}

static inline int flss(unsigned short val)
{
	int bit = 16;
	if (val == 0) return 0;
	if ((val & 0xff00) == 0) {
		bit -= 8;
		val <<= 8;
	}
	if ((val & 0xF000) == 0) {
		bit -= 4;
		val <<= 4;
	}
	if ((val & 0xC000) == 0) {
		bit -= 2;
		val <<= 2;
	}
	if ((val & 0x8000) == 0) {
		bit -= 1;
		val <<= 1;
	}
	return bit;
}
static inline void createReciprocal(unsigned short divisor, unsigned short* dst)
{
	if (divisor == 1)
	{
		dst[0] = 0;
		dst[1] = 1;
		dst[2] = 1;
	}
	int b = flss(divisor) - 1;
	int r = 2 * 8 + b;
	uint32_t fq = (1 << r) / divisor;
	uint32_t fr = (1 << r) % divisor;
	uint16_t c = divisor / 2;
	if (fr == 0)
	{
		fq >>= 1;
		r -= 1;
	}
	else if (fr <= (divisor / 2U))
	{
		c += 1;
	}
	else {
		fq += 1;
	}
	uint16_t s = (1 << (2 * 8 * 2 - r));
	dst[0] = c;
	dst[1] = fq;
	dst[2] = s;
}

static int VMX_CalculateBitrate(int targetMbps, bool min)
{
	float t = (float)targetMbps;
	t /= (60 * 8);
	t *= 1048576;
	if (min)
	{
		t *= 0.95;
	}
	else 
	{
		t *= 1.05;
	}
	return (int)t;
}

VMX_API VMX_INSTANCE* VMX_Create(VMX_SIZE dimensions, VMX_PROFILE profile, VMX_COLORSPACE colorSpace)
{
	if (dimensions.width < VMX_MIN_WIDTH) return NULL;
	if (dimensions.width > VMX_MAX_WIDTH) return NULL;
	if (dimensions.height < VMX_MIN_HEIGHT) return NULL;
	if (dimensions.height > VMX_MAX_HEIGHT) return NULL;
	if (dimensions.width % 2) return NULL;

	VMX_INSTANCE* instance = new VMX_INSTANCE;

	if (profile == VMX_PROFILE_DEFAULT) { profile = VMX_PROFILE_HQ; }

	instance->avx2 = 0;

#if defined(X64)
	#if defined(__GNUC__)
		if (__builtin_cpu_supports("avx2")) {
			instance->avx2 = 1;
		}
		if (!__builtin_cpu_supports("bmi2")) {
			instance->avx2 = 0;
		}
	#else
		int cpu_info7[4] = { 0 };
		__cpuidex(cpu_info7, 7, 0);

		//avx2
		if ((cpu_info7[1] & (1 << 5)) != 0)
		{
			instance->avx2 = 1;
		}

		//bmi2
		if ((cpu_info7[1] & (1 << 8)) == 0)
		{
			instance->avx2 = 0;
		}
	#endif
#endif

	if (colorSpace == VMX_COLORSPACE_UNDEFINED) {
		if (dimensions.height >= 720) {
			colorSpace = VMX_COLORSPACE_BT709;
		}
		else {
			colorSpace = VMX_COLORSPACE_BT601;
		}
	}

	instance->Profile = profile;
	instance->MinQuality = 80;
	instance->DCShift = 0;
	instance->Format = VMX_FORMAT_PROGRESSIVE;
	instance->ColorSpace = colorSpace;


	for (int i = 0; i < VMX_BR_TABLE_COUNT; i++)
	{
		if (VMX_BITRATE_TABLE[i][VMX_BR_PROFILE_INDEX] == profile)
		{
			if (dimensions.height >= VMX_BITRATE_TABLE[i][VMX_BR_RESOLUTION_INDEX])
			{
				instance->MinQuality = VMX_BITRATE_TABLE[i][VMX_BR_MINQ_INDEX];
				instance->DCShift = VMX_BITRATE_TABLE[i][VMX_BR_SHIFT_INDEX];
				instance->Threads = VMX_BITRATE_TABLE[i][VMX_BR_THREADS_INDEX];
				instance->TargetBytesPerFrameMin = VMX_CalculateBitrate(VMX_BITRATE_TABLE[i][VMX_BR_TARGET_INDEX], true);
				instance->TargetBytesPerFrameMax = VMX_CalculateBitrate(VMX_BITRATE_TABLE[i][VMX_BR_TARGET_INDEX], false);
				break;
			}
		}
	}

	unsigned int nthreads = std::thread::hardware_concurrency();
	if (dimensions.height >= 4320)
	{
		if (nthreads >= 16)
		{
			instance->Threads = 16;
		}

	}
	else if (dimensions.height >= 2160)
	{
		if (nthreads >= 8)
		{
			instance->Threads = 8;
		}
	}	

	//instance->Threads = 1;
	instance->Tasks = CreateTasks(instance->Threads);

	//128bit SIMD
	for (int i = 0; i < VMX_QUALITY_COUNT; i++)
	{
		instance->DecodeQualityPresets[i] = (unsigned short*)_mm_malloc(VMX_DECODE_MATRIX_COUNT * 2, 16);
		instance->EncodeQualityPresets[i] = (unsigned short*)_mm_malloc(VMX_ENCODE_MATRIX_COUNT * 2, 16);
		for (int y = 0; y < VMX_DECODE_MATRIX_COUNT; y++)
		{
			if (y == 0)
			{
				instance->DecodeQualityPresets[i][0] = VMX_DEFAULT_QUANTIZATION_MATRIX[0];
			}
			else {
				instance->DecodeQualityPresets[i][y] = VMX_DEFAULT_QUANTIZATION_MATRIX[y] * VMX_QUALITY[i];
			}
			unsigned short rc[3];
			createReciprocal(instance->DecodeQualityPresets[i][y], &rc[0]);
			instance->EncodeQualityPresets[i][y] = rc[0];
			instance->EncodeQualityPresets[i][y + 64] = rc[1];
			instance->EncodeQualityPresets[i][y + 128] = rc[2];
		}
	}

	//Duplicate matrices for 256bit
	for (int i = 0; i < VMX_QUALITY_COUNT; i++)
	{
		instance->DecodeQualityPresets256[i] = (unsigned short*)_mm_malloc(VMX_DECODE_MATRIX_COUNT * 4, 32);
		instance->EncodeQualityPresets256[i] = (unsigned short*)_mm_malloc(VMX_ENCODE_MATRIX_COUNT * 4, 32);
		for (int y = 0; y < VMX_DECODE_MATRIX_COUNT; y += 8)
		{
			memcpy(&instance->DecodeQualityPresets256[i][y * 2], &instance->DecodeQualityPresets[i][y], 16);
			memcpy(&instance->DecodeQualityPresets256[i][(y * 2) + 8], &instance->DecodeQualityPresets[i][y], 16);

			memcpy(&instance->EncodeQualityPresets256[i][y * 2], &instance->EncodeQualityPresets[i][y], 16);
			memcpy(&instance->EncodeQualityPresets256[i][(y * 2) + 8], &instance->EncodeQualityPresets[i][y], 16);

			memcpy(&instance->EncodeQualityPresets256[i][(y * 2) + 128], &instance->EncodeQualityPresets[i][y + 64], 16);
			memcpy(&instance->EncodeQualityPresets256[i][(y * 2) + 136], &instance->EncodeQualityPresets[i][y + 64], 16);

			memcpy(&instance->EncodeQualityPresets256[i][(y * 2) + 256], &instance->EncodeQualityPresets[i][y + 128], 16);
			memcpy(&instance->EncodeQualityPresets256[i][(y * 2) + 264], &instance->EncodeQualityPresets[i][y + 128], 16);

		}

	}

	instance->Planes[0].Index = 0;
	instance->Planes[1].Index = 1;
	instance->Planes[2].Index = 2;
	instance->Planes[3].Index = 3;

	instance->Planes[0].Size = dimensions;
	instance->Planes[1].Size.width = dimensions.width / 2;
	instance->Planes[1].Size.height = dimensions.height;
	instance->Planes[2].Size = instance->Planes[1].Size;
	instance->Planes[0].Stride = dimensions.width;
	instance->Planes[1].Stride = instance->Planes[1].Size.width;
	instance->Planes[2].Stride = instance->Planes[2].Size.width;
	//Alpha plane
	instance->Planes[3].Size = dimensions;
	instance->Planes[3].Stride = dimensions.width;

	//Preview Dimensions and alignment 
	instance->PreviewSize.width = dimensions.width >> 3;
	instance->PreviewSize.height = dimensions.height >> 3;
	VMX_ALIGN(instance->PreviewSize.width, 2);

	instance->PreviewSizeInterlaced = instance->PreviewSize;
	if (instance->PreviewSizeInterlaced.height % 2) instance->PreviewSizeInterlaced.height--;

	VMX_ALIGN(instance->Planes[0].Stride, 8);
	VMX_ALIGN(instance->Planes[1].Stride, 8);
	VMX_ALIGN(instance->Planes[2].Stride, 8);
	VMX_ALIGN(instance->Planes[3].Stride, 8);
	if (instance->Planes[1].Size.width % 16)
	{
		//We can't use AVX2 if uv plane width is not divisible by 16.
		//Even though there is sufficient space in the plane buffers and stride to avoid overflows, the resulting encoded file could have more 8x8 blocks
		//than non-AVX2 decoders expect, leading to a broken image.
		instance->avx2 = 0;
	}

	instance->AlignedHeight = dimensions.height;
	VMX_ALIGN(instance->AlignedHeight, 16);

	int planeLen = instance->Planes[0].Stride * instance->AlignedHeight * 2;
	instance->Planes[0].Data = (BYTE*)_mm_malloc(planeLen, VMX_ALIGNMENT);
	instance->Planes[1].Data = (BYTE*)_mm_malloc(planeLen, VMX_ALIGNMENT);
	instance->Planes[2].Data = (BYTE*)_mm_malloc(planeLen, VMX_ALIGNMENT);
	instance->Planes[3].Data = (BYTE*)_mm_malloc(planeLen, VMX_ALIGNMENT);

	memset(instance->Planes[0].Data, 0, planeLen);
	memset(instance->Planes[1].Data, 128, planeLen);
	memset(instance->Planes[2].Data, 128, planeLen);
	memset(instance->Planes[3].Data, 255, planeLen);

	instance->Planes[0].DataLowerPreview = instance->Planes[0].Data + ((instance->Planes[0].Stride) * (instance->AlignedHeight >> 4));
	instance->Planes[1].DataLowerPreview = instance->Planes[1].Data + ((instance->Planes[1].Stride) * (instance->AlignedHeight >> 4));
	instance->Planes[2].DataLowerPreview = instance->Planes[2].Data + ((instance->Planes[2].Stride) * (instance->AlignedHeight >> 4));
	instance->Planes[3].DataLowerPreview = instance->Planes[3].Data + ((instance->Planes[3].Stride) * (instance->AlignedHeight >> 4));

	instance->SliceCount = instance->AlignedHeight >> 4;
	instance->Slices = new VMX_SLICE_SET * [instance->SliceCount];

	int dcLen = instance->Planes[0].Stride * VMX_SLICE_HEIGHT * 2;
	int acLen = instance->Planes[0].Stride * VMX_SLICE_HEIGHT * 4;

	int offsets[4] = { 0,0,0,0 };
	int offsets16[4] = { 0,0,0,0 };
	for (int i = 0; i < instance->SliceCount; i++)
	{
		instance->Slices[i] = new VMX_SLICE_SET;
		instance->Slices[i]->DC.Stream = (BYTE*)_mm_malloc(dcLen, VMX_ALIGNMENT);
		instance->Slices[i]->AC.Stream = (BYTE*)_mm_malloc(acLen, VMX_ALIGNMENT);
		instance->Slices[i]->AC.StreamLength = 0;
		instance->Slices[i]->DC.StreamLength = 0;
		instance->Slices[i]->AC.MaxStreamLength = acLen;
		instance->Slices[i]->DC.MaxStreamLength = dcLen;
		memset(instance->Slices[i]->DC.Stream, 0xFF, dcLen);
		memset(instance->Slices[i]->AC.Stream, 0xFF, acLen);

		instance->Slices[i]->PixelSize = { dimensions.width, VMX_SLICE_HEIGHT };
		if (i == instance->SliceCount - 1) {
			instance->Slices[i]->PixelSize.height = VMX_SLICE_HEIGHT - (instance->AlignedHeight - dimensions.height);
		}

		instance->Slices[i]->PixelSizeInterlaced = { dimensions.width, VMX_SLICE_HEIGHT };
		if ((i == ((instance->SliceCount >> 1) - 1)) || (i == instance->SliceCount - 1)) {
			instance->Slices[i]->PixelSizeInterlaced.height = VMX_SLICE_HEIGHT - ((instance->AlignedHeight - dimensions.height) >> 1);
		}

		if (i >= (instance->SliceCount >> 1)) {
			instance->Slices[i]->LowerField = 1;
		}
		else {
			instance->Slices[i]->LowerField = 0;
		}

		instance->Slices[i]->Offset[0] = offsets[0];
		instance->Slices[i]->Offset[1] = offsets[1];
		instance->Slices[i]->Offset[2] = offsets[2];
		instance->Slices[i]->Offset[3] = offsets[3];

		instance->Slices[i]->Offset16[0] = offsets16[0];
		instance->Slices[i]->Offset16[1] = offsets16[1];
		instance->Slices[i]->Offset16[2] = offsets16[2];
		instance->Slices[i]->Offset16[3] = offsets16[3];

		offsets[0] += instance->Planes[0].Stride * VMX_SLICE_HEIGHT;
		offsets[1] += instance->Planes[1].Stride * VMX_SLICE_HEIGHT;
		offsets[2] += instance->Planes[2].Stride * VMX_SLICE_HEIGHT;
		offsets[3] += instance->Planes[3].Stride * VMX_SLICE_HEIGHT;

		offsets16[0] += instance->Planes[0].Stride * VMX_SLICE_HEIGHT * 2;
		offsets16[1] += instance->Planes[1].Stride * VMX_SLICE_HEIGHT * 2;
		offsets16[2] += instance->Planes[2].Stride * VMX_SLICE_HEIGHT * 2;
		offsets16[3] += instance->Planes[3].Stride * VMX_SLICE_HEIGHT * 2;

	}
	VMX_ResetStream(instance);
	VMX_SetQuality(instance, 80);
	return instance;
}

#define CHECKBUFF(checkNumBytes) \
{ \
	if ((b + checkNumBytes) > maxB) { return VMX_ERR_BUFFER_OVERFLOW; } \
} 

#define CHECKBUFF_SAVE(checkNumBytes) \
{ \
	if ((b + checkNumBytes) > maxB) return 0; \
} 

inline void VMX_ConfigureInterlaced(VMX_INSTANCE* instance, int interlaced) {
	instance->Format = VMX_FORMAT_PROGRESSIVE;
	if (interlaced) {
		int height = instance->Planes[0].Size.height;
		//Only these three resolutions will work with interlaced, as they need to be divisible by slice height * 2. 1080 is an exception as it is aligned to 1088
		if (height == 480 || height == 576 || height == 1080) {
			instance->Format = VMX_FORMAT_INTERLACED;
		}
	}
}

VMX_API VMX_ERR VMX_LoadFrom(VMX_INSTANCE* instance, BYTE* data, int dataLen)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (!data) return VMX_ERR_INVALID_PARAMETERS;
	if (!dataLen) return VMX_ERR_INVALID_PARAMETERS;

	BYTE* b = data;
	BYTE* maxB = data + dataLen;
	int offset = 0;
	int dcshift = 0;
	int format = 0;
	int sliceCount = 0;

	CHECKBUFF(5);

	if (b[0] == VMX_CODEC_FORMAT_PROGRESSIVE || b[0] == VMX_CODEC_FORMAT_INTERLACED || b[0] == VMX_CODEC_FORMAT_EXTENDED)
	{
		if (b[0] == VMX_CODEC_FORMAT_EXTENDED)
		{
			offset = 2;
			dcshift = b[1];
		}
		format = b[offset] - 1;
		sliceCount = b[offset + 2];
		if (sliceCount == 14 && instance->SliceCount == 270) sliceCount = 270; //Special case for 8K 4320p
		if (sliceCount == instance->SliceCount)
		{
			//Bypass min/max checking in setquality to ensure exact quality is used when decoding
			VMX_SetQualityInternal(instance, b[offset + 1]);
			b += (3 + offset);
			uint32_t len = 0;
			for (int i = 0; i < instance->SliceCount; i++)
			{
				VMX_SLICE_DATA d = instance->Slices[i]->DC;
				CHECKBUFF(4);
				len = *(uint32_t*)b;
				b += 4;
				CHECKBUFF(len);
				if (len > d.MaxStreamLength) return VMX_ERR_BUFFER_OVERFLOW;
				memcpy(d.Stream, b, len);
				b += len;
				instance->Slices[i]->DC.StreamLength = len;
			}
			//We may be loading a preview only image which only has DC
			if (b < maxB)
			{
				for (int i = 0; i < instance->SliceCount; i++)
				{
					VMX_SLICE_DATA d = instance->Slices[i]->AC;
					CHECKBUFF(4);
					len = *(uint32_t*)b;
					b += 4;
					CHECKBUFF(len);
					if (len > d.MaxStreamLength) return VMX_ERR_BUFFER_OVERFLOW;
					memcpy(d.Stream, b, len);
					b += len;
					instance->Slices[i]->AC.StreamLength = len;
				}
			}
			//instance->Format = (VMX_FORMAT)format;
			VMX_ConfigureInterlaced(instance, format);
			instance->DCShift = dcshift;
			return VMX_ERR_OK;
		}
		else {
			return VMX_ERR_INVALID_SLICE_COUNT;
		}
	}
	else {
		return VMX_ERR_INVALID_CODEC_FORMAT;
	}
	return VMX_ERR_OK;
}

VMX_API void VMX_DecodePlanePreview(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_DecodePlanePreviewInternal(instance, pPlane, s);
}

VMX_API void VMX_DecodePlane(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET * s)
{
	VMX_DecodePlaneInternal(instance, pPlane, s);
}

VMX_API void VMX_DecodePlane16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_DecodePlaneInternal16(instance, pPlane, s);
}

void VMX_DecodeSlices(VMX_INSTANCE* instance, int startIndex, int count)
{
	VMX_PLANE y = instance->Planes[0];
	VMX_PLANE u = instance->Planes[1];
	VMX_PLANE v = instance->Planes[2];
	VMX_PLANE a = instance->Planes[3];

	const short* colorTable;

	if (instance->Format == VMX_FORMAT_PROGRESSIVE) {
		int sliceStride = instance->ImageStride * VMX_SLICE_HEIGHT;
		int sliceStrideA;
		int sliceStrideU;

		switch (instance->ImageFormat) 
		{
		case VMX_IMAGE_P216:
			sliceStrideU = instance->ImageStrideU * VMX_SLICE_HEIGHT;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane16(instance, &y, s);
				VMX_DecodePlane16(instance, &u, s);
				VMX_DecodePlane16(instance, &v, s);
				VMX_PlanarToP216(y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2,
					instance->ImageData + (i * sliceStride), instance->ImageStride,
					instance->ImageDataU + (i * sliceStrideU), instance->ImageStrideU,
					s->PixelSize);
			}
			break;
		case VMX_IMAGE_PA16:
			sliceStrideU = instance->ImageStrideU * VMX_SLICE_HEIGHT;
			sliceStrideA = instance->ImageStrideA * VMX_SLICE_HEIGHT;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane16(instance, &y, s);
				VMX_DecodePlane16(instance, &u, s);
				VMX_DecodePlane16(instance, &v, s);
				VMX_DecodePlane16(instance, &a, s);
				VMX_PlanarToP216(y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2,
					instance->ImageData + (i * sliceStride), instance->ImageStride,
					instance->ImageDataU + (i * sliceStrideU), instance->ImageStrideU,
					s->PixelSize);
				VMX_PlanarToA16(a.Data + s->Offset16[3], a.Stride * 2,
					instance->ImageDataA + (i * sliceStrideA), instance->ImageStrideA, s->PixelSize);
			}
			break;
		case VMX_IMAGE_UYVY:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_PlanarToUYVY(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					instance->ImageData + (i * sliceStride), instance->ImageStride, s->PixelSize);
			}
			break;
		case VMX_IMAGE_UYVA:
			sliceStrideA = instance->ImageStrideA * VMX_SLICE_HEIGHT;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_DecodePlane(instance, &a, s);
				VMX_PlanarToUYVY(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					instance->ImageData + (i * sliceStride), instance->ImageStride, s->PixelSize);
				VMX_PlanarToA(a.Data + s->Offset[3],a.Stride,
					instance->ImageDataA + (i * sliceStrideA),instance->ImageStrideA,s->PixelSize);
			}
			break;
		case VMX_IMAGE_YUY2:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_PlanarToYUY2(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					instance->ImageData + (i * sliceStride), instance->ImageStride, s->PixelSize);
			}
			break;
		case VMX_IMAGE_BGRA:
			colorTable = YUV_RGB_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = YUV_RGB_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_DecodePlane(instance, &a, s);
				VMX_YUV4224ToBGRA(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride,
					instance->ImageData + (i * sliceStride), instance->ImageStride, s->PixelSize, colorTable);
			}
			break;
		case VMX_IMAGE_BGRX:
			colorTable = YUV_RGB_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = YUV_RGB_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_YUV4224ToBGRA(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride,
					instance->ImageData + (i * sliceStride), instance->ImageStride, s->PixelSize, colorTable);
			}
			break;
		default:
			break;
		}

	} else {

		int sourceStride = instance->ImageStride << 1;
		int sliceStride = sourceStride * VMX_SLICE_HEIGHT;
		int offset = 0;
		int interlacedOffset = -(instance->ImageStride * (instance->AlignedHeight - 1));

		int sliceStrideA;
		int offsetA;
		int sourceStrideA;
		int interlacedOffsetA;

		int sliceStrideU;
		int offsetU;
		int sourceStrideU;
		int interlacedOffsetU;

		switch (instance->ImageFormat)
		{
		case VMX_IMAGE_P216:
			sourceStrideU = instance->ImageStrideU << 1;
			sliceStrideU = sourceStrideU * VMX_SLICE_HEIGHT;
			offsetU = 0;
			interlacedOffsetU = -(instance->ImageStrideU * (instance->AlignedHeight - 1));
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) {
					offset = interlacedOffset;
					offsetU = interlacedOffsetU;
				}
				VMX_DecodePlane16(instance, &y, s);
				VMX_DecodePlane16(instance, &u, s);
				VMX_DecodePlane16(instance, &v, s);
				VMX_PlanarToP216(y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2,
					instance->ImageData + (i * sliceStride) + offset, sourceStride, 
					instance->ImageDataU + (i * sliceStrideU) + offsetU, sourceStrideU,
					s->PixelSizeInterlaced);
			}
			break;
		case VMX_IMAGE_PA16:
			sourceStrideU = instance->ImageStrideU << 1;
			sliceStrideU = sourceStrideU * VMX_SLICE_HEIGHT;
			offsetU = 0;
			interlacedOffsetU = -(instance->ImageStrideU * (instance->AlignedHeight - 1));

			offsetA = 0;
			sourceStrideA = instance->ImageStrideA << 1;
			sliceStrideA = sourceStrideA * VMX_SLICE_HEIGHT;
			interlacedOffsetA = -(instance->ImageStrideA * (instance->AlignedHeight - 1));

			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) {
					offset = interlacedOffset;
					offsetU = interlacedOffsetU;
					offsetA = interlacedOffsetA;
				}
				VMX_DecodePlane16(instance, &y, s);
				VMX_DecodePlane16(instance, &u, s);
				VMX_DecodePlane16(instance, &v, s);
				VMX_DecodePlane16(instance, &a, s);
				VMX_PlanarToP216(y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2,
					instance->ImageData + (i * sliceStride) + offset, sourceStride,
					instance->ImageDataU + (i * sliceStrideU) + offsetU, sourceStrideU,
					s->PixelSizeInterlaced);
				VMX_PlanarToA16(a.Data + s->Offset16[3], a.Stride * 2,
					instance->ImageDataA + (i * sliceStrideA) + offsetA, sourceStrideA, s->PixelSizeInterlaced);
			}
			break;
		case VMX_IMAGE_UYVY:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_PlanarToUYVY(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					instance->ImageData + (i * sliceStride) + offset, sourceStride, s->PixelSizeInterlaced);
			}
			break;
		case VMX_IMAGE_UYVA:
			offsetA = 0;
			sourceStrideA = instance->ImageStrideA << 1;
			sliceStrideA = sourceStrideA * VMX_SLICE_HEIGHT;
			interlacedOffsetA = -(instance->ImageStrideA * (instance->AlignedHeight - 1));
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) {
					offset = interlacedOffset; 
					offsetA = interlacedOffsetA;
				}
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_DecodePlane(instance, &a, s);
				VMX_PlanarToUYVY(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					instance->ImageData + (i * sliceStride) + offset, sourceStride, s->PixelSizeInterlaced);
				VMX_PlanarToA(a.Data + s->Offset[3], a.Stride,
					instance->ImageDataA + (i * sliceStrideA) + offsetA, sourceStrideA, s->PixelSizeInterlaced);
			}
			break;
		case VMX_IMAGE_YUY2:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_PlanarToYUY2(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					instance->ImageData + (i * sliceStride) + offset, sourceStride, s->PixelSizeInterlaced);
			}
			break;
		case VMX_IMAGE_BGRA:
			colorTable = YUV_RGB_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = YUV_RGB_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_DecodePlane(instance, &a, s);
				VMX_YUV4224ToBGRA(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride,
					instance->ImageData + (i * sliceStride) + offset, sourceStride, s->PixelSizeInterlaced, colorTable);
			}
			break;
		case VMX_IMAGE_BGRX:
			colorTable = YUV_RGB_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = YUV_RGB_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_DecodePlane(instance, &y, s);
				VMX_DecodePlane(instance, &u, s);
				VMX_DecodePlane(instance, &v, s);
				VMX_YUV4224ToBGRA(y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride,
					instance->ImageData + (i * sliceStride) + offset, sourceStride, s->PixelSizeInterlaced, colorTable);
			}
			break;
		default:
			break;
		}
	}
}

void VMX_DecodePlanesPreview(VMX_INSTANCE* instance, bool alpha)
{
	VMX_PLANE y = instance->Planes[0];
	VMX_PLANE u = instance->Planes[1];
	VMX_PLANE v = instance->Planes[2];
	VMX_PLANE a = instance->Planes[3];
	for (int i = 0; i < instance->SliceCount; i++)
	{
		VMX_SLICE_SET* s = instance->Slices[i];
		VMX_DecodePlanePreview(instance, &y, s);
		VMX_DecodePlanePreview(instance, &u, s);
		VMX_DecodePlanePreview(instance, &v, s);
		if (alpha) VMX_DecodePlanePreview(instance, &a, s);
	}
}

inline void VMX_DecodePlanes(VMX_INSTANCE* instance)
{
	int numThreads = instance->Threads;
	int totalSlices = instance->SliceCount;
	int slicesPerThread = totalSlices / numThreads;
	int count = slicesPerThread;
	for (int i = 0; i < numThreads; i++)
	{
		if (i == numThreads - 1) count = totalSlices - (i * slicesPerThread);
		instance->Tasks->tasks[i]->Push([instance, i, slicesPerThread, count] {VMX_DecodeSlices(instance, i * slicesPerThread, count); });
	}
	for (int i = 0; i < numThreads; i++)
	{
		instance->Tasks->tasks[i]->Join();
	}
}

VMX_API VMX_ERR VMX_DecodeP216(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_P216;
	instance->ImageDataU = dst + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideU = stride;
	VMX_DecodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodePA16(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_PA16;
	instance->ImageDataU = dst + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideU = stride;
	instance->ImageDataA = instance->ImageDataU + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideA = stride;
	VMX_DecodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodeUYVY(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_UYVY;
	VMX_DecodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodeUYVA(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_UYVA;
	instance->ImageDataA = dst + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideA = stride >> 1;
	VMX_DecodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodeYUY2(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_YUY2;
	VMX_DecodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodePreviewUYVY(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	VMX_SIZE previewSize = instance->PreviewSize;
	if (stride < (previewSize.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	VMX_DecodePlanesPreview(instance, false);
	if (instance->Format == VMX_FORMAT_INTERLACED)
	{
		previewSize = instance->PreviewSizeInterlaced;
		VMX_SIZE sz = { previewSize.width, previewSize.height >> 1 };
		VMX_PlanarToUYVY(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, dst, stride << 1, sz);
		dst += stride;
		VMX_PlanarToUYVY(instance->Planes[0].DataLowerPreview, instance->Planes[0].Stride, instance->Planes[1].DataLowerPreview, instance->Planes[1].Stride, instance->Planes[2].DataLowerPreview, instance->Planes[2].Stride, dst, stride << 1, sz);
	}
	else
	{
		VMX_PlanarToUYVY(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, dst, stride, previewSize);
	}
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodePreviewUYVA(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	VMX_SIZE previewSize = instance->PreviewSize;
	if (stride < (previewSize.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	VMX_DecodePlanesPreview(instance, true);
	BYTE* dstA = dst + (previewSize.height * stride);
	if (instance->Format == VMX_FORMAT_INTERLACED)
	{
		previewSize = instance->PreviewSizeInterlaced;
		VMX_SIZE sz = { previewSize.width, previewSize.height >> 1 };
		VMX_PlanarToUYVY(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, dst, stride << 1, sz);
		VMX_PlanarToA(instance->Planes[3].Data, instance->Planes[3].Stride, dstA, stride, sz);
		dst += stride;
		dstA += (stride >> 1);
		VMX_PlanarToUYVY(instance->Planes[0].DataLowerPreview, instance->Planes[0].Stride, instance->Planes[1].DataLowerPreview, instance->Planes[1].Stride, instance->Planes[2].DataLowerPreview, instance->Planes[2].Stride, dst, stride << 1, sz);
		VMX_PlanarToA(instance->Planes[3].DataLowerPreview, instance->Planes[3].Stride, dstA, stride, sz);
	}
	else
	{
		VMX_PlanarToUYVY(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, dst, stride, previewSize);
		VMX_PlanarToA(instance->Planes[3].Data, instance->Planes[3].Stride, dstA, stride >> 1, previewSize);
	}
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodePreviewYUY2(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	VMX_SIZE previewSize = instance->PreviewSize;
	if (stride < (previewSize.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	VMX_DecodePlanesPreview(instance, false);
	if (instance->Format == VMX_FORMAT_INTERLACED)
	{
		previewSize = instance->PreviewSizeInterlaced;
		VMX_SIZE sz = { previewSize.width, previewSize.height >> 1 };
		VMX_PlanarToYUY2(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, dst, stride << 1, sz);
		dst += stride;
		VMX_PlanarToYUY2(instance->Planes[0].DataLowerPreview, instance->Planes[0].Stride, instance->Planes[1].DataLowerPreview, instance->Planes[1].Stride, instance->Planes[2].DataLowerPreview, instance->Planes[2].Stride, dst, stride << 1, sz);
	}
	else
	{
		VMX_PlanarToYUY2(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, dst, stride, previewSize);
	}
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodePreviewBGRA(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	VMX_SIZE previewSize = instance->PreviewSize;
	if (stride < (previewSize.width * 4)) return VMX_ERR_INVALID_PARAMETERS;

	const short* colorTable = YUV_RGB_709;
	if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = YUV_RGB_601;

	VMX_ResetStream(instance);
	VMX_DecodePlanesPreview(instance, true);

	if (instance->Format == VMX_FORMAT_INTERLACED)
	{
		previewSize = instance->PreviewSizeInterlaced;
		VMX_SIZE sz = { previewSize.width, previewSize.height >> 1 };
		VMX_YUV4224ToBGRA(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, instance->Planes[3].Data, instance->Planes[3].Stride, dst, stride << 1, sz, colorTable);
		dst += stride;
		VMX_YUV4224ToBGRA(instance->Planes[0].DataLowerPreview, instance->Planes[0].Stride, instance->Planes[1].DataLowerPreview, instance->Planes[1].Stride, instance->Planes[2].DataLowerPreview, instance->Planes[2].Stride, instance->Planes[3].DataLowerPreview, instance->Planes[3].Stride, dst, stride << 1, sz, colorTable);
	}
	else
	{
		VMX_YUV4224ToBGRA(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, instance->Planes[3].Data, instance->Planes[3].Stride, dst, stride, previewSize, colorTable);
	}
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodePreviewBGRX(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	VMX_SIZE previewSize = instance->PreviewSize;
	if (stride < (previewSize.width * 4)) return VMX_ERR_INVALID_PARAMETERS;

	const short* colorTable = YUV_RGB_709;
	if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = YUV_RGB_601;

	VMX_ResetStream(instance);
	VMX_DecodePlanesPreview(instance, false);
	int planeLen = instance->Planes[0].Stride * previewSize.height;
	memset(instance->Planes[3].Data, 255, planeLen);

	if (instance->Format == VMX_FORMAT_INTERLACED)
	{
		previewSize = instance->PreviewSizeInterlaced;
		VMX_SIZE sz = { previewSize.width, previewSize.height >> 1 };
		VMX_YUV4224ToBGRA(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, instance->Planes[3].Data, instance->Planes[3].Stride, dst, stride << 1, sz, colorTable);
		dst += stride;
		VMX_YUV4224ToBGRA(instance->Planes[0].DataLowerPreview, instance->Planes[0].Stride, instance->Planes[1].DataLowerPreview, instance->Planes[1].Stride, instance->Planes[2].DataLowerPreview, instance->Planes[2].Stride, instance->Planes[3].DataLowerPreview, instance->Planes[3].Stride, dst, stride << 1, sz, colorTable);
	}
	else
	{
		VMX_YUV4224ToBGRA(instance->Planes[0].Data, instance->Planes[0].Stride, instance->Planes[1].Data, instance->Planes[1].Stride, instance->Planes[2].Data, instance->Planes[2].Stride, instance->Planes[3].Data, instance->Planes[3].Stride, dst, stride, previewSize, colorTable);
	}
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodeBGRA(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 4)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_BGRA;
	VMX_DecodePlanes(instance);	
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_DecodeBGRX(VMX_INSTANCE* instance, BYTE* dst, int stride)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 4)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = dst;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_BGRX;
	int planeLen = instance->Planes[0].Stride * instance->Planes[0].Size.height;
	memset(instance->Planes[3].Data, 255, planeLen);
	VMX_DecodePlanes(instance);
	return VMX_ERR_OK;
}

void VMX_AdjustBitrate(VMX_INSTANCE* instance, int prevFrameLength)
{
	if (!prevFrameLength) return;

	int targetMin = instance->TargetBytesPerFrameMin;
	int targetMax = instance->TargetBytesPerFrameMax;
	int qual = instance->Quality;
	int minQual = instance->MinQuality;

	if (targetMin && targetMax)
	{
		if (prevFrameLength < targetMin)
		{
			if (qual < minQual) { VMX_SetQuality(instance, minQual); }
			else if (qual < 76) { VMX_SetQuality(instance, qual + 4); }
			else if (qual < 92) { VMX_SetQuality(instance, qual + 2); }
			else if (qual < 99) { VMX_SetQuality(instance, qual + 1); }
		}
		else if (prevFrameLength > targetMax)
		{
			if (qual > 92) { VMX_SetQuality(instance, qual - 1); }
			else if (qual > minQual) { VMX_SetQuality(instance, qual - 2); }
			else { VMX_SetQuality(instance, minQual); }
		}
	}
}
VMX_API int VMX_GetEncodedPreviewLength(VMX_INSTANCE* instance)
{
	int len = 0;
	if (instance->DCShift > 0)
	{
		len += 5;
	}
	else 
	{
		len += 3;
	}
	for (int i = 0; i < instance->SliceCount; i++)
	{
		VMX_SLICE_DATA d = instance->Slices[i]->DC;
		len += d.StreamPos - d.Stream;
		len += 4;
	}
	return len;
}
VMX_API void VMX_BGRXToUYVY(BYTE* pSrc, int srcStride, BYTE* pDst, int dstStride, VMX_SIZE size)
{
	const ShortRGB* colorTable = RGB_YUV_709;
	if (size.height < 720) colorTable = RGB_YUV_601;
	VMX_BGRXToUYVYInternal(pSrc, srcStride, pDst, dstStride, size, colorTable);	
}
VMX_API int VMX_BGRXToUYVYConditional(BYTE* pSrc, BYTE* pSrcPrev, int srcStride, BYTE* pDst, int dstStride, VMX_SIZE size)
{
	const ShortRGB* colorTable = RGB_YUV_709;
	if (size.height < 720) colorTable = RGB_YUV_601;
	return VMX_BGRXToUYVYConditionalInternal(pSrc, pSrcPrev, srcStride, pDst, dstStride, size, colorTable);
}
VMX_API int VMX_GetThreads(VMX_INSTANCE* instance)
{
	if (!instance) return 0;
	return instance->Threads;
}
VMX_API void VMX_SetThreads(VMX_INSTANCE* instance, int numThreads)
{
	if (!instance) return;
	if (numThreads > 0) {
		if (numThreads != instance->Threads) {
			DestroyTasks(instance->Tasks);
			instance->Threads = numThreads;
			instance->Tasks = CreateTasks(numThreads);
		}
	}
}
VMX_API int VMX_SaveTo(VMX_INSTANCE* instance, BYTE* dst, int maxLen)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (!dst) return VMX_ERR_INVALID_PARAMETERS;
	if (!maxLen) return VMX_ERR_INVALID_PARAMETERS;

	BYTE* b = dst;
	BYTE* maxB = dst + maxLen;

	int dcshift = instance->DCShift;
	int slicecount = instance->SliceCount;
	int qual = instance->Quality;
	VMX_FORMAT fmt = instance->Format;

	CHECKBUFF_SAVE(5);

	if (dcshift > 0)
	{
		b[0] = VMX_CODEC_FORMAT_EXTENDED;
		b[1] = dcshift;

		b[2] = VMX_CODEC_FORMAT_PROGRESSIVE;
		if (fmt == VMX_FORMAT_INTERLACED) b[2] = VMX_CODEC_FORMAT_INTERLACED;
		b[3] = qual;
		b[4] = slicecount;
		b += 5;
	}
	else {
		b[0] = VMX_CODEC_FORMAT_PROGRESSIVE;
		if (fmt == VMX_FORMAT_INTERLACED) b[0] = VMX_CODEC_FORMAT_INTERLACED;
		b[1] = qual;
		b[2] = slicecount;
		b += 3;
	}
	int len = 0;
	for (int i = 0; i < slicecount; i++)
	{
		VMX_SLICE_DATA d = instance->Slices[i]->DC;
		len = d.StreamPos - d.Stream;
		CHECKBUFF_SAVE(len + 4);
		*(uint32_t*)b = len;
		b += 4;
		memcpy(b, d.Stream, len);
		b += len;
	}
	for (int i = 0; i < slicecount; i++)
	{
		VMX_SLICE_DATA d = instance->Slices[i]->AC;
		len = d.StreamPos - d.Stream;
		CHECKBUFF_SAVE(len + 4);
		*(uint32_t*)b = len;
		b += 4;
		memcpy(b, d.Stream, len);
		b += len;
	}
	len = b - dst;
	VMX_AdjustBitrate(instance, len);
	return len;
}

inline void VMX_EncodePlane(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_EncodePlaneInternal(instance, pPlane, s);
}

inline void VMX_EncodePlane16(VMX_INSTANCE* instance, VMX_PLANE* pPlane, VMX_SLICE_SET* s)
{
	VMX_EncodePlaneInternal16(instance, pPlane, s);
}

void VMX_EncodeSlices(VMX_INSTANCE* instance, int startIndex, int count)
{
	VMX_PLANE y = instance->Planes[0];
	VMX_PLANE u = instance->Planes[1];
	VMX_PLANE v = instance->Planes[2];
	VMX_PLANE a = instance->Planes[3];

	const ShortRGB* colorTable;

	if (instance->Format == VMX_FORMAT_PROGRESSIVE) {

		int sliceStride = instance->ImageStride * VMX_SLICE_HEIGHT;
		int sliceStrideU;
		int sliceStrideV;
		int sliceStrideA;
		switch (instance->ImageFormat)
		{
		case VMX_IMAGE_P216:
			sliceStrideU = (instance->ImageStrideU * VMX_SLICE_HEIGHT);
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_P216ToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					instance->ImageDataU + (i * sliceStrideU), instance->ImageStrideU,
					y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2, s->PixelSize);
				VMX_EncodePlane16(instance, &y, s);
				VMX_EncodePlane16(instance, &u, s);
				VMX_EncodePlane16(instance, &v, s);
			}
			break;
		case VMX_IMAGE_PA16:
			sliceStrideU = (instance->ImageStrideU * VMX_SLICE_HEIGHT);
			sliceStrideA = instance->ImageStrideA * VMX_SLICE_HEIGHT;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_P216ToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					instance->ImageDataU + (i * sliceStrideU), instance->ImageStrideU,
					y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2, s->PixelSize);
				VMX_A16ToPlanar(instance->ImageDataA + (i * sliceStrideA), instance->ImageStrideA,
					a.Data + s->Offset16[3], a.Stride * 2, s->PixelSize);
				VMX_EncodePlane16(instance, &y, s);
				VMX_EncodePlane16(instance, &u, s);
				VMX_EncodePlane16(instance, &v, s);
				VMX_EncodePlane16(instance, &a, s);
			}
			break;
		case VMX_IMAGE_UYVY:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_UYVYToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSize);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_UYVA:
			sliceStrideA = instance->ImageStrideA * VMX_SLICE_HEIGHT;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_UYVYToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSize);
				VMX_AToPlanar(instance->ImageDataA + (i * sliceStrideA), instance->ImageStrideA,
					a.Data + s->Offset[3], a.Stride, s->PixelSize);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
				VMX_EncodePlane(instance, &a, s);
			}
			break;
		case VMX_IMAGE_YUY2:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_YUY2ToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSize);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_NV12:
			sliceStrideU = (instance->ImageStrideU * VMX_SLICE_HEIGHT) >> 1;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_NV12ToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					instance->ImageDataU + (i * sliceStrideU), instance->ImageStrideU,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSize);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_YV12:
			sliceStrideU = (instance->ImageStrideU * VMX_SLICE_HEIGHT) >> 1;
			sliceStrideV = (instance->ImageStrideV * VMX_SLICE_HEIGHT) >> 1;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_YV12ToPlanar(instance->ImageData + (i * sliceStride), instance->ImageStride,
					instance->ImageDataU + (i * sliceStrideU), instance->ImageStrideU,
					instance->ImageDataV + (i * sliceStrideV), instance->ImageStrideV,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSize);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_YUVPLANAR422:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_BGRA:
			colorTable = RGB_YUV_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = RGB_YUV_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_BGRAToYUV4224(instance->ImageData + (i * sliceStride), instance->ImageStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, 
					a.Data + s->Offset[3], a.Stride,
					s->PixelSize, colorTable);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
				VMX_EncodePlane(instance, &a, s);
			}
			break;
		case VMX_IMAGE_BGRX:
			colorTable = RGB_YUV_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = RGB_YUV_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				VMX_BGRAToYUV4224(instance->ImageData + (i * sliceStride), instance->ImageStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride,
					s->PixelSize, colorTable);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		default:
			break;
		}
	}
	else {
		int sourceStride = instance->ImageStride << 1;
		int sliceStride = sourceStride * VMX_SLICE_HEIGHT;
		int offset = 0;
		int interlacedOffset = -(instance->ImageStride * (instance->AlignedHeight - 1));

		int sourceStrideU;
		int sliceStrideU;
		int offsetU;
		int interlacedOffsetU;

		int sourceStrideV;
		int sliceStrideV;
		int offsetV;
		int interlacedOffsetV;

		int sourceStrideA;
		int sliceStrideA;
		int offsetA;
		int interlacedOffsetA;

		switch (instance->ImageFormat)
		{
		case VMX_IMAGE_P216:
			sourceStrideU = instance->ImageStrideU << 1;
			sliceStrideU = (sourceStrideU * VMX_SLICE_HEIGHT);
			offsetU = 0;
			interlacedOffsetU = -(instance->ImageStrideU * (instance->AlignedHeight - 1));
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) { offset = interlacedOffset; offsetU = interlacedOffsetU; }
				VMX_P216ToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					instance->ImageDataU + (i * sliceStrideU) + offsetU, sourceStrideU,
					y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2, s->PixelSizeInterlaced);
				VMX_EncodePlane16(instance, &y, s);
				VMX_EncodePlane16(instance, &u, s);
				VMX_EncodePlane16(instance, &v, s);
			}
			break;
		case VMX_IMAGE_PA16:
			sourceStrideU = instance->ImageStrideU << 1;
			sliceStrideU = (sourceStrideU * VMX_SLICE_HEIGHT);
			offsetU = 0;
			interlacedOffsetU = -(instance->ImageStrideU * (instance->AlignedHeight - 1));

			sourceStrideA = instance->ImageStrideA << 1;
			sliceStrideA = sourceStrideA * VMX_SLICE_HEIGHT;
			offsetA = 0;
			interlacedOffsetA = -(instance->ImageStrideA * (instance->AlignedHeight - 1));

			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) { offset = interlacedOffset; offsetU = interlacedOffsetU; offsetA = interlacedOffsetA; }
				VMX_P216ToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					instance->ImageDataU + (i * sliceStrideU) + offsetU, sourceStrideU,
					y.Data + s->Offset16[0], y.Stride * 2,
					u.Data + s->Offset16[1], u.Stride * 2,
					v.Data + s->Offset16[2], v.Stride * 2, s->PixelSizeInterlaced);
				VMX_A16ToPlanar(instance->ImageDataA + (i * sliceStrideA) + offsetA, sourceStrideA,
					a.Data + s->Offset16[3], a.Stride * 2, s->PixelSizeInterlaced);
				VMX_EncodePlane16(instance, &y, s);
				VMX_EncodePlane16(instance, &u, s);
				VMX_EncodePlane16(instance, &v, s);
				VMX_EncodePlane16(instance, &a, s);
			}
			break;
		case VMX_IMAGE_UYVY:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_UYVYToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSizeInterlaced);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_UYVA:
			sourceStrideA = instance->ImageStrideA << 1;
			sliceStrideA = sourceStrideA * VMX_SLICE_HEIGHT;
			offsetA = 0;
			interlacedOffsetA = -(instance->ImageStrideA * (instance->AlignedHeight - 1));
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) {
					offset = interlacedOffset;
					offsetA = interlacedOffsetA;
				}
				VMX_UYVYToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSizeInterlaced);
				VMX_AToPlanar(instance->ImageDataA + (i * sliceStrideA) + offsetA, sourceStrideA,
					a.Data + s->Offset[3], a.Stride, s->PixelSizeInterlaced);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
				VMX_EncodePlane(instance, &a, s);
			}
			break;
		case VMX_IMAGE_YUY2:
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_YUY2ToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSizeInterlaced);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_NV12:
			sourceStrideU = instance->ImageStrideU << 1;
			sliceStrideU = (instance->ImageStrideU * VMX_SLICE_HEIGHT);
			offsetU = 0;
			interlacedOffsetU = -(instance->ImageStrideU * ((instance->AlignedHeight >> 1) - 1));
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) { offset = interlacedOffset; offsetU = interlacedOffsetU;  }
				VMX_NV12ToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					instance->ImageDataU + (i * sliceStrideU) + offsetU, sourceStrideU,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSizeInterlaced);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_YV12:
			sourceStrideU = instance->ImageStrideU << 1;
			sliceStrideU = (instance->ImageStrideU * VMX_SLICE_HEIGHT);
			offsetU = 0;
			interlacedOffsetU = -(instance->ImageStrideU * ((instance->AlignedHeight >> 1) - 1));

			sourceStrideV = instance->ImageStrideV << 1;
			sliceStrideV = (instance->ImageStrideV * VMX_SLICE_HEIGHT);
			offsetV = 0;
			interlacedOffsetV = -(instance->ImageStrideV * ((instance->AlignedHeight >> 1) - 1));

			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) { offset = interlacedOffset; offsetU = interlacedOffsetU; offsetV = interlacedOffsetV; }
				VMX_YV12ToPlanar(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					instance->ImageDataU + (i * sliceStrideU) + offsetU, sourceStrideU,
					instance->ImageDataV + (i * sliceStrideV) + offsetV, sourceStrideV,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride, s->PixelSizeInterlaced);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		case VMX_IMAGE_BGRA:
			colorTable = RGB_YUV_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = RGB_YUV_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_BGRAToYUV4224(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride, 
					s->PixelSizeInterlaced, colorTable);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
				VMX_EncodePlane(instance, &a, s);
			}
			break;
		case VMX_IMAGE_BGRX:
			colorTable = RGB_YUV_709;
			if (instance->ColorSpace == VMX_COLORSPACE_BT601) colorTable = RGB_YUV_601;
			for (int i = startIndex; i < (startIndex + count); i++)
			{
				VMX_SLICE_SET* s = instance->Slices[i];
				if (s->LowerField) offset = interlacedOffset;
				VMX_BGRAToYUV4224(instance->ImageData + (i * sliceStride) + offset, sourceStride,
					y.Data + s->Offset[0], y.Stride,
					u.Data + s->Offset[1], u.Stride,
					v.Data + s->Offset[2], v.Stride,
					a.Data + s->Offset[3], a.Stride,
					s->PixelSizeInterlaced, colorTable);
				VMX_EncodePlane(instance, &y, s);
				VMX_EncodePlane(instance, &u, s);
				VMX_EncodePlane(instance, &v, s);
			}
			break;
		default:
			break;
		}
	}

}
inline void VMX_EncodePlanes(VMX_INSTANCE* instance)
{
	int numThreads = instance->Threads;
	int totalSlices = instance->SliceCount;
	int slicesPerThread = totalSlices / numThreads;
	int count = slicesPerThread;
	for (int i = 0; i < numThreads; i++)
	{
		if (i == numThreads - 1) count = totalSlices - (i * slicesPerThread);
		instance->Tasks->tasks[i]->Push([instance, i, slicesPerThread, count] { VMX_EncodeSlices(instance, i * slicesPerThread, count); });
	}
	for (int i = 0; i < numThreads; i++)
	{
		instance->Tasks->tasks[i]->Join();
	}
}

VMX_API VMX_ERR VMX_EncodeP216(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = src;
	instance->ImageStride = stride;
	instance->ImageDataU = src + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideU = stride;
	instance->ImageFormat = VMX_IMAGE_P216;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}
VMX_API VMX_ERR VMX_EncodePA16(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = src;
	instance->ImageStride = stride;
	instance->ImageDataU = src + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideU = stride;
	instance->ImageDataA = instance->ImageDataU + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideA = stride;
	instance->ImageFormat = VMX_IMAGE_PA16;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodeUYVY(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = src;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_UYVY;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}
VMX_API VMX_ERR VMX_EncodeUYVA(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = src;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_UYVA;
	instance->ImageDataA = src + (stride * instance->Planes[0].Size.height);
	instance->ImageStrideA = stride >> 1;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodeYUY2(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 2)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = src;
	instance->ImageStride = stride;
	instance->ImageFormat = VMX_IMAGE_YUY2;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodeNV12(VMX_INSTANCE* instance, BYTE* srcY, int srcStrideY, BYTE* srcUV, int srcStrideUV, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (srcStrideY < (instance->Planes[0].Size.width)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = srcY;
	instance->ImageStride = srcStrideY;
	instance->ImageDataU = srcUV;
	instance->ImageStrideU = srcStrideUV;
	instance->ImageFormat = VMX_IMAGE_NV12;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodeYV12(VMX_INSTANCE* instance, BYTE* srcY, int srcStrideY, BYTE* srcU, int srcStrideU, BYTE* srcV, int srcStrideV, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (srcStrideY < (instance->Planes[0].Size.width)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	instance->ImageData = srcY;
	instance->ImageStride = srcStrideY;
	instance->ImageDataU = srcU;
	instance->ImageStrideU = srcStrideU;
	instance->ImageDataV = srcV;
	instance->ImageStrideV = srcStrideV;
	instance->ImageFormat = VMX_IMAGE_YV12;
	VMX_ConfigureInterlaced(instance, interlaced);
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodePlanar(VMX_INSTANCE* instance, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	VMX_ResetStream(instance);
	VMX_ConfigureInterlaced(instance, interlaced);
	instance->ImageFormat = VMX_IMAGE_YUVPLANAR422;
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodeBGRA(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 4)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	VMX_ConfigureInterlaced(instance, interlaced);
	instance->ImageFormat = VMX_IMAGE_BGRA;
	instance->ImageData = src;
	instance->ImageStride = stride;
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API VMX_ERR VMX_EncodeBGRX(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced)
{
	if (!instance) return VMX_ERR_INVALID_INSTANCE;
	if (stride < (instance->Planes[0].Size.width * 4)) return VMX_ERR_INVALID_PARAMETERS;
	VMX_ResetStream(instance);
	VMX_ConfigureInterlaced(instance, interlaced);
	instance->ImageFormat = VMX_IMAGE_BGRX;
	instance->ImageData = src;
	instance->ImageStride = stride;
	VMX_EncodePlanes(instance);
	return VMX_ERR_OK;
}

VMX_API float VMX_CalculatePSNR(BYTE* p1, BYTE* p2, int stride, int bytesPerPixel, VMX_SIZE sz)
{
	return VMX_CalculatePSNR_128(p1, p2, stride, bytesPerPixel, sz);
}

VMX_API int VMX_Test(VMX_INSTANCE* instance, short * src, short * dst)
{
	return 0;
}
