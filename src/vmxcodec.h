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
#include <cstdint>
#include <cstring>
#include "thread_tasks.h"
#ifdef _MSC_VER
#define VMX_API __declspec(dllexport)
#else
#define VMX_API extern "C" __attribute__((visibility("default")))
#endif
#if defined(_M_ARM64) | defined(__arm64) | defined(__aarch64__)
#define ARM64
#endif
#if defined(_M_X64) | defined(__x86_64__)
#define X64
#define AVX2 //Include AVX2 in all Intel builds as these functions will be dynamically called only if instruction set available.
#endif

const int VMX_SLICE_HEIGHT = 16;
const int VMX_QUALITY_COUNT = 25;
const int VMX_MAX_PLANES = 4;

typedef unsigned long long buffer_t;
typedef unsigned char       BYTE;

typedef struct {
	int width;
	int height;
} VMX_SIZE;

typedef enum {
	VMX_FORMAT_PROGRESSIVE,
	VMX_FORMAT_INTERLACED
} VMX_FORMAT;

typedef enum {
	VMX_ERR_OK,
	VMX_ERR_UNKNOWN,
	VMX_ERR_INVALID_CODEC_FORMAT,
	VMX_ERR_INVALID_SLICE_COUNT,
	VMX_ERR_BUFFER_OVERFLOW,
	VMX_ERR_INVALID_INSTANCE,
	VMX_ERR_INVALID_PARAMETERS
} VMX_ERR;

typedef enum {
	VMX_PROFILE_DEFAULT = 0,
	VMX_PROFILE_LQ = 33,
	VMX_PROFILE_SQ = 66,
	VMX_PROFILE_HQ = 99,
	VMX_PROFILE_OMT_LQ = 133,
	VMX_PROFILE_OMT_SQ = 166,
	VMX_PROFILE_OMT_HQ = 199
} VMX_PROFILE;

/**
* Defines currently supported color spaces used when converting between RGB and YUV
* Undefined = BT601 for SD (height < 720) and BT709 for HD (height >= 720)
*/
typedef enum {
	VMX_COLORSPACE_UNDEFINED = 0,
	VMX_COLORSPACE_BT601 = 601,
	VMX_COLORSPACE_BT709 = 709
} VMX_COLORSPACE;

typedef enum {
	VMX_CODEC_FORMAT_NONE,
	VMX_CODEC_FORMAT_PROGRESSIVE,
	VMX_CODEC_FORMAT_INTERLACED,
	VMX_CODEC_FORMAT_EXTENDED
} VMX_CODEC_FORMAT;

typedef enum {
	VMX_IMAGE_UYVY,
	VMX_IMAGE_YUY2,
	VMX_IMAGE_NV12,
	VMX_IMAGE_YV12,
	VMX_IMAGE_YUVPLANAR422,
	VMX_IMAGE_BGRA,
	VMX_IMAGE_BGRX,
	VMX_IMAGE_UYVA,
	VMX_IMAGE_P216,
	VMX_IMAGE_PA16,
} VMX_IMAGE_FORMAT;

struct VMX_SLICE_DATA
{
	unsigned char* Stream;
	unsigned char* StreamPos;
	int MaxStreamLength;
	int StreamLength;
	//buffer_t BitsLeft;
	int BitsLeft; //Bad data may lead to wraparound if unsigned. Going negative is preferred
	buffer_t Temp;
	buffer_t TempRead;
	VMX_SIZE Size[3];
};

struct VMX_SLICE_SET
{
	VMX_SLICE_DATA DC;
	VMX_SLICE_DATA AC;
	int Offset[VMX_MAX_PLANES];
	int Offset16[VMX_MAX_PLANES]; //Offset for working with 16bit data
	VMX_SIZE PixelSize; //Size in pixels of the slice, for copying from source image data. Due to alignment last slice will be only 8 pixels height for 1080 for example.
	VMX_SIZE PixelSizeInterlaced; //Same as above but for second field alignment adjustment at the mid point.
	int LowerField; //=1 when this is a lower field slice
	__declspec(align(64)) short TempBlock[128];
	__declspec(align(64)) short TempBlock2[128];
	__declspec(align(64)) short TempBlock3[128];
};

struct VMX_PLANE
{
	int Index;
	VMX_SIZE Size;
	int Stride; //Double this when dealing with 16bit data
	BYTE* Data;
	BYTE* DataLowerPreview;
};

struct VMX_INSTANCE
{
	int avx2;
	VMX_FORMAT Format;
	VMX_PROFILE Profile;
	VMX_COLORSPACE ColorSpace;

	int Quality;
	int MinQuality;
	int DCShift;

	unsigned short* DecodeQualityPresets[VMX_QUALITY_COUNT];
	unsigned short* EncodeQualityPresets[VMX_QUALITY_COUNT];
	unsigned short* DecodeMatrix;
	unsigned short* EncodeMatrix;

	unsigned short* DecodeQualityPresets256[VMX_QUALITY_COUNT];
	unsigned short* EncodeQualityPresets256[VMX_QUALITY_COUNT];
	unsigned short* DecodeMatrix256;
	unsigned short* EncodeMatrix256;

	VMX_PLANE Planes[VMX_MAX_PLANES];
	int SliceCount;
	int AlignedHeight;
	VMX_SLICE_SET** Slices;
	int TargetBytesPerFrameMin;
	int TargetBytesPerFrameMax;
	int Threads;
	ThreadTasks * Tasks;
	VMX_SIZE PreviewSize;
	VMX_SIZE PreviewSizeInterlaced;

	BYTE* ImageData;
	int ImageStride;

	BYTE* ImageDataU;
	int ImageStrideU;

	BYTE* ImageDataV;
	int ImageStrideV;

	BYTE* ImageDataA;
	int ImageStrideA;

	VMX_IMAGE_FORMAT ImageFormat;
};

const int VMX_DECODE_MATRIX_COUNT = 64;
const int VMX_ENCODE_MATRIX_COUNT = 192;

//Public exports

/**
* Create a new instance of the codec.
* @param[in] dimensions The pixel size of the image. Supports a minimum dimension of 16x16 and a maximum of 7680x4320 (8K).
* @param[in] profile Select a profile to use when encoding. 
* @param[in] colorSpace Specify the color space to use when converting between RGB and YUV. Default is BT709.
*/
VMX_API VMX_INSTANCE* VMX_Create(VMX_SIZE dimensions, VMX_PROFILE profile, VMX_COLORSPACE colorSpace);

/**
* Destroy and free up memory of instance created with VMX_Create
*/
VMX_API void VMX_Destroy(VMX_INSTANCE* instance);

/**
* Set the quality to use for the next frame.
* 
* If set just before encoding a frame, will override the bitrate targeting.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] q The quality value in the range of 0 (lowest quality) to 100 (highest quality).
*/
VMX_API void VMX_SetQuality(VMX_INSTANCE* instance, int q);

/**
* Retrieve the current quality setting used by the encoder
* 
* This is adjusted automatically on the fly to meet the bitrate requirements and will change from frame to frame
* 
* @param[in] instance The instance created using VMX_Create
* @return Returns the quality value
*/
VMX_API int VMX_GetQuality(VMX_INSTANCE* instance);

/**
* Get current internal encoding parameters.
* @param[in] instance The instance created using VMX_Create
* @param[out] frameMin If an encoded frame falls below this value, quality will be increased for next frame
* @param[out] frameMax If an encoded frame is over this value, quality will be decreased for the next frame
* @param[out] minQuality The quality will not fall below this value regardless of resulting bitrate
* @param[out] dcShift Determines the DC precision where 0= 11bit, 1 = 10bit, 2= 9bit, 3= 8bit
*/
VMX_API void VMX_GetEncodingParameters(VMX_INSTANCE* instance, int* frameMin, int* frameMax, int* minQuality, int* dcShift);

/**
* Set internal encoding parameters to apply from the next frame.
* 
* This can be used to fine tune encoding or set custom bitrate targets.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] frameMin If an encoded frame falls below this value, quality will be increased for next frame
* @param[in] frameMax If an encoded frame is over this value, quality will be decreased for the next frame
* @param[in] minQuality The quality will not fall below this value regardless of resulting bitrate
* @param[in] dcShift Determines the DC precision where 0= 11bit, 1 = 10bit, 2= 9bit, 3= 8bit
*/
VMX_API void VMX_SetEncodingParameters(VMX_INSTANCE* instance, int frameMin, int frameMax, int minQuality, int dcShift);

/**
* Decode frame into BGRA buffer. BGRA is the same as ARGB32 and A8R8G8B8 on Windows
* 
* This should only be used where there is an alpha channel. Otherwise use BGRX.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodeBGRA(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decode frame into BGRX buffer. BGRX is the same as RGB32 and X8R8G8B8 on Windows
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodeBGRX(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decode frame into a P216 4:2:2 buffer. 
* 
* This is a 16bit Y plane followed by an interlaved 16bit UV plane.
* 
* Only the most significant 10bits is valid, so the 6 least signifiant bits will be 0. Shifting the values to the right by 6 is sufficient to convert back to 10bit.
*
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodeP216(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decode frame into a PA16 4:2:2:4 buffer.
*
* This is a 16bit Y plane followed by an interleaved 16bit UV plane followed by a 16bit alpha plane.
*
* Only the most significant 10bits is valid, so the 6 least signifiant bits will be 0. Shifting the values to the right by 6 is sufficient to convert back to 10bit.
*
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodePA16(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decode frame into a UYVY buffer. This is uyvy422 in FFmpeg
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodeUYVY(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decode frame into a UYVA buffer. This is uyvy422 in FFmpeg, followed by an alpha plane with half the stride.
*
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodeUYVA(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decode frame into a YUY2 buffer. This is yuyv422 in FFmpeg
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer to write the decoded frame to
* @param[in] stride The stride of the destination buffer in bytes
*/
VMX_API VMX_ERR VMX_DecodeYUY2(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Decodes a special 1/8th preview of the compressed image
* 
* This is calculated by dividing the width and height of the instance by 8 and then making sure the width is at least divisible by 2
* 
* Interlaced frames are a special case where the height is subtracted by 1 if not divisible by 2
* 
* Example 1920x1080 becomes 240x135 for progressive, 240x134 for interlaced.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer
* @param[in] stride The stride in bytes of each row of pixels
*/
VMX_API VMX_ERR VMX_DecodePreviewBGRA(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Same as VMX_DecodePreviewBGRA except without alpha channel (alpha = 255)
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer
* @param[in] stride The stride in bytes of each row of pixels
*/
VMX_API VMX_ERR VMX_DecodePreviewBGRX(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Same as VMX_DecodePreviewBGRA except for UYVY output
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer
* @param[in] stride The stride in bytes of each row of pixels
*/
VMX_API VMX_ERR VMX_DecodePreviewUYVY(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Same as VMX_DecodePreviewBGRA except for UYVA output
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer
* @param[in] stride The stride in bytes of each row of pixels
*/
VMX_API VMX_ERR VMX_DecodePreviewUYVA(VMX_INSTANCE* instance, BYTE* dst, int stride);

/**
* Same as VMX_DecodePreviewBGRA except for YUY2 output
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The destination buffer
* @param[in] stride The stride in bytes of each row of pixels
*/
VMX_API VMX_ERR VMX_DecodePreviewYUY2(VMX_INSTANCE* instance, BYTE* dst, int stride);


/**
* Encode a BGRA image. This is the same as the ARGB32 in DirectShow or A8R8G8B8 in Direct3D
* 
* The Alpha channel is included in the encoded image. If you do not require alpha use the BGRX functions instead.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeBGRA(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);

/**
* Encode a BGRX image. This is the same as the RGB32 in DirectShow or X8R8G8B8 in Direct3D
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeBGRX(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);

/**
* Encode a UYVY image.
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeUYVY(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);

/**
* Encode a UYVA image.
* 
* This is a UYVY image followed immediately by an alpha plane with each line consisting of half the stride.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeUYVA(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);

/**
* Encode a YUY2 image.
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeYUY2(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);

/**
* Encode a NV12 image.
* @param[in] instance The instance created using VMX_Create
* @param[in] srcY The Y plane source pixels
* @param[in] srcStrideY The stride of the Y plane in bytes
* @param[in] srcUV The UV plane source pixels
* @param[in] srcStrideUV The stride of the UV plane in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeNV12(VMX_INSTANCE* instance, BYTE* srcY, int srcStrideY, BYTE* srcUV, int srcStrideUV, int interlaced);

/**
* Encode a YV12 or I420 image.
* @param[in] instance The instance created using VMX_Create
* @param[in] srcY The Y plane source pixels
* @param[in] srcStrideY The stride of the Y plane in bytes
* @param[in] srcU The U plane source pixels
* @param[in] srcStrideU The stride of the U plane in bytes
* @param[in] srcV The V plane source pixels
* @param[in] srcStrideV The stride of the V plane in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeYV12(VMX_INSTANCE* instance, BYTE* srcY, int srcStrideY, BYTE* srcU, int srcStrideU, BYTE* srcV, int srcStrideV, int interlaced);

/**
* Encode a 4:2:2 P216 image.
* 
* This is a 16bit Y plane followed by an interleaved 16bit UV plane.
* 
* Only the most significant 10bits is valid, so the 6 least signifiant bits must be 0. Shifting 10-bit values to the left by 6 is sufficient.
* 
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodeP216(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);

/**
* Encode a 4:2:2:4 PA16 image.
*
* This is a 16bit Y plane followed by an interleaved 16bit UV plane followed by a 16bit alpha plane.
*
* Only the most significant 10bits is valid, so the 6 least signifiant bits must be 0. Shifting 10-bit values to the left by 6 is sufficient.
*
* @param[in] instance The instance created using VMX_Create
* @param[in] src The source pixels
* @param[in] stride The stride of the source pixels in bytes
* @param[in] interlaced 1 if interlaced, 0 if progressive
*/
VMX_API VMX_ERR VMX_EncodePA16(VMX_INSTANCE* instance, BYTE* src, int stride, int interlaced);


VMX_API VMX_ERR VMX_EncodePlanar(VMX_INSTANCE* instance, int interlaced);

/**
* Load a compressed frame from a buffer.
* @param[in] instance The instance created using VMX_Create
* @param[in] data The compressed frame data
* @param[in] dataLen The length of the compressed frame data in bytes. 
*/
VMX_API VMX_ERR VMX_LoadFrom(VMX_INSTANCE* instance, BYTE* data, int dataLen);

/**
* Save a compressed frame to a buffer.
* @param[in] instance The instance created using VMX_Create
* @param[in] dst The buffer to write to
* @param[in] maxLen The maximum amount of data that can be written to the buffer. A buffer size of at least width*height*4 is recommended
*/
VMX_API int VMX_SaveTo(VMX_INSTANCE* instance, BYTE* dst, int maxLen);

/**
* Returns the portion of the compressed frame that is needed to decode a preview. 
*/
VMX_API int VMX_GetEncodedPreviewLength(VMX_INSTANCE* instance);

/**
* Helper function to convert a BGRX image to UYVY. The destination buffer must be 64byte aligned.
* @param[in] pSrc The source pixels in BGRA format. Alpha is ignored.
* @param[in] srcStride The stride in bytes of each row of source pixels.
* @param[in] pDst The destination buffer. This must be 64byte aligned.
* @param[in] dstStride The stride in bytes of each row of destination pixels.
* @param[in] size The dimensions of the pixels.
*/
VMX_API void VMX_BGRXToUYVY(BYTE* pSrc, int srcStride, BYTE* pDst, int dstStride, VMX_SIZE size);

/**
* Helper function to convert a BGRX image to UYVY. 
* 
* Only changed pixels between pSrc and pSrcPrev will be copied.
* 
* pDst will be left untouched if no pixels have changed, and no conversion will occur.
* 
* The destination buffer must be 64byte aligned.
* 
* @param[in] pSrc The source pixels in BGRA format. Alpha is ignored.
* @param[in] pSrcPrev The previous source pixels in BGRA format. Alpha is ignored.
* @param[in] srcStride The stride in bytes of each row of source pixels.
* @param[in] pDst The destination buffer. This must be 64byte aligned.
* @param[in] dstStride The stride in bytes of each row of destination pixels.
* @param[in] size The dimensions of the pixels.
* @return Returns 1 if any pixels have changed, 0 otherwise.
*/
VMX_API int VMX_BGRXToUYVYConditional(BYTE* pSrc, BYTE* pSrcPrev, int srcStride, BYTE* pDst, int dstStride, VMX_SIZE size);

/**
* Helper function to calculate the PSNR of two frames to assess image quality.
* 
* The first image should be the original image, and the second the compressed to compare.
*/
VMX_API float VMX_CalculatePSNR(BYTE* p1, BYTE* p2, int stride, int bytesPerPixel, VMX_SIZE sz);

/**
* Get the number of threads currently configured for encoding/decoding. These are automatically set based on the dimensions/profile.
* @param[in] instance The instance created using VMX_Create
*/
VMX_API int VMX_GetThreads(VMX_INSTANCE* instance);

/**
* Set the number of threads for encoding/decoding.
* @param[in] instance The instance created using VMX_Create
* @param[in] numThreads The number of threads to use. <= 0 is invalid and ignored.
*/
VMX_API void VMX_SetThreads(VMX_INSTANCE* instance, int numThreads);

VMX_API int VMX_Test(VMX_INSTANCE* instance, short* src, short* dst);
//Private functions
void VMX_ResetData(VMX_SLICE_DATA* s);
void VMX_ResetStream(VMX_INSTANCE* instance);


