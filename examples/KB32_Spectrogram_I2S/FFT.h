/*

  ESP32 FFT
  =========

  This provides a vanilla radix-2 FFT implementation and a test example.

  Author
  ------

  This code was written by [Robin Scheibler](http://www.robinscheibler.org) during rainy days in October 2017.

  License
  -------

  Copyright (c) 2017 Robin Scheibler

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

typedef enum
{
  FFT_REAL,
  FFT_COMPLEX
} fft_type_t;

typedef enum
{
  FFT_FORWARD,
  FFT_BACKWARD
} fft_direction_t;

#define FFT_OWN_INPUT_MEM 1
#define FFT_OWN_OUTPUT_MEM 2

typedef struct
{
  int size;  // FFT size
  float* input;  // pointer to input buffer
  float* output; // pointer to output buffer
  float* twiddle_factors;  // pointer to buffer holding twiddle factors
  fft_type_t type;   // real or complex
  fft_direction_t direction; // forward or backward
  unsigned int flags; // FFT flags
} fft_config_t;


fft_config_t* fft_init(int size, fft_type_t type, fft_direction_t direction, float* input, float* output);
void fft_destroy(fft_config_t* config);
void fft_execute(fft_config_t* config);
void fft(float* input, float* output, float* twiddle_factors, int n);
void ifft(float* input, float* output, float* twiddle_factors, int n);
void rfft(float* x, float* y, float* twiddle_factors, int n);
void irfft(float* x, float* y, float* twiddle_factors, int n);
void fft_primitive(float* x, float* y, int n, int stride, float* twiddle_factors, int tw_stride);
void split_radix_fft(float* x, float* y, int n, int stride, float* twiddle_factors, int tw_stride);
void ifft_primitive(float* input, float* output, int n, int stride, float* twiddle_factors, int tw_stride);
void fft8(float* input, int stride_in, float* output, int stride_out);
void fft4(float* input, int stride_in, float* output, int stride_out);


#define TWO_PI 6.28318530
#define USE_SPLIT_RADIX 1
#define LARGE_BASE_CASE 1



fft_config_t* fft_init(int size, fft_type_t type, fft_direction_t direction, float* input, float* output)
{
  /*
   * Prepare an FFT of correct size and types.
   *
   * If no input or output buffers are provided, they will be allocated.
   */
  int k, m;

  fft_config_t* config = (fft_config_t*)malloc(sizeof(fft_config_t));

  // Check if the size is a power of two
  if ((size & (size - 1)) != 0)  // tests if size is a power of two
    return NULL;

  // start configuration
  config->flags = 0;
  config->type = type;
  config->direction = direction;
  config->size = size;

  // Allocate and precompute twiddle factors
  config->twiddle_factors = (float*)malloc(2 * config->size * sizeof(float));

  float two_pi_by_n = TWO_PI / config->size;

  for (k = 0, m = 0; k < config->size; k++, m += 2)
  {
    config->twiddle_factors[m] = cosf(two_pi_by_n * k);    // real
    config->twiddle_factors[m + 1] = sinf(two_pi_by_n * k);  // imag
  }

  // Allocate input buffer
  if (input != NULL)
    config->input = input;
  else
  {
    if (config->type == FFT_REAL)
      config->input = (float*)malloc(config->size * sizeof(float));
    else if (config->type == FFT_COMPLEX)
      config->input = (float*)malloc(2 * config->size * sizeof(float));

    config->flags |= FFT_OWN_INPUT_MEM;
  }

  if (config->input == NULL)
    return NULL;

  // Allocate output buffer
  if (output != NULL)
    config->output = output;
  else
  {
    if (config->type == FFT_REAL)
      config->output = (float*)malloc(config->size * sizeof(float));
    else if (config->type == FFT_COMPLEX)
      config->output = (float*)malloc(2 * config->size * sizeof(float));

    config->flags |= FFT_OWN_OUTPUT_MEM;
  }

  if (config->output == NULL)
    return NULL;

  return config;
}

void fft_destroy(fft_config_t* config)
{
  if (config->flags & FFT_OWN_INPUT_MEM)
    free(config->input);

  if (config->flags & FFT_OWN_OUTPUT_MEM)
    free(config->output);

  free(config->twiddle_factors);
  free(config);
}

void fft_execute(fft_config_t* config)
{
  if (config->type == FFT_REAL && config->direction == FFT_FORWARD)
    rfft(config->input, config->output, config->twiddle_factors, config->size);
  else if (config->type == FFT_REAL && config->direction == FFT_BACKWARD)
    irfft(config->input, config->output, config->twiddle_factors, config->size);
  else if (config->type == FFT_COMPLEX && config->direction == FFT_FORWARD)
    fft(config->input, config->output, config->twiddle_factors, config->size);
  else if (config->type == FFT_COMPLEX && config->direction == FFT_BACKWARD)
    ifft(config->input, config->output, config->twiddle_factors, config->size);
}

void fft(float* input, float* output, float* twiddle_factors, int n)
{
  /*
   * Forward fast Fourier transform
   * DIT, radix-2, out-of-place implementation
   *
   * Parameters
   * ----------
   *  input (float *)
   *    The input array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  output (float *)
   *    The output array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  n (int)
   *    The FFT size, should be a power of 2
   */

#if USE_SPLIT_RADIX
  split_radix_fft(input, output, n, 2, twiddle_factors, 2);
#else
  fft_primitive(input, output, n, 2, twiddle_factors, 2);
#endif
}

void ifft(float* input, float* output, float* twiddle_factors, int n)
{
  /*
   * Inverse fast Fourier transform
   * DIT, radix-2, out-of-place implementation
   *
   * Parameters
   * ----------
   *  input (float *)
   *    The input array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  output (float *)
   *    The output array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  n (int)
   *    The FFT size, should be a power of 2
   */
  ifft_primitive(input, output, n, 2, twiddle_factors, 2);
}

void rfft(float* x, float* y, float* twiddle_factors, int n)
{

  // This code uses the two-for-the-price-of-one strategy
#if USE_SPLIT_RADIX
  split_radix_fft(x, y, n / 2, 2, twiddle_factors, 4);
#else
  fft_primitive(x, y, n / 2, 2, twiddle_factors, 4);
#endif

  // Now apply post processing to recover positive
  // frequencies of the real FFT
  float t = y[0];
  y[0] = t + y[1];  // DC coefficient
  y[1] = t - y[1];  // Center coefficient

  // Apply post processing to quarter element
  // this boils down to taking complex conjugate
  y[n / 2 + 1] = -y[n / 2 + 1];

  // Now process all the other frequencies
  int k;
  for (k = 2; k < n / 2; k += 2)
  {
    float xer, xei, x0r, xoi, c, s, tr, ti;

    c = twiddle_factors[k];
    s = twiddle_factors[k + 1];

    // even half coefficient
    xer = 0.5 * (y[k] + y[n - k]);
    xei = 0.5 * (y[k + 1] - y[n - k + 1]);

    // odd half coefficient
    x0r = 0.5 * (y[k + 1] + y[n - k + 1]);
    xoi = -0.5 * (y[k] - y[n - k]);

    tr = c * x0r + s * xoi;
    ti = -s * x0r + c * xoi;

    y[k] = xer + tr;
    y[k + 1] = xei + ti;

    y[n - k] = xer - tr;
    y[n - k + 1] = -(xei - ti);
  }
}

void irfft(float* x, float* y, float* twiddle_factors, int n)
{
  /*
   * Destroys content of input vector
   */
  int k;

  // Here we need to apply a pre-processing first
  float t = x[0];
  x[0] = 0.5 * (t + x[1]);
  x[1] = 0.5 * (t - x[1]);

  x[n / 2 + 1] = -x[n / 2 + 1];

  for (k = 2; k < n / 2; k += 2)
  {
    float xer, xei, x0r, xoi, c, s, tr, ti;

    c = twiddle_factors[k];
    s = twiddle_factors[k + 1];

    xer = 0.5 * (x[k] + x[n - k]);
    tr = 0.5 * (x[k] - x[n - k]);

    xei = 0.5 * (x[k + 1] - x[n - k + 1]);
    ti = 0.5 * (x[k + 1] + x[n - k + 1]);

    x0r = c * tr - s * ti;
    xoi = s * tr + c * ti;

    x[k] = xer - xoi;
    x[k + 1] = x0r + xei;

    x[n - k] = xer + xoi;
    x[n - k + 1] = x0r - xei;
  }

  ifft_primitive(x, y, n / 2, 2, twiddle_factors, 4);
}

void fft_primitive(float* x, float* y, int n, int stride, float* twiddle_factors, int tw_stride)
{
  /*
   * This code will compute the FFT of the input vector x
   *
   * The input data is assumed to be real/imag interleaved
   *
   * The size n should be a power of two
   *
   * y is an output buffer of size 2n to accomodate for complex numbers
   *
   * Forward fast Fourier transform
   * DIT, radix-2, out-of-place implementation
   *
   * For a complex FFT, call first stage as:
   * fft(x, y, n, 2, 2);
   *
   * Parameters
   * ----------
   *  x (float *)
   *    The input array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  y (float *)
   *    The output array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  n (int)
   *    The FFT size, should be a power of 2
   *  stride (int)
   *    The number of elements to skip between two successive samples
   *  tw_stride (int)
   *    The number of elements to skip between two successive twiddle factors
   */
  int k;
  float t;

#if LARGE_BASE_CASE
  // End condition, stop at n=8 to avoid one trivial recursion
  if (n == 8)
  {
    fft8(x, stride, y, 2);
    return;
  }
#else
  // End condition, stop at n=2 to avoid one trivial recursion
  if (n == 2)
  {
    y[0] = x[0] + x[stride];
    y[1] = x[1] + x[stride + 1];
    y[2] = x[0] - x[stride];
    y[3] = x[1] - x[stride + 1];
    return;
  }
#endif

  // Recursion -- Decimation In Time algorithm
  fft_primitive(x, y, n / 2, 2 * stride, twiddle_factors, 2 * tw_stride);             // even half
  fft_primitive(x + stride, y + n, n / 2, 2 * stride, twiddle_factors, 2 * tw_stride);  // odd half

  // Stitch back together

  // We can a few multiplications in the first step
  t = y[0];
  y[0] = t + y[n];
  y[n] = t - y[n];

  t = y[1];
  y[1] = t + y[n + 1];
  y[n + 1] = t - y[n + 1];

  for (k = 1; k < n / 2; k++)
  {
    float x1r, x1i, x2r, x2i, c, s;
    c = twiddle_factors[k * tw_stride];
    s = twiddle_factors[k * tw_stride + 1];

    x1r = y[2 * k];
    x1i = y[2 * k + 1];
    x2r = c * y[n + 2 * k] + s * y[n + 2 * k + 1];
    x2i = -s * y[n + 2 * k] + c * y[n + 2 * k + 1];

    y[2 * k] = x1r + x2r;
    y[2 * k + 1] = x1i + x2i;

    y[n + 2 * k] = x1r - x2r;
    y[n + 2 * k + 1] = x1i - x2i;
  }

}

void split_radix_fft(float* x, float* y, int n, int stride, float* twiddle_factors, int tw_stride)
{
  /*
   * This code will compute the FFT of the input vector x
   *
   * The input data is assumed to be real/imag interleaved
   *
   * The size n should be a power of two
   *
   * y is an output buffer of size 2n to accomodate for complex numbers
   *
   * Forward fast Fourier transform
   * Split-Radix
   * DIT, radix-2, out-of-place implementation
   *
   * For a complex FFT, call first stage as:
   * fft(x, y, n, 2, 2);
   *
   * Parameters
   * ----------
   *  x (float *)
   *    The input array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  y (float *)
   *    The output array containing the complex samples with
   *    real/imaginary parts interleaved [Re(x0), Im(x0), ..., Re(x_n-1), Im(x_n-1)]
   *  n (int)
   *    The FFT size, should be a power of 2
   *  stride (int)
   *    The number of elements to skip between two successive samples
   *  twiddle_factors (float *)
   *    The array of twiddle factors
   *  tw_stride (int)
   *    The number of elements to skip between two successive twiddle factors
   */
  int k;

#if LARGE_BASE_CASE
  // End condition, stop at n=2 to avoid one trivial recursion
  if (n == 8)
  {
    fft8(x, stride, y, 2);
    return;
  }
  else if (n == 4)
  {
    fft4(x, stride, y, 2);
    return;
  }
#else
  // End condition, stop at n=2 to avoid one trivial recursion
  if (n == 2)
  {
    y[0] = x[0] + x[stride];
    y[1] = x[1] + x[stride + 1];
    y[2] = x[0] - x[stride];
    y[3] = x[1] - x[stride + 1];
    return;
  }
  else if (n == 1)
  {
    y[0] = x[0];
    y[1] = x[1];
    return;
  }
#endif

  // Recursion -- Decimation In Time algorithm
  split_radix_fft(x, y, n / 2, 2 * stride, twiddle_factors, 2 * tw_stride);
  split_radix_fft(x + stride, y + n, n / 4, 4 * stride, twiddle_factors, 4 * tw_stride);
  split_radix_fft(x + 3 * stride, y + n + n / 2, n / 4, 4 * stride, twiddle_factors, 4 * tw_stride);

  // Stitch together the output
  float u1r, u1i, u2r, u2i, x1r, x1i, x2r, x2i;
  float t;

  // We can save a few multiplications in the first step
  u1r = y[0];
  u1i = y[1];
  u2r = y[n / 2];
  u2i = y[n / 2 + 1];

  x1r = y[n];
  x1i = y[n + 1];
  x2r = y[n / 2 + n];
  x2i = y[n / 2 + n + 1];

  t = x1r + x2r;
  y[0] = u1r + t;
  y[n] = u1r - t;

  t = x1i + x2i;
  y[1] = u1i + t;
  y[n + 1] = u1i - t;

  t = x2i - x1i;
  y[n / 2] = u2r - t;
  y[n + n / 2] = u2r + t;

  t = x1r - x2r;
  y[n / 2 + 1] = u2i - t;
  y[n + n / 2 + 1] = u2i + t;

  for (k = 1; k < n / 4; k++)
  {
    float u1r, u1i, u2r, u2i, x1r, x1i, x2r, x2i, c1, s1, c2, s2;
    c1 = twiddle_factors[k * tw_stride];
    s1 = twiddle_factors[k * tw_stride + 1];
    c2 = twiddle_factors[3 * k * tw_stride];
    s2 = twiddle_factors[3 * k * tw_stride + 1];

    u1r = y[2 * k];
    u1i = y[2 * k + 1];
    u2r = y[2 * k + n / 2];
    u2i = y[2 * k + n / 2 + 1];

    x1r = c1 * y[n + 2 * k] + s1 * y[n + 2 * k + 1];
    x1i = -s1 * y[n + 2 * k] + c1 * y[n + 2 * k + 1];
    x2r = c2 * y[n / 2 + n + 2 * k] + s2 * y[n / 2 + n + 2 * k + 1];
    x2i = -s2 * y[n / 2 + n + 2 * k] + c2 * y[n / 2 + n + 2 * k + 1];

    t = x1r + x2r;
    y[2 * k] = u1r + t;
    y[2 * k + n] = u1r - t;

    t = x1i + x2i;
    y[2 * k + 1] = u1i + t;
    y[2 * k + n + 1] = u1i - t;

    t = x2i - x1i;
    y[2 * k + n / 2] = u2r - t;
    y[2 * k + n + n / 2] = u2r + t;

    t = x1r - x2r;
    y[2 * k + n / 2 + 1] = u2i - t;
    y[2 * k + n + n / 2 + 1] = u2i + t;
  }

}


void ifft_primitive(float* input, float* output, int n, int stride, float* twiddle_factors, int tw_stride)
{

#if USE_SPLIT_RADIX
  split_radix_fft(input, output, n, stride, twiddle_factors, tw_stride);
#else
  fft_primitive(input, output, n, stride, twiddle_factors, tw_stride);
#endif

  int ks;

  int ns = n * stride;

  // reverse all coefficients from 1 to n / 2 - 1
  for (ks = stride; ks < ns / 2; ks += stride)
  {
    float t;

    t = output[ks];
    output[ks] = output[ns - ks];
    output[ns - ks] = t;

    t = output[ks + 1];
    output[ks + 1] = output[ns - ks + 1];
    output[ns - ks + 1] = t;
  }

  // Apply normalization
  float norm = 1. / n;
  for (ks = 0; ks < ns; ks += stride)
  {
    output[ks] *= norm;
    output[ks + 1] *= norm;
  }

}

inline void fft8(float* input, int stride_in, float* output, int stride_out)
{
  /*
   * Unrolled implementation of FFT8 for a little more performance
   */
  float a0r, a1r, a2r, a3r, a4r, a5r, a6r, a7r;
  float a0i, a1i, a2i, a3i, a4i, a5i, a6i, a7i;
  float b0r, b1r, b2r, b3r, b4r, b5r, b6r, b7r;
  float b0i, b1i, b2i, b3i, b4i, b5i, b6i, b7i;
  float t;
  float sin_pi_4 = 0.7071067812;

  a0r = input[0];
  a0i = input[1];
  a1r = input[stride_in];
  a1i = input[stride_in + 1];
  a2r = input[2 * stride_in];
  a2i = input[2 * stride_in + 1];
  a3r = input[3 * stride_in];
  a3i = input[3 * stride_in + 1];
  a4r = input[4 * stride_in];
  a4i = input[4 * stride_in + 1];
  a5r = input[5 * stride_in];
  a5i = input[5 * stride_in + 1];
  a6r = input[6 * stride_in];
  a6i = input[6 * stride_in + 1];
  a7r = input[7 * stride_in];
  a7i = input[7 * stride_in + 1];

  // Stage 1

  b0r = a0r + a4r;
  b0i = a0i + a4i;

  b1r = a1r + a5r;
  b1i = a1i + a5i;

  b2r = a2r + a6r;
  b2i = a2i + a6i;

  b3r = a3r + a7r;
  b3i = a3i + a7i;

  b4r = a0r - a4r;
  b4i = a0i - a4i;

  b5r = a1r - a5r;
  b5i = a1i - a5i;
  // W_8^1 = 1/sqrt(2) - j / sqrt(2)
  t = b5r + b5i;
  b5i = (b5i - b5r) * sin_pi_4;
  b5r = t * sin_pi_4;

  // W_8^2 = -j
  b6r = a2i - a6i;
  b6i = a6r - a2r;

  b7r = a3r - a7r;
  b7i = a3i - a7i;
  // W_8^3 = -1 / sqrt(2) + j / sqrt(2)
  t = sin_pi_4 * (b7i - b7r);
  b7i = -(b7r + b7i) * sin_pi_4;
  b7r = t;

  // Stage 2

  a0r = b0r + b2r;
  a0i = b0i + b2i;

  a1r = b1r + b3r;
  a1i = b1i + b3i;

  a2r = b0r - b2r;
  a2i = b0i - b2i;

  // * j
  a3r = b1i - b3i;
  a3i = b3r - b1r;

  a4r = b4r + b6r;
  a4i = b4i + b6i;

  a5r = b5r + b7r;
  a5i = b5i + b7i;

  a6r = b4r - b6r;
  a6i = b4i - b6i;

  // * j
  a7r = b5i - b7i;
  a7i = b7r - b5r;

  // Stage 3

  // X[0]
  output[0] = a0r + a1r;
  output[1] = a0i + a1i;

  // X[4]
  output[4 * stride_out] = a0r - a1r;
  output[4 * stride_out + 1] = a0i - a1i;

  // X[2]
  output[2 * stride_out] = a2r + a3r;
  output[2 * stride_out + 1] = a2i + a3i;

  // X[6]
  output[6 * stride_out] = a2r - a3r;
  output[6 * stride_out + 1] = a2i - a3i;

  // X[1]
  output[stride_out] = a4r + a5r;
  output[stride_out + 1] = a4i + a5i;

  // X[5]
  output[5 * stride_out] = a4r - a5r;
  output[5 * stride_out + 1] = a4i - a5i;

  // X[3]
  output[3 * stride_out] = a6r + a7r;
  output[3 * stride_out + 1] = a6i + a7i;

  // X[7]
  output[7 * stride_out] = a6r - a7r;
  output[7 * stride_out + 1] = a6i - a7i;

}

inline void fft4(float* input, int stride_in, float* output, int stride_out)
{
  /*
   * Unrolled implementation of FFT4 for a little more performance
   */
  float t1, t2;

  t1 = input[0] + input[2 * stride_in];
  t2 = input[stride_in] + input[3 * stride_in];
  output[0] = t1 + t2;
  output[2 * stride_out] = t1 - t2;

  t1 = input[1] + input[2 * stride_in + 1];
  t2 = input[stride_in + 1] + input[3 * stride_in + 1];
  output[1] = t1 + t2;
  output[2 * stride_out + 1] = t1 - t2;

  t1 = input[0] - input[2 * stride_in];
  t2 = input[stride_in + 1] - input[3 * stride_in + 1];
  output[stride_out] = t1 + t2;
  output[3 * stride_out] = t1 - t2;

  t1 = input[1] - input[2 * stride_in + 1];
  t2 = input[3 * stride_in] - input[stride_in];
  output[stride_out + 1] = t1 + t2;
  output[3 * stride_out + 1] = t1 - t2;
}


uint16_t mag_2_color(float mag) {
  //https://github.com/libigl/libigl/blob/master/include/igl/colormap.cpp
  const float color_map[256][3] = {
    {0.18995,0.07176,0.23217},
    {0.19483,0.08339,0.26149},
    {0.19956,0.09498,0.29024},
    {0.20415,0.10652,0.31844},
    {0.20860,0.11802,0.34607},
    {0.21291,0.12947,0.37314},
    {0.21708,0.14087,0.39964},
    {0.22111,0.15223,0.42558},
    {0.22500,0.16354,0.45096},
    {0.22875,0.17481,0.47578},
    {0.23236,0.18603,0.50004},
    {0.23582,0.19720,0.52373},
    {0.23915,0.20833,0.54686},
    {0.24234,0.21941,0.56942},
    {0.24539,0.23044,0.59142},
    {0.24830,0.24143,0.61286},
    {0.25107,0.25237,0.63374},
    {0.25369,0.26327,0.65406},
    {0.25618,0.27412,0.67381},
    {0.25853,0.28492,0.69300},
    {0.26074,0.29568,0.71162},
    {0.26280,0.30639,0.72968},
    {0.26473,0.31706,0.74718},
    {0.26652,0.32768,0.76412},
    {0.26816,0.33825,0.78050},
    {0.26967,0.34878,0.79631},
    {0.27103,0.35926,0.81156},
    {0.27226,0.36970,0.82624},
    {0.27334,0.38008,0.84037},
    {0.27429,0.39043,0.85393},
    {0.27509,0.40072,0.86692},
    {0.27576,0.41097,0.87936},
    {0.27628,0.42118,0.89123},
    {0.27667,0.43134,0.90254},
    {0.27691,0.44145,0.91328},
    {0.27701,0.45152,0.92347},
    {0.27698,0.46153,0.93309},
    {0.27680,0.47151,0.94214},
    {0.27648,0.48144,0.95064},
    {0.27603,0.49132,0.95857},
    {0.27543,0.50115,0.96594},
    {0.27469,0.51094,0.97275},
    {0.27381,0.52069,0.97899},
    {0.27273,0.53040,0.98461},
    {0.27106,0.54015,0.98930},
    {0.26878,0.54995,0.99303},
    {0.26592,0.55979,0.99583},
    {0.26252,0.56967,0.99773},
    {0.25862,0.57958,0.99876},
    {0.25425,0.58950,0.99896},
    {0.24946,0.59943,0.99835},
    {0.24427,0.60937,0.99697},
    {0.23874,0.61931,0.99485},
    {0.23288,0.62923,0.99202},
    {0.22676,0.63913,0.98851},
    {0.22039,0.64901,0.98436},
    {0.21382,0.65886,0.97959},
    {0.20708,0.66866,0.97423},
    {0.20021,0.67842,0.96833},
    {0.19326,0.68812,0.96190},
    {0.18625,0.69775,0.95498},
    {0.17923,0.70732,0.94761},
    {0.17223,0.71680,0.93981},
    {0.16529,0.72620,0.93161},
    {0.15844,0.73551,0.92305},
    {0.15173,0.74472,0.91416},
    {0.14519,0.75381,0.90496},
    {0.13886,0.76279,0.89550},
    {0.13278,0.77165,0.88580},
    {0.12698,0.78037,0.87590},
    {0.12151,0.78896,0.86581},
    {0.11639,0.79740,0.85559},
    {0.11167,0.80569,0.84525},
    {0.10738,0.81381,0.83484},
    {0.10357,0.82177,0.82437},
    {0.10026,0.82955,0.81389},
    {0.09750,0.83714,0.80342},
    {0.09532,0.84455,0.79299},
    {0.09377,0.85175,0.78264},
    {0.09287,0.85875,0.77240},
    {0.09267,0.86554,0.76230},
    {0.09320,0.87211,0.75237},
    {0.09451,0.87844,0.74265},
    {0.09662,0.88454,0.73316},
    {0.09958,0.89040,0.72393},
    {0.10342,0.89600,0.71500},
    {0.10815,0.90142,0.70599},
    {0.11374,0.90673,0.69651},
    {0.12014,0.91193,0.68660},
    {0.12733,0.91701,0.67627},
    {0.13526,0.92197,0.66556},
    {0.14391,0.92680,0.65448},
    {0.15323,0.93151,0.64308},
    {0.16319,0.93609,0.63137},
    {0.17377,0.94053,0.61938},
    {0.18491,0.94484,0.60713},
    {0.19659,0.94901,0.59466},
    {0.20877,0.95304,0.58199},
    {0.22142,0.95692,0.56914},
    {0.23449,0.96065,0.55614},
    {0.24797,0.96423,0.54303},
    {0.26180,0.96765,0.52981},
    {0.27597,0.97092,0.51653},
    {0.29042,0.97403,0.50321},
    {0.30513,0.97697,0.48987},
    {0.32006,0.97974,0.47654},
    {0.33517,0.98234,0.46325},
    {0.35043,0.98477,0.45002},
    {0.36581,0.98702,0.43688},
    {0.38127,0.98909,0.42386},
    {0.39678,0.99098,0.41098},
    {0.41229,0.99268,0.39826},
    {0.42778,0.99419,0.38575},
    {0.44321,0.99551,0.37345},
    {0.45854,0.99663,0.36140},
    {0.47375,0.99755,0.34963},
    {0.48879,0.99828,0.33816},
    {0.50362,0.99879,0.32701},
    {0.51822,0.99910,0.31622},
    {0.53255,0.99919,0.30581},
    {0.54658,0.99907,0.29581},
    {0.56026,0.99873,0.28623},
    {0.57357,0.99817,0.27712},
    {0.58646,0.99739,0.26849},
    {0.59891,0.99638,0.26038},
    {0.61088,0.99514,0.25280},
    {0.62233,0.99366,0.24579},
    {0.63323,0.99195,0.23937},
    {0.64362,0.98999,0.23356},
    {0.65394,0.98775,0.22835},
    {0.66428,0.98524,0.22370},
    {0.67462,0.98246,0.21960},
    {0.68494,0.97941,0.21602},
    {0.69525,0.97610,0.21294},
    {0.70553,0.97255,0.21032},
    {0.71577,0.96875,0.20815},
    {0.72596,0.96470,0.20640},
    {0.73610,0.96043,0.20504},
    {0.74617,0.95593,0.20406},
    {0.75617,0.95121,0.20343},
    {0.76608,0.94627,0.20311},
    {0.77591,0.94113,0.20310},
    {0.78563,0.93579,0.20336},
    {0.79524,0.93025,0.20386},
    {0.80473,0.92452,0.20459},
    {0.81410,0.91861,0.20552},
    {0.82333,0.91253,0.20663},
    {0.83241,0.90627,0.20788},
    {0.84133,0.89986,0.20926},
    {0.85010,0.89328,0.21074},
    {0.85868,0.88655,0.21230},
    {0.86709,0.87968,0.21391},
    {0.87530,0.87267,0.21555},
    {0.88331,0.86553,0.21719},
    {0.89112,0.85826,0.21880},
    {0.89870,0.85087,0.22038},
    {0.90605,0.84337,0.22188},
    {0.91317,0.83576,0.22328},
    {0.92004,0.82806,0.22456},
    {0.92666,0.82025,0.22570},
    {0.93301,0.81236,0.22667},
    {0.93909,0.80439,0.22744},
    {0.94489,0.79634,0.22800},
    {0.95039,0.78823,0.22831},
    {0.95560,0.78005,0.22836},
    {0.96049,0.77181,0.22811},
    {0.96507,0.76352,0.22754},
    {0.96931,0.75519,0.22663},
    {0.97323,0.74682,0.22536},
    {0.97679,0.73842,0.22369},
    {0.98000,0.73000,0.22161},
    {0.98289,0.72140,0.21918},
    {0.98549,0.71250,0.21650},
    {0.98781,0.70330,0.21358},
    {0.98986,0.69382,0.21043},
    {0.99163,0.68408,0.20706},
    {0.99314,0.67408,0.20348},
    {0.99438,0.66386,0.19971},
    {0.99535,0.65341,0.19577},
    {0.99607,0.64277,0.19165},
    {0.99654,0.63193,0.18738},
    {0.99675,0.62093,0.18297},
    {0.99672,0.60977,0.17842},
    {0.99644,0.59846,0.17376},
    {0.99593,0.58703,0.16899},
    {0.99517,0.57549,0.16412},
    {0.99419,0.56386,0.15918},
    {0.99297,0.55214,0.15417},
    {0.99153,0.54036,0.14910},
    {0.98987,0.52854,0.14398},
    {0.98799,0.51667,0.13883},
    {0.98590,0.50479,0.13367},
    {0.98360,0.49291,0.12849},
    {0.98108,0.48104,0.12332},
    {0.97837,0.46920,0.11817},
    {0.97545,0.45740,0.11305},
    {0.97234,0.44565,0.10797},
    {0.96904,0.43399,0.10294},
    {0.96555,0.42241,0.09798},
    {0.96187,0.41093,0.09310},
    {0.95801,0.39958,0.08831},
    {0.95398,0.38836,0.08362},
    {0.94977,0.37729,0.07905},
    {0.94538,0.36638,0.07461},
    {0.94084,0.35566,0.07031},
    {0.93612,0.34513,0.06616},
    {0.93125,0.33482,0.06218},
    {0.92623,0.32473,0.05837},
    {0.92105,0.31489,0.05475},
    {0.91572,0.30530,0.05134},
    {0.91024,0.29599,0.04814},
    {0.90463,0.28696,0.04516},
    {0.89888,0.27824,0.04243},
    {0.89298,0.26981,0.03993},
    {0.88691,0.26152,0.03753},
    {0.88066,0.25334,0.03521},
    {0.87422,0.24526,0.03297},
    {0.86760,0.23730,0.03082},
    {0.86079,0.22945,0.02875},
    {0.85380,0.22170,0.02677},
    {0.84662,0.21407,0.02487},
    {0.83926,0.20654,0.02305},
    {0.83172,0.19912,0.02131},
    {0.82399,0.19182,0.01966},
    {0.81608,0.18462,0.01809},
    {0.80799,0.17753,0.01660},
    {0.79971,0.17055,0.01520},
    {0.79125,0.16368,0.01387},
    {0.78260,0.15693,0.01264},
    {0.77377,0.15028,0.01148},
    {0.76476,0.14374,0.01041},
    {0.75556,0.13731,0.00942},
    {0.74617,0.13098,0.00851},
    {0.73661,0.12477,0.00769},
    {0.72686,0.11867,0.00695},
    {0.71692,0.11268,0.00629},
    {0.70680,0.10680,0.00571},
    {0.69650,0.10102,0.00522},
    {0.68602,0.09536,0.00481},
    {0.67535,0.08980,0.00449},
    {0.66449,0.08436,0.00424},
    {0.65345,0.07902,0.00408},
    {0.64223,0.07380,0.00401},
    {0.63082,0.06868,0.00401},
    {0.61923,0.06367,0.00410},
    {0.60746,0.05878,0.00427},
    {0.59550,0.05399,0.00453},
    {0.58336,0.04931,0.00486},
    {0.57103,0.04474,0.00529},
    {0.55852,0.04028,0.00579},
    {0.54583,0.03593,0.00638},
    {0.53295,0.03169,0.00705},
    {0.51989,0.02756,0.00780},
    {0.50664,0.02354,0.00863},
    {0.49321,0.01963,0.00955},
    {0.47960,0.01583,0.01055}
  };

  if (mag > 255)
    mag = 255;

  uint16_t red = 32.0f * color_map[(int)mag][0];
  uint16_t green = 64.0f * color_map[(int)mag][1];
  uint16_t blue = 32.0f * color_map[(int)mag][2];

  uint16_t colour = red << 11 | green << 5 | blue;


  return colour;
}