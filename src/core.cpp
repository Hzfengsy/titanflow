#include <iostream>
#include <cassert>
#include <omp.h>
#include <cblas.h>
using namespace std;

extern "C" {
void matmul(const float *A, const float *B, float *C, int m, int p, int n, bool TA, bool TB)
{
	const enum CBLAS_ORDER Order = CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA = TA ? CblasTrans : CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB = TB ? CblasTrans : CblasNoTrans;
	const int M = m; //A的行数，C的行数
	const int N = n; //B的列数，C的列数
	const int K = p; //A的列数，B的行数
	const float alpha = 1;
	const float beta = 0;
	const int lda = TA ? M : K; //A的列
	const int ldb = TB ? K : N; //B的列
	const int ldc = N; //C的列
	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void conv2d(const float *x, const float *w, 
			float *out, int N,
			int in_height, int in_width,
			int filter_height, int filter_width,
			int out_height, int out_width, 
			int strides_1, int strides_2, 
			int in_channel, int out_channel)
{
	
	int mm = out_height * out_width;
	int pp = filter_height * filter_width * in_channel;
	int nn = out_channel;
	#pragma omp parallel for
	for (int n = 0; n < N; n++)
	{
		float *x_cols = new float[out_height * out_width * filter_height * filter_width * in_channel]();
		float *cols_rf = x_cols;
		const float *x_rf = x + n * in_height * in_width * in_channel;
		for (int i = filter_height; i <= in_height; i += strides_1)
			for (int j = filter_width; j <= in_width; j += strides_2)
			{
				int indx = 0;
				for (int ii = i - filter_height; ii < i; ii++)
					for (int jj = j - filter_width; jj < j; jj++)
						for (int t = 0; t < in_channel; t++)
						{
							cols_rf[indx++] = x_rf[ii * in_width * in_channel + jj * in_channel + t];
						}
				cols_rf += filter_height * filter_width * in_channel;
			}
		matmul(x_cols, w, out + n * out_height * out_width * out_channel, mm, pp, nn, false, false);
		delete[] x_cols;
	}
	
}

void conv2d_dw(const float *x, const float *dout,
			   float *out, int N,
			   int in_height, int in_width,
			   int filter_height, int filter_width,
			   int out_height, int out_width, 
			   int strides_1, int strides_2, 
			   int in_channel, int out_channel)
{
	int mm = out_height * out_width;
	int pp = filter_height * filter_width * in_channel;
	int nn = out_channel;
	for (int i = 0; i < pp * nn; i++) out[i] = 0;
	#pragma omp parallel for
	for (int n = 0; n < N; n++)
	{
		float *x_cols = new float[out_height * out_width * filter_height * filter_width * in_channel];
		const float *x_rf = x + n * in_height * in_width * in_channel;
		float *cols_rf = x_cols;
		for (int i = filter_height; i <= in_height; i += strides_1)
			for (int j = filter_width; j <= in_width; j += strides_2)
			{
				int indx = 0;
				for (int ii = i - filter_height; ii < i; ii++)
					for (int jj = j - filter_width; jj < j; jj++)
						for (int t = 0; t < in_channel; t++)
						{
							cols_rf[indx++] = x_rf[ii * in_width * in_channel + jj * in_channel + t];
						}
				cols_rf += filter_height * filter_width * in_channel;
			}
		float *temp = new float[pp * nn]();
		matmul(x_cols, dout + n * out_height * out_width * out_channel, temp, pp, mm, nn, true, false);
		for (int i = 0; i < pp * nn; i++) out[i] += temp[i];
		delete[] temp;
		delete[] x_cols;
	}
}

void conv2d_dx(const float *w, const float *dout,
			   float *out, int N,
			   int in_height, int in_width,
			   int filter_height, int filter_width,
			   int out_height, int out_width, 
			   int strides_1, int strides_2, 
			   int in_channel, int out_channel)
{
	int mm = out_height * out_width;
	int pp = out_channel;
	int nn = filter_height * filter_width * in_channel;
	for (int i = 0; i < N * in_height * in_width * in_channel; i++) out[i] = 0;
	#pragma omp parallel for
	for (int n = 0; n < N; n++)
	{
		const float *dout_cols = dout + n * out_height * out_width * out_channel;
		float *out_cols = out + n * in_height * in_width * in_channel;
		float *grad_x_cols = new float[mm * nn]();
		// cout << n << " " << flush;
		matmul(dout_cols, w, grad_x_cols, mm, pp, nn, false, true);
		// cout << n << endl << flush;
		float *x_rf = grad_x_cols;
		for (int i = filter_height; i <= in_height; i += strides_1)
			for (int j = filter_width; j <= in_width; j += strides_2)
			{
				int indx = 0;
				for (int ii = i - filter_height; ii < i; ii++)
					for (int jj = j - filter_width; jj < j; jj++)
						for (int t = 0; t < in_channel; t++)
						{
							out_cols[ii * in_width * in_channel + jj * in_channel + t] += x_rf[indx];
							indx++;
						}
				x_rf += nn;
			}
		delete[] grad_x_cols;
	}
		// # dout_cols = N * out_height * out_width, out_channel
		// dout_cols = dout.reshape(tnode.N, tnode.out_height * tnode.out_width, tnode.out_channel)
		// # grad_x_cols = N * out_height * out_width, filter_height * filter_width * in_channel
		// dx_padded = np.zeros((tnode.N, tnode.in_height, tnode.in_width, tnode.in_channel), dtype = input_vals[0].dtype)

		// tmp_shape = (1, tnode.filter_height, tnode.filter_width, tnode.in_channel)
		// for n in range(tnode.N):
		// 	dout_sub = dout_cols[n]
		// 	grad_x_cols = np.dot(dout_sub, tnode.w_cols.T)
		// 	idx = 0
		// 	for i in range(tnode.filter_height, tnode.in_height + 1, tnode.strides[1]):
		// 		for j in range(tnode.filter_width, tnode.in_width + 1, tnode.strides[2]):
		// 			tmp = grad_x_cols[idx, :].reshape(tmp_shape)
		// 			dx_padded[n: n + 1, i - tnode.filter_height: i, j - tnode.filter_width: j, :] += tmp
		// 			idx += 1
}
}