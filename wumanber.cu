#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <bitset>
#include <ctime>

using namespace std;

// #define DEBUG

#define WORD 32
#define ALPHABET_INITIAL ' '
#define ALPHABET_FINAL '~'
#define ASIZE (int) (ALPHABET_FINAL - ALPHABET_INITIAL + 1)

#define THREADS_PER_BLOCK 512
#define MAX_P_LEN 32
// #define STREAM_COUNT 5
#define MAX_K 5

__device__ unsigned int dagger1(unsigned int u1, unsigned int x1, unsigned int u2, unsigned int x2)
{
	return u1 + u2;
}

__device__ unsigned int dagger2(unsigned int u1, unsigned int x1, unsigned int u2, unsigned int x2)
{
	return (x1 << u2) | x2;
}

__device__ unsigned int dagger3(unsigned int x1, unsigned int x2)
{
	return x1 | x2;
}

__device__ unsigned int dagger4(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
	return (y1 & ~x2) | y2;
}

__global__ void wuManber_GPU(unsigned int *convText, int t_len, unsigned int *convPattern, int p_len, int k, unsigned int *d_AF, unsigned int *d_AS, unsigned int *d_AW, int pos)
{
	unsigned int src_index = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned int shared_convPattern[MAX_P_LEN];
	__shared__ unsigned int AF[MAX_K+1][THREADS_PER_BLOCK];
	__shared__ unsigned int AS[MAX_K+1][THREADS_PER_BLOCK];
	__shared__ unsigned int AW[MAX_K+1][THREADS_PER_BLOCK];

	int tx = threadIdx.x;

	if(tx < p_len)
		shared_convPattern[tx] = convPattern[tx];

	__syncthreads();

	//shiftOR algorithm 1

	AF[0][tx] = 1;
	AS[0][tx] = 0;

	for (int i = 0; i < p_len; ++i)
	{
		if(convText[src_index] == shared_convPattern[i])
			AS[0][tx] |= 1 << i;
	}

	AS[0][tx] = ~AS[0][tx];

	if(tx == 0 && blockIdx.x == 0)
	{
		AF[0][tx] = 0;
		AS[0][tx] = ~0;
	}

	__syncthreads();

	int stride = 1;
	while(stride < THREADS_PER_BLOCK)
	{
		int index = (tx+1)*stride*2 - 1;
		if(index < THREADS_PER_BLOCK)
		{
			unsigned int tempF = dagger1(AF[0][index], AS[0][index], AF[0][index-stride], AS[0][index-stride]);
			unsigned int tempS = dagger2(AF[0][index], AS[0][index], AF[0][index-stride], AS[0][index-stride]);

			AF[0][index] = tempF;
			AS[0][index] = tempS;
		}
		stride = stride*2;

		__syncthreads();
	}

	stride = THREADS_PER_BLOCK/4;
	while(stride > 0)
	{
		int index = (tx+1)*stride*2 - 1;
		if(index + stride < THREADS_PER_BLOCK)
		{
			unsigned int tempF = dagger1(AF[0][index+stride], AS[0][index+stride], AF[0][index], AS[0][index]);
			unsigned int tempS = dagger2(AF[0][index+stride], AS[0][index+stride], AF[0][index], AS[0][index]);

			AF[0][index] = tempF;
			AS[0][index] = tempS;
		}
		stride = stride / 2;
		__syncthreads();
	}

	unsigned int tempN = 0;

	for(int i = 1; i <= k; i++)
	{
		tempN = AF[i-1][tx-1] & (AF[i-1][tx-1] << 1) & (AF[i-1][tx] << 1);

		AW[i][0] = 0 << p_len;
		AS[i][0] = 1 << (p_len - k);
		AF[i][0] = 0;

		if(tx != 0)
		{
			AW[i][tx] = ~tempN;
			AS[i][tx] &= tempN;
			AF[i][tx] = 1;
		}

		__syncthreads();

		int stride = 1;
		while(stride < THREADS_PER_BLOCK)
		{
			int index = (tx+1)*stride*2 - 1;
			if(index < THREADS_PER_BLOCK)
			{
				unsigned int tempF = dagger1(AF[i][index], AS[i][index], AF[i][index-stride], AS[i][index-stride]);
				unsigned int tempW = dagger3(AW[i][index] << AF[i][index-stride], AW[i][index-stride]);
				unsigned int tempS = dagger4(AW[i][index] << AF[i][index-stride], AS[i][index] << AF[i][index-stride], AW[i][index-stride], AS[i][index-stride]);

				AF[i][index] = tempF;
				AS[i][index] = tempS;
				AW[i][index] = tempW;
			}
			stride = stride*2;

			__syncthreads();
		}

		stride = THREADS_PER_BLOCK/4;
		while(stride > 0)
		{
			int index = (tx+1)*stride*2 - 1;
			if(index + stride < THREADS_PER_BLOCK)
			{
				unsigned int tempF = dagger1(AF[i][index+stride], AS[i][index+stride], AF[i][index], AS[i][index]);
				unsigned int tempW = dagger3(AW[i][index+stride] << AF[i][index], AW[i][index]);
				unsigned int tempS = dagger4(AW[i][index+stride] << AF[i][index], AS[i][index+stride] << AF[i][index], AW[i][index], AS[i][index]);

				AF[i][index] = tempF;
				AS[i][index] = tempS;
				AW[i][index] = tempW;
			}
			stride = stride / 2;
			__syncthreads();
		}

		d_AF[i*THREADS_PER_BLOCK + pos * t_len + src_index] = AF[i][tx];
		d_AS[i*THREADS_PER_BLOCK + pos * t_len + src_index] = AS[i][tx];
		d_AW[i*THREADS_PER_BLOCK + pos * t_len + src_index] = AF[i][tx];
	}
}

__global__ void wuManber_halo_GPU(int t_len, int k, unsigned int *d_AF, unsigned int *d_AS, unsigned int *d_AW, unsigned int *R, int pos)
{
	__shared__ unsigned int AF[MAX_K+1][THREADS_PER_BLOCK];
	__shared__ unsigned int AS[MAX_K+1][THREADS_PER_BLOCK];
	__shared__ unsigned int AW[MAX_K+1][THREADS_PER_BLOCK];

	int tx = threadIdx.x;
	int store = (pos)*t_len - 1;

	for(int i = 1; i <= k; i++)
	{
		unsigned int tempF = dagger1(d_AF[i*THREADS_PER_BLOCK + store], d_AS[i*THREADS_PER_BLOCK + tx], d_AF[i*THREADS_PER_BLOCK + tx], d_AS[i*THREADS_PER_BLOCK + tx]);
		unsigned int tempW = dagger3(d_AW[i*THREADS_PER_BLOCK + store] << d_AF[i*THREADS_PER_BLOCK + tx], d_AW[i*THREADS_PER_BLOCK + tx]);
		unsigned int tempS = dagger4(d_AW[i*THREADS_PER_BLOCK + store] << d_AF[i*THREADS_PER_BLOCK + tx], d_AS[i*THREADS_PER_BLOCK + store] << d_AF[i*THREADS_PER_BLOCK + tx], d_AW[i*THREADS_PER_BLOCK + tx], d_AS[i*THREADS_PER_BLOCK + tx]);

		AF[i][tx] = tempF;
		AS[i][tx] = tempS;
		AW[i][tx] = tempW;

		R[tx] = AF[i][tx] + AS[i][tx] + AW[i][tx];
	}
}

unsigned int charToUInt(char c)
{
	return (unsigned int) (c - ALPHABET_INITIAL);
}

char UintToChar(unsigned int i)
{
	// return (char) (i);
	// printf("Hi\n");

	// printf("%u\n", (i + charToUInt(ALPHABET_INITIAL)));
	return (char)(i) + ALPHABET_INITIAL;
}

char bringInRange(char c)
{
	if(c > ALPHABET_FINAL || c < ALPHABET_INITIAL)
	{
		// cout << c << " " << charToUInt(c) << endl;
		return ' ';
	}
	return c;
}

void mapStringToInt(char input[], unsigned int converted[], size_t length)
{
	for (int i = 0; i < length; ++i)
	{
		if(input[i] > ALPHABET_FINAL || input[i] < ALPHABET_INITIAL)
		{
			printf("Error: String contains invalid characters\n");
			exit(0);
		}
		converted[i] = charToUInt(input[i]);
	}
	return;
}

int main(int argc, const char **argv)
{
	#ifndef DEBUG
	if(argc != 5)
	{
		printf("Usage: %s <path/to/text/file> <path/to/pattern/file> <k = max error> <number of streams>\n", argv[0]);
		exit(0);
	}
	#endif

	#ifndef DEBUG
		FILE *t_fp = fopen(argv[1],"r");
	#else
		FILE *t_fp = fopen("data/t_vvl.txt", "r");
	#endif
	if (!t_fp)
	{
		printf("Unable to open text file.\n");
		exit(0);
	}

	#ifndef DEBUG
		FILE *p_fp = fopen(argv[2],"r");
	#else
		FILE *p_fp = fopen("data/p_sample.txt", "r");
	#endif
	if (!p_fp)
	{
		printf("Unable to open pattern file.\n");
		exit(0);
	}

	int k, STREAM_COUNT;

	stringstream strValue;
	strValue << argv[3];
	strValue >> k;

	stringstream strValue2;
	strValue2 << argv[4];
	strValue2 >> STREAM_COUNT;

	size_t t_len = 0, p_len = 0;
	while (getc(t_fp) != EOF)
	{
		t_len++;
	}
	rewind(t_fp);

	while (getc(p_fp) != EOF)
	{
		p_len++;
	}
	rewind(p_fp);

	t_len -= 1;
	p_len -= 1;

	// cout << p_len << " " << t_len<< endl;

	char *text = (char *) malloc(t_len);
	char *pattern = (char *) malloc(p_len);

	for (int l = 0; l < p_len; l++)
	{
		pattern[l] = bringInRange(getc(p_fp));
	}

	for (int l = 0; l < t_len; l++)
	{
		text[l] = bringInRange(getc(t_fp));
	}

	fclose(t_fp);
	fclose(p_fp);

	// cout << t_len << endl;
	// cout << text << endl;
	// cout << p_len << endl;
	// cout << pattern << endl;

	unsigned int **AF = new unsigned int*[MAX_K+1];
	unsigned int **AS = new unsigned int*[MAX_K+1];
	unsigned int **AW = new unsigned int*[MAX_K+1];

	for(int i = 0; i < MAX_K+1; i++)
	{
		AF[i] = new unsigned int[t_len];
		AS[i] = new unsigned int[t_len];
		AW[i] = new unsigned int[t_len];
	}

	unsigned int* convText = new unsigned int[t_len];
	mapStringToInt(text, convText, t_len);

	unsigned int convPattern[p_len];
	mapStringToInt(pattern, convPattern, p_len);

	free(text);
	free(pattern);

	if(p_len > WORD)
	{
		perror("Error: Use pattern length <= word size");
		return 0;
	}

	/****** GPU Execution ********/
	unsigned int* d_AF;
	unsigned int* d_AS;
	unsigned int* d_AW;
	unsigned int* d_convText;
	unsigned int* d_convPattern;
	unsigned int* R;

	cudaMalloc((void**)&d_AF, k * t_len * sizeof(unsigned int *));
	cudaMalloc((void**)&d_AS, k * t_len * sizeof(unsigned int *));
	cudaMalloc((void**)&d_AW, k * t_len * sizeof(unsigned int *));
	cudaMalloc((void**)&R, k * t_len * sizeof(unsigned int));

	cudaMalloc(&d_convText, t_len/STREAM_COUNT * sizeof(unsigned int));
	cudaMalloc(&d_convPattern, p_len * sizeof(unsigned int));

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaStream_t streams[STREAM_COUNT + 1];

	for(int i = 1; i <= STREAM_COUNT; i++)
	{
		cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
		cudaMemcpyAsync(d_convText, convText + STREAM_COUNT * i, t_len/STREAM_COUNT * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]);
		wuManber_GPU <<<(t_len/(THREADS_PER_BLOCK * STREAM_COUNT)) + 1, THREADS_PER_BLOCK, 0, streams[i]>>>(d_convText, t_len/STREAM_COUNT, d_convPattern, p_len, k, d_AF, d_AS, d_AW, i);
	}

	cudaStreamCreateWithFlags(&streams[STREAM_COUNT], cudaStreamNonBlocking);
	cudaMemcpyAsync(d_convText, convText + (STREAM_COUNT * STREAM_COUNT), t_len%STREAM_COUNT * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[STREAM_COUNT]);
	wuManber_GPU <<<((t_len%STREAM_COUNT)/(THREADS_PER_BLOCK)) + 1, THREADS_PER_BLOCK, 0, streams[STREAM_COUNT]>>>(d_convText, t_len%STREAM_COUNT, d_convPattern, p_len, k, d_AF, d_AS, d_AW, STREAM_COUNT);

	cudaDeviceSynchronize();

	for(int i = 1; i <= STREAM_COUNT; i++)
	{
		wuManber_halo_GPU <<<(t_len/(THREADS_PER_BLOCK * STREAM_COUNT)) + 1, THREADS_PER_BLOCK, 0, streams[i]>>>(t_len/STREAM_COUNT, k, d_AF, d_AS, d_AW, R, i-1);
	}

	wuManber_halo_GPU <<<((t_len%STREAM_COUNT)/(THREADS_PER_BLOCK)) + 1, THREADS_PER_BLOCK, 0, streams[STREAM_COUNT]>>>(t_len%STREAM_COUNT, k, d_AF, d_AS, d_AW, R, STREAM_COUNT);

	cudaDeviceSynchronize();

	for(int i = 1; i <= STREAM_COUNT; i++)
		cudaStreamDestroy(streams[i]);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);

	// printf("GPU found %d matches	\n", count);
	printf("GPU Total Time for matching keywords: %fms\n", elapsedTime);


	delete [] convText;
	delete [] AF;
	delete [] AS;
	delete [] AW;

	cudaFree(d_convText);
	cudaFree(d_convPattern);
	// cudaFree(d_M);
	// cudaFree(d_AF);
	// cudaFree(d_AS);

	return 0;
}
