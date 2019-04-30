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
// #define STREAM_COUNT 20

__device__ unsigned int dagger1(unsigned int u1, unsigned int x1, unsigned int u2, unsigned int x2)
{
	return u1 + u2;
}

__device__ unsigned int dagger2(unsigned int u1, unsigned int x1, unsigned int u2, unsigned int x2)
{
	return (x1 << u2) | x2;
}

__global__ void shiftOR_GPU(unsigned int *convText, int t_len, unsigned int *convPattern, int p_len, unsigned int *d_AS, unsigned int *d_AF, int pos)
{
	unsigned int src_index = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned int shared_convPattern[MAX_P_LEN];
	__shared__ int AF[THREADS_PER_BLOCK];
	__shared__ int AS[THREADS_PER_BLOCK];

	int tx = threadIdx.x;

	if(tx < p_len)
		shared_convPattern[tx] = convPattern[tx];

	__syncthreads();

	AF[tx] = 1;
	AS[tx] = 0;

	for (int i = 0; i < p_len; ++i)
	{
		if(convText[src_index] == shared_convPattern[i])
			AS[tx] |= 1 << i;
	}

	AS[tx] = ~AS[tx];

	if(tx == 0 && blockIdx.x == 0)
	{
		AF[tx] = 0;
		AS[tx] = ~0;
	}

	__syncthreads();

	int stride = 1;
	while(stride < THREADS_PER_BLOCK)
	{
		int index = (tx+1)*stride*2 - 1;
		if(index < THREADS_PER_BLOCK)
		{
			unsigned int tempF = dagger1(AF[index], AS[index], AF[index-stride], AS[index-stride]);
			unsigned int tempS = dagger2(AF[index], AS[index], AF[index-stride], AS[index-stride]);

			AF[index] = tempF;
			AS[index] = tempS;
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
			unsigned int tempF = dagger1(AF[index+stride], AS[index+stride], AF[index], AS[index]);
			unsigned int tempS = dagger2(AF[index+stride], AS[index+stride], AF[index], AS[index]);

			AF[index] = tempF;
			AS[index] = tempS;
		}
		stride = stride / 2;
		__syncthreads();
	}

	d_AF[src_index] = AF[tx];
	d_AS[src_index] = AS[tx];
}

__global__ void shiftOR_halo_GPU(int t_len, unsigned int *d_AF, unsigned int *d_AS, unsigned int *R, int pos)
{
	__shared__ unsigned int AF[THREADS_PER_BLOCK];
	__shared__ unsigned int AS[THREADS_PER_BLOCK];
	int tx = threadIdx.x;

	unsigned int tempF = dagger1(d_AF[tx], d_AS[tx], d_AF[tx], d_AS[tx]);
	unsigned int tempS = dagger2(d_AF[tx], d_AS[tx], d_AF[tx], d_AS[tx]);

	AF[tx] = tempF;
	AS[tx] = tempS;

	R[tx] = AF[tx] + AS[tx];
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

int countZero(unsigned int *M, int t_len, int count)
{
	for (int i = 0; i < t_len; ++i)
	{
		if(!M[i])
			count++;
	}
	return count;
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

void preSO(unsigned int *pattern, int p_len, unsigned int *S)
{
	for (int i = 0; i < ASIZE; ++i)
	{
		S[i] = 0;
	}

	for (int i = 0; i < p_len; ++i)
	{
		S[pattern[i]] |= 1 << i;
	}

	for (int i = 0; i < ASIZE; ++i)
	{
		S[i] = ~S[i];
	}

	return;
}

int shiftOR(unsigned int *pattern, int p_len, unsigned int *text, int t_len)
{
	unsigned int state;
	unsigned int S[ASIZE];
	int hit;

	unsigned int ctr = 0;

	/* pre-processing */
	preSO(pattern, p_len, S);

	#ifdef DEBUG
		cout << "Pre-processing Done\n";
	#endif

	/* searching */
	state = ~0;
	hit = (1 << (p_len - 1));
	for (int i = 0; i < t_len; ++i)
	{
		state = ((state << 1) & ~0) | S[text[i]];

		#ifdef DEBUG
			// cout << bitset<32>(state) << " & ["  << UintToChar(text[i]) << "] : " << bitset<32>(S[text[i]]) << endl;
		#endif

		if(!(state & hit))
		{
			// cout << "Found at position " <<  i - p_len + 1 << endl;
			ctr++;
		}
	}
	return ctr;
}

int main(int argc, const char **argv)
{
	#ifndef DEBUG
		if(argc != 4)
		{
			printf("Usage: %s <path/to/text/file> <path/to/pattern/file> <number of streams>\n", argv[0]);
			exit(0);
		}
	#endif

	#ifndef DEBUG
		FILE *t_fp = fopen(argv[1],"r");
	#else
		FILE *t_fp = fopen("data/t_l.txt", "r");
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

	// const char* value = "1234567";
	stringstream strValue;
	strValue << argv[3];

	int STREAM_COUNT;
	strValue >> STREAM_COUNT;

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

	unsigned int* M = new unsigned int[t_len];
	unsigned int* AF = new unsigned int[t_len];
	unsigned int* AS = new unsigned int[t_len];

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

	/****** CPU Execution ********/
	const clock_t begin_time = clock();
	int count = shiftOR(convPattern, p_len, convText, t_len);
	float runTime = (float)( clock() - begin_time ) /  CLOCKS_PER_SEC;

	printf("CPU found %d matches\n", count);
	printf("CPU Time for matching keywords: %fms\n\n", runTime*1000);


	/****** GPU Execution ********/
	unsigned int* d_AF;
	unsigned int* d_AS;
	unsigned int* d_convText;
	unsigned int* d_convPattern;
	unsigned int* R;

	cudaMalloc(&d_AF, t_len * sizeof(unsigned int));
	cudaMalloc(&d_AS, t_len * sizeof(unsigned int));
	cudaMalloc(&R, t_len * sizeof(unsigned int));

	cudaMalloc(&d_convText, t_len/STREAM_COUNT * sizeof(unsigned int));
	cudaMalloc(&d_convPattern, p_len * sizeof(unsigned int));

	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaStream_t streams[STREAM_COUNT + 1];

	cudaHostRegister(convText, t_len * sizeof(unsigned int), 0);

	for(int i = 1; i <= STREAM_COUNT; i++)
	{
		cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
		cudaMemcpyAsync(d_convText, convText + STREAM_COUNT * i, t_len/STREAM_COUNT * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]);
		shiftOR_GPU <<<(t_len/(THREADS_PER_BLOCK * STREAM_COUNT)) + 1, THREADS_PER_BLOCK, 0, streams[i]>>>(d_convText, t_len/STREAM_COUNT, d_convPattern, p_len, d_AF, d_AS, i);
	}

	cudaStreamCreateWithFlags(&streams[STREAM_COUNT], cudaStreamNonBlocking);
	cudaMemcpyAsync(d_convText, convText + (STREAM_COUNT * STREAM_COUNT), t_len%STREAM_COUNT * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[STREAM_COUNT]);
	shiftOR_GPU <<<((t_len%STREAM_COUNT)/(THREADS_PER_BLOCK)) + 1, THREADS_PER_BLOCK, 0, streams[STREAM_COUNT]>>>(d_convText, t_len%STREAM_COUNT, d_convPattern, p_len, d_AF, d_AS, STREAM_COUNT);

	cudaDeviceSynchronize();

	for(int i = 1; i <= STREAM_COUNT; i++)
	{
		shiftOR_halo_GPU <<<(t_len/(THREADS_PER_BLOCK * STREAM_COUNT)) + 1, THREADS_PER_BLOCK, 0, streams[i]>>>(t_len/STREAM_COUNT, d_AF, d_AS, R, i-1);
	}

	shiftOR_halo_GPU <<<((t_len%STREAM_COUNT)/(THREADS_PER_BLOCK)) + 1, THREADS_PER_BLOCK, 0, streams[STREAM_COUNT]>>>(t_len%STREAM_COUNT, d_AF, d_AS, R, STREAM_COUNT);

	cudaDeviceSynchronize();

	for(int i = 1; i <= STREAM_COUNT; i++)
		cudaStreamDestroy(streams[i]);

	cudaHostUnregister(convText);

	// cudaMemcpy(convText, d_convText, t_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(M, R, t_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	// countZero(M, t_len, count);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);

	printf("GPU found %d matches	\n", count);
	printf("GPU Total Time for matching keywords: %fms\n", elapsedTime);


	delete [] convText;
	delete [] M;
	delete [] AF;
	delete [] AS;

	cudaFree(d_convText);
	cudaFree(d_convPattern);

	return 0;
}
