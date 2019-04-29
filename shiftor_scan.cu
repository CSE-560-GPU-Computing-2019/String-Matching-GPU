#include <iostream>
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

#define THREADS_PER_BLOCK 1024
#define MAX_P_LEN 32

__device__ unsigned int dagger1(unsigned int u1, unsigned int x1, unsigned int u2, unsigned int x2)
{
	return u1 + u2;
}

__device__ unsigned int dagger2(unsigned int u1, unsigned int x1, unsigned int u2, unsigned int x2)
{
	return (x1 << u2) | x2;
}

__global__ void shiftOR_GPU(unsigned int *convText, int t_len, unsigned int *convPattern, int p_len)
{
	unsigned int src_index = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned int shared_convPattern[MAX_P_LEN];
	__shared__ int AF[THREADS_PER_BLOCK];
	__shared__ int AS[THREADS_PER_BLOCK];

	if(threadIdx.x < p_len)
		shared_convPattern[threadIdx.x] = convPattern[threadIdx.x];

	__syncthreads();

	// printf("Hi. Pattern loaded.\n");

	// if(src_index > t_len)
		// printf("Thread: %d %d %d\n", threadIdx.x, src_index, t_len);

	AF[threadIdx.x] = 1;
	AS[threadIdx.x] = 0;

	for (int i = 0; i < p_len; ++i)
	{
		if(convText[src_index] == shared_convPattern[i])
			AS[threadIdx.x] |= 1 << i;
	}

	AS[threadIdx.x] = ~AS[threadIdx.x];

	if(threadIdx.x == 0 && blockIdx.x == 0)
	{
		AF[threadIdx.x] = 0;
		AS[threadIdx.x] = ~0;
	}

	// if (threadIdx.x == THREADS_PER_BLOCK - 1)
	// 	data[THREADS_PER_BLOCK * 1] = d_text[src_index + 1 * THREADS_PER_BLOCK - THREADS_PER_BLOCK + 1];

	__syncthreads();


	int stride = 1;
	while(stride < THREADS_PER_BLOCK)
	{
		int index = (threadIdx.x+1)*stride*2 - 1;
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
		int index = (threadIdx.x+1)*stride*2 - 1;
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

	// AF[src_index] = AF[threadIdx.x];
	// AS[src_index] = AS[threadIdx.x];
	// if(threadIdx.x < t_len)
	// 	printf("%d\n", AF[threadIdx.x]);
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
		if(argc != 3)
		{
			printf("Usage: %s <path/to/text/file> <path/to/pattern/file>\n", argv[0]);
			exit(0);
		}
	#endif

	#ifndef DEBUG
		FILE *t_fp = fopen(argv[1],"r");
	#else
		FILE *t_fp = fopen("data/t_s.txt", "r");
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
	// unsigned int* d_M;
	// unsigned int* d_AF;
	// unsigned int* d_AS;
	unsigned int* d_convText;
	unsigned int* d_convPattern;

	// cudaMalloc(&d_M, t_len * sizeof(unsigned int));
	// cudaMalloc(&d_AF, t_len * sizeof(unsigned int));
	// cudaMalloc(&d_AS, t_len * sizeof(unsigned int));

	cudaMalloc(&d_convText, t_len * sizeof(unsigned int));
	cudaMalloc(&d_convPattern, p_len * sizeof(unsigned int));

	cudaEvent_t start, stop;
	cudaEvent_t start_small, stop_small;
	float elapsedTime, elapsedTime_small;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaMemcpy(d_convText, convText, t_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_convPattern, convPattern, p_len * sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start_small);
	cudaEventCreate(&stop_small);
	cudaEventRecord(start_small,0);

	shiftOR_GPU <<<(t_len/THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(d_convText, t_len, d_convPattern, p_len);

	cudaEventRecord(stop_small,0);
	cudaEventSynchronize(stop_small);
	cudaEventElapsedTime(&elapsedTime_small, start_small,stop_small);

	// cudaMemcpy(M, d_convText, t_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	// countZero(M, t_len, count);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start,stop);

	printf("GPU found %d matches	\n", count);
	printf("GPU Kernel Time for matching keywords: %fms\n", elapsedTime_small);
	printf("GPU Total Time for matching keywords: %fms\n", elapsedTime);


	delete [] convText;
	delete [] M;
	delete [] AF;
	delete [] AS;

	cudaFree(d_convText);
	cudaFree(d_convPattern);
	// cudaFree(d_M);
	// cudaFree(d_AF);
	// cudaFree(d_AS);

	return 0;
}
