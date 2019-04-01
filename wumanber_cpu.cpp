#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <bitset>

using namespace std;

// #define DEBUG

#define WORD 32
#define ALPHABET_INITIAL ' '
#define ALPHABET_FINAL 'z'
#define ASIZE (int) (ALPHABET_FINAL - ALPHABET_INITIAL + 1)
// #define k 1
#define shift(x) 1 << (31 - x)
#define CHAR_MAX 255

char bringInRange(char c)
{
	if(c > ALPHABET_FINAL || c < ALPHABET_INITIAL)
	{
		// cout << c << " " << charToUInt(c) << endl;
		return ' ';
	}
	return c;
}

void WuManber(char *pattern, int p_len, char *text, int t_len, int k)
{
	// cout << t_len << endl;
	// cout << text << endl;
	// cout << p_len << endl;
	// cout << pattern << endl;
	// cout << cas << endl;

	if(p_len > WORD - 1)
	{
		perror("Error: Use pattern length <= word size");
		return;
	}

	unsigned long alphabets[256];

	for(int i = 0; i < 256; i++)
	{
		alphabets[i] = ~0;
	}

	for(int i = 0 ; i < p_len; i++)
		alphabets[pattern[i]] = alphabets[pattern[i]] & ~(1UL << i);
	
	unsigned long R[k+1];

	for(int i = 0; i <= k; i++)
		R[i] = ~1;

	unsigned long temptextstore, temperrorstore;

	for (int i = 0; i < t_len; i++)
	{
		temptextstore = R[0];

		R[0] = (R[0] | alphabets[text[i]]) << 1;

		for(int j = 1; j <= k; j++)
		{
			temperrorstore = R[j];

			R[j] |= alphabets[text[i]];
			R[j] &= temptextstore;
			R[j] <<= 1;

			temptextstore = (((R[j] << 1) | shift(0)) & alphabets[text[i]]) |
					  (R[j-1] |
					  (R[j] << 1) |
					  (R[j-1] << 1)) ;

			temptextstore ^= (temperrorstore ^ temptextstore);

			temptextstore = temperrorstore;
		}

		temptextstore = 1UL << p_len;
		temptextstore &= R[k];

		if((~temptextstore) == -1)
			cout<<(i + 1 - p_len)<<endl;
	}
}

int main(int argc, char const *argv[])
{
	if(argc != 3)
	{
		printf("Usage: %s <path/to/text/file> <path/to/pattern/file>\n", argv[0]);
		exit(0);
	}

	FILE *t_fp = fopen(argv[1],"r");
	if (!t_fp)
	{
		printf("Unable to open text file.\n");
		exit(0);
	}

	FILE *p_fp = fopen(argv[2],"r");
	if (!p_fp)
	{
		printf("Unable to open pattern file.\n");
		exit(0);
	}

	size_t t_len = -1, p_len = -1;
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

	// unsigned int* convText = new unsigned int[t_len];
	// mapStringToInt(text, convText, t_len);

	// unsigned int convPattern[p_len];
	// mapStringToInt(pattern, convPattern, p_len);

	WuManber(pattern, p_len, text, t_len, 1);

	free(text);
	free(pattern);

	// delete [] convText;

	return 0;
}
