#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <bitset>

using namespace std;

#define DEBUG

#define WORD 32
#define ALPHABET_INITIAL ' '
#define ALPHABET_FINAL 'z'
#define ASIZE (int) (ALPHABET_FINAL - ALPHABET_INITIAL + 1)

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
	// printf("%lu\n", length);

	for (int i = 0; i < length; ++i)
	{
		if(input[i] > ALPHABET_FINAL || input[i] < ALPHABET_INITIAL)
		{
			printf("Error: String contains invalid characters\n");
			exit(0);
		}
		converted[i] = charToUInt(input[i]);

		// printf("%u\n", converted[i]);
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

void shiftOR(unsigned int *pattern, int p_len, unsigned int *text, int t_len)
{
	unsigned int state;
	unsigned int S[ASIZE];
	int hit;

	if(p_len > WORD)
	{
		perror("Error: Use pattern length <= word size");
		return;
	}

	/* pre-processing */
	preSO(pattern, p_len, S);

	/* searching */
	state = ~0;
	hit = (1 << (p_len - 1));
	for (int i = 0; i < t_len; ++i)
	{
		state = ((state << 1) & ~0) | S[text[i]];

		#ifdef DEBUG
			cout << bitset<32>(state) << " & ["  << UintToChar(text[i]) << "] : " << bitset<32>(S[text[i]]) << endl;
		#endif

		if(!(state & hit))
		{
			cout << "Found at position " <<  i - p_len + 1 << endl;
		}
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

	cout << t_len << endl;
	cout << text << endl;
	cout << p_len << endl;
	cout << pattern << endl;

	unsigned int convText[t_len];
	mapStringToInt(text, convText, t_len);

	unsigned int convPattern[p_len];
	mapStringToInt(pattern, convPattern, p_len);

	free(text);
	free(pattern);

	shiftOR(convPattern, p_len, convText, t_len);

	return 0;
}
