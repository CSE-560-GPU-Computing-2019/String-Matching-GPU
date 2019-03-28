#include <iostream>
#include <stdio.h>
#include <string>
#include <assert.h>
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

void mapStringToInt(char input[], unsigned int converted[], size_t length)
{
	// printf("%lu\n", length);

	for (int i = 0; i < length; ++i)
	{
		assert(input[i] <= ALPHABET_FINAL);
		converted[i] = charToUInt(input[i]);

		// printf("%u\n", converted[i]);
	}
	return;
}

void preSO(unsigned int pattern[], int p_len, unsigned int S[])
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

void shiftOR(unsigned int pattern[], int p_len, unsigned int text[], int t_len)
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
	char text[] = "test string is the best";
	size_t t_len = sizeof(text)/sizeof(char) - 1;

	char pattern[] = "est";
	size_t p_len = sizeof(pattern)/sizeof(char) - 1;

	unsigned int convText[t_len];
	mapStringToInt(text, convText, t_len);

	unsigned int convPattern[p_len];
	mapStringToInt(pattern, convPattern, p_len);

	shiftOR(convPattern, p_len, convText, t_len);
	return 0;
}
