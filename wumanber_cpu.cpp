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
#define k 3
#define shift(x) 1 << (31 - x)

char bringInRange(char c)
{
	if(c > ALPHABET_FINAL || c < ALPHABET_INITIAL)
	{
		// cout << c << " " << charToUInt(c) << endl;
		return ' ';
	}
	return c;
}

void WuManber(char *pattern, int p_len, char *text, int t_len, int cas)
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

	int alphabets[256];

	if(cas == 1)
	{
		for(int i = 0; i < p_len; i++)
		{
			alphabets[pattern[i]] = alphabets[pattern[i]] | shift(i);
		}
	}
	else
	{
		for(int i = 0; i < p_len; i++)
		{
			if(pattern[i] >= 'A' && pattern[i] <= 'Z')
				alphabets[(pattern[i] - 'A' + 'a')] = alphabets[(pattern[i] - 'A' + 'a')] | shift(i);
			else
				alphabets[pattern[i]] = alphabets[pattern[i]] | shift(i);
		}		
	}

	int R[k+1][t_len+1];

	for(int i = 0; i <= k; i++)
		R[i][0] = 0;

	for(int i = 1; i <= t_len; i++)
		R[0][i] = alphabets[text[i-1]] & ((R[0][i-1] << 1 ) | shift(0));

	for(int i = 1; i <= k; i++)
	{
		// R[i][0] |= shift(i);

		for(int j = 0; j < i; j++)
			R[j][0] = R[j][0] | shift(j);
	}

	for (int i = 1; i <= k; i++)
	{
		for(int j = 1; j <= t_len; j++)
		{
			R[i][j] = (((R[i][j-1] << 1) | shift(0)) & alphabets[text[j-1]]) |
					  (R[i-1][j-1] |
					  (R[i-1][j] << 1) |
					  (R[i-1][j-1] << 1)) ;
		}
	}

	for(int i = 0; i <= t_len; i++)
		cout<<R[k][i]<<" "<<text[i]<<endl;
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
	p_len++;
	t_len++;

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

	int cas;

	cout<<"Case Sensitive(Yes - 1, No - 0): ";

	scanf("%d", &cas);

	WuManber(pattern, p_len, text, t_len, cas);

	free(text);
	free(pattern);

	return 0;
}
