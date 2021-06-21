#include <stdio.h>

#include "parameters.h"

extern float GPU_kernel(float *B, float *A, IndexSave *indsave);

void genNumbers(float *A, float *B, int size)
{

	for (int i = 0; i < size; i++)
	{
		A[i] = 1.0 * ((rand() % 256) / 256.0);
		B[i] = 1.0 * ((rand() % 256) / 256.0);
	}
}

void function_1(float *B, float *A, float *C)
{

	for (int i = 0; i < SIZE; i++)
	{
		C[i] = (B[i] - A[i]) * (B[i] - A[i]);
	}
}

bool verify(float a, float b)
{
	if (a != b)
		return true;
	return false;
}

void printIndex(IndexSave *indsave, float *B, float *C, float *a, float *b)
{
	for (int i = 0; i < SIZE; i++)
	{
		printf("%d,im here4!\n", i);
		printf("%d : blockInd_x=%d,threadInd_x=%d,head=%d,stripe=%d", i, (indsave[i]).blockInd_x, (indsave[i]).threadInd_x, (indsave[i]).head, (indsave[i]).stripe);
		printf(" || GPU result=%f,CPU result=%f\n", B[i], C[i]);
		*a += C[i];
		*b += B[i];
	}
}

int main()

{
	// random seed
	float *A = new float[SIZE];
	// random number sequence computed by GPU
	float *B = new float[SIZE];
	// random number sequence computed by CPU
	float *C = new float[SIZE];
	// Indices saver (for checking correctness)
	IndexSave *indsave = new IndexSave[SIZE];

	genNumbers(A, B, SIZE);

	/* CPU side*/
	function_1(B, A, C);

	/* GPU side*/
	float elapsedTime = GPU_kernel(B, A, indsave);
	float lossc = 0;
	float lossg = 0;

	/*Show threads execution info*/
	printIndex(indsave, B, C, &lossc, &lossg);

	printf("==============================================\n");
	/* verify the result*/
	if (verify(lossg, lossc))
	{
		printf("wrong answer\n");
	}
	printf("GPU time = %5.2f ms\n", elapsedTime);

	/*Please press any key to exit the program*/
	getchar();
}
