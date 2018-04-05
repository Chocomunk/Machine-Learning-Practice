#include "stdafx.h"
#include "CppUnitTest.h"
#include "../Machine Learning Practice/cuda_wrapper.h"
#include "testUtil.h"

#define TEST_METHOD_PRIORITY(methodName, priority) \
BEGIN_TEST_METHOD_ATTRIBUTE(methodName) \
	TEST_PRIORITY(priority) \
END_TEST_METHOD_ATTRIBUTE() \
TEST_METHOD(methodName)

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MachineLearningPracticeTests
{		
	TEST_CLASS(MatAddTest)
	{
	public:

		static const int TRIALS = 1;
		static const int N_TDIM = 1000;
		float* inVec = new float[N_TDIM];
		float* outVec = new float[N_TDIM];

		MatAddTest() {
			for (int i = 0; i < N_TDIM; i++) {
				inVec[i] = i;
				outVec[i] = 2 * i;
			}
		}
		~MatAddTest() {
			delete outVec;
			delete inVec;
		}

		TEST_METHOD_PRIORITY(MatAddGPUFirst, 1)
		{
			float* a = new float[2]{ 1,0 };
			float* b = new float[2]{ 0,1 };
			float* out = new float[2]{ 1,1 };
			float* c;
			c = matAdd(c_dim3(2), a, b);
			Assert::IsTrue(vecIsEqual(2, c, out));
			delete a;
			delete b;
			delete c;
			delete out;
		}
		
		TEST_METHOD_PRIORITY(MatAddGPUTest, 2)
		{
			for(int a=0; a<TRIALS; a++){
				float* c;
				c = matAdd(c_dim3(N_TDIM), inVec, inVec);
				Assert::IsTrue(vecIsEqual(N_TDIM, c, outVec));
				delete c;
			}
		}

		TEST_METHOD_PRIORITY(MatAddCPUTest, 3)
		{
			for (int a = 0; a < TRIALS; a++) {
				float* c = new float[N_TDIM];
				for (int j = 0; j <N_TDIM; j++) {
					c[j] = inVec[j] + inVec[j];
				}
				Assert::IsTrue(vecIsEqual(N_TDIM, c, outVec));
				delete c;
			}
		}

	};

	TEST_CLASS(MatDotTest)
	{
	public:

		static const int TRIALS = 1;
		static const int N_TDIM = 1000;
		float* inVec = new float[N_TDIM];
		float* outVec = new float[N_TDIM];

		MatDotTest() {
			for (int i = 0; i < N_TDIM; i++) {
				inVec[i] = i;
				outVec[i] = i * i;
			}
		}
		~MatDotTest() {
			delete outVec;
			delete inVec;
		}

		TEST_METHOD_PRIORITY(MatDotGPUFirst, 1)
		{
			float* a = new float[2]{ 1,0 };
			float* b = new float[2]{ 0,1 };
			float* out = new float[2]{ 0,0 };
			float* c;
			c = matDot(c_dim3(2), a, b);
			Assert::IsTrue(vecIsEqual(2, c, out));
			delete a;
			delete b;
			delete c;
			delete out;
		}
		
		TEST_METHOD_PRIORITY(MatDotGPUTest, 2)
		{
			for(int a=0; a<TRIALS; a++){
				float *c = matDot(c_dim3(N_TDIM), inVec, inVec);
				Assert::IsTrue(vecIsEqual(N_TDIM, c, outVec));
				delete c;
			}
		}

		TEST_METHOD_PRIORITY(MatDotCPUTest, 3)
		{
			for (int a = 0; a < TRIALS; a++) {
				float* c = new float[N_TDIM];
				for (int j = 0; j <N_TDIM; j++) {
					c[j] = inVec[j] * inVec[j];
				}
				Assert::IsTrue(vecIsEqual(N_TDIM, c, outVec));
				delete c;
			}
		}

	};
}