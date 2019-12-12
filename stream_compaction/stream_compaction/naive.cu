#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int* odata, int* idata)
        {
            int k = blockIdx.x*blockDim.x+threadIdx.x;
            if(k>=n)
                return;
            int start = powf(2.0,1.0*(d-1));
            if(k>=start)
                odata[k] = idata[k-start]+idata[k];
            else
                odata[k] = idata[k];
        }

         __global__ void kernInclu2Exclu(int n, int* odata, int* idata)
         {
            int k = blockIdx.x*blockDim.x+threadIdx.x;
            if(k>=n)
                return;
           if(k==0)
                odata[0] = 0;
            else
                odata[k] = idata[k-1];
         }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            int* tmp;
            int blockSize = 16;
            cudaMalloc((void**)&dev_idata, n*sizeof(int));
            cudaMalloc((void**)&dev_odata, n*sizeof(int));
            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            //checkCUDAErrorWithLine("copy idata failed!");
            cudaMemcpy(dev_odata, odata, n*sizeof(int), cudaMemcpyHostToDevice);
            //checkCUDAErrorWithLine("copy odata failed!");
            dim3 fullBlocksPerGrid((n+blockSize-1)/blockSize);

            timer().startGpuTimer();
            // TODO
            for(int d=1;d<=ilog2ceil(n);d++)
            {
                kernNaiveScan<<<blockSize,fullBlocksPerGrid>>>(n,d,dev_odata,dev_idata);
                tmp = dev_odata;
                dev_odata = dev_idata;
                dev_idata = tmp;
            }
            //from inclusive to exclusive
            kernInclu2Exclu<<<blockSize,fullBlocksPerGrid>>>(n,dev_odata,dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
            //checkCUDAErrorWithLine("get odata failed!");
            cudaFree(dev_idata); cudaFree(dev_odata);
        }
    }
}
