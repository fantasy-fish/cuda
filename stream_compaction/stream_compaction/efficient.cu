#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__
        __global__ void kernUpSweep(int n, int d, int* idata)
        {
            int k = blockIdx.x*blockDim.x+threadIdx.x;
            if(k>=n)
                return;
            int step = powf(2.0,1.0*(d+1));
            if(k%step==0)
            {
                if(k+step-1<n)
                    idata[k+step-1] += idata[k+step/2-1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* idata)
        {
            int k = blockIdx.x*blockDim.x+threadIdx.x;
            if(k>=n)
                return;
            int step = powf(2.0,1.0*(d+1));
            if(k%step==0)
            {
                if(k+step-1<n)
                {
                    int t = idata[k+step/2-1];
                    idata[k+step/2-1] = idata[k+step-1];
                    idata[k+step-1] += t;
                }
            }
        }

        __global__ void kernSetElement(int* idata, int index, int value)
        {
            idata[index] = value;
        }

        __global__ void Init(int n, int* data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            data[index] = 0;
        }

        bool IsPowerOfTwo(int number) {
            if (number == 0)
                return false;
            for (int power = 1; power > 0; power = power << 1) 
            {
                if (power == number)
                    return true;
                if (power > number)
                    return false;
            }
            return false;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_idata;
            int blockSize = 16;
            int pow2n = n;
            if(!IsPowerOfTwo(n))
                pow2n = static_cast<int>(pow(2.0, 1.0 * (ilog2ceil(n) + 1)));
            dim3 fullBlocksPerGrid((pow2n+blockSize-1)/blockSize);

            cudaMalloc((void**)&dev_idata, pow2n*sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
            Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_idata);
            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            //checkCUDAErrorWithLine("copy idata failed!");
            
            timer().startGpuTimer();
            // TODO
            for(int d=0;d<ilog2ceil(pow2n);d++)
                kernUpSweep<<<blockSize,fullBlocksPerGrid>>>(pow2n,d,dev_idata);
            kernSetElement<<<dim3(1),dim3(1)>>>(dev_idata,pow2n-1,0);
             for(int d=ilog2ceil(pow2n)-1;d>=0;d--)
                kernDownSweep<<<blockSize,fullBlocksPerGrid>>>(pow2n,d,dev_idata);
            //from inclusive to exclusive
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
            //checkCUDAErrorWithLine("get odata failed!");
            cudaFree(dev_idata); 

        }

        void scan_notimer(int n, int *odata, const int *idata) {
            int* dev_idata;
            int blockSize = 16;
            int pow2n = n;
            if(!IsPowerOfTwo(n))
                pow2n = static_cast<int>(pow(2.0, 1.0 * (ilog2ceil(n) + 1)));
            dim3 fullBlocksPerGrid((pow2n+blockSize-1)/blockSize);

            cudaMalloc((void**)&dev_idata, pow2n*sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
            Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_idata);
            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            //checkCUDAErrorWithLine("copy idata failed!");
            
            //timer().startGpuTimer();
            // TODO
            for(int d=0;d<ilog2ceil(pow2n);d++)
                kernUpSweep<<<blockSize,fullBlocksPerGrid>>>(pow2n,d,dev_idata);
            kernSetElement<<<dim3(1),dim3(1)>>>(dev_idata,pow2n-1,0);
             for(int d=ilog2ceil(pow2n)-1;d>=0;d--)
                kernDownSweep<<<blockSize,fullBlocksPerGrid>>>(pow2n,d,dev_idata);
            //from inclusive to exclusive
            //timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n*sizeof(int), cudaMemcpyDeviceToHost);
            //checkCUDAErrorWithLine("get odata failed!");
            cudaFree(dev_idata); 

        }
        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int blockSize = 16;
            int* dev_idata;
            int* dev_bool;
            int* scan_out;
            int* dev_odata;

            int pow2n = n;
            if (!IsPowerOfTwo(n)) {
                pow2n = static_cast<int>(pow(2.0, 1.0 * (ilog2ceil(n) + 1)));
            }
            dim3 fullBlocksPerGrid((pow2n + blockSize - 1) / blockSize);

            cudaMalloc((void**)&dev_idata, pow2n*sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
            Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_idata);
            cudaMalloc((void**)&dev_bool, pow2n*sizeof(int));
            //checkCUDAErrorWithLine("cudaMalloc dev_bool failed!");
            Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_bool);
            cudaMalloc((void**)&scan_out, pow2n * sizeof(int));
            //checkCUDAError("cudaMalloc scan_out failed!");
            Init << <fullBlocksPerGrid, blockSize >> > (pow2n, scan_out);
            cudaMalloc((void**)&dev_odata, pow2n * sizeof(int));
            //checkCUDAError("cudaMalloc dev_odata failed!");
            Init << <fullBlocksPerGrid, blockSize >> > (pow2n, dev_odata);

            cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
            //checkCUDAError("cudaMemcpy dev_idata failed!");

            timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean<<<blockSize,fullBlocksPerGrid>>>(pow2n,dev_bool,dev_idata);
            scan_notimer(pow2n, scan_out, dev_bool);
            Common::kernScatter<<<blockSize,fullBlocksPerGrid>>>(pow2n, dev_odata, dev_idata, dev_bool, scan_out);
            timer().endGpuTimer();
            
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
            //checkCUDAError("get odata failed!");

            int count = -1;
            for (int i = 0; i < n; i++) {
                if (odata[i] == 0) {
                    count = i;
                    break;
                }
            }
            cudaFree(dev_idata); cudaFree(dev_bool);
            cudaFree(dev_odata); cudaFree(scan_out);
            return count;
        }
    }
}
