#include <iostream>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for(int k=1;k<n;k++)
                odata[k] = odata[k-1]+idata[k-1];
	        timer().endCpuTimer();
        }

        void scan_notimer(int n, int *odata, const int *idata) {
            // cpu scan without timer
            odata[0] = 0;
            for(int k=1;k<n;k++)
                odata[k] = odata[k-1]+idata[k-1];
        }
        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
            int k=0;
            for(int i=0;i<n;i++)
            {
                if(idata[i]!=0)
                    odata[k++] = idata[i]; 
            }
	        timer().endCpuTimer();
            return k;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
	        // TODO
            int *tmp1 = new int[n];
            int *tmp2 = new int[n];
            for(int i=0;i<n;i++)
            {
                if(idata[i]!=0)
                    tmp1[i]=1; 
                else
                    tmp1[i]=0;
            }
            scan_notimer(n,tmp2,tmp1);

            for(int i =0; i<n; i++)
            {
                if(tmp1[i]==1)
                    odata[tmp2[i]]=idata[i];
            }
            int num_elt = tmp2[n-1];
	        timer().endCpuTimer();
            free(tmp2);
            free(tmp1);
            return num_elt;
        }
    }
}
