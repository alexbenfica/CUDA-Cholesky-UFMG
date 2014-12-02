/*  Device code for Cholesky decomposition. */
#ifndef _CHOL_KERNEL_H_
#define _CHOL_KERNEL_H_
#include "chol.h"


__global__ void chol_kernel_cudaUFMG_sqrt(float * U) {
    // Get a thread identifier 
    int tx = blockIdx.x * blockDim.x + threadIdx.x;    
    int tx_diag = tx * MATRIX_SIZE + tx;    
    U[tx_diag] = sqrt(U[tx_diag]);    
}





__global__ void chol_kernel_cudaUFMG_division(float * U, int elem_per_thr) {
    // Get a thread identifier 
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tn = ty * blockDim.x * gridDim.x + tx;    
    
    
    //#define DEBUGDIV

    #ifdef DEBUGDIV
    
    int dbg = 0;
    if(blockIdx.x == 4){
        if(blockIdx.y == 5){
            if(threadIdx.x == 2){
                if(threadIdx.y == 1){
                    dbg = 1;
                    printf("\n\n");
                    printf("\ntx=%d \nty=%d \ntn=%d", tx, ty, tn);
                }
            }
        }        
    }
    
    #endif
    
    
    for(unsigned i=0;i<elem_per_thr;i++){
        int iel = tn * elem_per_thr + i;
        int xval = iel % MATRIX_SIZE;
        int yval = iel / MATRIX_SIZE;

        if(xval == yval){        
            continue;
        }
        
        
        #ifdef     DEBUGDIV
        if(dbg == 1){
            if(i==37){
            printf("\niel=%d \nxval=%d \nyval=%d", iel, xval, yval);
            }
        }
        #endif

        
        // if on the lower diagonal...
        if(yval > xval){
            xval = MATRIX_SIZE - xval - 1;
            yval = MATRIX_SIZE - yval - 1;
        }
        
        
        int iU = xval + yval * MATRIX_SIZE;
        int iDiag = yval + yval * MATRIX_SIZE;
        
        #ifdef     DEBUGDIV
        if(dbg == 1){
            if(i==37){
                printf("\nxtrans=%d \nytrans=%d \niU=%d \niDiag=%d", xval, yval, iU, iDiag);
                printf("\n\n");
            }
        }
        #endif        
        
        U[iU] /= U[iDiag];                    
        
    }
    
}













__global__ void chol_kernel_cudaUFMG_elimination(float * U, int k) {

    //This call acts as a single K iteration
    //Each block does a single i iteration
    //Need to consider offset, 
    int i = (k+1) + blockIdx.x;
    
    //Each thread does some part of j
    //Stide in units of 'stride'
    //Thread 0 does 0, 16, 32
    //Thread 1 does 1, 17, 33
    //..etc.
    int jstart = i + threadIdx.x;
    int jstep = blockDim.x;
    
    // Pre-calculate indexes
    int kM = k * MATRIX_SIZE;
    int iM = i * MATRIX_SIZE;
    int ki = kM + i;
    
    // Fabricio... dá para otimizar tirando U[ki] pra fora do loop... o que acha?
    // Mas nao funciona! Ahh math!    
    
    //Do work for this i iteration
    //Want to stride across
    for (int j=jstart; j<MATRIX_SIZE; j+=jstep) {        
        U[iM + j] -= U[ki] * U[kM + j];
    }
}

    
    
    
    

































__global__ void chol_kernel_cudaUFMG_zero(float * U, int elem_per_thr) {
    // Get a thread identifier 
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tn = ty * blockDim.x * gridDim.x + tx;    
    
    for(unsigned i=0;i<elem_per_thr;i++){
        int iel = tn * elem_per_thr + i;
        int xval = iel % MATRIX_SIZE;
        int yval = iel / MATRIX_SIZE;

        if(xval == yval){        
            continue;
        }        
        
        // if on the upper diagonal...
        if(yval < xval){
            xval = MATRIX_SIZE - xval - 1;
            yval = MATRIX_SIZE - yval - 1;
        }
        int iU = xval + yval * MATRIX_SIZE;
        U[iU] = 0;                            
    }
    
}
































__global__ void chol_kernel(float * U, int ops_per_thread) {
    //Determine the boundaries for this thread
    //Get a thread identifier
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    //Iterators
    unsigned int i, j, k;
    //unsigned int size = MATRIX_SIZE*MATRIX_SIZE;
    unsigned int num_rows = MATRIX_SIZE;

    //Contents of the A matrix should already be in U

    //Perform the Cholesky decomposition in place on the U matrix
    for (k = 0; k < num_rows; k++) {
        //Only one thread does squre root and division
        if (tx == 0) {
            // Take the square root of the diagonal element
            U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
            //Don't bother doing check...live life on the edge!

            // Division step
            for (j = (k + 1); j < num_rows; j++) {
                U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
            }
        }

        //Sync threads!!!!! (only one thread block so, ok)
        __syncthreads();

        //Elimination step
        
        int istart = ( k + 1 )  +  tx * ops_per_thread;
        int iend = istart + ops_per_thread;
        
        for (i = istart; i < iend; i++) {
            //Do work  for this i iteration
            for (j = i; j < num_rows; j++) {
                U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
            }
        }
        

        //Sync threads!!!!! (only one thread block so, ok)
        __syncthreads();
    }

    //Sync threads!!!!! (only one thread block so, ok)
    __syncthreads();


    
    
    //As the final step, zero out the lower triangular portion of U
    //for(i = 0; i < U.num_rows; i++)
    //Each thread does so many iterations of zero out loop
    //Starting index for this thread
    int istart = tx * ops_per_thread;
    //Ending index for this thread
    int iend = istart + ops_per_thread;

    //Check boundaries, else do nothing
    for (i = istart; i < iend; i++) {
        //Do work  for this i iteration
        for (j = 0; j < i; j++) {
            U[i * num_rows + j] = 0.0;
        }
    }
   
    
    //Don't sync, will sync outside here	
}








//Division step

__global__ void
chol_kernel_optimized_div_old(float * U, int k, int stride) {
    //General thread id
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    //Iterators
    unsigned int j;
    unsigned int num_rows = MATRIX_SIZE;

    //Only let one thread do this
    if (tx == 0) {
        // Take the square root of the diagonal element
        U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
        //Don't bother doing check...live life on the edge!

        // Division step
        for (j = (k + 1); j < num_rows; j++) {
            U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
        }
    }
}

//Division step

__global__ void
chol_kernel_optimized_div(float * U, int k, int stride) {
    //With stride...

    //General thread id
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    //Iterators
    unsigned int j;
    unsigned int num_rows = MATRIX_SIZE;

    //Only let one thread do this
    if (tx == 0) {
        // Take the square root of the diagonal element
        U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
        //Don't bother doing check...live life on the edge!
    }

    //Each thread does some part of j
    //Stide in units of 'stride'
    //Thread 0 does 0, 16, 32
    //Thread 1 does 1, 17, 33
    //..etc.
    int offset = (k + 1); //From original loop
    int jstart = threadIdx.x + offset;
    int jstep = stride;

    //Only continue if in bounds?
    //Top limit on i for whole (original) loop
    int jtop = num_rows - 1;
    //Bottom limit on i for whole (original) loop
    int jbottom = (k + 1);

    //Do work for this i iteration
    //Division step
    //Only let one thread block do this
    if (blockIdx.x == 0) {
        for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
            U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
        }
    }
}

__global__ void
chol_kernel_optimized(float * U, int k, int stride) {
    //With stride...

    //Iterators
    unsigned int j;
    unsigned int num_rows = MATRIX_SIZE;


    //This call acts as a single K iteration
    //Each block does a single i iteration
    //Need to consider offset, 
    int i = blockIdx.x + (k + 1);
    //Each thread does some part of j
    //Stide in units of 'stride'
    //Thread 0 does 0, 16, 32
    //Thread 1 does 1, 17, 33
    //..etc.
    int offset = i; //From original loop
    int jstart = threadIdx.x + offset;
    int jstep = stride;

    //Only continue if in bounds?
    //Top limit on i for whole (original) loop
    int jtop = num_rows - 1;
    //Bottom limit on i for whole (original) loop
    int jbottom = i;

    //Do work for this i iteration
    //Want to stride across
    for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
        U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
    }
}

__global__ void
chol_kernel_optimized_no_stride(float * U, int k, int stride) {
    //Iterators
    unsigned int j;
    unsigned int num_rows = MATRIX_SIZE;

    //TODO USE STRIDE	

    //This call acts as a single K iteration
    //Each block does a single i iteration
    //Need to consider offset, 
    int i = blockIdx.x + (k + 1);
    //Each thread does some part of j
    //Split j based on stride and thread index
    //Index 0 is j= 0-15
    //Index 1 is j=16-31
    int offset = i;
    int jstart = (threadIdx.x * stride) + offset;
    int jend = jstart + (stride - 1);

    //Only continue if in bounds?
    //Top limit on i for whole (original) loop
    int jtop = num_rows - 1;
    //Bottom limit on i for whole (original) loop
    int jbottom = i;
    //Check boundaries, else do nothing
    if (!((jstart >= jbottom) && (jend <= jtop))) {
        return; //This thread does nothing now
    }

    //Do work  for this i iteration
    //Want to stride across
    for (j = jstart; j <= jend; j++) {
        U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
    }
}

#endif // #ifndef _CHOL_KERNEL_H_
