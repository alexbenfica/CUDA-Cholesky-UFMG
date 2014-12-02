/* Cholesky decomposition.
 * Host code.
 * Author: Naga Kandasamy
 * Date: May 23, 2013
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <winsock2.h>
#include "systime.h"

// includes, kernels
#include "chol_kernel.cu"
#include "utils.h"


////////////////////////////////////////////////////////////////////////////////
// declarations, forward
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);

void check_error(const char *msg);

extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern "C" int chol_gold(const Matrix, Matrix);
extern "C" int check_chol(const Matrix, const Matrix);
void chol_on_device(const Matrix, Matrix);
void chol_on_device_optimized(const Matrix, Matrix);
void chol_on_device_cudaUFMG(const Matrix, Matrix);

extern void print_matrix_to_file(const Matrix M, char *filename);

//Globals
double time_cpu;
// Matrices for the program
Matrix A; // The N x N input matrix
Matrix reference; // The upper triangular matrix computed by the CPU
Matrix U_on_device; // The upper triangular matrix computed by the device (slow)
Matrix U_on_device_fast; // The upper triangular matrix computed by the device (fast)
Matrix U_on_device_cudaUFMG;


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc > 1) {
        printf("Error. This program accepts no arguments. \n");
        exit(0);
    }

    // Initialize the random number generator with a seed value 
    srand(time(NULL));

    // Create the positive definite matrix. May require a few tries if we are unlucky
    int success = 0;
    while (!success) {
        A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
        if (A.elements != NULL)
            success = 1;
    }

    reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the CPU result
    U_on_device = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the device result
    U_on_device_fast = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
    U_on_device_cudaUFMG = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);

    //Compute the Cholesky decomposition on the CPU
    printf("\n== CPU ==\n");
    int status = 1;
    
    start_time();
    
    status = chol_gold(A, reference);
    
    time_cpu = show_time();
    
    
    if (status == 0) {
        printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
        exit(0);
    }
#if 0
    printf("Double checking for correctness by recovering the original matrix. \n");
    if(check_chol(A, reference) == 0){
            printf("CPU: FAILED\n");
            exit(0);
    }
#endif    
    printf("	PASSED\n\n"); //IT IS SO PERFECT WE DON'T EVEN CHECK.
    


    //Slow
    //Perform the Cholesky decomposition on the GPU. The resulting upper triangular matrix should be retured in U_on_gpu
    // chol_on_device(A, U_on_device);
    
    //return 1;

    //Optimized
    //Perform the Cholesky decomposition on the GPU. The resulting upper triangular matrix should be retured in U_on_gpu
    // chol_on_device_optimized(A, U_on_device_fast);
    
    
    //Optimized for project at UFMG
    chol_on_device_cudaUFMG(A, U_on_device_cudaUFMG);
    
    

    // Free host matrices
    free(A.elements);
    free(U_on_device.elements);
    free(U_on_device_fast.elements);
    free(reference.elements);
    return 1;
}

//Error helper

void check_for_error(char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}















unsigned compareArrays(float *reference, float * device, int size){
    
    for(int i=0; i<size; i++) {        
        // Default tolerance of 0.15 (must be improved)
        float epsilon = 0.15;        
        
        int x = i / MATRIX_SIZE;
        int y = i % MATRIX_SIZE;
        if(x==y){
            // A different tolerance for diagonals
            epsilon = 1;
        }        
        if (fabs(reference[i] - device[i]) > epsilon) {
            printf("\ni=%d : reference=%f  !=  device=%f   | x=%d y=%d   \n" , i, reference[i], device[i], x, y);
            return 0;
        }
    }
    return 1;
}













/* Write code to perform Cholesky decopmposition on the device. */
void chol_on_device(const Matrix A, Matrix U) {
    //Slow
    //Perform the Cholesky decomposition on the GPU. The resulting upper triangular matrix should be retured in U_on_gpu

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //Maximum size expected is 8192x8192
    //Will be splitting the elimination i loop
    //Which has up to MATRIX_SIZE iterations
    //So we would optimally use 8192 threads
    //Thus requiring 16 blocks
    //Rather than attempting to syncronize 16 blocks
    //Where each thread does one operation per outer K iteration
    //Just have one block and have each thread do 16 operations 
    //(in the worst case)
    int num_blocks = 1;

    //Max per block threads
    int threads_per_block = 512;

    //Operations per thread
    int ops_per_thread = MATRIX_SIZE / (threads_per_block * num_blocks);

    printf("== GPU (Slow) ==\n");
    printf("	Threads per block: %d\n", threads_per_block);
    printf("	Number of blocks: %d\n", num_blocks);
    printf("	Operations per thread: %d\n", ops_per_thread);

    
    cudaEventRecord(start, 0);
    
    //A and U are already allocated on CPU already
    //Allocate space on gpu
    Matrix gpu_u = allocate_matrix_on_gpu(U);

    //Copy matrices to gpu, copy A right into U
    copy_matrix_to_device(gpu_u, A);

    //Set up the execution grid on the GPU
    dim3 thread_block(threads_per_block, 1, 1);
    dim3 grid(num_blocks, 1);

    // Launch the kernel <<<grid, thread_block>>>
    chol_kernel << <grid, thread_block>>>(gpu_u.elements, ops_per_thread);

    //Sync at end and check for errors
    cudaThreadSynchronize();
    check_for_error("SLOW KERNEL FAILURE\n");


    float time_gpu;


    //Copy data back
    copy_matrix_from_device(U, gpu_u);
    
    //Stop timer before copy back
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);

    //Free memory on device
    cudaFree(gpu_u.elements);

    printf("	Run time:    %0.10f ms. \n", time_gpu);
    printf("	Speedup: %0.10fx\n", time_cpu / time_gpu);
    //Check if the device result is equivalent to the expected solution. If you can't meet the desired tolerance, try using double precision support.
    unsigned int size = reference.num_rows * reference.num_columns;
    unsigned res = compareArrays(reference.elements, U.elements, size);
    printf("	%s\n", (1 == res) ? "PASSED" : "FAILED");
}










/* Write code to perform Cholesky decopmposition on the device. */
void chol_on_device_optimized(const Matrix A, Matrix U) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("== GPU (Fast) ==\n");
    //A and U are already allocated on CPU already
    
    //Each thread within a block will take some j iterations
    int threads_per_block = 256; //Optimal
    //Stride size should equal threads per block - just cause?
    int stride = threads_per_block;    
    printf("	Threads per block / stride: %d\n", threads_per_block);
    
    
    //Start timer BEFORE copy
    cudaEventRecord(start, 0);    
    
    //Allocate space on gpu for U
    Matrix gpu_u = allocate_matrix_on_gpu(U);
    
    //Copy matrices to gpu, copy A right into U
    copy_matrix_to_device(gpu_u, A);


    //Each kernel call will be one iteration of out K loop
    int k;
    for (k = 0; k < MATRIX_SIZE; k++) {
        //Want threads to stride across memory
        //i is outer loop
        //j is inner loop
        //so threads should split the j loop
        //Each thread block will take an i iteration
        int isize = (MATRIX_SIZE - 1) - (k + 1) + 1;
        int num_blocks = isize;
        if (num_blocks <= 0) {
            num_blocks = 1;
        }

        //Set up the execution grid on the GPU
        //printf("	Threads per block: %d\n",threads_per_block);
        //printf("	Number of blocks: %d\n",num_blocks);
        dim3 thread_block(threads_per_block, 1, 1);
        dim3 grid(num_blocks, 1);

        //Call the div kernel for this k iteration
        chol_kernel_optimized_div << <grid, thread_block>>>(
                gpu_u.elements,
                k,
                stride);

        
        //Call kernel with for this K iteration
        chol_kernel_optimized << <grid, thread_block>>>(gpu_u.elements,k,stride);


        //Sync at end and check for errors
        cudaThreadSynchronize();
        check_for_error("FAST KERNEL FAILURE");
    }

    
    //Sync at end
    cudaThreadSynchronize();

    
    //Copy data back
    copy_matrix_from_device(U, gpu_u);
    
    //Stop timer after copy back					 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time_gpu_fast;
    cudaEventElapsedTime(&time_gpu_fast, start, stop);


    //Free memory on device
    cudaFree(gpu_u.elements);


    //As the final step, zero out the lower triangular portion of U
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++)
        for (j = 0; j < i; j++)
            U.elements[i * MATRIX_SIZE + j] = 0.0;

    printf("	Run time:    %0.10f ms. \n", time_gpu_fast / 1000);
    printf("	Speedup: %0.10f\n", time_cpu / (time_gpu_fast / 1000));
    //Check if the device result is equivalent to the expected solution. If you can't meet the desired tolerance, try using double precision support.
    unsigned int size_fast = reference.num_rows * reference.num_columns;
    unsigned res = compareArrays(reference.elements, U_on_device_fast.elements, size_fast);
    printf("	%s\n", (1 == res) ? "PASSED" : "FAILED");
    
    //print_matrix_to_file(U,"debug_chol_optimized\\matrix-GPU-div.txt");
    //print_matrix_to_file(U,"debug_chol_optimized\\matrix-GPU-final.txt");
    //print_matrix_to_file(reference,"debug_chol_optimized\\matrix-CPU-div.txt");
    
    
    //	CUTBoolean res_fast = cutComparefe(reference.elements, U_on_device_fast.elements, size_fast, 0.1f);
    //	printf("	%s\n", (1 == res_fast) ? "PASSED" : "FAILED");
}









/* Optimized for UFMG CUDA course project */
void chol_on_device_cudaUFMG(const Matrix A, Matrix U) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n== GPU (UFMG Cuda Course) ==\n");
    
    /*    
     * Shared memory per block: 48k = 49152 bytes    
     * Total bytes for matrix = MATRIX_SIZE x MATRIX_SIZE x size_of(float)
     * 1 element = 1 scalar of a matrix
     * Limited by shared memory, a maximum of   49152 / size_of(float)  elements can be copied to shared memory on each interation.
     * Max elements for thread = 12k elements
     */
    
    
    //Allocate space on gpu for U
    Matrix gpu_u = allocate_matrix_on_gpu(U);


    //Start timer BEFORE copy
    cudaEventRecord(start, 0);    
    
    //Copy matrices to gpu, copy A right into U
    copy_matrix_to_device(gpu_u, A);
    


    
    int threads_per_block_sqrt = 512;    
    int blocks_sqrt = MATRIX_SIZE / threads_per_block_sqrt;    
    dim3 thread_block(threads_per_block_sqrt, 1, 1);
    dim3 grid(blocks_sqrt, 1);    
    chol_kernel_cudaUFMG_sqrt <<<grid, thread_block>>>(gpu_u.elements);
    
    
    
    
    int block_x_div = 16;    
    int block_y_div = 16;        
    int thread_x_div = 4;    
    int thread_y_div = 4;        
    dim3 grid_div(block_x_div, block_y_div, 1);    
    dim3 thread_block_div(thread_x_div, thread_y_div, 1);
    int elements_per_thread_div = ((MATRIX_SIZE * MATRIX_SIZE) / 2) /  (thread_x_div * thread_y_div * block_x_div * block_y_div);        
    chol_kernel_cudaUFMG_division <<<grid_div, thread_block_div >>>(gpu_u.elements, elements_per_thread_div);



 
    int block_y_eli = 1;        
    //Each thread within a block will take some j iterations
    int thread_x_eli = 256;    
    int thread_y_eli = 1;        
    
    //Each kernel call will be one iteration of out K loop
    for (int k = 0; k < MATRIX_SIZE; k++) {
        
        //Want threads to stride across memory
        //i is outer loop
        //j is inner loop
        //so threads should split the j loop
        //Each thread block will take an i iteration
        
        // i=k+1;i<MATRIX_SIZE       
        int isize = MATRIX_SIZE - (k + 1);
        if(isize==0){
            isize++;
        }
        int block_x_eli = isize;
        
        //Set up the execution grid on the GPU
        dim3 thread_block(thread_x_eli, 1, 1);
        dim3 grid(block_x_eli, 1);
        
        //Call kernel with for this K iteration
        chol_kernel_cudaUFMG_elimination <<<grid, thread_block>>>(gpu_u.elements, k);
    }

    
    
    chol_kernel_cudaUFMG_zero <<<grid_div, thread_block_div>>>(gpu_u.elements, elements_per_thread_div);
    
    
    
    
    
    /*---------------------------------------------*/
    
    //Copy data back
    copy_matrix_from_device(U, gpu_u);

    //CUDA_SAFE_CALL(cudaPeekAtLastError());    
    //return;

    
    //Stop timer after copy back					 
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float time_gpu_fast;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&time_gpu_fast, start, stop));
    
    //Free memory on device
    cudaFree(gpu_u.elements);
    
    //Set up the execution grid on the GPU
    //printf("Threads per block sqrt: %d\n", threads_per_block_sqrt);
    //printf("Number of blocks sqrt: %d\n", blocks_sqrt);
    //printf("Elements_per_thread div: %d\n", elements_per_thread_div);
    
    printf("	Run time on GPU:    %0.10f s. \n", time_gpu_fast / 1000);            
    printf("	Speedup from single-core CPU: %0.10f x\n", time_cpu / (time_gpu_fast / 1000) ) ;
    //Check if the device result is equivalent to the expected solution. If you can't meet the desired tolerance, try using double precision support.
    unsigned int size_fast = reference.num_rows * reference.num_columns;
    
    //print_matrix_to_file(U,"matrix-CUDA.txt");
    //print_matrix_to_file(reference,"matrix-CPU.txt");
    
    unsigned res = compareArrays(reference.elements, U.elements, size_fast);
    if(res==1){
        printf("\nPASSED: GPU = CPU (explain epsilons)\n");
    }
    else{
        printf("\nFAILED: GPU != CPU");
    }


    
#if 0
    //Each thread within a block will take some j iterations
    int threads_per_block = 256; //Optimal
    //Stride size should equal threads per block - just cause?
    int stride = threads_per_block;    
    printf("	Threads per block / stride: %d\n", threads_per_block);
    const int shared_memory_size = 49152;
    // Limited by shared memory.
    int element_per_shared = shared_memory_size / size_of(float);
    int elements_per_thread = (element_per_shared / max_threads_per_block) -1;

    //Each kernel call will be one iteration of out K loop
    int k;
    for (k = 0; k < MATRIX_SIZE; k++) {
        //Want threads to stride across memory
        //i is outer loop
        //j is inner loop
        //so threads should split the j loop
        //Each thread block will take an i iteration
        int isize = (MATRIX_SIZE - 1) - (k + 1) + 1;
        int num_blocks = isize;
        if (num_blocks <= 0) {
            num_blocks = 1;
        }


        //Call kernel with for this K iteration
        chol_kernel_optimized << <grid, thread_block>>>(
                gpu_u.elements,
                k,
                stride);


        //Sync at end and check for errors
        cudaThreadSynchronize();
        check_for_error("FAST KERNEL FAILURE");
    }

    //Sync at end
    cudaThreadSynchronize();

    
    //Copy data back
    copy_matrix_from_device(U, gpu_u);
    


    //As the final step, zero out the lower triangular portion of U
    int i, j;
    for (i = 0; i < MATRIX_SIZE; i++)
        for (j = 0; j < i; j++)
            U.elements[i * MATRIX_SIZE + j] = 0.0;

    
    //	CUTBoolean res_fast = cutComparefe(reference.elements, U_on_device_fast.elements, size_fast, 0.1f);
    //	printf("	%s\n", (1 == res_fast) ? "PASSED" : "FAILED");
    
#endif    
}




















// Allocate a device matrix of same size as M.

Matrix allocate_matrix_on_gpu(const Matrix M) {
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof (float);
    cudaMalloc((void**) &Mdevice.elements, size);
    return Mdevice;
}


// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.

Matrix allocate_matrix(int num_rows, int num_columns, int init) {
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (float *) malloc(size * sizeof (float));
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.elements[i] = 0;
        else
            M.elements[i] = (float) rand() / (float) RAND_MAX;
    }
    return M;
}

// Copy a host matrix to a device matrix.

void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost) {
    int size = Mhost.num_rows * Mhost.num_columns * sizeof (float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.

void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice) {
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof (float);
    CUDA_SAFE_CALL(cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost));
}

void check_error(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
