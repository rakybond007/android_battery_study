#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

typedef struct print_cpu_util_data {
	long double** pre_utils;
    long double** cur_utils;
    float* total_util;
} print_cpu_util_data;

typedef struct matmul_data {
    float* total_util;
} matmul_data;

float readCpuUtil(long double** pre_utils, long double** cur_utils){
    long double idle_diff;
    long double total_diff;
    long double diff;
    float *utils = (float*)malloc(sizeof(float)*9);
    FILE *fp;
    fp = fopen("/proc/stat", "r");
    for (int idx = 0; idx < 9; idx++) {
        fscanf(fp, "%*s %Lf %Lf %Lf %Lf %Lf %Lf %Lf", &cur_utils[idx][0], &cur_utils[idx][1], &cur_utils[idx][2], &cur_utils[idx][3], &cur_utils[idx][4], &cur_utils[idx][5], &cur_utils[idx][6]);
    }
    fclose(fp);
    for (int i = 0; i < 9; i++) {
        total_diff = 0;
        idle_diff = 0;
        for (int j = 0; j < 8; j++){
            diff = cur_utils[i][j] - pre_utils[i][j];
            total_diff += diff;
            if (j == 3 || j == 4){
                idle_diff += diff;
            }
            pre_utils[i][j] = cur_utils[i][j];
        }
        utils[i] = 100.0 - (idle_diff / total_diff * 100);
    }
    return utils[0];
}

void* print_cpu_total_util(void *print_cpu_util_data_void) {
	float util;
    print_cpu_util_data *data = (print_cpu_util_data*) print_cpu_util_data_void;
    for (int i=0; i<100; i++){
        struct timeval stop, start;
	    gettimeofday(&start, NULL);
        util = readCpuUtil (data->pre_utils, data->cur_utils);
        printf("util : %f\n", util);
        *(data->total_util) = util;
        gettimeofday(&stop, NULL);
        printf("* Microseconds taken: %lu\n", stop.tv_usec - start.tv_usec);
        usleep(500000 - (stop.tv_usec - start.tv_usec));
    }
}

void* matmul_thread(void *matmul_data_void) {
    matmul_data *data = (matmul_data*) matmul_data_void;
    for (int i=0; i<100; i++){
        if (*(data->total_util) < 30) {
            printf("Need to make more load! %f\n", *(data->total_util));
            usleep(500000);
        }
    }
}

int main(int argc, char **argv) {
    long double **pre_utils = (long double**)malloc(sizeof(long double*) * 9);
    long double **cur_utils = (long double**)malloc(sizeof(long double*) * 9);
    float total_util;
    for (int i = 0; i < 9; i++){
        pre_utils[i] = (long double*)malloc(sizeof(long double) * 10);
        cur_utils[i] = (long double*)malloc(sizeof(long double) * 10);
    }
    FILE *fp;
    fp = fopen("/proc/stat", "r");
    for (int idx = 0; idx < 9; idx++) {
        fscanf(fp, "%*s %Lf %Lf %Lf %Lf %Lf %Lf %Lf", &pre_utils[idx][0], &pre_utils[idx][1], &pre_utils[idx][2], &pre_utils[idx][3], &pre_utils[idx][4], &pre_utils[idx][5], &pre_utils[idx][6]);
    }
    sleep(1);
    fclose(fp);

    pthread_t **threads = malloc(sizeof(pthread_t*) * 2);
	int thread_count = 1;
    print_cpu_util_data* data = malloc(sizeof(print_cpu_util_data));
    matmul_data* data2 = malloc(sizeof(matmul_data));
	data->pre_utils = pre_utils;
    data->cur_utils = cur_utils;
    data->total_util = &total_util;
    data2->total_util = &total_util;
    pthread_t *thread1 = malloc(sizeof(pthread_t));
    threads[0] = thread1;
    pthread_t *thread2 = malloc(sizeof(pthread_t));
    threads[1] = thread2;
    if(pthread_create(thread1, NULL, print_cpu_total_util, data)) {
        fprintf(stderr, "Error creating thread\n");
    }
    if(pthread_create(thread2, NULL, matmul_thread, data2)) {
        fprintf(stderr, "Error creating thread\n");
    }
	for (int i = 0; i < thread_count; ++i) {
		pthread_join(*threads[i], NULL);
		free(threads[i]);
	}

    int cpu_cores = 8;
    int rows_num = 1024;
    int cols_num = 1024;
	int **values = malloc(sizeof(int*) * cpu_cores);
    for (int c = 0; c < cpu_cores; ++c) {
		values[c] = malloc(sizeof(int) * cols_num * rows_num);
	}
    for (int c = 0; c < cpu_cores; ++c) {
        for (int i = 0; i < rows_num; ++i) {
            for (int j = 0; j < cols_num; ++j) {
                values[c][i*cols_num+j] = 1;
            }
        }
    }

	return 0;
}