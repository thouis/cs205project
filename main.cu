//
//  main.cpp
//  pomdp
//
//  Created by Jaemin Cheun on 10/23/15.
//  Copyright © 2015 Jaemin Cheun. All rights reserved.
//

#include <iostream>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;

#define BELIEF_POINT_COUNT (4096)
#define ITERATIONS (30)

#define EPSILON (0.0000001)

#define STATE_COUNT (10)
#define ACTION_COUNT (3)
#define OBSERVATION_COUNT (2)

#define BLOCKS (32)
#define THREADS_PR_BLOCK (BELIEF_POINT_COUNT/BLOCKS)

int hyperplane_action[BELIEF_POINT_COUNT];


typedef struct{
    int actions;
    int states;
    int observations;
    int belief_point_count;
    
    int *hyperplane_action;
    
    float discount;
    
    float **B;
    float *R;
    float *O;
    float *T;
    
    float **V;
    float **Vtmp;
}POMDP;

float RandomNumber(float Min, float Max)
{
    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}


double dist(float *a, float *b, int length){
    double result = 0;
    for (int i = 0; i < length; ++i){
        result += pow((a[i] - b[i]), 2.0);
    }
    return sqrt(result);
}

float dot(float * a, float *b, int length){
    float sum = 0;
    for (int i = 0; i < length; ++i){
        sum += a[i] * b[i];
    }
    return sum;
}

float ** init_B_uniform(int count, int states){
    float **B = new float*[count];
    
    int bp;
    for (bp = 0; bp < states; ++bp){
        B[bp] = new float[states];
        memset(B[bp], 0, states* sizeof(float));
        B[bp][bp] = 1;
    }
    
    double ** dist_matrix = new double*[count];
    for (int i = 0; i < count; ++i){
        dist_matrix[i] = new double[count];
    }
    
    for (int i = 0; i < count; ++i){
        for (int j = 0; j < count; ++j){
            dist_matrix[i][j] = 0;
        }
    }
    double max_dist = 0;
    int index1 = -1;
    int index2 = -1;
    
    for(int i = 0; i < states; ++i){
        for (int j = i+1; j < states; ++j){
            dist_matrix[i][j] = dist(B[i], B[j], states);
            
            if (dist_matrix[i][j] > max_dist){
                max_dist = dist_matrix[i][j];
                index1 = i, index2 = j;
            }
        }
    }
    
    for(; bp < count; ++bp){
        B[bp] = new float[states];
        
        for (int k = 0; k < states; ++k){
            B[bp][k] = (B[index1][k] + B[index2][k]) / 2.0;
        }
        for (int i = 0; i < bp; ++i){
            dist_matrix[i][bp] = dist(B[bp], B[i], states);
        }
        dist_matrix[index1][index2] = 0;
        max_dist = 0;
        for(int i = 0; i <= bp; ++i){
            for (int j = i + 1; j <= bp; ++j){
                if (dist_matrix[i][j] > max_dist){
                    max_dist = dist_matrix[i][j];
                    index1 = i; index2= j;
                }
            }
        }
    }
    return B;
}

float *init_B_uniform_vector(int count, int states){
    float **BB = init_B_uniform(count,states);
    float *B = new float[count*states];
    for (int i = 0; i < count; ++i){
        for (int j = 0; j < states; ++j){
            B[states * i + j] =  BB[i][j];
        }
    }
    return B;
}

__global__ void init_pbvi(float **V, float **Vtmp, float initial_value, int states){
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    V[b] = new float[states];
    Vtmp[b] = new float[states];

    for (int i = 0; i < states; ++i){
        V[b][i] = initial_value;
        Vtmp[b][i] = initial_value;
    }
}

__global__ void pbvi_shared(float **dev_V, float **dev_Vtmp, float *dev_T, float *dev_O, float *dev_R, float *dev_B, 
    int *best_action, int states, int observations, int actions, float discount){

    int b = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float s_T[STATE_COUNT*STATE_COUNT*ACTION_COUNT];
    __shared__ float s_O[STATE_COUNT*OBSERVATION_COUNT*ACTION_COUNT];
    __shared__ float s_R[STATE_COUNT*ACTION_COUNT];
    __shared__ float s_B[STATE_COUNT*THREADS_PR_BLOCK];

    for (int i = 0; i < states; ++i){
        s_B[states * threadIdx.x + i] = dev_B[states *b + i];
    }

    if(states * states * actions <= THREADS_PR_BLOCK){
        if(threadIdx.x < states * states * actions){
            s_T[threadIdx.x] = dev_T[threadIdx.x];
        }
    }
    else{
        int iterations = (int)((states * states * actions + THREADS_PR_BLOCK - 1)/ THREADS_PR_BLOCK);
        int index;
        for (int i = 0; i < iterations; ++i){
            index = threadIdx.x * iterations + i;
            if(index < states*states*actions){
                s_T[index] = dev_T[index];
            }
        }
    }
    if(states * observations * actions <= THREADS_PR_BLOCK){
        if(threadIdx.x < states * observations * actions){
            s_O[threadIdx.x] = dev_O[threadIdx.x];
        }
    }
    else{
        int iterations = (int)((states * observations * actions + THREADS_PR_BLOCK - 1)/ THREADS_PR_BLOCK);
        int index;
        for (int i = 0; i < iterations; ++i){
            index = threadIdx.x * iterations + i;
            if(index < states*observations*actions){
                s_O[index] = dev_O[index];
            }
        }
    }
    if(states * actions <= THREADS_PR_BLOCK){
        if(threadIdx.x < states * actions){
            s_R[threadIdx.x] = dev_R[threadIdx.x];
        }
    }
    else{
        int iterations = (int)((states * actions + THREADS_PR_BLOCK - 1)/ THREADS_PR_BLOCK);
        int index;
        for (int i = 0; i < iterations; ++i){
            index = threadIdx.x * iterations + i;
            if(index < states*actions){
                s_R[index] = dev_R[index];
            }
        }
    }
    best_action[b] = -1;

    float *hyperplane = new float[states];
    float *local_hyperplane = new float[states];
    float *best_local_hyperplane = new float[states];
    
    float max = 0;
    for (int action = 0; action < actions; ++action){
        memset(hyperplane, 0, states * sizeof(float));
        for (int observation = 0; observation < observations; ++ observation){
            float local_max = 0;
            memset(best_local_hyperplane, 0, states * sizeof(float));
            
            /* for each hyper plane */
            for (int i = 0; i < BELIEF_POINT_COUNT; ++i){
                for (int s = 0; s < states; ++s){
                    local_hyperplane[s] =
                        s_O[states * observations * action + observation] *
                        s_T[states * states * action + states * s] *
                        dev_V[i][0];
                }
                for (int sp = 1; sp < states; ++sp){
                    for (int s = 0; s < states; ++s){
                        local_hyperplane[s] +=
                        s_O[states * observations * action + observations * sp + observation] *
                        s_T[states * states * action + states * s + sp] *
                        dev_V[i][sp];
                    }
                }

                float local_value = 0;
                for (int i = 0; i < states; ++i){
                    local_value += local_hyperplane[i] * dev_B[states * b + i];
                }
                if (local_value > local_max){
                    local_max = local_value;
                    float *tmp = local_hyperplane;
                    local_hyperplane = best_local_hyperplane;
                    best_local_hyperplane = tmp;
                }
            }
            for (int j = 0; j < states; ++j){
                hyperplane[j] += best_local_hyperplane[j];
            }
        }
        for (int j = 0; j < states; ++j){
            hyperplane[j] = s_R[states * action + j] + discount * hyperplane[j];
        }
        float value = 0;
        for (int i = 0; i < states; ++i){
            value += hyperplane[i] * s_B[states * b + i];
        }
        
        if(value > max){
            best_action[b] = action;
            max = value; 
            float *tmp = hyperplane;
            hyperplane = dev_Vtmp[b];
            dev_Vtmp[b] = tmp;
        }
    }
    delete hyperplane;
    delete local_hyperplane;
    delete best_local_hyperplane;

}

__global__ void pbvi(float **dev_V, float **dev_Vtmp, float *dev_T, float *dev_O, float *dev_R, float *dev_B, 
    int *best_action, int states, int observations, int actions, float discount){

    int b = threadIdx.x + blockIdx.x * blockDim.x;
    best_action[b] = -1;

    float *hyperplane = new float[states];
    float *local_hyperplane = new float[states];
    float *best_local_hyperplane = new float[states];
    
    float max = 0;
    for (int action = 0; action < actions; ++action){
        memset(hyperplane, 0, states * sizeof(float));
        for (int observation = 0; observation < observations; ++ observation){
            float local_max = 0;
            memset(best_local_hyperplane, 0, states * sizeof(float));
            
            /* for each hyper plane */
            for (int i = 0; i < BELIEF_POINT_COUNT; ++i){
                for (int s = 0; s < states; ++s){
                    local_hyperplane[s] =
                        dev_O[states * observations * action + observation] *
                        dev_T[states * states * action + states * s] *
                        dev_V[i][0];
                }
                for (int sp = 1; sp < states; ++sp){
                    for (int s = 0; s < states; ++s){
                        local_hyperplane[s] +=
                        dev_O[states * observations * action + observations * sp + observation] *
                        dev_T[states * states * action + states * s + sp] *
                        dev_V[i][sp];
                    }
                }

                float local_value = 0;
                for (int i = 0; i < states; ++i){
                    local_value += local_hyperplane[i] * dev_B[states * b + i];
                }
                if (local_value > local_max){
                    local_max = local_value;
                    float *tmp = local_hyperplane;
                    local_hyperplane = best_local_hyperplane;
                    best_local_hyperplane = tmp;
                }
            }
            for (int j = 0; j < states; ++j){
                hyperplane[j] += best_local_hyperplane[j];
            }
        }
        for (int j = 0; j < states; ++j){
            hyperplane[j] = dev_R[states * action + j] + discount * hyperplane[j];
        }
        float value = 0;
        for (int i = 0; i < states; ++i){
            value += hyperplane[i] * dev_B[states * b + i];
        }
        
        if(value > max){
            best_action[b] = action;
            max = value; 
            float *tmp = hyperplane;
            hyperplane = dev_Vtmp[b];
            dev_Vtmp[b] = tmp;
        }
    }
    delete hyperplane;
    delete local_hyperplane;
    delete best_local_hyperplane;
}
__global__ void pbvi_get(float **dev_V, float *dev_result, int states){
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < states; ++i){
        dev_result[states * b + i] = dev_V[b][i];
    }
}

void pbvi_shared(POMDP pomdp){
    float minimum = pomdp.R[0];
    
    for (int action = 0; action < pomdp.actions; ++action){
        for (int state = 0; state < pomdp.states; ++state){
            if (pomdp.R[pomdp.states * action + state] < minimum){
                minimum = pomdp.R[pomdp.states * action + state];
            }
        }
    }
    
    float initial_value = 0;
    initial_value = (1/(1-pomdp.discount)) *minimum;
    
    float *B = init_B_uniform_vector(BELIEF_POINT_COUNT, pomdp.states);
    
    float **dev_V;
    cudaMalloc((void**) &dev_V, BELIEF_POINT_COUNT * sizeof(float *));

    float **dev_Vtmp; 
    cudaMalloc((void**) &dev_Vtmp, BELIEF_POINT_COUNT * sizeof(float *));

    float *dev_R;
    cudaMalloc((void**) &dev_R, pomdp.actions * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_R, pomdp.R, pomdp.actions * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_O;
    cudaMalloc((void**) &dev_O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *));
    cudaMemcpy(dev_O, pomdp.O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_T;
    cudaMalloc((void**) &dev_T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_T, pomdp.T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_B;
    cudaMalloc((void**) &dev_B, BELIEF_POINT_COUNT * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_B, pomdp.B, BELIEF_POINT_COUNT * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    int *dev_best_action;
    cudaMalloc((void **) &dev_best_action, BELIEF_POINT_COUNT * sizeof(int));

    init_pbvi<<<BLOCKS, THREADS_PR_BLOCK>>>(dev_V, dev_Vtmp, initial_value, pomdp.states);
    cudaDeviceSynchronize();

    for(int i = 0; i < ITERATIONS; ++i){
        pbvi_shared<<<BLOCKS, THREADS_PR_BLOCK>>>
        (dev_V, dev_Vtmp, dev_T, dev_O, dev_R, dev_B, 
            dev_best_action, pomdp.states, pomdp.observations,pomdp.actions, pomdp.discount);
        cudaDeviceSynchronize();
        swap(dev_V, dev_Vtmp);
    }
    float *dev_result;
    cudaMalloc((void **) &dev_result, pomdp.states * BELIEF_POINT_COUNT * sizeof(float));

    pbvi_get<<<BLOCKS, THREADS_PR_BLOCK>>>(dev_V, dev_result, pomdp.states);
    float *result = new float[pomdp.states * BELIEF_POINT_COUNT];
    int *best_action = new int[BELIEF_POINT_COUNT];
    cudaMemcpy(result, dev_result, pomdp.states * BELIEF_POINT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_action, dev_best_action, BELIEF_POINT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_best_action);
    cudaFree(dev_result);
    cudaFree(dev_V);
    cudaFree(dev_Vtmp);
    cudaFree(dev_R);
    cudaFree(dev_O);
    cudaFree(dev_T);
    cudaFree(dev_B);
    
}

void pbvi_base(POMDP pomdp){
    
    float minimum = pomdp.R[0];
    
    for (int action = 0; action < pomdp.actions; ++action){
        for (int state = 0; state < pomdp.states; ++state){
            if (pomdp.R[pomdp.states * action + state] < minimum){
                minimum = pomdp.R[pomdp.states * action + state];
            }
        }
    }
    
    float initial_value = 0;
    initial_value = (1/(1-pomdp.discount)) *minimum;
    
    float *B = init_B_uniform_vector(BELIEF_POINT_COUNT, pomdp.states);
    
    float **dev_V;
    cudaMalloc((void**) &dev_V, BELIEF_POINT_COUNT * sizeof(float *));

    float **dev_Vtmp; 
    cudaMalloc((void**) &dev_Vtmp, BELIEF_POINT_COUNT * sizeof(float *));

    float *dev_R;
    cudaMalloc((void**) &dev_R, pomdp.actions * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_R, pomdp.R, pomdp.actions * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_O;
    cudaMalloc((void**) &dev_O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *));
    cudaMemcpy(dev_O, pomdp.O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_T;
    cudaMalloc((void**) &dev_T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_T, pomdp.T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_B;
    cudaMalloc((void**) &dev_B, BELIEF_POINT_COUNT * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_B, pomdp.B, BELIEF_POINT_COUNT * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    int *dev_best_action;
    cudaMalloc((void **) &dev_best_action, BELIEF_POINT_COUNT * sizeof(int));

    init_pbvi<<<BLOCKS, THREADS_PR_BLOCK>>>(dev_V, dev_Vtmp, initial_value, pomdp.states);
    cudaDeviceSynchronize();

    for(int i = 0; i < ITERATIONS; ++i){
        pbvi<<<BLOCKS, THREADS_PR_BLOCK>>>
        (dev_V, dev_Vtmp, dev_T, dev_O, dev_R, dev_B, 
            dev_best_action, pomdp.states, pomdp.observations,pomdp.actions, pomdp.discount);
        cudaDeviceSynchronize();
        swap(dev_V, dev_Vtmp);
    }
    float *dev_result;
    cudaMalloc((void **) &dev_result, pomdp.states * BELIEF_POINT_COUNT * sizeof(float));

    pbvi_get<<<BLOCKS, THREADS_PR_BLOCK>>>(dev_V, dev_result, pomdp.states);
    float *result = new float[pomdp.states * BELIEF_POINT_COUNT];
    int *best_action = new int[BELIEF_POINT_COUNT];
    cudaMemcpy(result, dev_result, pomdp.states * BELIEF_POINT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_action, dev_best_action, BELIEF_POINT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_best_action);
    cudaFree(dev_result);
    cudaFree(dev_V);
    cudaFree(dev_Vtmp);
    cudaFree(dev_R);
    cudaFree(dev_O);
    cudaFree(dev_T);
    cudaFree(dev_B);
    
}

int main() {

    POMDP pomdp;
    
    // number of states
    pomdp.states = STATE_COUNT;
    
    // number of actions
    pomdp.actions = ACTION_COUNT;
    
    // number of observations
    pomdp.observations = OBSERVATION_COUNT;
    
    // discount factor
    pomdp.discount = 0.95;
    
    float T[pomdp.states*pomdp.actions*pomdp.states];
    
    for (int action = 0; action < pomdp.actions; ++action){
        for(int state = 0; state < pomdp.states; ++state){
            float total = 0.0;
            for (int sp = 0; sp < pomdp.states; ++sp){
                T[action*pomdp.states * pomdp.states + state * pomdp.states + sp] = RandomNumber(0, 10); 
                total +=  T[action*pomdp.states * pomdp.states + state * pomdp.states + sp];
            }
            for (int sp = 0; sp < pomdp.states; ++sp){
                T[action*pomdp.states * pomdp.states + state * pomdp.states + sp] = T[action*pomdp.states * pomdp.states + state * pomdp.states + sp] / total;
            }
        }
    }
    pomdp.T = T;
    
    // index of Reward R(s,a) is action * num_state + state
    float R[pomdp.actions * pomdp.states];

    for (int action = 0; action < pomdp.actions; ++action){
        for (int state = 0; state < pomdp.states; ++state){
            R[action * pomdp.states + state] = RandomNumber(-10, 10);
        }
    }
    pomdp.R = R;
    
    // index of Observation Functino O(s',a,o) = action * num_state * num_observation + s' * num_obs + obs
    float O[pomdp.actions * pomdp.states * pomdp.observations];
    for (int action = 0; action < pomdp.actions; ++action){
        for(int state = 0; state < pomdp.states; ++state){
            float total = 0.0;
            for (int observation = 0; observation < pomdp.observations; ++observation){
                O[action*pomdp.states * pomdp.observations + state * pomdp.observations + observation] = RandomNumber(0, 10); 
                total +=  O[action*pomdp.states * pomdp.observations + state * pomdp.observations + observation];
            }
            for (int observation = 0; observation < pomdp.observations; ++observation){
                O[action*pomdp.states * pomdp.observations + state * pomdp.observations + observation] = O[action*pomdp.states * pomdp.observations + state * pomdp.observations + observation] / total;
            }
        }
    }
    pomdp.O = O;

    timeval start_time, end_time;
    float elapsedTime;
    gettimeofday(&start_time, NULL);
    pbvi_base(pomdp);
    gettimeofday(&end_time, NULL);
    elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
    elapsedTime += (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    cout << elapsedTime << endl;

    gettimeofday(&start_time, NULL);
    pbvi_shared(pomdp);
    gettimeofday(&end_time, NULL);
    elapsedTime = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
    elapsedTime += (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    cout << elapsedTime << endl;

    timeval t1, t2;
    double elapsedTime1;
    //start timer
    gettimeofday(&t1, NULL);

    
    
    float minimum = pomdp.R[0];
    
    for (int action = 0; action < pomdp.actions; ++action){
        for (int state = 0; state < pomdp.states; ++state){
            if (pomdp.R[pomdp.states * action + state] < minimum){
                minimum = pomdp.R[pomdp.states * action + state];
            }
        }
    }
    
    float initial_value = 0;
    initial_value = (1/(1-pomdp.discount)) *minimum;
    
    pomdp.V = new float*[BELIEF_POINT_COUNT];
    pomdp.Vtmp = new float*[BELIEF_POINT_COUNT];
    
    for (int b = 0; b < BELIEF_POINT_COUNT; ++b){
        pomdp.V[b] = new float[pomdp.states];
        pomdp.Vtmp[b] = new float[pomdp.states];
        
        for (int state = 0; state < pomdp.states; ++ state){
            pomdp.V[b][state] = initial_value;
            pomdp.Vtmp[b][state] = initial_value;
        }
    }
    
    float **B = init_B_uniform(BELIEF_POINT_COUNT, pomdp.states);
    
    float *hyperplane = new float[pomdp.states];
    float *local_hyperplane = new float[pomdp.states];
    float *best_local_hyperplane = new float[pomdp.states];
    float *best_hyperplane = new float[pomdp.states];
    
    for (int i = 0; i < ITERATIONS; ++i){
        for (int b = 0; b < BELIEF_POINT_COUNT; ++b){
            float max = 0;
            for (int action = 0; action < pomdp.actions; ++action){
                memset(hyperplane, 0, pomdp.states * sizeof(float));
                for (int observation = 0; observation < pomdp.observations; ++ observation){
                    float local_max = 0;
                    memset(best_local_hyperplane, 0, pomdp.states * sizeof(float));
                    
                    /* for each hyper plane */
                    for (int i = 0; i < BELIEF_POINT_COUNT; ++i){
                        for (int s = 0; s < pomdp.states; ++s){
                            local_hyperplane[s] =
                                pomdp.O[pomdp.states * pomdp.observations * action + observation] *
                                pomdp.T[pomdp.states * pomdp.states * action + pomdp.states * s] *
                                pomdp.V[i][0];
                        }
                        for (int sp = 1; sp < pomdp.states; ++sp){
                            for (int s = 0; s < pomdp.states; ++s){
                                local_hyperplane[s] +=
                                pomdp.O[pomdp.states * pomdp.observations * action + pomdp.observations * sp + observation] *
                                pomdp.T[pomdp.states * pomdp.states * action + pomdp.states * s + sp] *
                                pomdp.V[i][sp];
                            }
                        }
                        
                        float local_value = dot(local_hyperplane, B[b], pomdp.states);
                        if (local_value > local_max){
                            local_max = local_value;
                            swap(local_hyperplane, best_local_hyperplane);
                        }
                    }
                    for (int j = 0; j < pomdp.states; ++j){
                        hyperplane[j] += best_local_hyperplane[j];
                    }
                }
                for (int j = 0; j < pomdp.states; ++j){
                    hyperplane[j] = pomdp.R[pomdp.states * action + j] + pomdp.discount * hyperplane[j];
                }
                float value = dot(hyperplane, B[b], pomdp.states);
                
                if(value > max){
                    max = value;
                    swap(hyperplane, pomdp.Vtmp[b]);
                    hyperplane_action[b] = action;
                }
            }
            swap(pomdp.V, pomdp.Vtmp);
        }
    }
//    for (int i = 0; i < BELIEF_POINT_COUNT; ++i){
//        cout << hyperplane_action[i] << endl;
//    }
    gettimeofday(&t2, NULL);
    elapsedTime1 = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime1 += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << elapsedTime1 << endl;
    
}
