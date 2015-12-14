//
//  main.cu
//  PBVI solver (serial, gpu global memory, gpu shared memory) algorithms used to solve POMDPs, and a simulation on tiger world problem
//
//  Created by Jaemin Cheun on 10/23/15.
//  Copyright © 2015 Jaemin Cheun. All rights reserved.
//

#include <iostream>
#include <string.h>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <limits>
#include <cmath>
#include <algorithm>
#include "pomdp.h"
#include <map>
#include <random>
#include <iostream>

using namespace std;

// Initialize belief counts, iterations, number of blocks
#define NUM_BELIEF (512)
#define ITERATIONS (10)
#define BLOCKS (32)
#define THREADS_PER_BLOCK (NUM_BELIEF/BLOCKS)
#define NUM_STATES (2)
#define NUM_ACTIONS (3)
#define NUM_OBSERVATIONS (2)

// creates an random integer between min and max
float RandomNumber(float Min, float Max)
{
    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

// calculates the distance between array A and array B
float dist(float *a, float *b, int length){
    float total = 0;
    for (int i = 0; i < length; ++i){
        total += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(total);
}

// creates a random belief sample of belief points given num_belief and states
float ** init_B_random(int num_belief, int states){
    float **B = new float*[num_belief];
    int belief_point;
    for (belief_point = 0; belief_point < states; ++belief_point){
        B[belief_point] = new float[states];
        memset(B[belief_point], 0, states * sizeof(float));
        B[belief_point][belief_point] = 1;
    }
    srand(3);
    for(; belief_point < num_belief; ++belief_point){
        B[belief_point] = new float[states];
        float sum = 0;
        for (int state = 0; state < states; ++state){
            B[belief_point][state] = rand() % 1000+ 1;
            sum += B[belief_point][state];
        }

        for (int state = 0; state < states; ++state){
            B[belief_point][state] /= sum;
        }
    }
    return B;
}

float *init_B_random_vector(int num_belief, int states){
    float **BB = init_B_random(num_belief,states);
    float *B = new float[num_belief*states];
    for (int belief_point = 0; belief_point < num_belief; ++belief_point){
        for (int state = 0; state < states; ++state){
            B[states * belief_point + state] =  BB[belief_point][state];
        }
    }
    return B;
}

// initialize PBVI algorithm
__global__ void init_pbvi(float **V, float **Vtmp, float init_value, int states){
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    V[b] = new float[states];
    Vtmp[b] = new float[states];
    for (int state = 0; state < states; ++state){
        V[b][state] = init_value;
        Vtmp[b][state] = init_value;
    }
}

// update V and Vtmp using shared GPU
__global__ void pbvi_shared(float **dev_V, float **dev_Vtmp, float *dev_T, float *dev_O, float *dev_R, float *dev_B, 
    int *dev_best_action, int states, int observations, int actions, float discount){

    int b = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_T[NUM_STATES*NUM_STATES*NUM_ACTIONS];
    __shared__ float shared_O[NUM_STATES*NUM_OBSERVATIONS*NUM_ACTIONS];
    __shared__ float shared_R[NUM_STATES*NUM_ACTIONS];
    __shared__ float shared_B[NUM_STATES*THREADS_PER_BLOCK];

    for (int state = 0; state < states; ++state){
        shared_B[states * threadIdx.x + b] = dev_B[states *b + state];
    }

    // Transferring T to shared memory
    if(states * states * actions <= THREADS_PER_BLOCK){

        // make sure it does not overflow
        if(threadIdx.x < states * states * actions){
            shared_T[threadIdx.x] = dev_T[threadIdx.x];
        }
    }
    else{
        int iterations = (int)((states * states * actions + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK);
        int index;
        for (int i = 0; i < iterations; ++i){
            index = threadIdx.x * iterations + i;
            if(index < states * states * actions){
                shared_T[index] = dev_T[index];
            }
        }
    }

    // Transferring O to shared memory
    if(states*observations*actions <= THREADS_PER_BLOCK){

        //make sure it does not overflow
        if(threadIdx.x < states*observations*actions){
            shared_O[threadIdx.x] = dev_O[threadIdx.x];
        }
    }
    else{
        int iterations = (int)((states*observations*actions + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK);
        int index;
        for (int i = 0; i < iterations; ++i){
            index = threadIdx.x * iterations + i;
            if(index < states*observations*actions){
                shared_O[index] = dev_O[index];
            }
        }
    }

    // Transferring R to shared memory
    if(states*actions <= THREADS_PER_BLOCK){

        // make sure it does not overflow
        if(threadIdx.x < states*actions){
            shared_R[threadIdx.x] = dev_R[threadIdx.x];
        }
    }
    else{
        int iterations = (int)((states*actions + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK);
        int index;
        for (int i = 0; i < iterations; ++i){
            index = threadIdx.x * iterations + i;
            if(index < states*actions){
                shared_R[index] = dev_R[index];
            }
        }
    }

    // initialize actions 
    dev_best_action[b] = -1;

    float *hyperplane = new float[states];
    float *local_hyperplane = new float[states];
    float *best_local_hyperplane = new float[states];
    
    float max = -INFINITY;
    for (int action = 0; action < actions; ++action){
        memset(hyperplane, 0, states * sizeof(float));
        for (int observation = 0; observation < observations; ++ observation){
            float local_max = -INFINITY;
            memset(best_local_hyperplane, 0, states * sizeof(float));   
            for (int i = 0; i < NUM_BELIEF; ++i){
                for (int s = 0; s < states; ++s){
                    local_hyperplane[s] =
                        shared_O[states * observations * action + observation] *
                        shared_T[states * states * action + states * s] *
                        dev_V[i][0];
                }
                for (int next_state = 1; next_state < states; ++next_state){
                    for (int s = 0; s < states; ++s){
                        local_hyperplane[s] +=
                        shared_O[states * observations * action + observations * next_state + observation] *
                        shared_T[states * states * action + states * s + next_state] *
                        dev_V[i][next_state];
                    }
                }
                float local_value = 0;
                for (int state = 0; state < states; ++state){
                    local_value += local_hyperplane[state] * shared_B[states * b + state];
                }

                if (local_value > local_max){
                    local_max = local_value;
                    float *tmp = local_hyperplane;
                    local_hyperplane = best_local_hyperplane;
                    best_local_hyperplane = tmp;
                }
            }
            for (int state = 0; state < states; ++state){
                hyperplane[state] += best_local_hyperplane[state];
            }
        }
        for (int state = 0; state < states; ++state){
            hyperplane[state] = shared_R[states * action + state] + discount * hyperplane[state];
        }
        float value = 0;
        for (int state = 0; state < states; ++state){
            value += hyperplane[state] * shared_B[states * b + state];
        }
        if(value > max){
            max = value;
            float *tmp = hyperplane;
            hyperplane = dev_Vtmp[b];
            dev_Vtmp[b] = tmp;
            dev_best_action[b] = action;
        }
    }
    delete hyperplane;
    delete local_hyperplane;
    delete best_local_hyperplane;

}

//update V and Vtmp using Global Memory
__global__ void pbvi_global(float **dev_V, float **dev_Vtmp, float *dev_T, float *dev_O, float *dev_R, float *dev_B, 
    int *dev_best_action, int states, int observations, int actions, float discount){

    int b = threadIdx.x + blockIdx.x * blockDim.x;
    dev_best_action[b] = -1;

    float *hyperplane = new float[states];
    float *local_hyperplane = new float[states];
    float *best_local_hyperplane = new float[states];
    
    float max = -INFINITY;
    for (int action = 0; action < actions; ++action){
        memset(hyperplane, 0, states * sizeof(float));
        for (int observation = 0; observation < observations; ++ observation){
            float local_max = -INFINITY;
            memset(best_local_hyperplane, 0, states * sizeof(float));   
            /* for each hyper plane */
            for (int i = 0; i < NUM_BELIEF; ++i){
                for (int s = 0; s < states; ++s){
                    local_hyperplane[s] =
                        dev_O[states * observations * action + observation] *
                        dev_T[states * states * action + states * s] *
                        dev_V[i][0];
                }
                for (int next_state = 1; next_state < states; ++next_state){
                    for (int s = 0; s < states; ++s){
                        local_hyperplane[s] +=
                        dev_O[states * observations * action + observations * next_state + observation] *
                        dev_T[states * states * action + states * s + next_state] *
                        dev_V[i][next_state];
                    }
                }
                float local_value = 0;
                for (int state = 0; state < states; ++state){
                    local_value += local_hyperplane[state] * dev_B[states * b + state];
                }

                if (local_value > local_max){
                    local_max = local_value;
                    float *tmp = local_hyperplane;
                    local_hyperplane = best_local_hyperplane;
                    best_local_hyperplane = tmp;
                }
            }
            for (int state = 0; state < states; ++state){
                hyperplane[state] += best_local_hyperplane[state];
            }
        }
        for (int state = 0; state < states; ++state){
            hyperplane[state] = dev_R[states * action + state] + discount * hyperplane[state];
        }
        float value = 0;
        for (int state = 0; state < states; ++state){
            value += hyperplane[state] * dev_B[states * b + state];
        }
        if(value > max){
            max = value;
            float *tmp = hyperplane;
            hyperplane = dev_Vtmp[b];
            dev_Vtmp[b] = tmp;
            dev_best_action[b] = action;
        }
    }
    delete hyperplane;
    delete local_hyperplane;
    delete best_local_hyperplane;
}

// Gets the result
__global__ void pbvi_get(float **dev_V, float *dev_result, int states){
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < states; ++i){
        dev_result[states * b + i] = dev_V[b][i];
    }
}

// Solves POMDP using Shared Memory GPU PBVI
void pbvi_shared(POMDP pomdp,float*result, int*best_action){    
    
    float **dev_V;
    cudaMalloc((void**) &dev_V, NUM_BELIEF * sizeof(float *));

    float **dev_Vtmp; 
    cudaMalloc((void**) &dev_Vtmp, NUM_BELIEF * sizeof(float *));

    float *dev_T;
    cudaMalloc((void**) &dev_T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_T, pomdp.T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_O;
    cudaMalloc((void**) &dev_O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *));
    cudaMemcpy(dev_O, pomdp.O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_R;
    cudaMalloc((void**) &dev_R, pomdp.actions * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_R, pomdp.R, pomdp.actions * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_B;
    cudaMalloc((void**) &dev_B, NUM_BELIEF * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_B, pomdp.B, NUM_BELIEF * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    int *dev_best_action;
    cudaMalloc((void **) &dev_best_action, NUM_BELIEF * sizeof(int));

    init_pbvi<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_V, dev_Vtmp, pomdp.init_value, pomdp.states);
    cudaDeviceSynchronize();

    for(int i = 0; i < ITERATIONS; ++i){
        pbvi_shared<<<BLOCKS, THREADS_PER_BLOCK>>>
        (dev_V, dev_Vtmp, dev_T, dev_O, dev_R, dev_B, 
            dev_best_action, pomdp.states, pomdp.observations,pomdp.actions, pomdp.discount);
        cudaDeviceSynchronize();
        swap(dev_V, dev_Vtmp);
    }
    float *dev_result;
    cudaMalloc((void **) &dev_result, pomdp.states * NUM_BELIEF * sizeof(float));

    pbvi_get<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_V, dev_result, pomdp.states);

    cudaMemcpy(result, dev_result, pomdp.states * NUM_BELIEF * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_action, dev_best_action, NUM_BELIEF * sizeof(float), cudaMemcpyDeviceToHost);
}

// Solves POMDP using Global Memory GPU PBVI
void pbvi_global(POMDP pomdp, float*result, int*best_action){

    float **dev_V;
    cudaMalloc((void**) &dev_V, NUM_BELIEF * sizeof(float *));

    float **dev_Vtmp; 
    cudaMalloc((void**) &dev_Vtmp, NUM_BELIEF * sizeof(float *));

    float *dev_T;
    cudaMalloc((void**) &dev_T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_T, pomdp.T, pomdp.actions * pomdp.states * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_O;
    cudaMalloc((void**) &dev_O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *));
    cudaMemcpy(dev_O, pomdp.O, pomdp.actions * pomdp.states * pomdp.observations * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_R;
    cudaMalloc((void**) &dev_R, pomdp.actions * pomdp.states * sizeof(float *));
    cudaMemcpy(dev_R, pomdp.R, pomdp.actions * pomdp.states * sizeof(float *), cudaMemcpyHostToDevice);

    float *dev_B;
    cudaMalloc((void**) &dev_B, NUM_BELIEF * pomdp.states * sizeof(float));
    cudaMemcpy(dev_B, pomdp.B, NUM_BELIEF * pomdp.states * sizeof(float), cudaMemcpyHostToDevice);

    int *dev_best_action;
    cudaMalloc((void **) &dev_best_action, NUM_BELIEF * sizeof(int));

    init_pbvi<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_V, dev_Vtmp, pomdp.init_value, pomdp.states);
    cudaDeviceSynchronize();

    for(int i = 0; i < ITERATIONS; ++i){
        pbvi_global<<<BLOCKS, THREADS_PER_BLOCK>>>
        (dev_V, dev_Vtmp, dev_T, dev_O, dev_R, dev_B, 
            dev_best_action, pomdp.states, pomdp.observations,pomdp.actions, pomdp.discount);
        cudaDeviceSynchronize();
        swap(dev_V, dev_Vtmp);
    }
    float *dev_result;
    cudaMalloc((void **) &dev_result, pomdp.states * NUM_BELIEF * sizeof(float));

    pbvi_get<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_V, dev_result, pomdp.states);
    cudaMemcpy(result, dev_result, pomdp.states * NUM_BELIEF * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(best_action, dev_best_action, NUM_BELIEF * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_best_action);
    cudaFree(dev_result);
    cudaFree(dev_V);
    cudaFree(dev_Vtmp);
    cudaFree(dev_R);
    cudaFree(dev_O);
    cudaFree(dev_T);
    cudaFree(dev_B);
}


// Solves POMDP using serial PBVI
void pbvi_cpu(POMDP pomdp, float* result, int* best_action){
    float **V = new float*[NUM_BELIEF];
    float **Vtmp = new float*[NUM_BELIEF];
    
    for (int b = 0; b < NUM_BELIEF; ++b){
        V[b] = new float[pomdp.states];
        Vtmp[b] = new float[pomdp.states];
        
        for (int state = 0; state < pomdp.states; ++state){
            V[b][state] = pomdp.init_value;
            Vtmp[b][state] = pomdp.init_value;
        }
    }
    
    float *hyperplane = new float[pomdp.states];
    float *local_hyperplane = new float[pomdp.states];
    float *best_local_hyperplane = new float[pomdp.states];
    
    for (int iteration = 0; iteration < ITERATIONS; ++iteration){
        for (int b = 0; b < NUM_BELIEF; ++b){
            float max = -INFINITY;;
            for (int action = 0; action < pomdp.actions; ++action){
                memset(hyperplane, 0, pomdp.states * sizeof(float));
                for (int observation = 0; observation < pomdp.observations; ++ observation){
                    float local_max = -INFINITY;;
                    memset(best_local_hyperplane, 0, pomdp.states * sizeof(float));   
                    /* for each hyper plane */
                    for (int i = 0; i < NUM_BELIEF; ++i){
                        for (int s = 0; s < pomdp.states; ++s){
                            local_hyperplane[s] =
                                pomdp.O[pomdp.states * pomdp.observations * action + observation] *
                                pomdp.T[pomdp.states * pomdp.states * action + pomdp.states * s] *
                                V[i][0];
                        }
                        for (int next_state = 1; next_state < pomdp.states; ++next_state){
                            for (int s = 0; s < pomdp.states; ++s){
                                local_hyperplane[s] +=
                                pomdp.O[pomdp.states * pomdp.observations * action + pomdp.observations * next_state + observation] *
                                pomdp.T[pomdp.states * pomdp.states * action + pomdp.states * s + next_state] *
                                V[i][next_state];
                            }
                        }
                        float local_value = 0;
                        for (int state = 0; state < pomdp.states; ++state){
                            local_value += local_hyperplane[state] * pomdp.B[pomdp.states * b + state];
                        }

                        if (local_value > local_max){
                            local_max = local_value;
                            swap(local_hyperplane, best_local_hyperplane);
                        }
                    }
                    for (int state = 0; state < pomdp.states; ++state){
                        hyperplane[state] += best_local_hyperplane[state];
                    }
                }
                for (int state = 0; state < pomdp.states; ++state){
                    hyperplane[state] = pomdp.R[pomdp.states * action + state] + pomdp.discount * hyperplane[state];
                }
                float value = 0;
                for (int state = 0; state < pomdp.states; ++state){
                    value += hyperplane[state] * pomdp.B[pomdp.states * b + state];
                }
                
                if(value > max){
                    max = value;
                    swap(hyperplane, Vtmp[b]);
                    best_action[b] = action;
                }
            }
        }
        swap(V, Vtmp);
    }
    for (int b = 0; b < NUM_BELIEF; ++b){
        for(int i = 0; i < pomdp.states; ++i){
            result[pomdp.states * b + i] = V[b][i];
        }       
    }
}

// updates the belief state given action and observation
float* update_belief(POMDP pomdp, int action, int observation, float *cur_belief){
    float* new_belief = new float[pomdp.states];
    memset(new_belief, 0, pomdp.states * sizeof(float));
    for(int new_state = 0; new_state < pomdp.states; ++new_state){
        for (int state = 0; state < pomdp.states; ++state){
            new_belief[new_state] += pomdp.T[action*pomdp.states * pomdp.states + state * pomdp.states + new_state] * cur_belief[state];
        }
        new_belief[new_state] *= pomdp.O[action * pomdp.states * pomdp.observations + new_state * pomdp.observations + observation];
    }
    float sum = 0;
    for(int new_state = 0; new_state < pomdp.states; ++new_state){
        sum += new_belief[new_state];
    }
    for(int new_state = 0; new_state < pomdp.states; ++new_state){
        new_belief[new_state] /= sum;
    }
    return new_belief;
}

// gives the best action given the current belief state
int get_action(POMDP pomdp, float *result, int *best_action,float* belief){
    float *vals = new float[NUM_BELIEF];
    memset(vals, 0, NUM_BELIEF * sizeof(float));
    for (int belief_point = 0; belief_point < NUM_BELIEF; ++belief_point){
        for (int state = 0; state < pomdp.states; ++state){
            vals[belief_point] += result[belief_point*pomdp.states + state] * belief[state];
        }
    }
    int aindx = -1;
    float max_value = -INFINITY;
    for (int belief_point = 0; belief_point< NUM_BELIEF; ++belief_point){
        if (vals[belief_point] > max_value){
            max_value = vals[belief_point];
            aindx = belief_point;
        }
    }
    return best_action[aindx];
}


// PBVI Agent for solving POMDPs
void simulate(POMDP pomdp, float *result, int *best_action, float *initial_belief, int& true_state, int& reward){
    int action = get_action(pomdp, result, best_action, initial_belief);
    int temp_reward = pomdp.R[action * pomdp.states + true_state];
    int trash = rand();
    reward += temp_reward;
    float* prob_next_state = new float[pomdp.states];
    memset(prob_next_state, 0, pomdp.states * sizeof(float));
    for (int next_state = 0; next_state < pomdp.states; ++next_state){
        prob_next_state[next_state] = pomdp.T[action*pomdp.states * pomdp.states + true_state * pomdp.states + next_state];
    }
    double val = (float)rand() / RAND_MAX;
    int next_state;
    if (val < prob_next_state[0])
        next_state = 0;
    else
        next_state = 1;

    int obs;
    float* prob_observation = new float[pomdp.observations];
    memset(prob_observation, 0, pomdp.observations * sizeof(float));
    for (int observation = 0; observation < pomdp.observations; ++observation){
        prob_observation[observation] = pomdp.O[action*pomdp.states * pomdp.observations + next_state * pomdp.observations + observation];
    }
    val = (float)rand() / RAND_MAX;
    if (val < prob_observation[0])       
        obs = 0;
    else

        obs = 1;

    float* new_belief = new float[pomdp.states];
    new_belief = update_belief(pomdp, action, obs, initial_belief);
    memcpy(initial_belief, new_belief, pomdp.states * sizeof(float));
    true_state = next_state;
}

// Random Agent for Solving POMDPs
void simulate_random(POMDP pomdp, float *initial_belief, int& true_state, int& reward){
    int trash = rand();
    int action = (rand() % (pomdp.actions -1 ));
    int temp_reward = pomdp.R[action * pomdp.states + true_state];
    
    reward += temp_reward;
    float* prob_next_state = new float[pomdp.states];
    memset(prob_next_state, 0, pomdp.states * sizeof(float));
    for (int next_state = 0; next_state < pomdp.states; ++next_state){
        prob_next_state[next_state] = pomdp.T[action*pomdp.states * pomdp.states + true_state * pomdp.states + next_state];
    }
    double val = (float)rand() / RAND_MAX;
    int next_state;
    if (val < prob_next_state[0])
        next_state = 0;
    else
        next_state = 1;

    int obs;
    float* prob_observation = new float[pomdp.observations];
    memset(prob_observation, 0, pomdp.observations * sizeof(float));
    for (int observation = 0; observation < pomdp.observations; ++observation){
        prob_observation[observation] = pomdp.O[action*pomdp.states * pomdp.observations + next_state * pomdp.observations + observation];
    }
    val = (float)rand() / RAND_MAX;
    if (val < prob_observation[0])       
        obs = 0;
    else

        obs = 1;

    float* new_belief = new float[pomdp.states];
    new_belief = update_belief(pomdp, action, obs, initial_belief);
    memcpy(initial_belief, new_belief, pomdp.states * sizeof(float));
    true_state = next_state;
}

// average of a list
float avg(float * score, int size)
{
    float total = 0;
    
    for (int i = 0; i < size; ++i)
    {
        total += score[i];
    }
    
    return ( total / (float)size );
}

int main() {  


    POMDP pomdp;
    
    // number of states
    pomdp.states = 2;
    
    // number of actions
    pomdp.actions = 3;
    
    // number of observations
    pomdp.observations = 2;
    
    // discount factor
    pomdp.discount = 0.75;

    // transition matrix
    float T[] = {1,0,0,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    pomdp.T = T;

    // observation matrix
    float O[] = {0.85,0.15,0.15,0.85,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    pomdp.O = O;

    // reward matrix
    float R[] = {-1,-1,-100,10,10,-100};
    pomdp.R = R;

    // belief points
    float *B = init_B_random_vector(NUM_BELIEF, pomdp.states);
    pomdp.B = B;

    // initial value
    float minimum = pomdp.R[0];
    
    for (int action = 0; action < pomdp.actions; ++action){
        for (int state = 0; state < pomdp.states; ++state){
            if (pomdp.R[pomdp.states * action + state] < minimum){
                minimum = pomdp.R[pomdp.states * action + state];
            }
        }
    }
    pomdp.init_value = (1/(1-pomdp.discount)) *minimum;
    

    // time for serial implementation
    timeval start_time_cpu, end_time_cpu;
    float elapsedTime_cpu;
    gettimeofday(&start_time_cpu, NULL);
    
    float *result_cpu = new float[pomdp.states * NUM_BELIEF];
    int *best_action_cpu = new int[NUM_BELIEF];
    pbvi_cpu(pomdp, result_cpu, best_action_cpu);

    gettimeofday(&end_time_cpu, NULL);
    elapsedTime_cpu = (end_time_cpu.tv_sec - start_time_cpu.tv_sec) * 1000.0;
    elapsedTime_cpu += (end_time_cpu.tv_usec - start_time_cpu.tv_usec) / 1000.0;
    printf("\n Solving the problem with serial implementation of PBVI takes %f ms \n", elapsedTime_cpu);


    // time for global memory implementation
    timeval start_time_global, end_time_global;
    float elapsedTime_global;
    gettimeofday(&start_time_global, NULL);
    
    float *result_global = new float[pomdp.states * NUM_BELIEF];
    int *best_action_global = new int[NUM_BELIEF];
    pbvi_global(pomdp, result_global, best_action_global);

    gettimeofday(&end_time_global, NULL);
    elapsedTime_global = (end_time_global.tv_sec - start_time_global.tv_sec) * 1000.0;
    elapsedTime_global += (end_time_global.tv_usec - start_time_global.tv_usec) / 1000.0;
    printf("\n Solving the problem with global gpu implementation of PBVI takes %f ms \n", elapsedTime_global);


    // time for shared memory implementation
    timeval start_time_shared, end_time_shared;
    float elapsedTime_shared;
    gettimeofday(&start_time_shared, NULL);
    
    float *result_shared = new float[pomdp.states * NUM_BELIEF];
    int *best_action_shared = new int[NUM_BELIEF];
    pbvi_shared(pomdp, result_shared, best_action_shared);

    gettimeofday(&end_time_shared, NULL);
    elapsedTime_shared = (end_time_shared.tv_sec - start_time_shared.tv_sec) * 1000.0;
    elapsedTime_shared += (end_time_shared.tv_usec - start_time_shared.tv_usec) / 1000.0;
    printf("\n Solving the problem with shared gpu implementation of PBVI takes %f ms \n", elapsedTime_shared);
    
    // Calculating Average reward for PBVI agent for 100000 steps
    struct timeval rand_time; 
    gettimeofday(&rand_time,NULL);
    srand((rand_time.tv_sec * 1000) + (rand_time.tv_usec / 1000));
    float *rewardshared_pbvi = new float[100];
    for (int k = 0; k <100; ++k){

        //
        int true_state =rand()%2;

        //initiali belief
        float belief[] = {0.5,0.5};
        int reward = 0;
        for(int i = 0; i < 10000; ++i){
            simulate(pomdp, result_global, best_action_global, belief, true_state, reward);
        }
        rewardshared_pbvi[k] = reward;
    }
    printf("\n Average reward for PBVI Agent for 10000 steps: %f \n", avg(rewardshared_pbvi, 100));


    // Calculating Average reward for Random agent for 100000 steps
    float *rewardshared_rand = new float[100];
    for (int k = 0; k <100; ++k){
        int true_state =rand()%2;
        float belief[] = {0.5,0.5};
        int reward = 0;
        for(int i = 0; i < 10000; ++i){
            simulate_random(pomdp, belief, true_state, reward);
        }
        rewardshared_rand[k] = reward;
    }
    printf("\n Average reward for Random Agent for 10000 steps: %f \n", avg(rewardshared_rand, 100));
    
}
