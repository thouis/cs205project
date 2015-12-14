
// Global Headers
#include <stdio.h>

typedef struct {
    int actions;
    int states;
    int observations;
    int *hyperplane_action;
    float discount;
    float init_value;
    float *B;
    float *R;
    float *O;
    float *T;
}POMDP;
