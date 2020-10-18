#include <stdint.h>
#include "../neural_network_parameters.h"
#include "../math/fixed_point_ops.h"

#ifndef BUDGET_DISTRIBUTION_GUARD
#define BUDGET_DISTRIBUTION_GUARD

#define PRIOR_COUNT 1

struct BudgetDistribution {
    int32_t levelCounts[NUM_OUTPUTS];
    int32_t observedEnergy[NUM_OUTPUTS];
    int32_t observedCounts[NUM_OUTPUT_FEATURES];
    int32_t estimatedCounts[NUM_OUTPUT_FEATURES];
    int32_t classCounts[NUM_OUTPUT_FEATURES][NUM_OUTPUTS];
};
typedef struct BudgetDistribution BudgetDistribution;

struct ConfidenceBound {
    int32_t lower;
    int32_t upper;
};
typedef struct ConfidenceBound ConfidenceBound;

BudgetDistribution *init_distribution(BudgetDistribution *distribution, int32_t classCounts[NUM_OUTPUT_FEATURES][NUM_OUTPUTS], int32_t maxSteps, uint16_t precision);
ConfidenceBound get_budget(int32_t target, int32_t step, int32_t maxSteps, int32_t priorEnergy[NUM_OUTPUTS], BudgetDistribution *distribution, uint16_t precision);
void update_distribution(uint16_t classIdx, uint16_t level, int32_t energy, BudgetDistribution *distribution, uint16_t precision);

#endif
