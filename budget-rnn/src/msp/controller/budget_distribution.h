#include <stdint.h>
#include "../neural_network_parameters.h"
#include "../math/fixed_point_ops.h"

#ifndef BUDGET_DISTRIBUTION_GUARD
#define BUDGET_DISTRIBUTION_GUARD

#define PRIOR_COUNT 1
#define UPDATE_WINDOW 1

struct BudgetDistribution {
    int32_t levelCounts[NUM_OUTPUTS];  // Counts number of samples ending at each level (fixed point)
    int32_t observedEnergy[NUM_OUTPUTS];  // Sum of energy observed per level (fixed point)
    int32_t observedCounts[NUM_OUTPUT_FEATURES];  // Counts of observed labels based on model predictions (fixed point)
    int32_t estimatedCounts[NUM_OUTPUT_FEATURES];  // Estimated count of each label from validation set (fixed point)
    int32_t classCounts[NUM_OUTPUT_FEATURES][NUM_OUTPUTS];  // Observed + Prior counts of level distribution for each label (fixed point) 
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
