#include "budget_distribution.h"

static int32_t energyEstimates[NUM_OUTPUTS];


BudgetDistribution *init_distribution(BudgetDistribution *distribution, int32_t classCounts[NUM_OUTPUT_FEATURES][NUM_OUTPUTS], int32_t maxSteps, uint16_t precision) {
    uint16_t i;

    for (i = NUM_OUTPUTS; i > 0; i--) {
        distribution->levelCounts[i-1] = 0;
        distribution->observedEnergy[i-1] = 0;
    }

    for (i = NUM_OUTPUT_FEATURES; i > 0; i--) {
        distribution->observedCounts[i-1] = 0;
    }

    uint16_t j;
    int32_t totalCount = 0;
    for (i = NUM_OUTPUT_FEATURES; i > 0; i--) {
        for (j = NUM_OUTPUTS; j > 0; j--) {
            distribution->classCounts[i-1][j-1] = classCounts[i-1][j-1];
            totalCount = fp32_add(classCounts[i-1][j-1], totalCount);
        }
    }

    int32_t countSum, frac;
    for (i = NUM_OUTPUT_FEATURES; i > 0; i--) {
        countSum = 0;

        for (j = NUM_OUTPUTS; j > 0; j--) {
            countSum = fp32_add(classCounts[i-1][j-1], countSum);
        }

        frac = fp32_div(countSum, totalCount, precision);
        distribution->estimatedCounts[i-1] = fp32_mul(frac, int_to_fp32(maxSteps, precision), precision);
    }

    return distribution;
}


ConfidenceBound get_budget(int32_t target, int32_t step, int32_t maxSteps, int32_t priorEnergy[NUM_OUTPUTS], BudgetDistribution *distribution, uint16_t precision) {
    /**
     * Computes the budget using the current distribution.
     *
     * Args:
     *  target: The target energy (Joules) budget (over all steps) in fixed-point representation
     *  step: The current step (minimum 1) as a standard integer
     *  maxSteps: The maximum number of steps as a standard integer
     *  priorEnergy: The estimated energy readings as fixed-point values
     *  distribution: The current distribution
     *  precision: Number of fractional bits in the fixed-point representation
     *
     * Returns:
     *  The lower and upper budget values.
     */

    // Estimate the energy consumed by each level
    uint16_t i = NUM_OUTPUTS;
    int32_t prior, denom, numerator;
    int32_t priorCount = int_to_fp32(PRIOR_COUNT, precision);
    for (; i > 0; i--) {
        prior = fp32_mul(priorCount, priorEnergy[i-1], precision);
        denom = fp32_add(priorCount, distribution->levelCounts[i-1]);
        numerator = fp32_add(prior, distribution->observedEnergy[i-1]);
        energyEstimates[i-1] = fp32_div(numerator, denom, precision);
    }

    // Compute the expected power and variance
    int32_t expectedRest = 0;
    int32_t varianceRest = 0;
    int32_t stepDelta = maxSteps - step;

    // Estimate the number remaining. This is a normalizing constant which
    // accounts for max(x, 0) clipping
    int32_t estRemaining = 0;
    int32_t countDiff;
    for (i = NUM_OUTPUT_FEATURES; i > 0; i--) {
        countDiff = fp32_sub(distribution->estimatedCounts[i-1], distribution->observedCounts[i-1]);

        if (countDiff < 0) {
            countDiff = 0;
        }

        estRemaining = fp32_add(countDiff, estRemaining);
    }

    uint16_t j;
    uint16_t classIdx;
    int32_t classCount;
    int32_t energyMean;
    int32_t energyVar;
    int32_t energyDiff;
    int32_t remainingFrac;
    int32_t weight;

    for (j = NUM_OUTPUT_FEATURES; j > 0; j--) {
        classIdx = j - 1;
        int32_t *classLevelCounts = distribution->classCounts[classIdx];

        // Count the number of elements
        classCount = 0;
        for (i = NUM_OUTPUTS; i > 0; i--) {
            classCount = fp32_add(classLevelCounts[i-1], classCount);
        }

        energyMean = 0;
        for (i = NUM_OUTPUTS; i > 0; i--) {
            weight = fp32_div(classLevelCounts[i-1], classCount, precision);
            energyMean = fp32_add(fp32_mul(weight, energyEstimates[i-1], precision), energyMean);
        }

        energyVar = 0;
        for (i = NUM_OUTPUTS; i > 0; i--) {
            weight = fp32_div(classLevelCounts[i-1], classCount, precision);
            energyDiff = fp32_sub(energyEstimates[i-1], energyMean);
            energyDiff = fp32_mul(energyDiff, energyDiff, precision);
            energyVar = fp32_add(fp32_mul(weight, energyDiff, precision), energyVar);
        }

        countDiff = fp32_sub(distribution->estimatedCounts[classIdx], distribution->observedCounts[classIdx]);
        if (countDiff < 0) {
            countDiff = 0;
        }

        remainingFrac = fp32_div(countDiff, estRemaining, precision);
        expectedRest = fp32_add(fp32_mul(remainingFrac, energyMean, precision), expectedRest);
        
        weight = fp32_div(countDiff, int_to_fp32(step, precision), precision);
        weight = fp32_mul(weight, weight, precision);
        varianceRest = fp32_add(fp32_mul(energyVar, weight, precision), varianceRest);
    }

    int32_t energyRest = (expectedRest * stepDelta) / 1000;  // Expected energy on the remaining sequences (in J)
    int32_t expectedEnergy = fp32_sub(target, energyRest) / step;
    expectedEnergy *= 1000; // Convert back to mJ

    if (expectedEnergy < priorEnergy[0]) {
        expectedEnergy = priorEnergy[0];
    } else if (expectedEnergy > priorEnergy[NUM_OUTPUTS-1]) {
        expectedEnergy = priorEnergy[NUM_OUTPUTS-1];
    }

    int32_t estimatorVar = varianceRest / step;
    int32_t estimatorStd = fp32_sqrt(estimatorVar, precision);

    ConfidenceBound bounds = { expectedEnergy - estimatorStd, expectedEnergy + estimatorStd };
    return bounds;
}


void update_distribution(uint16_t classIdx, uint16_t level, int32_t energy, BudgetDistribution *distribution, uint16_t precision) {
    int32_t one = int_to_fp32(1, precision);
    distribution->observedCounts[classIdx] = fp32_add(distribution->observedCounts[classIdx], one);
    distribution->classCounts[classIdx][level] = fp32_add(distribution->classCounts[classIdx][level], one);
    distribution->levelCounts[level] = fp32_add(distribution->levelCounts[level], one);
    distribution->observedEnergy[level] = fp32_add(distribution->observedEnergy[level], energy);
}
