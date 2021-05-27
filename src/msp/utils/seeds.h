/*
 * This file contains the hashing seeds for all layers and variables. This matches the Python implementation
 * used during model training. This alignment ensures that the implementations
 * mirror each other.
 */

#ifndef HASH_SEEDS_GUARD
#define HASH_SEEDS_GUARD

    static const char *TRANSFORM_SEED = "tr";
    static const char *EMBEDDING_SEED = "em";
    static const char *AGGREGATE_SEED = "ag";
    static const char *OUTPUT_SEED = "ou";
    static const char *UPDATE_SEED = "up";
    static const char *RESET_SEED = "rs";
    static const char *CANDIDATE_SEED = "cd";
    static const char *FUSION_SEED = "fs";

#endif
