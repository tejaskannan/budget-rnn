#include <stdint.h>
#include "math/matrix.h"
#include "layers/cells.h"
#include "layers/layers.h"
#include "math/matrix_ops.h"
#include "math/fixed_point_ops.h"

#ifndef NEURAL_NETWORK_GUARD
#define NEURAL_NETWORK_GUARD

static int FIXED_POINT_PRECISION = 8;

enum ModelClass { STANDARD = 0, ADAPTIVE = 1 };
enum ModelType { VANILLA = 0, SAMPLE = 1, BOW = 2, RNN = 3, BIRNN = 4 };
enum OutputType { REGRESSION = 0, BINARY_CLASSIFICATION = 1, MULTI_CLASSIFICATION = 2 };
static enum ModelClass MODEL_CLASS = STANDARD;
static enum ModelType MODEL_TYPE = RNN;
static enum OutputType OUTPUT_TYPE = MULTI_CLASSIFICATION;
static int16_t STATE_SIZE = 16;
static int16_t SEQ_LENGTH = 20;
static int16_t SAMPLES_PER_SEQ = 20;
static int16_t NUM_INPUT_FEATURES = 3;
static int16_t NUM_OUTPUT_FEATURES = 6;
static int16_t INPUT_MEAN[3] = { 168,1855,106 };
static int16_t INPUT_STD[3] = { 1753,1727,1218 };
static int16_t OUTPUT_MEAN[6] = { 0,0,0,0,0,0 };
static int16_t OUTPUT_STD[6] = { 256,256,256,256,256,256 };

static int16_t EMBEDDING_BIAS_0_NUM_DIMS = 1;
static int16_t EMBEDDING_BIAS_0_DIMS[1] = { 16 };
static int16_t EMBEDDING_BIAS_0[16] = { -35,53,74,-25,-24,55,-35,59,19,1,-73,50,64,-31,0,-20 };
static matrix EMBEDDING_BIAS_0_MAT_VAR = { 16, 1, EMBEDDING_BIAS_0 };
static matrix * EMBEDDING_BIAS_0_MAT = &EMBEDDING_BIAS_0_MAT_VAR;

static int16_t EMBEDDING_KERNEL_0_NUM_DIMS = 2;
static int16_t EMBEDDING_KERNEL_0_DIMS[2] = { 16,3 };
static int16_t EMBEDDING_KERNEL_0[48] = { 6,-52,-76,-103,8,-97,-73,-158,-70,19,-90,-53,-8,35,181,62,29,54,154,-165,-1,29,-114,7,-101,-89,135,-56,-37,79,-5,138,16,51,110,-71,26,147,-104,-31,2,125,107,-24,42,-80,1,-50 };
static matrix EMBEDDING_KERNEL_0_MAT_VAR = { 16, 3, EMBEDDING_KERNEL_0 };
static matrix * EMBEDDING_KERNEL_0_MAT = &EMBEDDING_KERNEL_0_MAT_VAR;

static int16_t OUTPUT_HIDDEN_BIAS_0_NUM_DIMS = 1;
static int16_t OUTPUT_HIDDEN_BIAS_0_DIMS[1] = { 32 };
static int16_t OUTPUT_HIDDEN_BIAS_0[32] = { 20,-33,-12,-13,14,31,26,25,-18,-6,-13,-22,-12,32,23,-70,62,-29,24,-29,19,0,18,-2,41,-9,-29,-2,-40,-22,17,21 };
static matrix OUTPUT_HIDDEN_BIAS_0_MAT_VAR = { 32, 1, OUTPUT_HIDDEN_BIAS_0 };
static matrix * OUTPUT_HIDDEN_BIAS_0_MAT = &OUTPUT_HIDDEN_BIAS_0_MAT_VAR;

static int16_t OUTPUT_HIDDEN_KERNEL_0_NUM_DIMS = 2;
static int16_t OUTPUT_HIDDEN_KERNEL_0_DIMS[2] = { 32,16 };
static int16_t OUTPUT_HIDDEN_KERNEL_0[512] = { 48,-20,22,22,-33,8,-3,-3,71,54,-42,60,53,4,-73,-60,10,-18,-43,22,-46,-71,32,-23,30,6,42,-66,12,32,56,-29,-34,96,20,7,-77,13,29,22,-4,65,-48,-62,-65,69,-26,-31,33,-64,36,-37,-24,52,44,37,9,-4,67,-12,-82,-35,25,-18,-116,-51,-25,28,-6,-112,62,-79,-122,76,1,-104,20,-60,14,-27,1,-35,-39,-12,15,-37,5,53,-8,-14,48,-29,1,-67,-28,-15,25,-19,8,41,-45,0,3,2,74,-48,1,93,72,-19,-111,-104,-78,25,-33,88,-100,-93,3,-22,-21,104,-50,19,43,-25,-49,22,19,-86,44,-35,15,-37,-31,100,-18,-47,37,-4,7,-85,8,15,-36,111,28,67,-23,-49,37,39,-43,-32,-76,35,-19,59,11,38,-40,-119,14,-44,-21,30,36,47,58,-54,91,67,-69,-39,39,12,13,-50,-4,9,7,95,15,23,83,-44,46,122,24,45,6,-8,13,19,54,-58,19,50,-9,36,61,-42,6,49,18,-29,-40,22,-2,-2,129,36,16,-25,-106,-7,53,-37,-3,34,91,5,10,-48,57,-42,115,43,54,33,-76,-52,-30,25,-17,9,16,23,-128,-35,61,15,-69,-19,-34,-31,50,-37,-2,4,-40,39,-33,26,73,23,-13,43,27,34,9,36,-12,36,60,-52,20,13,72,53,-86,-81,109,-35,-18,29,35,91,-41,55,45,-48,-32,93,-2,-12,2,-23,98,-89,-36,-22,108,72,6,48,10,-24,60,-28,-47,-2,-24,-50,-2,7,0,-11,-21,-6,157,18,18,-22,33,-77,-33,44,129,126,1,8,90,-25,43,-17,-107,-1,-30,-17,-31,30,15,-17,-129,-68,69,-107,-6,-70,78,71,0,87,8,-101,81,-4,-26,-97,16,-18,0,-25,41,-19,-52,12,-24,-11,-74,-12,70,-32,56,-8,-14,-102,35,5,-118,-10,-24,20,97,29,23,-18,-58,-27,23,-26,62,59,15,-41,63,20,9,51,15,-57,14,-1,23,54,4,-28,-55,5,-73,59,-47,81,-77,11,-52,-45,-68,52,11,9,-8,89,3,-19,-23,-96,-6,-56,85,-19,7,72,-1,-12,96,21,26,-78,-28,-7,-15,-98,-67,-14,24,-25,-35,70,32,-90,75,-16,24,-74,-19,-7,-21,-54,-113,39,43,19,132,5,-52,25,-13,-46,-40,26,76,108,-28,-20,-80,-51,60,-19,37,-55,-48,-17,-19,-77,18,-25,21,68,-67,-7,24,-3,-21,-98,13,33,-79,102,-39,-78,14,4,-13,-7,-8,-30,53,8,-8,-17,-152,18,64,51,2,6,89,-13,-73,-138 };
static matrix OUTPUT_HIDDEN_KERNEL_0_MAT_VAR = { 32, 16, OUTPUT_HIDDEN_KERNEL_0 };
static matrix * OUTPUT_HIDDEN_KERNEL_0_MAT = &OUTPUT_HIDDEN_KERNEL_0_MAT_VAR;

static int16_t OUTPUT_HIDDEN_BIAS_1_NUM_DIMS = 1;
static int16_t OUTPUT_HIDDEN_BIAS_1_DIMS[1] = { 32 };
static int16_t OUTPUT_HIDDEN_BIAS_1[32] = { -76,9,-5,-69,82,-48,-54,-46,-22,-6,57,-12,-51,84,-69,-21,-94,64,19,-53,38,51,-82,48,-47,78,13,-25,-37,-64,-3,-57 };
static matrix OUTPUT_HIDDEN_BIAS_1_MAT_VAR = { 32, 1, OUTPUT_HIDDEN_BIAS_1 };
static matrix * OUTPUT_HIDDEN_BIAS_1_MAT = &OUTPUT_HIDDEN_BIAS_1_MAT_VAR;

static int16_t OUTPUT_HIDDEN_KERNEL_1_NUM_DIMS = 2;
static int16_t OUTPUT_HIDDEN_KERNEL_1_DIMS[2] = { 32,32 };
static int16_t OUTPUT_HIDDEN_KERNEL_1[1024] = { 41,52,-1,48,46,-58,37,-30,54,10,-25,32,-38,-14,-86,64,-12,-21,-20,61,12,20,-90,65,-55,42,-49,-15,72,12,-36,-40,31,31,-20,-48,54,-54,-41,1,-10,-17,-83,34,32,-15,-49,43,-18,31,-2,-54,-8,6,29,-10,-2,26,-49,-61,-29,32,56,20,16,-47,61,-10,68,-29,41,16,-78,29,-46,-84,-25,-14,2,16,60,-23,-49,50,-51,-120,34,-21,10,77,-34,-69,31,-58,-5,-25,-54,-56,30,60,-93,-7,59,22,-1,-26,58,-2,59,-7,-29,-26,18,0,2,-17,44,-47,-66,-40,-21,32,-14,3,-22,58,-61,-2,-53,-31,-8,-22,-32,-15,49,39,-50,-43,-24,-35,21,-38,-11,-6,45,52,-36,-34,23,-22,-31,76,42,71,-36,-5,2,-22,37,-35,20,55,97,-78,32,-30,-49,35,-117,79,-41,17,-43,41,46,102,-34,-78,-92,-71,-13,-80,-20,-48,-13,81,-38,-54,14,6,75,46,-65,-18,26,12,6,54,-16,6,-21,58,30,40,10,12,13,-17,-50,20,-11,28,-57,-3,21,29,28,-1,37,-75,54,-19,55,-5,2,-69,12,18,-19,35,-65,-37,21,-31,-64,-19,44,-55,-5,58,46,69,-28,8,-29,17,5,31,75,20,20,28,-21,-61,29,0,58,-44,-48,-2,-63,-35,48,-37,-5,20,36,57,14,-55,-15,49,-45,88,-9,-24,9,-10,49,-52,54,-43,5,28,-31,16,6,46,-18,-42,-41,-19,-10,-22,55,-38,75,-63,25,43,-29,31,50,51,5,10,7,-15,-22,26,-60,44,1,-35,-23,-17,-22,0,-32,32,-12,-20,-52,3,128,5,-49,35,-60,27,0,-110,-66,-22,-46,-74,-87,-118,-32,42,42,-18,72,-61,0,76,-36,-48,27,103,-3,0,-39,33,-55,44,38,67,-25,6,68,-36,-11,33,3,-7,24,-61,4,-28,57,30,20,18,-56,16,-58,-63,-61,-7,-3,-10,27,-40,-93,22,-73,-34,-10,55,-18,-3,-52,60,47,-29,-55,-58,-91,39,-113,-13,-48,84,-31,7,-83,0,15,18,45,-11,99,84,52,-133,22,-11,53,8,-44,50,-10,-11,30,-7,-37,-14,11,59,71,-73,38,48,-61,-4,-6,47,71,-26,71,47,-43,-57,-28,52,-43,69,-62,75,23,-2,15,-55,-64,11,49,-35,27,-58,-44,-69,-42,-22,38,32,39,-4,-66,14,-29,87,-30,13,-30,17,48,45,-4,-77,-73,44,29,57,34,-9,26,32,-46,12,-4,-36,52,12,-60,43,-86,-46,47,65,-73,-31,-44,48,-47,4,42,29,19,65,-11,-44,40,67,-47,-47,-67,-51,-14,64,-63,-1,-10,4,-49,15,-36,40,-30,12,-4,74,-39,9,0,81,-12,-24,-21,-38,32,39,-70,1,67,10,-50,4,-9,-29,96,40,47,9,3,-7,9,84,9,-42,38,45,30,-131,96,12,-7,-58,77,16,-44,-37,-45,-61,-21,96,-29,-68,-52,97,-61,104,-5,-47,115,-100,65,59,32,-7,-42,21,-2,-14,112,-22,30,40,27,8,30,-46,80,52,18,-13,-57,-42,31,-42,66,-28,-30,-49,1,19,8,17,-23,20,5,-12,40,34,9,-13,-45,29,-4,-68,-36,30,-28,72,11,10,19,-74,10,-11,-16,-9,-44,7,-31,-45,6,2,26,-63,-5,-44,40,61,69,-9,-4,-84,-5,-33,4,-10,-49,9,53,-16,-42,-67,-50,-57,-31,-12,-29,11,-56,-18,-44,48,2,23,-14,-89,61,39,0,-2,58,9,32,41,66,-20,22,-20,-23,-17,23,21,90,56,8,63,34,-18,40,57,-13,31,-27,33,-30,-58,50,-47,21,-52,-11,-39,39,42,-95,-52,-28,-50,-53,66,21,10,29,30,32,7,61,-9,-17,48,7,42,10,13,-20,-3,29,-13,-41,41,-66,72,15,76,-33,-28,-20,34,58,-101,91,-12,19,4,8,-41,12,-38,-38,-49,-15,83,25,-16,-30,67,18,21,38,-59,17,41,0,55,72,-38,-3,-17,38,68,-8,-8,22,6,-38,39,-24,9,63,-46,-34,21,-89,-20,27,1,91,-86,-7,-7,47,92,-70,35,-53,6,-82,-16,28,-2,61,-57,-45,-72,-43,-65,-79,-23,-81,49,-90,-7,-27,-1,40,47,-32,-69,30,-16,2,5,-44,-6,67,-77,7,79,78,6,-7,-36,-48,47,65,-13,-49,-12,1,44,52,-83,97,75,19,-14,-7,18,32,-1,-53,-21,-48,-42,-15,-33,-47,4,20,85,49,29,-12,-8,37,0,53,18,-59,28,-7,-27,51,17,-34,25,21,-46,-45,-39,-82,34,32,0,27,85,-83,19,1,-45,38,-21,-72,-73,-85,39,-41,31,-27,87,-75,-43,17,-4,-48,-5,-45,-29,125,20,19,-4,-24,35,52,-18,-22,-42,-60,-10,22,-38,-14,4,54,-7,-15,64,-45,58,-46,-26,-87,7,-36,63,-32,5,-32,-49,10,-50,26,-47,18,22,-49,1,4,-12,58,3,-63,-60,25,21,13,54,29,1,-38,27,-41,63,30,23,29,-46,-8,11,19,-5,36,-37,-51,40,-44,39,-44,14,36,-24,-2,-40,-24,-7,-56,34,-38,-48,-98,61,-51,-24,34,-6,18,61,-40,-35,-51,-31,-59,-54,78,63,-33,-81 };
static matrix OUTPUT_HIDDEN_KERNEL_1_MAT_VAR = { 32, 32, OUTPUT_HIDDEN_KERNEL_1 };
static matrix * OUTPUT_HIDDEN_KERNEL_1_MAT = &OUTPUT_HIDDEN_KERNEL_1_MAT_VAR;

static int16_t OUTPUT_KERNEL_0_NUM_DIMS = 2;
static int16_t OUTPUT_KERNEL_0_DIMS[2] = { 6,32 };
static int16_t OUTPUT_KERNEL_0[192] = { -74,5,20,6,52,18,-56,-35,-20,13,-17,-18,26,59,-38,-27,-56,6,-3,30,12,45,-67,62,-38,63,-52,-8,-35,-50,0,-38,-71,-66,-66,73,-21,-68,-60,55,51,86,-92,24,24,50,-41,-34,-46,3,68,-43,-7,97,-37,77,69,15,49,87,-40,-49,65,-29,-8,-60,-65,-1,-111,-90,45,64,-34,21,23,72,128,-96,107,109,-14,-99,102,-40,-101,77,87,-77,4,-79,62,-11,48,-14,-7,91,56,-16,6,69,-22,30,43,58,-18,7,-54,14,131,-70,78,113,95,-95,13,97,-66,-3,-32,-37,9,41,-47,52,54,93,-44,59,-57,-28,-4,-8,51,2,-48,-19,-37,15,-27,20,60,56,-29,11,-42,-19,22,-9,-18,38,-50,35,-21,43,-38,6,8,-23,-42,-24,-50,-32,7,42,21,8,5,15,7,-5,-24,24,42,40,-15,0,-39,-13,17,21,7,33,-57,24,-33,50,-25,-3,-29,-16,-23,-11 };
static matrix OUTPUT_KERNEL_0_MAT_VAR = { 6, 32, OUTPUT_KERNEL_0 };
static matrix * OUTPUT_KERNEL_0_MAT = &OUTPUT_KERNEL_0_MAT_VAR;

static int16_t TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0_NUM_DIMS = 1;
static int16_t TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0_DIMS[1] = { 16 };
static int16_t TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0[16] = { -35,24,8,-13,-22,-15,-15,7,-21,35,-36,-16,2,33,-8,-23 };
static matrix TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0_MAT_VAR = { 16, 1, TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0 };
static matrix * TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0_MAT = &TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0_MAT_VAR;

static int16_t TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0_NUM_DIMS = 2;
static int16_t TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0_DIMS[2] = { 16,32 };
static int16_t TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0[512] = { 10,43,45,49,22,-77,31,-16,-26,14,-18,-51,-62,64,-53,38,-107,-64,-5,-22,35,71,-65,60,48,-55,16,80,8,-21,0,-14,4,34,-41,48,-36,-21,-55,21,32,-30,42,31,34,75,-19,18,-91,-142,-8,118,-114,-28,31,-39,-94,80,-39,-10,37,30,-37,18,-31,-28,-92,0,87,45,-54,10,33,-11,79,17,20,48,42,19,53,-37,-31,69,4,-4,-103,-27,6,-6,-24,60,127,-39,-76,-121,59,-2,20,5,-30,-19,24,-42,57,10,-32,32,38,-11,-46,-30,-20,-19,17,-7,-85,-100,-35,-74,-53,38,-115,-14,70,-23,-87,-45,77,-76,-43,117,-40,7,-7,92,-2,26,-30,-43,0,23,68,-57,71,-5,-7,-48,-32,97,34,39,53,-98,0,15,-22,-66,-28,32,27,-62,-57,-71,3,-26,32,-98,-3,44,60,14,23,-17,3,39,40,-49,-13,-14,14,-75,3,47,56,-84,81,72,23,-86,18,-47,-60,1,47,55,89,5,55,-29,106,63,30,-69,11,9,38,17,-62,-2,-173,-18,-21,-16,-102,-24,-83,43,-16,-19,-2,50,80,55,18,32,48,-24,-3,1,-53,-37,73,44,35,25,-37,42,14,-44,132,-55,31,-81,103,0,28,14,108,-153,-16,59,-49,-49,-55,-18,46,-22,-34,79,-26,-52,14,-82,28,-67,49,-9,18,-76,36,80,37,-63,72,-4,6,53,23,22,-69,-79,78,49,28,-72,-58,-67,32,-63,-8,-3,-31,68,-46,-17,13,-51,-10,47,48,15,40,-66,-27,81,3,-20,-81,-72,84,-43,-29,-43,-58,-60,11,19,-6,-18,77,-69,-14,69,1,-60,77,83,27,5,-28,-59,20,-1,60,-41,96,-42,-3,-67,37,22,31,152,6,-5,-73,5,-47,-17,12,39,27,28,-64,-4,-31,-63,11,-77,9,-87,68,45,9,38,-53,-14,65,-75,80,1,17,59,-62,22,14,-4,84,-134,30,-14,-49,-78,1,40,47,142,-63,-58,27,74,-72,-13,14,10,8,8,44,75,6,36,78,53,-22,51,-35,44,11,59,-78,-14,-72,14,-67,-82,-40,22,-39,7,42,-31,-73,-12,-89,28,25,73,104,-74,28,34,-29,35,21,56,-5,30,-20,-69,-77,74,-1,-68,82,-95,25,29,60,-3,19,66,-26,-47,-40,-8,25,-53,-90,-12,26,24,17,36,-26,-17,-58,-70,-11,-52,138,24,-12,40,4,-34,-111,-3,-39,180,-77,-41,-54,-20,61,60,-60,-8,-3,-38,22,34,14,40,30,9,47,-20,-159,-70,-20,-20,104,-49,-58,21,14,-72,-12,-18,73,-3 };
static matrix TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0_MAT_VAR = { 16, 32, TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0 };
static matrix * TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0_MAT = &TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0_MAT_VAR;

static int16_t TRANSFORM_LAYER_CELL_GATES_BIAS_0_0_NUM_DIMS = 1;
static int16_t TRANSFORM_LAYER_CELL_GATES_BIAS_0_0_DIMS[1] = { 32 };
static int16_t TRANSFORM_LAYER_CELL_GATES_BIAS_0_0[32] = { 278,232,205,133,205,228,222,177,223,213,226,258,191,262,195,203,128,139,162,113,142,155,140,132,96,137,170,134,113,161,184,122 };
static matrix TRANSFORM_LAYER_CELL_GATES_BIAS_0_0_MAT_VAR = { 32, 1, TRANSFORM_LAYER_CELL_GATES_BIAS_0_0 };
static matrix * TRANSFORM_LAYER_CELL_GATES_BIAS_0_0_MAT = &TRANSFORM_LAYER_CELL_GATES_BIAS_0_0_MAT_VAR;

static int16_t TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0_NUM_DIMS = 2;
static int16_t TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0_DIMS[2] = { 32,32 };
static int16_t TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0[1024] = { 51,22,69,106,-26,-41,38,127,-51,-114,-148,3,40,-88,14,2,41,-47,-33,39,98,15,-93,-19,95,-119,15,52,16,-106,49,-28,138,49,64,143,-113,-17,-8,-87,-40,-73,44,-10,68,15,-125,109,165,-50,96,46,11,50,-96,62,172,-40,-71,150,-33,33,-94,-114,-13,-70,31,13,114,96,21,88,-33,0,-9,9,53,-10,115,-81,-43,6,-19,-1,-3,-102,91,-138,-110,73,23,-193,-21,49,92,175,-3,-210,-106,52,123,-38,177,61,-83,78,-17,-157,-168,53,220,-82,72,-139,54,-114,85,55,85,-9,-36,46,46,37,-29,-34,44,-13,-15,43,-51,-20,89,-41,-123,-35,67,107,87,-15,-7,64,-94,31,88,57,16,-20,23,-4,20,82,-10,-21,-128,-34,23,42,-44,-41,28,-49,-9,29,-32,12,-20,0,-88,4,11,51,78,-19,83,-39,78,-15,-27,20,80,97,-62,23,56,-47,-78,19,-4,-3,-16,55,-9,-36,-102,-60,4,77,-74,-43,173,148,17,-41,-51,40,61,0,-15,-3,-74,-114,-10,-95,117,-32,-4,97,24,-119,4,-7,34,53,36,-63,-98,8,72,35,202,-52,-177,-195,26,75,81,-38,139,-115,-93,-38,73,-23,82,79,27,-58,96,-39,42,-41,32,95,-62,-9,-75,20,6,8,-54,1,-134,-62,30,-5,113,58,43,39,-67,110,157,8,20,-20,5,146,-16,94,59,-117,-11,114,56,-66,17,-68,118,-81,-53,-7,-133,61,4,40,-96,-57,-7,169,36,-1,-13,-21,28,-61,34,9,93,120,-71,-31,179,-127,-49,25,58,22,14,-38,-94,-57,-14,-89,23,27,-30,13,153,61,-74,-138,-46,15,46,-1,56,-90,20,-31,5,100,-14,134,1,-91,41,-19,-24,12,-73,20,-5,26,-8,-93,23,-9,-63,-65,-35,8,109,17,89,-27,-97,-16,90,-10,24,-73,-22,83,18,52,48,8,16,116,48,47,-31,16,24,-130,-105,19,40,147,138,-3,-45,-61,99,-49,-39,-2,116,-228,11,1,-65,-96,44,-4,83,-44,-40,19,100,-56,-64,-76,60,48,-35,23,48,-43,-80,-16,-4,-5,-105,-66,63,89,96,-5,31,-38,-7,87,9,4,-34,20,-77,-34,-9,10,15,-26,98,53,-143,-12,36,-83,-61,-12,29,67,89,112,-88,19,14,-19,-61,60,46,-30,-64,-53,-108,40,-29,-19,121,-187,-123,100,-31,-102,-3,135,81,119,-115,-89,-41,-104,110,24,67,55,35,102,-21,-132,-97,132,3,33,-73,52,-117,-69,-45,-53,95,-68,-98,58,-32,-168,-32,79,74,-25,-85,-81,-143,-61,-17,48,24,-98,-106,-102,93,80,102,56,31,-51,18,-33,-12,-36,52,7,7,34,62,0,71,2,-7,-54,23,5,-2,47,-32,2,-58,-60,-102,-43,54,-4,144,40,66,74,-103,45,-31,-57,-20,-14,-1,-5,-11,-14,-35,-50,43,65,0,42,39,1,200,76,179,236,-128,-202,91,70,-3,-112,-50,-26,0,-158,-63,134,-9,-36,9,-71,-23,31,34,20,-8,-31,0,30,-84,30,0,22,111,2,15,44,31,-49,29,-149,44,30,93,-71,34,-8,-54,38,29,32,4,-34,-74,-37,53,10,74,-1,-76,33,24,18,-38,-7,-40,61,0,-136,47,-73,-76,-161,-21,1,166,11,66,-31,-114,94,58,-9,-23,-34,25,58,-2,5,77,-27,56,-6,5,11,34,48,101,17,19,48,56,-57,8,157,31,-22,-46,-75,-90,32,-23,0,-9,-22,18,-89,15,61,21,1,-51,11,-71,-37,-36,11,-85,40,-14,-21,14,56,160,13,58,-20,180,79,-22,-120,-85,139,-30,-32,29,81,-76,82,-31,-37,-28,-37,29,-25,-33,53,101,110,-75,-3,100,-48,-159,-3,-6,-82,20,-29,-2,-111,0,-12,-18,-36,8,-15,-3,23,-38,-10,31,39,74,6,-39,-14,-30,-24,-116,-16,81,14,-71,-98,-6,-140,73,43,30,-18,18,30,8,-90,0,125,-17,-28,-5,-20,-4,36,-12,29,27,-29,106,14,5,57,81,37,-6,-19,-28,-121,-147,-8,16,114,-25,-60,-89,-56,118,124,55,-30,103,-38,45,-69,-58,10,32,56,93,-29,55,-58,50,-40,20,-51,21,9,-93,38,-102,-106,7,-47,-129,-77,-22,27,136,59,53,30,-104,55,-32,39,-51,0,-15,0,-11,2,89,-38,56,17,6,63,2,-7,-39,-82,65,-12,26,-24,61,73,20,56,-66,-87,-59,21,101,0,-40,48,-42,35,-62,32,31,-28,55,4,-11,90,45,29,-33,-14,-54,-112,-114,19,87,-73,-91,-50,27,50,72,-29,-42,87,0,9,25,30,-57,32,-25,-29,-19,41,123,-31,-131,3,-3,114,-38,12,-47,50,-72,-104,32,-71,-74,-122,-147,-52,67,59,100,-49,-70,17,33,10,-44,-58,19,20,-28,68,-9,-80,34,22,-35,9,19,36,53,87,29,155,-80,-156,24,-40,-23,-35,-16,-33,-29,36,-85,94,-17,-9,0,-31,19,79,-40,40,16,45,-34,-24,-27,6,0,-18,-90,-117,-226,-205,193,59,-114,-227,-71,32,197,58,60,178,101,-82,100,19,-98,77,-4,-49,11,-13,96,-1,3,92,94,16,-71,32 };
static matrix TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0_MAT_VAR = { 32, 32, TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0 };
static matrix * TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0_MAT = &TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0_MAT_VAR;

int16_t execute_model(matrix **inputs, int16_t seqLength);

#endif
