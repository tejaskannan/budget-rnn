#include <stdint.h>
#include "math/matrix.h"

#ifndef NEURAL_NETWORK_PARAMS_GUARD
#define NEURAL_NETWORK_PARAMS_GUARD

#define FIXED_POINT_PRECISION 10
#define STATE_SIZE 14
#define NUM_INPUT_FEATURES 10
#define NUM_OUTPUT_FEATURES 2
#define SEQ_LENGTH 20
#define NUM_OUTPUTS 10
#define STRIDE_LENGTH 1
#define SAMPLES_PER_SEQ 2
#define UGRNN_TRANSFORM
#define IS_SAMPLE_RNN
#define VECTOR_COLS 1

static int16_t THRESHOLDS[1][10] = { { 0,0,0,0,0,0,0,0,0,0 } };
static float BUDGETS[1] = { 100 };
static dtype EMBEDDING_LAYER_KERNEL[140] = { 223,-616,-148,-555,38,-80,282,68,-167,-400,-142,217,532,-472,353,-414,-343,-71,478,-145,419,-288,-535,362,190,426,-447,-399,109,204,-30,303,-550,281,-404,-279,-98,197,354,-126,444,52,-363,-157,-203,-370,278,156,265,307,-646,-360,445,717,-504,202,312,231,-280,-539,-48,172,13,-375,124,-321,-291,-116,132,14,-90,-169,-295,286,406,-149,85,-26,-550,-262,329,-556,164,331,-530,374,-557,144,-383,119,323,458,113,-441,-30,171,-509,-472,-189,529,456,367,282,530,-188,-267,-296,-409,94,525,191,-340,411,470,-76,-51,-7,-438,-214,-254,381,430,285,13,250,2,357,-476,213,281,-448,389,432,230,-200,116,348,403,-613,-633 };
static matrix EMBEDDING_LAYER_KERNEL_MAT_VAR = { EMBEDDING_LAYER_KERNEL, 14, 10 };
static matrix * EMBEDDING_LAYER_KERNEL_MAT = &EMBEDDING_LAYER_KERNEL_MAT_VAR;

static dtype EMBEDDING_LAYER_BIAS[14] = { -421,434,211,-158,466,-157,-345,321,686,-143,-119,-403,396,-373 };
static matrix EMBEDDING_LAYER_BIAS_MAT_VAR = { EMBEDDING_LAYER_BIAS, 14, 1 };
static matrix * EMBEDDING_LAYER_BIAS_MAT = &EMBEDDING_LAYER_BIAS_MAT_VAR;

static dtype RNN_CELL_W_TRANSFORM[784] = { 239,208,338,-92,47,180,2,-207,-395,206,-285,330,159,-79,-202,78,-380,-249,-566,-57,-150,-122,141,5,228,267,282,137,-205,414,90,-377,362,-195,216,237,-291,405,422,-71,343,50,-384,-57,224,-67,-371,-172,67,-489,-321,-194,-151,-69,154,102,-230,216,70,-11,53,83,158,37,137,-141,-264,241,-18,-113,523,-458,-468,367,-25,-363,-213,-271,-432,-542,-485,-275,-42,191,-213,-122,-95,-49,-297,-207,305,327,-167,13,145,202,-366,242,-110,216,-86,144,-214,-356,194,-67,-28,-113,-47,106,-7,-278,124,-238,177,-35,122,-370,-181,-98,-272,-254,-90,-4,-66,-64,253,-150,-68,-178,-65,-175,335,180,-220,-279,-218,-127,26,-252,-95,7,-125,88,0,-224,-30,71,-21,-467,73,-158,114,207,317,104,79,-96,325,-2,304,197,150,371,126,-273,90,246,-112,-144,-201,504,-356,128,-149,-333,104,89,-169,-54,421,-337,378,-176,207,-133,-141,-371,271,-242,129,-464,-18,132,-63,-29,-210,293,324,239,240,-125,-270,-350,240,261,253,-42,-221,180,-563,-56,-127,-106,120,-171,187,78,-197,366,-89,-373,7,-386,298,-36,13,150,-212,225,-150,238,-222,-44,-292,-215,-101,-208,-155,344,-115,200,-267,4,66,-7,114,225,72,-236,-35,-124,275,375,223,142,306,134,112,12,-313,2,102,217,-64,-211,73,-99,-277,227,421,-92,-279,-78,125,-644,-227,218,-287,148,-472,437,-513,8,234,-135,-123,-406,-101,221,544,-118,-47,-24,315,-235,65,72,-238,78,-116,71,-64,257,-289,-218,117,333,170,8,-202,-294,173,251,26,147,6,286,240,-147,285,305,-173,31,150,180,298,-222,329,-279,-261,338,294,-184,-197,-173,222,308,-248,-126,394,-259,-222,133,119,-147,-34,182,295,56,-49,-24,-113,-327,-263,310,305,135,262,264,173,-100,-419,2,140,188,-8,304,-176,-201,212,-55,32,-57,318,33,-336,375,12,-223,-419,290,248,193,-278,68,-428,311,167,7,396,44,-77,405,18,-345,236,-308,262,-103,-207,-161,83,-79,76,395,427,192,-349,-185,126,165,63,-250,311,183,11,-39,-152,62,320,24,-460,-160,33,-11,-186,184,-148,-267,-85,205,129,-18,168,-238,150,68,-51,-28,-313,-146,-195,-274,11,125,-137,-256,-326,98,73,-29,-94,-107,-94,-175,139,-110,-266,63,0,-362,-35,-146,-152,394,-273,475,-36,313,186,-381,386,-41,-17,-109,-255,-282,153,-180,188,-313,-364,-287,229,-152,-162,-208,-166,46,-99,-227,113,316,-90,461,-159,60,260,30,-120,-236,248,258,29,99,325,-273,-4,105,142,155,-382,278,-317,10,216,153,58,28,76,213,-152,121,84,-82,36,-150,-130,-9,-32,31,-125,-251,-13,314,-182,-15,-57,-203,108,-265,-265,88,-77,-185,-118,-275,-77,136,-160,-139,401,76,-38,-208,-247,313,143,-79,117,197,-541,-164,-107,-63,266,-310,-191,-30,255,249,-167,437,-230,-43,-178,119,-3,-183,121,19,-111,456,-236,-407,162,-112,200,-17,-338,-89,-119,17,256,-113,-169,128,225,59,95,221,207,155,19,-5,138,-320,227,51,-28,-164,-61,-474,-273,-400,241,-174,268,-322,-25,-85,263,149,228,202,250,-113,-77,248,-162,-237,117,-11,-2,-71,181,-57,-24,265,217,-90,216,248,-267,78,101,316,-234,228,206,-89,98,334,-90,-48,205,-457,328,185,106,-122,2,390,-165,342,-233,-113,-97,-46,-295,341,-41,-346,-448,-292,164,137,346,-225,-291,-23,430,154,-72,119,266,164,-138,-326,-100,-453,229,-320,257,270,-220,-357,146,-153,-149,-323,-60,27,-69,151,48,210,-326,-31,350,-362,230,-108,227,-145,242,127,28,93,-189,-118,-190,-70,162,-186,-128,163,-20,385,-145,53,-50,-159,196,-198,-303,37,31,-197,227,-293,-20,-29,55,-303,-241,-2,-65,-3,-232,113,-212,271,-97,166,168,322,-151,-210,-190,-103,387,-86,-206,-299,193,-43,85,367,230,346,153,-311,268,98,83,-41,136,213,-49,-224,34,-301 };
static matrix RNN_CELL_W_TRANSFORM_MAT_VAR = { RNN_CELL_W_TRANSFORM, 28, 28 };
static matrix * RNN_CELL_W_TRANSFORM_MAT = &RNN_CELL_W_TRANSFORM_MAT_VAR;

static dtype RNN_CELL_B_TRANSFORM[28] = { -291,-140,158,-148,-467,4,240,-827,-269,165,-31,-92,-442,-574,314,-433,40,-373,-178,-163,425,244,-105,-376,-386,-93,-20,-405 };
static matrix RNN_CELL_B_TRANSFORM_MAT_VAR = { RNN_CELL_B_TRANSFORM, 28, 1 };
static matrix * RNN_CELL_B_TRANSFORM_MAT = &RNN_CELL_B_TRANSFORM_MAT_VAR;

static dtype STOP_PREDICTION_HIDDEN_0_KERNEL[56] = { -471,251,-287,-397,-74,356,-51,158,-156,272,-405,372,-145,516,446,99,311,574,-535,-496,144,-567,-450,-591,-146,310,-69,-314,11,-113,84,-262,517,541,-480,510,-174,433,363,-3,-142,369,-158,-200,513,-430,410,-48,68,-253,533,457,552,110,218,-543 };
static matrix STOP_PREDICTION_HIDDEN_0_KERNEL_MAT_VAR = { STOP_PREDICTION_HIDDEN_0_KERNEL, 4, 14 };
static matrix * STOP_PREDICTION_HIDDEN_0_KERNEL_MAT = &STOP_PREDICTION_HIDDEN_0_KERNEL_MAT_VAR;

static dtype STOP_PREDICTION_HIDDEN_0_BIAS[4] = { -357,683,-700,637 };
static matrix STOP_PREDICTION_HIDDEN_0_BIAS_MAT_VAR = { STOP_PREDICTION_HIDDEN_0_BIAS, 4, 1 };
static matrix * STOP_PREDICTION_HIDDEN_0_BIAS_MAT = &STOP_PREDICTION_HIDDEN_0_BIAS_MAT_VAR;

static dtype STOP_PREDICTION_OUTPUT_KERNEL[4] = { 746,904,-142,-591 };
static matrix STOP_PREDICTION_OUTPUT_KERNEL_MAT_VAR = { STOP_PREDICTION_OUTPUT_KERNEL, 1, 4 };
static matrix * STOP_PREDICTION_OUTPUT_KERNEL_MAT = &STOP_PREDICTION_OUTPUT_KERNEL_MAT_VAR;

static dtype STOP_PREDICTION_OUTPUT_BIAS[1] = { 629 };
static matrix STOP_PREDICTION_OUTPUT_BIAS_MAT_VAR = { STOP_PREDICTION_OUTPUT_BIAS, 1, 1 };
static matrix * STOP_PREDICTION_OUTPUT_BIAS_MAT = &STOP_PREDICTION_OUTPUT_BIAS_MAT_VAR;

static dtype OUTPUT_LAYER_HIDDEN_0_KERNEL[280] = { -388,-338,275,611,36,-205,-461,-15,-292,168,-547,-28,-92,-581,292,-69,544,444,16,264,85,183,37,-84,-1,16,-451,-399,-127,173,-490,-356,-160,-240,-5,492,85,-293,-112,-128,457,525,-184,346,137,103,285,-107,-470,-112,100,-132,-655,178,-253,90,-277,-104,-367,-244,261,-420,548,-261,-231,195,126,-56,-12,103,54,311,460,309,-194,416,309,78,-304,343,181,236,-344,-188,157,-269,691,434,73,-233,-242,-442,-142,172,-143,-346,427,-619,60,184,108,-348,-159,-321,-329,-241,-158,82,369,17,-24,535,197,-101,-311,-156,58,249,157,105,-220,-197,277,68,-238,420,119,33,-372,-316,-220,-69,198,-148,-70,349,-110,-221,366,401,-63,337,326,-91,390,-90,207,-421,423,445,-425,352,-500,55,-399,220,576,166,-140,326,128,-431,-47,38,-427,97,341,-60,-291,-336,667,272,-483,137,135,121,-117,86,-40,9,-271,-27,135,-330,-536,-93,79,155,81,-211,161,-102,141,90,157,175,-67,239,-35,-560,148,321,-114,247,-76,-290,276,-132,506,-229,429,237,-69,106,-289,-136,557,136,46,-265,228,-297,-317,65,253,-213,-318,-39,336,136,581,334,122,-493,545,-441,-16,-125,-6,-325,-208,165,225,-398,57,146,237,-239,502,354,-82,367,-420,144,550,42,385,-150,-556,-425,240,-93,-376,24,-47,-402,-250,-7,80,-233,309,94,504,-229,231,25,150,76,381,475 };
static matrix OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT_VAR = { OUTPUT_LAYER_HIDDEN_0_KERNEL, 20, 14 };
static matrix * OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT = &OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT_VAR;

static dtype OUTPUT_LAYER_HIDDEN_0_BIAS[20] = { 331,281,288,-530,-485,147,461,685,199,-599,-630,551,-213,163,640,-323,-329,13,591,465 };
static matrix OUTPUT_LAYER_HIDDEN_0_BIAS_MAT_VAR = { OUTPUT_LAYER_HIDDEN_0_BIAS, 20, 1 };
static matrix * OUTPUT_LAYER_HIDDEN_0_BIAS_MAT = &OUTPUT_LAYER_HIDDEN_0_BIAS_MAT_VAR;

static dtype OUTPUT_LAYER_OUTPUT_KERNEL[40] = { 405,-128,-171,535,169,343,452,-545,-52,-417,52,353,555,-246,-316,-400,310,-90,578,364,-129,-480,243,81,524,-513,-436,-106,356,63,-218,-568,183,384,383,336,555,487,4,574 };
static matrix OUTPUT_LAYER_OUTPUT_KERNEL_MAT_VAR = { OUTPUT_LAYER_OUTPUT_KERNEL, 2, 20 };
static matrix * OUTPUT_LAYER_OUTPUT_KERNEL_MAT = &OUTPUT_LAYER_OUTPUT_KERNEL_MAT_VAR;

static dtype OUTPUT_LAYER_OUTPUT_BIAS[2] = { -120,-19 };
static matrix OUTPUT_LAYER_OUTPUT_BIAS_MAT_VAR = { OUTPUT_LAYER_OUTPUT_BIAS, 2, 1 };
static matrix * OUTPUT_LAYER_OUTPUT_BIAS_MAT = &OUTPUT_LAYER_OUTPUT_BIAS_MAT_VAR;
#endif
