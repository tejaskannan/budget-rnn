#include <stdint.h>
#include "math/matrix.h"

#ifndef NEURAL_NETWORK_PARAMS_GUARD
#define NEURAL_NETWORK_PARAMS_GUARD

#define FIXED_POINT_PRECISION 9
#define STATE_SIZE 20
#define NUM_INPUT_FEATURES 2
#define NUM_OUTPUT_FEATURES 10
#define SEQ_LENGTH 8
#define NUM_OUTPUTS 4
#define STRIDE_LENGTH 4
#define SAMPLES_PER_SEQ 2
#define NUM_BUDGETS 11
#define UGRNN_TRANSFORM
#define IS_BUDGET_RNN
#define VECTOR_COLS 1

static int16_t THRESHOLDS[11][4] = { { 272,0,0,0 },{ 290,0,0,0 },{ 304,0,0,0 },{ 320,0,0,0 },{ 322,282,24,19 },{ 320,326,0,0 },{ 345,347,0,0 },{ 322,367,0,0 },{ 351,360,331,0 },{ 475,366,352,0 },{ 423,364,441,0 } };
static int32_t BUDGETS[11] = { 3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192 };
static int16_t AVG_ENERGY[11] = { 3187,3665,4133,4593,5035,5585,6141,6446,7114,7576,8142 };
static int32_t ENERGY_ESTIMATES[4] = { 2745,4673,6602,8530 };
static int16_t LABEL_COUNTS[11][10][4] = { { { 119,1,0,0 },{ 86,61,0,0 },{ 150,20,0,0 },{ 145,3,0,0 },{ 145,4,0,0 },{ 127,36,0,0 },{ 81,99,0,0 },{ 116,4,0,0 },{ 51,59,0,0 },{ 106,48,0,0 } },{ { 104,17,0,0 },{ 21,128,0,0 },{ 70,97,0,0 },{ 122,26,0,0 },{ 118,36,0,0 },{ 118,45,0,0 },{ 21,133,0,0 },{ 105,18,0,0 },{ 28,106,0,0 },{ 57,91,0,0 } },{ { 53,77,0,0 },{ 1,144,0,0 },{ 9,159,0,0 },{ 41,103,0,0 },{ 102,63,0,0 },{ 88,63,0,0 },{ 12,137,0,0 },{ 66,59,0,0 },{ 9,139,0,0 },{ 28,108,0,0 } },{ { 13,124,0,0 },{ 0,147,0,0 },{ 0,179,0,0 },{ 0,151,0,0 },{ 33,128,0,0 },{ 7,124,0,0 },{ 2,142,0,0 },{ 0,128,0,0 },{ 2,160,0,0 },{ 4,117,0,0 } },{ { 0,138,3,0 },{ 0,118,26,0 },{ 0,177,1,0 },{ 0,152,0,0 },{ 0,158,5,0 },{ 0,93,36,0 },{ 0,42,100,0 },{ 0,130,0,0 },{ 0,108,57,0 },{ 0,71,46,0 } },{ { 13,35,97,0 },{ 0,93,57,0 },{ 0,95,72,0 },{ 0,104,47,0 },{ 33,66,60,0 },{ 7,52,72,0 },{ 2,17,120,0 },{ 0,114,20,0 },{ 2,51,111,0 },{ 4,21,96,0 } },{ { 0,6,145,0 },{ 0,60,102,0 },{ 0,18,137,0 },{ 0,47,105,0 },{ 0,37,123,0 },{ 0,39,91,0 },{ 0,3,129,0 },{ 0,96,42,0 },{ 0,33,131,0 },{ 0,10,107,0 } },{ { 0,0,151,0 },{ 0,29,133,0 },{ 0,2,152,0 },{ 0,7,145,0 },{ 0,2,158,0 },{ 0,15,115,0 },{ 0,0,132,0 },{ 0,36,103,0 },{ 0,21,143,0 },{ 0,6,111,0 } },{ { 0,0,74,78 },{ 0,39,82,40 },{ 0,4,128,22 },{ 0,18,30,103 },{ 0,11,43,106 },{ 0,23,51,57 },{ 0,1,108,23 },{ 0,58,62,23 },{ 0,26,76,59 },{ 0,7,45,64 } },{ { 0,0,44,108 },{ 0,30,56,75 },{ 0,2,123,29 },{ 0,9,17,125 },{ 0,5,13,142 },{ 0,18,44,69 },{ 0,0,17,115 },{ 0,40,54,49 },{ 0,21,61,79 },{ 0,7,30,79 } },{ { 0,0,0,152 },{ 0,31,0,130 },{ 0,3,0,151 },{ 0,11,0,140 },{ 0,6,0,154 },{ 0,19,0,112 },{ 0,0,0,132 },{ 0,47,0,96 },{ 0,23,0,138 },{ 0,7,0,109 } } };
static dtype EMBEDDING_LAYER_KERNEL[40] = { 356,-502,-312,659,-395,366,67,-579,-253,-401,220,441,155,46,-397,-268,-309,-370,237,148,-354,-162,-335,-189,453,57,362,-469,77,-579,-436,-42,47,-467,318,219,-145,-473,532,98 };
static matrix EMBEDDING_LAYER_KERNEL_MAT_VAR = { EMBEDDING_LAYER_KERNEL, 20, 2 };
static matrix * EMBEDDING_LAYER_KERNEL_MAT = &EMBEDDING_LAYER_KERNEL_MAT_VAR;

static dtype EMBEDDING_LAYER_BIAS[20] = { -105,97,-369,77,-143,-93,541,-283,565,-167,90,312,-376,-87,-59,-145,13,357,-339,130 };
static matrix EMBEDDING_LAYER_BIAS_MAT_VAR = { EMBEDDING_LAYER_BIAS, 20, 1 };
static matrix * EMBEDDING_LAYER_BIAS_MAT = &EMBEDDING_LAYER_BIAS_MAT_VAR;

static dtype RNN_CELL_W_TRANSFORM[1600] = { -529,-46,2,-377,-229,35,-364,-41,-211,-119,-84,21,-158,91,352,-191,148,279,15,-229,-330,-311,-142,-385,-197,64,50,36,-166,23,-313,-231,64,-403,-135,-140,-164,-106,75,-95,-13,-547,-404,490,218,-25,122,-434,-362,347,133,359,-377,278,157,-182,-49,360,39,147,-640,-15,233,-467,-83,-8,-325,-96,-192,128,-140,-142,71,-435,-415,-143,-371,-63,-12,-239,94,190,-141,-16,59,-283,10,-194,-84,263,79,-284,-59,-68,296,-353,100,-39,321,424,22,65,370,-3,65,-237,-373,59,8,-234,28,2,2,-125,-122,68,-147,-491,163,-483,318,-212,-176,91,-58,-697,-70,-193,32,132,-322,-306,-13,-271,182,-648,-159,-92,450,408,-312,86,264,-469,-79,-20,-59,59,-164,-67,65,-1,-285,-352,-271,117,-447,-170,-155,-282,99,-41,-484,610,58,-31,53,186,-144,262,166,208,-73,460,43,46,-184,-189,51,370,-133,-162,74,-199,35,166,-192,-95,-310,-8,-284,-205,-4,136,6,-195,-99,46,-133,-156,-190,376,333,-294,-484,-46,-399,322,-217,211,-105,-69,-21,255,193,-279,-422,-216,34,-29,-75,-136,-140,-95,-26,177,-19,-58,-127,62,-168,-298,79,79,-155,-257,-136,-94,-134,-7,-317,-316,213,190,-71,-126,-115,-413,-119,231,-200,177,45,184,184,-556,-124,-45,285,-278,-75,-177,167,63,-100,78,-15,19,-225,9,-66,-216,156,29,14,-327,34,-173,6,17,-244,-111,-299,-70,171,392,-174,-124,-621,346,151,277,87,-77,297,-112,152,632,-82,16,-127,-377,-78,184,66,-67,-262,11,-246,14,-231,-308,291,-41,108,-369,22,47,227,-6,135,-165,-201,9,-20,-131,-129,-130,-238,256,-240,-156,-327,238,40,-1,-444,-43,63,289,-211,155,267,-252,-234,-53,-338,-55,-64,-95,-132,-187,27,-183,-290,75,-375,-320,-22,-245,-75,392,56,-458,-155,125,-93,410,23,-47,-142,-252,-62,-169,252,-232,-72,-46,293,240,225,-422,38,-73,52,-165,-356,229,-176,-17,-60,-95,-82,63,-64,101,-14,-192,177,13,-162,-46,-70,46,290,213,-170,141,-64,-325,261,-53,-397,13,293,-109,424,781,-165,-172,271,-254,-11,174,-79,-293,-80,100,-99,4,-151,-223,-2,144,76,71,91,-182,295,-186,-50,143,-109,59,-219,-26,116,187,321,-12,-36,135,-136,98,133,-263,49,157,48,55,-203,-126,108,-233,-125,252,-227,9,-421,245,-166,-359,472,-145,-304,-220,-311,-52,24,67,-356,-63,-70,-238,-67,-50,-101,-274,222,66,173,111,-219,195,212,132,89,304,22,-382,109,-279,-96,-236,5,-30,-90,191,-157,-9,47,-141,-8,163,114,123,57,-28,52,184,131,78,-84,16,-19,-191,137,-194,298,-24,298,-368,339,81,-3,-89,120,399,61,308,-351,86,274,-252,-221,-64,-201,141,14,-242,-84,9,-340,-358,-316,-43,-26,-78,-75,-172,-436,573,-365,-312,236,310,-341,281,-372,-112,-227,-256,-279,-97,343,-41,-45,-32,210,270,230,-310,-86,223,193,-145,-271,61,-188,-151,-24,-247,-147,138,201,27,232,-222,309,-88,181,-13,-133,134,16,138,-98,423,-171,248,15,-30,19,-199,61,-36,-441,-336,-9,438,-33,-367,-87,-99,300,-406,-211,336,-41,62,-9,-129,-91,75,175,16,101,-277,425,-111,-192,-24,-166,-396,17,1,-205,-153,81,-18,-4,-31,-263,256,56,-199,169,354,-35,-406,101,-426,-230,-22,-263,-43,-225,-239,-185,-76,-467,-396,22,319,-44,-267,29,-17,-5,25,-209,362,-209,-351,123,416,166,376,-229,-51,60,-196,82,251,-35,134,29,53,-295,335,-277,-439,-124,-109,272,-361,-356,300,1,-110,78,41,-136,-277,-223,-11,-112,-403,344,-369,53,-98,78,-379,-529,-256,18,349,95,194,-108,-154,-103,480,214,-485,17,-119,106,393,205,-72,192,115,-76,-7,-287,94,-384,-32,-347,-569,212,286,157,-215,-102,-202,103,-107,-189,180,-79,-149,-150,-170,-158,-288,19,61,-194,139,101,-282,100,-38,-102,17,144,200,-330,86,549,-191,-132,-72,-380,315,60,-270,185,139,-76,-502,-122,244,-218,-388,66,-239,13,110,-150,65,-36,170,205,-3,96,-188,72,104,165,190,-269,-83,123,-196,81,309,-54,128,-205,89,-24,186,95,-134,-160,-43,88,84,-82,128,-198,-40,-174,245,-92,300,199,385,-328,301,233,132,68,194,-85,-13,-94,82,-30,-174,26,347,-213,-241,-82,-86,-36,-39,207,-89,-152,172,48,-126,-142,318,-137,-100,186,118,-54,11,36,-155,-18,-78,19,-164,113,92,80,127,22,216,-321,58,62,46,-190,302,197,-227,-131,-148,223,-258,-223,151,59,-149,-206,318,15,-145,-231,125,-80,-151,387,-116,-262,37,-210,113,-3,109,105,-129,-236,283,-66,33,136,-30,-115,99,-186,210,84,-23,-91,-223,-93,-160,-19,370,113,-292,-63,190,340,-298,12,20,240,-13,108,184,96,-46,-1,-125,84,-38,181,-186,-15,42,59,19,38,107,-38,-438,210,139,-128,43,79,-314,-140,288,-281,-194,-209,67,-195,-87,27,-146,-48,-124,-106,-192,161,-60,83,141,-226,-202,-111,-13,-162,-177,-198,-180,31,3,-245,186,-43,226,-129,147,-181,143,102,-191,168,-20,156,154,271,247,-81,-235,55,119,303,114,-46,107,-138,209,-53,-119,-59,123,-2,-5,195,197,21,-146,125,-193,295,-274,-125,292,220,-322,250,-241,-31,-9,180,-87,-43,108,47,50,-177,77,-189,-201,-37,249,-45,-94,-70,-32,174,-123,184,-66,131,-48,-197,-180,25,-63,-163,107,-239,133,222,74,-94,205,-24,147,23,306,-476,286,129,259,-63,84,-125,-67,-22,-28,-159,348,197,237,263,14,47,223,-200,-315,-221,125,-27,-148,82,260,10,70,-19,58,-4,21,-125,-25,-221,57,68,-392,-129,-272,195,-86,108,213,2,243,268,-188,183,-50,92,-106,13,219,0,36,-84,185,127,73,-7,-119,-137,-60,-138,122,172,51,179,142,-22,-2,-88,137,109,-164,256,210,-46,47,282,-174,45,25,69,-147,140,136,264,215,-221,-54,130,-381,-103,254,203,53,-103,-29,118,55,35,78,209,39,155,31,195,-122,292,-9,155,-172,-160,255,121,-112,166,-141,-146,109,174,69,-229,150,139,131,117,-4,148,51,-255,-47,-221,-33,-14,-113,1,-24,116,-142,229,176,-197,-325,-208,239,-153,-105,75,-10,-156,122,-278,43,-16,123,-221,303,21,-277,-122,180,-100,302,102,261,-106,-66,138,-270,165,-48,-124,187,301,8,-109,-81,159,-38,151,-13,-27,53,312,-3,105,-27,199,-160,-32,-284,-285,250,-104,121,-59,-106,141,-61,144,-51,197,-43,70,368,289,-39,28,-261,-273,-30,-301,-171,10,236,78,54,0,164,53,-162,126,-83,-347,-209,-268,213,-58,158,29,-15,-106,207,194,129,2,-123,-160,-26,-30,315,-404,82,96,-99,225,209,184,-269,109,-196,62,-26,196,-37,110,51,52,-19,108,76,-191,-32,167,121,-68,-109,170,134,-101,180,-11,-23,297,186,-38,-85,-129,382,-233,-95,23,-386,-24,245,-164,1,-153,-144,86,-257,12,237,192,-17,-55,-74,-42,-105,-92,87,-148,42,70,-247,17,71,129,-135,-189,392,109,-3,-198,-233,-194,-105,-25,266,-83,159,170,73,224,87,33,-236,256,9,-343,344,128,-78,-17,221,-115,63,195,-55,207,133,67,-185,-214,194,-114,-160,-44,-164,-1,-207,88,-49,85,-27,231,-35,-217,541,-52,135,-91,-182,167,-179,117,-133,-211,271,85,-42,-49,-74,-104,-172,-83,-135,98,-41,-71,184,-143,-129,-93,-138,-154,-72,-15,0,-140,-125,-415,-16,78,226,-19,-83,14,-122,-34,-17,-308,-330,56,97,176,353,-89,-248,-274,-226,-245,25,-4,-99,144,35,245,-271,-70,148,-196,-420,-158,-55,-16,-91,102,-48,382,202,-310,163,98,-255,121,-265,251,196,-91,350,-25,227,-132,42,-52,-439,93,146,280,-266,22,202,-139,-9,-130,140,-41,-44,-84,-125,170,374,24,-138,31,108,34,-77,129,79,-1,-218,133,275,71,277,74,-128,196,44,42,116,14,308,437,40,-244,299,11,141,210,28,3,88,-144,176,79,283,84,164,252,-27,-60,-3,-170,11,15,52 };
static matrix RNN_CELL_W_TRANSFORM_MAT_VAR = { RNN_CELL_W_TRANSFORM, 40, 40 };
static matrix * RNN_CELL_W_TRANSFORM_MAT = &RNN_CELL_W_TRANSFORM_MAT_VAR;

static dtype RNN_CELL_B_TRANSFORM[40] = { 74,-131,-99,-176,-320,-190,-62,-176,-49,-480,-54,-153,-17,12,-149,-145,-20,-80,-147,-71,121,25,174,146,9,-55,80,-56,6,-189,273,-106,192,-29,-18,143,25,13,-126,-206 };
static matrix RNN_CELL_B_TRANSFORM_MAT_VAR = { RNN_CELL_B_TRANSFORM, 40, 1 };
static matrix * RNN_CELL_B_TRANSFORM_MAT = &RNN_CELL_B_TRANSFORM_MAT_VAR;

static dtype RNN_CELL_W_FUSION[800] = { 220,166,-216,-63,113,-97,-125,-99,-111,-244,-199,-468,38,-318,110,110,-255,71,252,-49,-553,-69,-155,11,48,-117,-465,83,-506,415,-207,226,-767,215,193,-272,174,255,34,-23,-434,138,-330,-2,-242,-125,-595,94,-100,385,-304,292,-315,273,538,-192,-223,236,519,-337,-113,407,73,49,-245,168,-261,515,-443,16,-383,-362,-37,-286,547,-8,136,76,340,-141,197,-10,198,-189,-212,411,-15,372,286,-213,115,-73,-1,-447,-189,-183,296,21,-53,-159,93,168,-208,-300,296,211,176,25,137,314,164,-249,66,-145,-33,-38,204,129,-182,52,-62,187,206,-83,106,-56,46,269,-29,4,25,3,97,-31,-35,155,-46,88,-55,-57,495,399,38,-314,-74,123,213,170,38,-73,-138,-701,660,-722,-238,-444,-392,-78,119,-84,-32,115,-90,41,162,27,-25,-59,-340,-112,-225,344,60,-246,-20,-425,-20,15,74,96,81,332,116,-219,9,219,-14,107,-64,320,-76,-489,515,-262,-152,-286,-89,-295,-203,28,121,172,-197,-251,-45,107,33,443,11,-40,41,-82,153,63,-52,252,184,327,81,168,-152,-37,194,-211,-55,205,-82,96,95,73,14,-101,51,158,-267,-245,261,440,-11,215,181,187,56,-186,411,272,187,157,-345,16,152,-295,-140,36,120,-1,189,-88,-275,137,-86,268,-485,51,-227,-220,-414,35,-446,673,-415,-305,78,117,300,-198,-70,232,621,198,29,-31,89,-232,-346,-169,-127,394,359,-59,-299,114,-84,-63,20,54,7,75,135,-68,-48,-76,163,106,-206,-175,86,-111,27,-110,-153,3,-200,214,-314,-190,-67,193,-22,209,-262,-64,-286,158,-522,-131,-417,283,207,340,4,104,-492,155,572,-345,98,7,674,-205,-1,-204,-265,31,-16,-52,-16,193,-270,357,118,-318,128,-20,172,-186,-27,683,-113,-146,49,291,-208,-253,-390,-365,-65,-201,-77,71,-174,-13,-331,123,290,-705,-13,176,201,121,-360,51,91,-131,144,-228,-150,109,-79,202,200,-30,-399,220,184,-352,294,562,-77,366,-116,-508,-372,293,49,-271,36,-275,-42,167,79,397,-23,431,178,154,381,130,89,-30,36,-70,-1,254,-16,-131,-125,-108,390,-126,192,97,-131,16,-6,-31,0,-147,336,-195,-149,-369,-293,282,93,-385,-99,-448,-137,211,-88,425,-91,13,-91,35,-22,-144,59,320,-147,174,160,-238,-112,-58,-74,-65,119,-282,23,-325,-126,108,100,113,-174,-257,109,74,-304,-287,-259,-3,-125,-368,-202,-150,-286,351,167,361,41,40,220,-55,3,70,-19,238,131,136,-265,73,244,57,20,-259,106,60,36,169,198,-180,582,171,-319,-102,240,-52,-209,-710,-384,560,392,78,100,-52,-9,84,153,250,-22,40,-145,501,248,88,-204,37,401,139,33,-244,119,-220,-134,-141,251,-144,-47,-265,445,-195,66,-100,-193,-276,113,-218,85,338,-56,-223,-217,55,55,13,-55,8,-39,-448,8,-287,2,83,-43,-137,209,77,-74,226,-293,6,-239,-138,-360,80,-395,210,-303,289,-26,102,385,-261,-240,-1,-187,343,103,-217,88,-188,171,72,-27,-152,-171,-131,-263,129,10,-223,30,-171,-26,-81,-131,-184,-12,-115,-478,225,-154,11,-77,-54,467,355,-98,-165,26,-77,-5,-99,-88,94,186,-243,-69,212,-345,-203,-234,-27,-216,28,155,65,-133,267,-306,114,-62,-322,-125,360,7,314,-193,-107,-146,-84,-75,74,37,-117,46,-197,188,-113,109,169,-128,443,258,278,39,119,-38,-466,-134,-210,131,-2,-325,165,233,140,-30,-268,-70,18,257,-345,-191,76,56,76,-59,250,32,-241,-265,270,-108,-34,130,166,36,-29,-60,459,212,-528,361,444,218,251,50,-247,-179,-137,-509,-28,-72,-74,241,-123,-105,-235,239,-99,-86,-378,-19,36,98,233,338,52,-11,-146,-181,-175,-69,179,17,257,-17,-217,210,254,-34,-704,4,359,-64,219,-333,275,27,0,-54,-158,-203,96,-146,-27,-271,-20,-259,222,300,-490,-73,-37,48,136,199,369,82,-240,-31,13,-231,231,-358,67,-329,-223,-168,-195,213,-310,-146,-20,167,-124 };
static matrix RNN_CELL_W_FUSION_MAT_VAR = { RNN_CELL_W_FUSION, 20, 40 };
static matrix * RNN_CELL_W_FUSION_MAT = &RNN_CELL_W_FUSION_MAT_VAR;

static dtype RNN_CELL_B_FUSION[20] = { -96,-220,-158,-122,203,-342,-201,178,-456,-432,-283,303,-22,200,-391,270,-212,58,166,-31 };
static matrix RNN_CELL_B_FUSION_MAT_VAR = { RNN_CELL_B_FUSION, 20, 1 };
static matrix * RNN_CELL_B_FUSION_MAT = &RNN_CELL_B_FUSION_MAT_VAR;

static dtype STOP_PREDICTION_HIDDEN_0_KERNEL[480] = { 125,-1,-146,-148,175,-154,132,-168,-95,-27,-142,102,151,170,157,-85,-87,-30,135,127,-127,-166,105,12,179,8,176,-35,-168,-66,-184,-102,141,48,33,-153,33,79,-55,71,160,-78,-176,160,-155,127,69,-99,-17,-150,-22,151,117,-53,85,157,132,78,-72,7,-71,60,-58,-150,40,95,161,-188,-131,-65,-22,-7,-83,58,25,55,100,178,-139,-176,-40,-135,-83,-18,-188,-29,-60,67,-93,-17,-58,-79,82,143,48,-38,0,139,-20,82,-86,111,-109,-188,-91,135,-66,-96,-92,-61,119,4,65,-42,-103,-137,2,110,-34,-7,-57,104,-47,-71,-155,-19,110,-24,46,45,-155,63,151,-70,-186,-54,88,187,76,-164,-17,52,-183,79,121,-57,148,-161,48,57,-128,-48,-33,-55,62,-19,-130,-131,-94,-107,-164,91,-43,-163,-143,-117,-56,175,101,-6,113,155,118,-131,-173,10,125,-155,77,-122,28,27,45,172,181,-17,-174,153,-147,-103,141,97,-106,11,75,-17,88,9,-54,176,-175,105,61,47,-131,29,-67,171,-95,184,-95,-183,-180,60,-68,-53,82,187,-84,162,26,-16,-80,19,-114,2,39,51,-16,63,-102,-24,-86,158,27,182,-7,-18,157,60,-61,-121,161,-158,-72,-25,-36,17,55,-126,-90,-6,-95,155,179,-176,140,83,-9,35,144,61,-70,13,173,-3,-43,-105,-8,188,-171,-145,-76,-95,-91,92,46,-97,114,140,76,34,3,-55,-19,-72,-52,-127,168,-14,181,9,123,85,30,-19,100,50,173,164,115,-45,12,-181,136,-128,69,5,146,43,-37,-2,-24,-102,-62,42,-135,-111,182,-178,-173,96,184,-112,185,-151,52,-32,136,-167,119,125,-127,158,112,-34,1,173,-170,-165,-115,106,182,85,134,-72,124,138,144,128,10,-85,-53,143,-135,57,-82,-56,-131,-82,-50,-126,33,-181,-135,114,149,-122,-51,-25,-70,-35,-122,33,177,46,-73,-62,138,-51,79,89,33,-63,-115,-169,-146,112,-142,28,-128,156,-56,106,-114,150,142,75,-9,-35,13,155,-172,29,-45,-79,20,94,-127,10,36,-122,-137,-49,-113,-99,-8,-170,80,-185,140,156,114,47,-61,-189,132,103,-181,103,93,-24,-98,46,-103,151,98,-180,-105,-101,-161,1,111,-13,38,178,143,1,167,-26,-33,10,-51,-117,153,-28,-16,94,140,-149,141,164,83,121,-35,135,-1,37,137,174,-13,-2,-102,-18,28,10,-37,133,77,182 };
static matrix STOP_PREDICTION_HIDDEN_0_KERNEL_MAT_VAR = { STOP_PREDICTION_HIDDEN_0_KERNEL, 24, 20 };
static matrix * STOP_PREDICTION_HIDDEN_0_KERNEL_MAT = &STOP_PREDICTION_HIDDEN_0_KERNEL_MAT_VAR;

static dtype STOP_PREDICTION_HIDDEN_0_BIAS[24] = { 171,-192,-124,71,5,255,224,-236,60,-155,248,-256,-321,-76,53,-288,-260,-138,-19,-48,-228,-98,-183,-107 };
static matrix STOP_PREDICTION_HIDDEN_0_BIAS_MAT_VAR = { STOP_PREDICTION_HIDDEN_0_BIAS, 24, 1 };
static matrix * STOP_PREDICTION_HIDDEN_0_BIAS_MAT = &STOP_PREDICTION_HIDDEN_0_BIAS_MAT_VAR;

static dtype STOP_PREDICTION_OUTPUT_KERNEL[24] = { 193,117,31,-205,162,199,-187,-101,27,172,210,221,-245,-69,221,-108,1,-117,-111,-246,-151,214,244,-176 };
static matrix STOP_PREDICTION_OUTPUT_KERNEL_MAT_VAR = { STOP_PREDICTION_OUTPUT_KERNEL, 1, 24 };
static matrix * STOP_PREDICTION_OUTPUT_KERNEL_MAT = &STOP_PREDICTION_OUTPUT_KERNEL_MAT_VAR;

static dtype STOP_PREDICTION_OUTPUT_BIAS[1] = { 107 };
static matrix STOP_PREDICTION_OUTPUT_BIAS_MAT_VAR = { STOP_PREDICTION_OUTPUT_BIAS, 1, 1 };
static matrix * STOP_PREDICTION_OUTPUT_BIAS_MAT = &STOP_PREDICTION_OUTPUT_BIAS_MAT_VAR;

static dtype OUTPUT_LAYER_HIDDEN_0_KERNEL[640] = { 102,491,61,190,-37,293,115,315,-468,-355,-57,-81,8,-186,19,48,-69,-55,-103,223,40,163,68,-86,438,-283,162,-242,489,137,207,50,70,-188,453,18,47,-310,493,20,71,141,209,-51,-276,308,-138,52,-402,-329,-98,168,-50,-230,-261,33,-172,44,-335,212,344,593,-138,97,37,-62,270,376,92,130,-274,-6,246,73,303,-52,-260,-261,585,371,111,379,42,-15,-121,267,155,-74,-452,-248,-661,-106,156,-271,68,165,-286,-146,-24,-369,-87,-81,106,24,-279,-130,-208,16,-86,-27,-53,263,-114,-285,-147,128,42,58,21,478,-50,-80,72,358,89,134,56,-142,66,-424,-13,-169,243,95,-363,356,-186,-163,225,-443,159,-363,-297,-205,3,161,114,-209,265,-218,-66,-332,79,-40,-260,118,232,-577,102,-156,137,-562,-130,174,-121,319,267,-344,-53,-177,165,-471,321,128,-207,-222,74,21,-485,-211,-267,-298,-75,-305,66,5,232,-408,242,235,-108,-301,59,-44,-75,-37,146,19,11,-416,87,-79,290,-69,-304,-370,-225,93,426,-161,220,386,-347,156,-313,-148,-23,-58,62,296,-104,86,170,-367,-308,350,68,69,284,-239,231,212,-347,266,-388,105,261,385,-235,136,188,257,305,460,160,-190,36,-152,105,-444,67,234,392,-11,-172,330,-476,-344,261,75,205,-156,-139,427,398,-187,110,-158,-91,-175,7,-124,298,183,246,-38,-5,-229,178,-276,-82,-395,-53,134,359,-220,-51,-166,33,43,226,7,-22,44,-50,-10,333,74,-221,95,10,141,30,96,-33,75,-305,383,-414,7,-228,277,-87,175,-4,-34,1,158,-187,225,113,-221,71,140,-68,-6,-146,-240,344,-513,361,97,-94,62,-771,26,245,-25,175,151,310,532,369,69,261,145,237,-76,180,-267,109,45,296,128,-123,192,-204,-293,480,-281,155,-182,-320,247,-153,48,151,470,-340,180,-22,-274,-187,169,-99,30,-25,93,-342,-88,26,179,61,-90,-194,-264,-57,-8,21,94,227,216,-388,23,225,-390,438,147,169,326,-18,-169,-44,231,-125,91,214,540,-219,256,-47,107,-462,232,-99,-80,297,378,-124,48,-127,-389,-162,382,375,49,-65,-27,-86,-236,303,-192,433,-80,-3,129,154,210,55,-348,-420,-97,24,-298,7,-451,-3,-144,218,380,23,85,-450,-394,157,-227,406,157,-54,64,-28,0,-250,-114,-427,190,172,275,-224,-160,-181,-505,71,42,-71,14,-34,-103,-320,-116,163,21,-130,28,23,-237,52,289,295,237,113,5,-303,10,103,-105,63,-119,79,450,183,-143,-172,390,-169,80,-304,251,-220,-365,-60,-147,-81,167,24,160,-185,-120,42,-97,45,-43,170,117,192,-411,-275,146,-214,347,-229,262,-145,-394,257,224,155,-159,-45,284,-246,-187,-154,294,212,-319,-199,-19,240,166,18,60,-96,252,324,-127,-27,281,-53,-339,61,200,109,-180,12,400,178,21,-111,-468,-270,-103,167,-23,-211,209,-110,-178,-228,92,-8,219,139,-154,-269,-295,-137,150,266,-37,228,49,-304,263,-211,129,-87,-258,415,319,237,173,229,376,238,-387,129,241,-207,-211,-17,392,-212,202,-93,197,-57,34,155,160,140,-252,324,-6,-185,-199,-34,-224,327,-244,369,384,-136,201,-591,-189,75,-19,197,-52 };
static matrix OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT_VAR = { OUTPUT_LAYER_HIDDEN_0_KERNEL, 32, 20 };
static matrix * OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT = &OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT_VAR;

static dtype OUTPUT_LAYER_HIDDEN_0_BIAS[32] = { -165,160,87,-186,-237,-434,176,296,-228,390,194,281,-120,-343,-82,264,86,-29,-236,276,-9,-137,186,128,-70,157,-128,-147,281,-264,110,-35 };
static matrix OUTPUT_LAYER_HIDDEN_0_BIAS_MAT_VAR = { OUTPUT_LAYER_HIDDEN_0_BIAS, 32, 1 };
static matrix * OUTPUT_LAYER_HIDDEN_0_BIAS_MAT = &OUTPUT_LAYER_HIDDEN_0_BIAS_MAT_VAR;

static dtype OUTPUT_LAYER_OUTPUT_KERNEL[320] = { -34,264,-267,268,-11,186,138,104,-531,90,170,-37,302,6,-513,-251,80,340,-743,166,-755,-719,187,-200,244,9,155,-66,80,512,134,200,-144,130,-82,-325,-67,-88,-470,76,82,126,144,132,-156,-48,62,-431,202,-626,-277,152,61,-87,218,-252,51,-454,115,-31,104,26,-62,255,-436,-139,-267,-368,-98,-76,-232,-920,114,118,-639,142,-497,-177,59,170,-634,-443,138,26,199,182,277,-419,-127,-220,290,286,-513,-148,-5,-444,-499,9,-844,215,-147,-150,-405,159,100,140,-98,-318,-683,-203,-195,-87,-895,-673,342,110,-149,-194,156,118,222,125,291,-471,-460,-139,-543,-698,261,252,16,312,277,-241,-99,-105,-550,-383,-411,-405,306,544,-292,494,-875,450,-186,-86,-137,-197,-639,-47,63,213,358,-90,149,-321,240,-1013,54,-798,47,-452,188,-119,145,190,197,61,-571,-50,29,226,-33,-33,119,-589,291,-623,145,173,-288,219,-269,389,-159,-162,184,-289,-216,-533,-53,-222,244,-152,33,491,151,22,-486,-622,218,-435,368,-240,51,246,269,-145,-439,-190,-212,-43,84,-573,-72,130,265,-190,-693,258,156,246,225,-863,190,111,-76,173,-372,-459,101,-736,227,240,-211,-320,-38,329,129,-238,316,184,212,-125,-136,-32,118,179,-265,-249,39,358,197,214,-393,122,-119,-344,-296,-122,234,56,205,71,43,-233,286,423,60,-29,245,203,96,-104,-59,307,-128,-704,-149,74,-2,129,-232,-203,1,239,79,-135,41,86,268,-189,203,115,161,218,-82,382,139,-484,-291,-317,220,176,-302,-273,-230,94,-408,122,-613,-566,-611,-351,235,-64,-1255,285 };
static matrix OUTPUT_LAYER_OUTPUT_KERNEL_MAT_VAR = { OUTPUT_LAYER_OUTPUT_KERNEL, 10, 32 };
static matrix * OUTPUT_LAYER_OUTPUT_KERNEL_MAT = &OUTPUT_LAYER_OUTPUT_KERNEL_MAT_VAR;

static dtype OUTPUT_LAYER_OUTPUT_BIAS[10] = { -198,120,293,227,186,-15,-353,-120,-244,-13 };
static matrix OUTPUT_LAYER_OUTPUT_BIAS_MAT_VAR = { OUTPUT_LAYER_OUTPUT_BIAS, 10, 1 };
static matrix * OUTPUT_LAYER_OUTPUT_BIAS_MAT = &OUTPUT_LAYER_OUTPUT_BIAS_MAT_VAR;

static dtype AGGREGATION_LAYER_KERNEL[40] = { 228,-124,215,-174,-23,210,-282,164,-123,-22,218,-143,19,-7,122,233,437,-133,-108,304,56,-119,-66,-135,-76,-150,-139,-65,113,-41,186,-29,14,-134,48,61,-115,-28,143,-115 };
static matrix AGGREGATION_LAYER_KERNEL_MAT_VAR = { AGGREGATION_LAYER_KERNEL, 1, 40 };
static matrix * AGGREGATION_LAYER_KERNEL_MAT = &AGGREGATION_LAYER_KERNEL_MAT_VAR;

static dtype AGGREGATION_LAYER_BIAS[1] = { -243 };
static matrix AGGREGATION_LAYER_BIAS_MAT_VAR = { AGGREGATION_LAYER_BIAS, 1, 1 };
static matrix * AGGREGATION_LAYER_BIAS_MAT = &AGGREGATION_LAYER_BIAS_MAT_VAR;
#endif
