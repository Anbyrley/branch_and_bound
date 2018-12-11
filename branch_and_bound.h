/** @file branch_and_bound.h
*   @brief Contains functions for the branch and bound program.
*
*
*  @author Alex N. Byrley (anbyrley)
*  @date July 2017
*  @bug No known bugs
*/

#ifndef BRANCH_AND_BOUND_H
#define BRANCH_AND_BOUND_H

#define DEBUG_BB 0
#define USE_TEST_DICTIONARY 0
#define MAX_NUM_BB_LEVELS 100

//================================================================================================//
//===================================STANDARD INCLUDES============================================//
//================================================================================================//

#include "macros.h"
#include "helper.h"
#include "dictionary.h"
#include "machine_learning.h"


//================================================================================================//
//====================================DATA STRUCTURES=============================================//
//================================================================================================//

//================================================================================================//
/** @struct bb_node_t
*   @brief This structure is the typedef the bb_node_t object.
*/
//================================================================================================//
typedef struct bb_node_s bb_node_t;
typedef struct bb_node_s{
	unsigned int level;
	unsigned int num_descendants;
	unsigned int sibling_rank;
	unsigned int feature;
	double value;
	bb_node_t* parent;
	bb_node_t* descendants[MAX_NUM_FEATURES];
} bb_node_t;


//================================================================================================//
/** @struct bb_tree_t
*   @brief This structure is the typedef the bb_tree_t object.
*/
//================================================================================================//
typedef struct bb_tree_s bb_tree_t;
typedef struct bb_tree_s{
	unsigned int loop_count;
	unsigned int num_levels;
	unsigned int num_features;
	unsigned int subset_size;
	unsigned int num_available_features;
	double current_best_objective;
	unsigned int features[MAX_NUM_FEATURES];
	unsigned int available_features[MAX_NUM_FEATURES];
	unsigned int current_feature_path[MAX_NUM_FEATURES];
	unsigned int optimal_features[MAX_NUM_FEATURES];
	dataset_t full_dataset;
	dataset_t subset;
	bb_node_t root;
	bb_node_t* current_node;
	dictionary_t test_feature_dictionary; //this is to search for feature combo jvalues for testing
} bb_tree_t;

//need to set feature group j_values manually


//================================================================================================//
//==================================FUNCTION DECLARATIONS=========================================//
//================================================================================================//

//================================================================================================//
/**
* @brief This function initializes a branch and bound node.
*
* @param[in,out] bb_node_t* self
*
* @return NONE
*/
//================================================================================================//
void initialize_bb_node(bb_node_t*);


//================================================================================================//
/**
* @brief This function prints a branch and bound node.
*
* @param[in,out] bb_node_t* self
*
* @return NONE
*/
//================================================================================================//
void print_bb_node(bb_node_t*);


//================================================================================================//
/**
* @brief This function initializes a branch and bound tree.
*
* @param[in,out] bb_tree_t* self
* @param[in] unsigned int* features
* @param[in] unsigned int num_features
* @param[in] unsigned int subset_size
*
* @return NONE
*/
//================================================================================================//
void initialize_bb_tree(bb_tree_t*,unsigned int*,unsigned int,unsigned int);


//================================================================================================//
/**
* @brief This function prints a branch and bound tree's optimal features.
*
* @param[in] bb_tree_t* self
* @param[in] FILE* file
*
* @return NONE
*/
//================================================================================================//
void print_bb_tree_optimal_features(bb_tree_t*, FILE*);


//================================================================================================//
/**
* @brief This function checks to see if we should move up the bb tree's call stack.
*
* @param[in] bb_tree_t* self
*
* @return unsigned int should_unwind
*/
//================================================================================================//
unsigned int should_bb_tree_unwind(bb_tree_t*);


//================================================================================================//
/**
* @brief This function initializes the next node in the bb tree and sets it to the current node.
*
* @param[in,out] bb_tree_t* self
*
* @return NONE
*/
//================================================================================================//
void initialize_bb_tree_next_node(bb_tree_t*);


//================================================================================================//
/**
* @brief This function tests the current node to see if we need to go further or back up.
*
* @param[in,out] bb_tree_t* self
*
* @return NONE
*/
//================================================================================================//
void test_bb_tree_current_node(bb_tree_t*);


//================================================================================================//
/**
* @brief This function moves up a level in the bb tree.
*
* @param[in,out] bb_tree_t* self
*
* @return NONE
*/
//================================================================================================//
void backtrack_bb_tree(bb_tree_t*);


//================================================================================================//
/**
* @brief This function updates the bb tree bound and then backtracks.
*
* @param[in,out] bb_tree_t* self
*
* @return NONE
*/
//================================================================================================//
void update_bb_bound(bb_tree_t*);


//================================================================================================//
/**
* @brief Edit this function to manually enter values for each node in a bb_tree.
*
* @param[in,out] bb_tree_t* self
*
* @return NONE
*/
//================================================================================================//
void setup_bb_test_dictionary(bb_tree_t*);


//================================================================================================//
/**
* @brief This function tests bb_tree functionality.
*
* @return NONE
*/
//================================================================================================//
void test_bb();


#endif //BRANCH_AND_BOUND_H//
