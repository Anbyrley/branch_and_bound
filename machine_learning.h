/** @file machine_learning.h
*   @brief Contains functions for working with datasets using machine learning.
*
*
*  @author Alex N. Byrley (anbyrley)
*  @date January 2017
*  @bug No known bugs
*/

#ifndef MACHINE_LEARNING_H
#define MACHINE_LEARNING_H


//================================================================================================//
//===================================STANDARD INCLUDES============================================//
//================================================================================================//

#include <math.h>
#include <complex.h>


//================================================================================================//
//======================================MACROS====================================================//
//================================================================================================//

#define MAX_NUM_SOM_NODES 75
#define MAX_NUM_SOM_ITERATIONS 2*(MAX_NUM_SOM_NODES*MAX_NUM_SOM_NODES)

#define MAX_NUM_BALANCED_RECORDS 50000
#define MAX_NUM_RECORDS 200000
#define MAX_NUM_ATTRIBUTES 6
#define MAX_NUM_FEATURES 65
#define MAX_NUM_CLASSES 2

#define MAX_CG_ITERATIONS 2500
#define MAX_CG_ERROR_TOLERANCE (10e-16)



//================================================================================================//
//=====================================LOCAL INCLUDES=============================================//
//================================================================================================//

#include "macros.h"
#include "helper.h"
#include "random_numbers.h"
#include "linear_algebra.h"
#include "dictionary.h"

//================================================================================================//
//=====================================DATA STRUCTURES============================================//
//================================================================================================//

//================================================================================================//
/** @struct som_node_t
*   @brief This structure is the typedef for the som_node_t object.
*/
//================================================================================================//
typedef struct som_node_s som_node_t;
typedef struct som_node_s{
	unsigned int *num_features;
	unsigned int num_times_bmu;
	unsigned int location[2];
	unsigned int class_total;
	double distance;
	double distance_factor;
	double average_class;
	double weights[MAX_NUM_FEATURES];
} som_node_t;


//================================================================================================//
/** @struct som_t
*   @brief This structure is the typedef for the som_t object.
*/
//================================================================================================//
typedef struct som_s som_t;
typedef struct som_s{
	unsigned int num_iterations;
	unsigned int num_nodes;
	unsigned int num_features;
	unsigned int input_class;
	unsigned int num_x_nodes, num_y_nodes;
	double initial_radius, radius;
	double initial_learning_rate, learning_rate;
	double radius_decay_constant;
	double input[MAX_NUM_FEATURES];
	som_node_t* bmu;
	som_node_t nodes[MAX_NUM_SOM_NODES/2][MAX_NUM_SOM_NODES/2];
} som_t;


//================================================================================================//
/** @struct dataset_t
*   @brief This structure is the typedef for the dataset_t object.
*/
//================================================================================================//
typedef struct dataset_s dataset_t;
typedef struct dataset_s{
	unsigned int num_records;
	unsigned int num_features;
	unsigned int num_attributes;
	unsigned int num_classes;
	unsigned int classes[MAX_NUM_RECORDS];
	unsigned int class_sizes[MAX_NUM_CLASSES];
	char record_attribute_names[MAX_NUM_ATTRIBUTES][80];
	double class_probabilities[MAX_NUM_CLASSES];
	double record_attributes[MAX_NUM_RECORDS][MAX_NUM_ATTRIBUTES];
	data_type type;
	union{
		double dbl[MAX_NUM_RECORDS][MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_RECORDS][MAX_NUM_FEATURES];
	}record;
	union{
		double dbl[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	}covariance_matrix;
	union{
		double dbl[MAX_NUM_CLASSES][MAX_NUM_FEATURES*MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_CLASSES][MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	}class_covariance_matrix;
	union{
		double dbl[MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_FEATURES];
	}record_centroid;
	union{
		double dbl[MAX_NUM_CLASSES][MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_CLASSES][MAX_NUM_FEATURES];
	}class_centroid;
	union{
		double dbl[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	}within_class_scatter_matrix;
	union{
		double dbl[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
		double complex cmplx[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	}between_class_scatter_matrix;
} dataset_t;


//================================================================================================//
//==================================FUNCTION DECLARATIONS=========================================//
//================================================================================================//

//-----------------------------------------------------------------------------------------------//
//=====================================SOM FUNCTIONS=============================================//
//-----------------------------------------------------------------------------------------------//

//================================================================================================//
/**
* @brief This function initializes an som_node_t object.
*
* @param[in,out] som_node_t* self
* @param[in] unsigned int x_location
* @param[in] unsigned int y_location
* @param[in] unsigned int* num_features
* @param[in] random_number_generator_t* rng
*
* @return NONE
*/
//================================================================================================//
void initialize_som_node(som_node_t*,unsigned int,unsigned int,unsigned int*,random_number_generator_t*);


//================================================================================================//
/**
* @brief This function calculates the distance between the node and the bmu.
*
* @param[in,out] som_node_t* self
* @param[in] som_node_t* bmu
*
* @return NONE
*/
//================================================================================================//
void calculate_som_node_distance(som_node_t*,som_node_t*);


//================================================================================================//
/**
* @brief This function update the weights for an som node.
*
* @param[in,out] som_node_t* self
* @param[in] som_t* som
*
* @return NONE
*/
//================================================================================================//
void update_som_node_weights(som_node_t*,som_t*);


//================================================================================================//
/**
* @brief This function initializes an som_t object.
*
* @param[in,out] som_t* self
* @param[in] unsigned int num_x_nodes
* @param[in] unsigned int num_y_nodes
* @param[in] unsigned int num_features
* @param[in] unsigned int num_iterations
* @param[in] double initial_learning_rate
* @param[in] random_number_generator_t* rng
*
* @return NONE
*/
//================================================================================================//
void initialize_som(som_t*,unsigned int,unsigned int,unsigned int,
					unsigned int,double,random_number_generator_t*);


//================================================================================================//
/**
* @brief This function initializes an som_t object.
*
* @param[in,out] som_t* self
* @param[in] double* input
* @param[in] unsigned int num_features
* @param[in] unsigned int input_class
*
* @return NONE
*/
//================================================================================================//
void set_som_input(som_t*,double*,unsigned int,unsigned int);


//================================================================================================//
/**
* @brief This function updates the som's radius.
*
* @param[in,out] som_t* self
* @param[in] unsigned int iteration
*
* @return NONE
*/
//================================================================================================//
void update_som_radius(som_t*,unsigned int);


//================================================================================================//
/**
* @brief This function updates the som's learning rate.
*
* @param[in,out] som_t* self
* @param[in] unsigned int iteration
*
* @return NONE
*/
//================================================================================================//
void update_som_learning_rate(som_t*,unsigned int);


//================================================================================================//
/**
* @brief This function updates the som's best matching unit.
*
* @param[in,out] som_t* self
*
* @return NONE
*/
//================================================================================================//
void find_som_best_matching_unit(som_t*);

//------------------------------------------------------------------------------------------------//
//=================================OPTIMIZATION FUNCTIONS=========================================//
//------------------------------------------------------------------------------------------------//

//================================================================================================//
/**
* @brief This function runs the conjugate gradient method on a system of double precision equations.
*
* NOTE: The system A must be positive definite or this method will diverge.
*
* @param[in] double* A
* @param[in] double* b
* @param[in] double* M1
* @param[in] unsigned int dimension
* @param[out] double* x
*
* @return NONE
*/
//================================================================================================//
void run_conjugate_gradient_dbl(double*,double*,double*,unsigned int,double*);


//------------------------------------------------------------------------------------------------//
//====================================DATASET FUNCTIONS===========================================//
//------------------------------------------------------------------------------------------------//

//================================================================================================//
/**
* @brief This function initializes a dataset_t object.
*
* @param[in,out] dataset_t* self
* @param[in] unsigned int num_features
* @param[in] unsigned int num_attributes
* @param[in] unsigned int num_classes
* @param[in] data_type type
*
* @return NONE
*/
//================================================================================================//
void initialize_dataset(dataset_t*,unsigned int,unsigned int,unsigned int,data_type);


//================================================================================================//
/**
* @brief This function resets a dataset_t object.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void reset_dataset(dataset_t*);


//================================================================================================//
/**
* @brief This function copies a dataset_t object.
*
* @param[in] dataset_t* original
* @param[out] dataset_t* copy
*
* @return NONE
*/
//================================================================================================//
void copy_dataset_dbl(dataset_t*,dataset_t*);


//================================================================================================//
/**
* @brief This function loads a dataset_t object from a printed file of a dataset.
* NOTE: The datset must be initialized first!
*
* @param[in] char* filename
* @param[out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void load_dataset_dbl(char*,dataset_t*);


//================================================================================================//
/**
* @brief This function finds the indices of the records that correspond to an attribute level.
*
* @param[in] dataset_t* self
* @param[in] unsigned int attribute_index
* @param[in] double attribute_level
* @param[out] unsigned int* record_indices
* @param[out] unsigned int* num_levels
*
* @return NONE
*/
//================================================================================================//
void get_dataset_record_indices_by_attribute_dbl(dataset_t*,unsigned int,double,
												 unsigned int*,unsigned int*);


//================================================================================================//
/**
* @brief This function subsets a dataset via a feature vector of 0s (discard) and 1s (keep).
*
* @param[in] dataset_t* self
* @param[in] unsigned int* feature_indices
* @param[out] dataset_t* subset
*
* @return NONE
*/
//================================================================================================//
void subset_dataset_by_feature_dbl(dataset_t*,unsigned int*,dataset_t*);


//================================================================================================//
/**
* @brief This function subsets a dataset via an attribute.
*
* @param[in] dataset_t* self
* @param[in] unsigned int attribute_index
* @param[in] double* attribute_levels
* @param[in] unsigned int num_attribute_levels
* @param[out] dataset_t* subset
*
* @return NONE
*/
//================================================================================================//
void subset_dataset_by_attribute_dbl(dataset_t*,unsigned int,double*,unsigned int,dataset_t*);


//================================================================================================//
/**
* @brief This function gets the indices of all records in a dataset for a given class.
*
* @param[in] dataset_t* self
* @param[in] unsigned int class_level
* @param[out] unsigned int* indices
* @param[out] unsigned int* num_indices
*
* @return NONE
*/
//================================================================================================//
void get_dataset_indices_by_class(dataset_t*,unsigned int,unsigned int*,unsigned int*);


//================================================================================================//
/**
* @brief This function gets the indices of all records in a dataset for a given attribute.
*
* @param[in] dataset_t* self
* @param[in] unsigned int attribute_index
* @param[in] double* attribute_levels
* @param[in] unsigned int num_attribute_levels
* @param[out] unsigned int* indices
* @param[out] unsigned int* num_indices
*
* @return NONE
*/
//================================================================================================//
void get_dataset_indices_by_attribute_dbl(dataset_t*,unsigned int,double*,
										  unsigned int,unsigned int*,unsigned int*);


//================================================================================================//
/**
* @brief This function subsets a dataset via a class.
*
* @param[in] dataset_t* self
* @param[in] unsigned int class
* @param[out] dataset_t* subset
*
* @return NONE
*/
//================================================================================================//
void subset_dataset_by_class_dbl(dataset_t*,unsigned int,dataset_t*);


//================================================================================================//
/**
* @brief This function creates test indices for k-folds analysis.
*
* NOTE: The format is [[fold1], [fold2], ..., [foldn-1]] each w/ length self->num_records/num_folds.
*
* @param[in] dataset_t* self
* @param[in] unsigned int num_folds
* @param[in] unsigned int* test_indices
* @param[in,out] random_number_generator_t* rng
*
* @return NONE
*/
//================================================================================================//
void make_dataset_test_indices(dataset_t*,unsigned int,unsigned int*,random_number_generator_t*);


//================================================================================================//
/**
* @brief This function finds the levels of an attribute in a dataset.
*
* @param[in] dataset_t* self
* @param[in] unsigned int attribute_index
* @param[out] double* attribute_levels
* @param[out] unsigned int* num_levels
*
* @return NONE
*/
//================================================================================================//
void find_attribute_levels_dbl(dataset_t*,unsigned int,double*,unsigned int*);


//================================================================================================//
/**
* @brief This function balances the number of records across class and attribute.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void balance_dataset_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function balances the number of records for a given class.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void balance_dataset_classes_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function balances the number of records for a given attribute.
*
* @param[in,out] dataset_t* self
* @param[in] unsigned int attribute_index
*
* @return NONE
*/
//================================================================================================//
void balance_dataset_attribute_dbl(dataset_t*,unsigned int);


//================================================================================================//
/**
* @brief This function gets a record from the dataset.
*
* @param[in] dataset_t* self
* @param[in] unsigned int index
* @param[out] double* record
*
* @return NONE
*/
//================================================================================================//
void get_dataset_record_dbl(dataset_t*,unsigned int,double*);


//================================================================================================//
/**
* @brief This function removes a record from the dataset.
*
* @param[in,out] dataset_t* self
* @param[in] unsigned int index
*
* @return NONE
*/
//================================================================================================//
void remove_dataset_record_dbl(dataset_t*,unsigned int);


//================================================================================================//
/**
* @brief This function adds a record to the dataset.
*
* @param[in,out] dataset_t* self
* @param[in] double* record
* @param[in] double* attributes
* @param[in] unsigned int* class
*
* @return NONE
*/
//================================================================================================//
void add_dataset_record_dbl(dataset_t*,double*,double*,unsigned int*);


//================================================================================================//
/**
* @brief This function copies all records of a feature to an array.
*
* @param[in,out] dataset_t* self
* @param[in] unsigned int feature_index
* @param[out] double* feature_records
*
* @return NONE
*/
//================================================================================================//
void copy_dataset_feature_dbl(dataset_t*,unsigned int,double*);


//================================================================================================//
/**
* @brief This function sets all records of a feature into an array.
*
* @param[in,out] dataset_t* self
* @param[in] double* feature_records
* @param[in] unsigned int feature_index
*
* @return NONE
*/
//================================================================================================//
void set_dataset_feature_dbl(dataset_t*,double*,unsigned int);


//================================================================================================//
/**
* @brief This function removes all records of a feature in a dataset.
*
* @param[in,out] dataset_t* self
* @param[in] unsigned int feature_index
*
* @return NONE
*/
//================================================================================================//
void remove_dataset_feature_dbl(dataset_t*,unsigned int);


//================================================================================================//
/**
* @brief This function sets the class codes in the dataset.
*
* @param[in,out] dataset_t* self
* @param[in] unsigned int* classes
*
* @return NONE
*/
//================================================================================================//
void set_dataset_classes_dbl(dataset_t*,unsigned int*);


//================================================================================================//
/**
* @brief This function finds the number of classes in the dataset by looking through records.
*
* @param[in] dataset_t* self
* @param[out] unsigned int* unique_classes
* @param[out] unsigned int* num_classes
*
* @return NONE
*/
//================================================================================================//
void find_dataset_class_levels_dbl(dataset_t*,unsigned int*,unsigned int*);


//================================================================================================//
/**
* @brief This function prints the dataset records to a file.
*
* @param[in,out] dataset_t* self
* @param[in] FILE* file
*
* @return NONE
*/
//================================================================================================//
void print_dataset_records_dbl(dataset_t*,FILE*);


//================================================================================================//
/**
* @brief This function normalizes each feature to have zero mean and unit variance.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_records_zscores_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function compresses the dataset records to the range [0,1].
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compress_dataset_records_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the number of records in each class in the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_class_sizes(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the class probabilities for each class in the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_class_probabilities(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the global centroid for all records in the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_record_centroid_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the class centroids for all classes in the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_class_centroids_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the global covariance for all records in the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_covariance_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the class covariance for all classes in the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_class_covariances_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the within class scatter matrix for the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_dataset_within_class_scatters_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function computes the between class scatter matrix for the dataset.
*
* @param[in,out] dataset_t* self
*
* @return NONE
*/
//================================================================================================//
void compute_between_class_scatters_dbl(dataset_t*);


//================================================================================================//
/**
* @brief This function generates a dataset filled with random gaussian features.
*
* @param[in] unsigned int num_classes
* @param[in] unsigned int num_records
* @param[in] unsigned int num_features
* @param[out] dataset_t* self
* @param[in,out] random_number_generator_t* rng
*
* @return NONE
*/
//================================================================================================//
void generate_random_dataset_dbl(unsigned int,unsigned int,unsigned int,
								 dataset_t*,random_number_generator_t*);


//================================================================================================//
/**
* @brief This function computes J = |Sb + Sw|/|Sw|.
*
* @param[in] dataset_t* self
*
* @return double J
*/
//================================================================================================//
double compute_dataset_seperability_criterion_1(dataset_t*);


//================================================================================================//
/**
* @brief This function computes standard PCA for all records the dataset.
*
* @param[in,out] dataset_t* self
* @param[in] double percent_variance
*
* @return NONE
*/
//================================================================================================//
void run_dataset_global_max_variance_pca_dbl(dataset_t*,double);


//================================================================================================//
/**
* @brief This function computes PCA using the within class scatter for all records the dataset.
*
* NOTE: KL2 In Statistical Pattern Recognition 2nd Edition by Andrew R Webb pg 330
*
* @param[in,out] dataset_t* self
* @param[in] double percent_variance
*
* @return NONE
*/
//================================================================================================//
void run_dataset_class_max_variance_pca_dbl(dataset_t*,double);


//================================================================================================//
/**
* @brief This function computes PCA using the between class scatter for all records the dataset.
*
* NOTE: KL3 In Statistical Pattern Recognition 2nd Edition by Andrew R Webb pg 331
*
* @param[in,out] dataset_t* self
* @param[in] double percent_variance
*
* @return NONE
*/
//================================================================================================//
void run_dataset_class_mean_variance_pca_dbl(dataset_t*,double);


//================================================================================================//
/**
* @brief This function computes PCA using min entropy measure for all records the dataset.
*
* NOTE: KL4 In Statistical Pattern Recognition 2nd Edition by Andrew R Webb pg 331
* NOTE: Percent Entropy should be small
*
* @param[in,out] dataset_t* self
* @param[in] double percent_entropy
*
* @return NONE
*/
//================================================================================================//
void run_dataset_class_min_entropy_pca_dbl(dataset_t*,double);


//================================================================================================//
/**
* @brief This function computes PCA using class compression measure for all records the dataset.
*
* NOTE: KL5 In Statistical Pattern Recognition 2nd Edition by Andrew R Webb pg 331
* NOTE: Percent Entropy should be small
*
* @param[in,out] dataset_t* self
* @param[in] double within_percent_variance
* @param[in] double between_percent_variance
*
* @return NONE
*/
//================================================================================================//
void run_dataset_class_compressed_variance_pca_dbl(dataset_t*,double,double);


//================================================================================================//
/**
* @brief This function creates a dataset that can be separated by various entropy means.
*
* @param[in] unsigned int num_classes
* @param[in] unsigned int num_records
* @param[in] unsigned int num_features
* @param[out] dataset_t* self
* @param[in,out] random_number_generator_t* rng
*
* @return NONE
*/
//================================================================================================//
void generate_seperable_dataset_dbl(unsigned int,unsigned int,unsigned int,
									dataset_t*,random_number_generator_t*);


#endif //MACHINE_LEARNING_H//
