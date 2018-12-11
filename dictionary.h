/** @file dictionary.h
*   @brief Contains functions for the data dictionary.
*
*
*  @author Alex N. Byrley (anbyrley)
*  @date February 2016
*  @bug No known bugs
*/

#ifndef DICTIONARY_H
#define DICTIONARY_H

#define DICTIONARY_KEY_STRING_BASE 103
#define DICTIONARY_HASH_MULTIPLIER (31)
#define MAX_DICTIONARY_KEY_LENGTH 100
#define MAX_DICTIONARY_VALUE_LENGTH 100

//================================================================================================//
//===================================STANDARD INCLUDES============================================//
//================================================================================================//

#include "macros.h"
#include "helper.h"
#include "random_numbers.h"


//================================================================================================//
//====================================DATA STRUCTURES=============================================//
//================================================================================================//

//================================================================================================//
/** @enum data_type
*   @brief This enum flags whether the data has a certain type.
*/
//================================================================================================//
typedef enum{
	CHAR,
	UINT,
	DOUBLE, 
	COMPLEX,
	VOID
} data_type;


//================================================================================================//
/** @struct dictionary_entry_t
*   @brief This structure holds the various types of entries for the dictionary object.
*/
//================================================================================================//
typedef struct dictionary_entry_s dictionary_entry_t;
typedef struct dictionary_entry_s{
	unsigned int available; //if available == 1, then the cell is empty and is available for use
	unsigned int value_length;
	unsigned int key_length;
	char key_char[MAX_DICTIONARY_KEY_LENGTH];
	unsigned int key_uint[MAX_DICTIONARY_KEY_LENGTH];
	unsigned int value_uint[MAX_DICTIONARY_VALUE_LENGTH];
	double value_dbl[MAX_DICTIONARY_VALUE_LENGTH];
} dictionary_entry_t;


//================================================================================================//
/** @struct dictionary_t
*   @brief This structure is the typedef the dictionary_t object.
*/
//================================================================================================//
typedef struct dictionary_s dictionary_t;
typedef struct dictionary_s{
	data_type key_type;	
	data_type value_type;
    unsigned int num_entries; //num_entries will count how many get added, if > MAX throw error
	//maybe should have a IF MAX > LIMIT use MALLOC instead?
	dictionary_entry_t entries[MAX_DICTIONARY_ENTRIES]; //so we will access via entries[key].data->dbl[1]
} dictionary_t;



//================================================================================================//
//==================================FUNCTION DECLARATIONS=========================================//
//================================================================================================//


//================================================================================================//
/**
* @brief This function initializes a dictionary entry.
*
* @param[in,out] dictionary_entry_t* self
*
* @return NONE
*/
//================================================================================================//
void initialize_dictionary_entry(dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function prints a dictionary to stdout.
*
* @param[in] dictionary_entry_t* self
* @param[in] data_type key_type
* @param[in] data_type value_type
* @param[out] FILE* file
*
* @return NONE
*/
//================================================================================================//
void print_dictionary_entry(dictionary_entry_t*,data_type,data_type,FILE*);


//================================================================================================//
/**
* @brief This function sets a dictionary entry's key as a char array.
*
* @param[in,out] dictionary_entry_t* self
* @param[in] char* key
* @param[in] unsigned int key_length
*
* @return NONE
*/
//================================================================================================//
void set_dictionary_entry_key_char(dictionary_entry_t*,char*,unsigned int);


//================================================================================================//
/**
* @brief This function sets a dictionary entry's key as an unsigned integer array.
*
* @param[in,out] dictionary_entry_t* self
* @param[in] unsigned int* key
* @param[in] unsigned int key_length
*
* @return NONE
*/
//================================================================================================//
void set_dictionary_entry_key_uint(dictionary_entry_t*,unsigned int*,unsigned int);


//================================================================================================//
/**
* @brief This function sets a dictionary entry's value as an unsigned int array.
*
* @param[in,out] dictionary_entry_t* self
* @param[in] unsigned int* value
* @param[in] unsigned int value_length
*
* @return NONE
*/
//================================================================================================//
void set_dictionary_entry_value_uint(dictionary_entry_t*,unsigned int*,unsigned int);


//================================================================================================//
/**
* @brief This function sets a dictionary entry's value as a double array.
*
* @param[in,out] dictionary_entry_t* self
* @param[in] double* value
* @param[in] unsigned int value_length
*
* @return NONE
*/
//================================================================================================//
void set_dictionary_entry_value_dbl(dictionary_entry_t*,double*,unsigned int);


//================================================================================================//
/**
* @brief This function appends uints to a dictionary entry's value array.
*
* @param[in,out] dictionary_entry_t* self
* @param[in] unsigned int* values
* @param[in] unsigned int num_values
*
* @return NONE
*/
//================================================================================================//
void append_dictionary_entry_values_uint(dictionary_entry_t*,unsigned int*,unsigned int);


//================================================================================================//
/**
* @brief This function removes uints to a dictionary entry's value array.
*
* @param[in,out] dictionary_entry_t* self
* @param[in] unsigned int* values
* @param[in] unsigned int num_values
*
* @return NONE
*/
//================================================================================================//
void remove_dictionary_entry_values_uint(dictionary_entry_t*,unsigned int*,unsigned int);


//================================================================================================//
/**
* @brief This function copies a dictionary entry.
*
* @param[in] dictionary_entry_t* original
* @param[out] dictionary_entry_t* copy
*
* @return NONE
*/
//================================================================================================//
void copy_dictionary_entry(dictionary_entry_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function resets a dictionary entry.
*
* @param[in,out] dictionary_entry_t* original
*
* @return NONE
*/
//================================================================================================//
void reset_dictionary_entry(dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function checks if two entries are equal.
*
* @param[in] dictionary_entry_t* entry_one
* @param[in] dictionary_entry_t* entry_two
*
* @return unsigned int equal
*/
//================================================================================================//
unsigned int are_dictionary_entries_equal(dictionary_entry_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function checks if two entries char keys are equal.
*
* @param[in] dictionary_entry_t* entry_one
* @param[in] dictionary_entry_t* entry_two
*
* @return unsigned int equal_keys
*/
//================================================================================================//
unsigned int are_dictionary_entry_keys_equal_char(dictionary_entry_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function checks if two entries unsigned int keys are equal.
*
* @param[in] dictionary_entry_t* entry_one
* @param[in] dictionary_entry_t* entry_two
*
* @return unsigned int equal_keys
*/
//================================================================================================//
unsigned int are_dictionary_entry_keys_equal_uint(dictionary_entry_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function returns the minimum unsigned int value of the entry.
*
* @param[in] dictionary_entry_t* self
*
* @return unsigned int minimum_value
*/
//================================================================================================//
unsigned int get_dictionary_entry_minimum_value_uint(dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function sorts the values in the dictionary entry.
*
* @param[in] dictionary_entry_t* self
*
* @return NONE
*/
//================================================================================================//
void sort_dictionary_entry_values_uint(dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function initializes a dictionary
*
* @param[in,out] dictionary_t* self
* @param[in] data_type key_type
* @param[in] data_type value_type

* @return NONE 
*/
//================================================================================================//
void initialize_dictionary(dictionary_t*,data_type,data_type);


//================================================================================================//
/**
* @brief This function prints a dictionary to stdout
*
* @param[in] dictionary_t* self
* @param[out] FILE* file
*
* @return NONE 
*/
//================================================================================================//
void print_dictionary(dictionary_t*,FILE*);


//================================================================================================//
/**
* @brief This function prints a dictionary's keys
*
* @param[in] dictionary_t* self
*
* @return NONE 
*/
//================================================================================================//
void print_dictionary_keys(dictionary_t*);



void print_dictionary_by_level( dictionary_t* self );


//================================================================================================//
/**
* @brief This function prints all entries of a certain key length from a dictionary to stdout.
*
* @param[in] dictionary_t* self
* @param[in] unsigned int key_length 
*
* @return NONE 
*/
//================================================================================================//
void print_dictionary_by_key_length(dictionary_t*,unsigned int);


void get_dictionary_best_entry_by_key_length( dictionary_t* self,
											  unsigned int key_length,
											  dictionary_entry_t* entry );

//================================================================================================//
/**
* @brief This function hashes a key of chars
*
* @param[in] char* key
* @param[in] unsigned int key_length
*
* @return long long int hash
*/
//================================================================================================//
long long int hash_array_char(char*,unsigned int);


//================================================================================================//
/**
* @brief This function hashes a key of uints
*
* @param[in] unsigned int* key
* @param[in] unsigned int key_length
*
* @return long long int hash
*/
//================================================================================================//
long long int hash_array_uint(unsigned int*,unsigned int);


//================================================================================================//
/**
* @brief This function hashes a key of uints
*
* @param[in] dictionary_t* self
* @param[in] dictionary_entry_t* entry 
*
* @return unsigned int has_entry
*/
//================================================================================================//
unsigned int has_dictionary_entry(dictionary_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function gets a reference to an entry via an char key.
*
* @param[in] dictionary_t* self
* @param[in] char* key
* @param[in] unsigned int key_length
* @param[out] dictionary_entry_t** entry
*
* @return NONE
*/
//================================================================================================//
void get_dictionary_entry_by_key_char(dictionary_t*,char*,unsigned int,dictionary_entry_t**);


//================================================================================================//
/**
* @brief This function gets a reference to an entry via an unsigned integer key.
*
* @param[in] dictionary_t* self
* @param[in] unsigned int* key
* @param[in] unsigned int key_length
* @param[out] dictionary_entry_t** entry
*
* @return NONE
*/
//================================================================================================//
void get_dictionary_entry_by_key_uint(dictionary_t*,unsigned int*,unsigned int,dictionary_entry_t**);


//================================================================================================//
/**
* @brief This function adds an entry to the dictionary.
*
* @param[in,out] dictionary_t* self
* @param[in] dictionary_entry_t* entry
*
* @return NONE
*/
//================================================================================================//
void add_dictionary_entry(dictionary_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function appends to an entry in the dictionary.
*
* @param[in,out] dictionary_t* self
* @param[in] dictionary_entry_t* entry
*
* @return NONE
*/
//================================================================================================//
void append_dictionary_entry(dictionary_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function deletes an entry from the dictionary.
*
* @param[in,out] dictionary_t* self
* @param[in] dictionary_entry_t* entry
*
* @return NONE
*/
//================================================================================================//
void delete_dictionary_entry(dictionary_t*,dictionary_entry_t*);


//================================================================================================//
/**
* @brief This function generates keys with random values in the power set of a [1,...,num_features].
*
* @param[in,out] dictionary_t* self
* @param[in] unsigned int num_features
* @param[in] random_number_generator_t* rng
*
* @return NONE
*/
//================================================================================================//
void generate_dictionary_power_set(dictionary_t*,unsigned int,random_number_generator_t*);


//================================================================================================//
/**
* @brief This function tests basic dictionary operations.
*
* @return NONE
*/
//================================================================================================//
void test_dictionary();

void test_dictionary_power_set();
void test_dictionary_string_keys();

#endif //DICTIONARY_H//
