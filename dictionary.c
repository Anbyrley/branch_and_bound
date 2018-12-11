#include "dictionary.h"

/*
	This is should be a hashmap
*/


//================================================================================================//
//========================================ENTRY METHODS===========================================//
//================================================================================================//

void initialize_dictionary_entry( dictionary_entry_t* self )
{

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- initialize_dictionary_entry!\n");
		quit();
	}

	//===Initialize===//
	self->available = 1;
	self->value_length = 0;
	self->key_length = 0;
	initialize_array_char(self->key_char, MAX_DICTIONARY_KEY_LENGTH);
	initialize_array_uint(self->key_uint, MAX_DICTIONARY_KEY_LENGTH);
	initialize_array_uint(self->value_uint, MAX_DICTIONARY_VALUE_LENGTH);
	initialize_array_dbl(self->value_dbl, MAX_DICTIONARY_VALUE_LENGTH);

	return;
}

void print_dictionary_entry( dictionary_entry_t* self,
							 data_type key_type,
							 data_type value_type,
							 FILE* file )
{

	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- print_dictionary_entry!\n");
		quit();
	}

	fprintf(file, "Key: ");
	if (key_type == UINT){
		for (i=0; i<self->key_length; i++){
			fprintf(file, "%d ", self->key_uint[i]);
		}	
	}
	else if (key_type == CHAR){
		for (i=0; i<self->key_length; i++){
			fprintf(file, "%c", self->key_char[i]);
		}	
	}
	else{
		fprintf(stderr, "Error:: Key Type is Invalid! In Function -- print_dictionary_entry!\n");
		quit();
	}
	fprintf(file, " ");

	fprintf(file, "Value: ");
	if (value_type == DOUBLE){
		for (i=0; i<self->value_length; i++){
			fprintf(file, "%lf ", self->value_dbl[i]);
		}	
	}
	else if (value_type == UINT){
		for (i=0; i<self->value_length; i++){
			fprintf(file, "%d ", self->value_uint[i]);
		}	
	}
	else{
		fprintf(stderr, "Error:: Value Type is Invalid! In Function -- print_dictionary_entry!\n");
		quit();
	}
	fprintf(file, "\n");

	return;
}

void set_dictionary_entry_key_char( dictionary_entry_t* self,
							   		char* key,
							   		unsigned int key_length )
{
	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- set_dictionary_entry_key_char!\n");
		quit();
	}
	if (key == NULL){
		fprintf(stderr, "Error:: Key is NULL! In Function -- set_dictionary_entry_key_char!\n");
		quit();
	}
	if (key_length == 0){
		fprintf(stderr, "Error:: Key Length is Invalid! In Function -- set_dictionary_entry_key_char!\n");
		quit();
	}

	//===Copy Over===//
	copy_array_char(key, key_length, self->key_char);
	self->key_length = key_length;
	return;
}


void set_dictionary_entry_key_uint( dictionary_entry_t* self,
							   		unsigned int* key,
							   		unsigned int key_length )
{
	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- set_dictionary_entry_key_uint!\n");
		quit();
	}
	if (key == NULL){
		fprintf(stderr, "Error:: Key is NULL! In Function -- set_dictionary_entry_key_uint!\n");
		quit();
	}
	if (key_length == 0){
		fprintf(stderr, "Error:: Key Length is Invalid! In Function -- set_dictionary_entry_key_uint!\n");
		quit();
	}

	//===Copy Over===//
	copy_array_uint(key, key_length, self->key_uint);
	self->key_length = key_length;
	return;
}

void set_dictionary_entry_value_uint( dictionary_entry_t* self,
							   		  unsigned int* value,
							   		  unsigned int value_length )
{
	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- set_dictionary_entry_value_uint!\n");
		quit();
	}
	if (value == NULL){
		fprintf(stderr, "Error:: Value is NULL! In Function -- set_dictionary_entry_value_uint!\n");
		quit();
	}
	if (value_length == 0){
		fprintf(stderr, "Error:: Value Length is Invalid! In Function -- set_dictionary_entry_value_uint!\n");
		quit();
	}

	//===Copy Over===//
	copy_array_uint(value, value_length, self->value_uint);
	self->value_length = value_length;

	return;
}


void set_dictionary_entry_value_dbl( dictionary_entry_t* self,
							   		 double* value,
							   		 unsigned int value_length )
{
	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- set_dictionary_entry_value_dbl!\n");
		quit();
	}
	if (value == NULL){
		fprintf(stderr, "Error:: Value is NULL! In Function -- set_dictionary_entry_value_dbl!\n");
		quit();
	}
	if (value_length == 0){
		fprintf(stderr, "Error:: Value Length is Invalid! In Function -- set_dictionary_entry_value_dbl!\n");
		quit();
	}

	//===Copy Over===//
	copy_array_dbl(value, value_length, self->value_dbl);
	self->value_length = value_length;

	return;
}

void append_dictionary_entry_values_uint( dictionary_entry_t* self,
										  unsigned int* values,
										  unsigned int num_values )
{
	unsigned int i, k;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self is NULL! In Function -- append_dictionary_entry_values_uint!\n");
		quit();
	}
	if (values == NULL){
		fprintf(stderr, "Error:: Values is NULL! In Function -- append_dictionary_entry_values_uint!\n");
		quit();
	}
	if (self->value_length + num_values >= MAX_DICTIONARY_VALUE_LENGTH){
		fprintf(stderr, "Error:: Num Values is Too Large! In Function -- append_dictionary_entry_values_uint!\n");
		quit();
	}

	//===Append===//
 	k = 0;
	for (i=self->value_length; i<self->value_length+num_values; i++){
		self->value_uint[i] = values[k++];
	}
	self->value_length += num_values;
	
	return;
}

void remove_dictionary_entry_values_uint( dictionary_entry_t* self,
										  unsigned int* values,
										  unsigned int num_values )
{
	int k;
	unsigned int j, num_indices;
	unsigned int indices[MAX_DICTIONARY_VALUE_LENGTH];

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self is NULL! In Function -- remove_dictionary_entry_values_uint!\n");
		quit();
	}
	if (values == NULL){
		fprintf(stderr, "Error:: Values is NULL! In Function -- remove_dictionary_entry_values_uint!\n");
		quit();
	}
	if ((int)((int)self->value_length - (int)num_values) < 0 ){
		fprintf(stderr, "Error:: Num Values is Too Large! In Function -- remove_dictionary_entry_values_uint!\n");
		quit();
	}

	//===Remove==//
	k =	num_values-1;
	while(k >= 0){	
		//===Find And Remove===//
		find_indices_where_equal_uint(self->value_uint,self->value_length,values[k],&num_indices,indices);
		for(j=0; j<num_indices; j++){
			remove_from_array_uint(self->value_uint,&(self->value_length),indices[j]);
		}
		k--;
	}

	return;
}


void copy_dictionary_entry( dictionary_entry_t* original,
							dictionary_entry_t* copy )
{

	//===Sanity Check===//
	if (original == NULL){
		fprintf(stderr, "Error:: Original is NULL! In Function -- copy_dictionary_entry!\n");
		quit();
	}
	if (copy == NULL){
		fprintf(stderr, "Error:: Copy is NULL! In Function -- copy_dictionary_entry!\n");
		quit();
	}

	//===Copy Each===//
	copy->available = original->available;
	copy->value_length = original->value_length;
	copy->key_length = original->key_length;
	copy_array_uint(original->value_uint, original->value_length, copy->value_uint);
	copy_array_dbl(original->value_dbl, original->value_length, copy->value_dbl);
	copy_array_char(original->key_char, original->key_length, copy->key_char);
	copy_array_uint(original->key_uint, original->key_length, copy->key_uint);


	return;
}

void reset_dictionary_entry( dictionary_entry_t* self )
{
	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- reset_dictionary_entry!\n");
		quit();
	}
	//===Reset===//
	initialize_dictionary_entry(self);
	return;
}

unsigned int are_dictionary_entries_equal( dictionary_entry_t* entry_one,
										   dictionary_entry_t* entry_two )
{

	unsigned int equal;

	//===Sanity Check===//
	if (entry_one == NULL){
		fprintf(stderr, "Error:: Entry One is NULL! In Function -- are_dictionary_entries_equal!\n");
		quit();
	}
	if (entry_two == NULL){
		fprintf(stderr, "Error:: Entry Two is NULL! In Function -- are_dictionary_entries_equal!\n");
		quit();
	}

	//===Try To Prove Unequality===//
	equal = 1;

	//===Compare Value Lengths===//
	if (entry_one->value_length != entry_two->value_length){
		equal = 0;
		return equal;
	}
	if (entry_one->key_length != entry_two->key_length){
		equal = 0;
		return equal;
	}

	//===Check Equality===//	
	equal = are_arrays_equal_dbl(entry_one->value_dbl, entry_two->value_dbl, entry_one->value_length);
	if (equal == 0) return equal;
	equal = are_arrays_equal_uint(entry_one->value_uint, entry_two->value_uint, entry_one->value_length);
	if (equal == 0) return equal;
	equal = are_arrays_equal_char(entry_one->key_char, entry_two->key_char, entry_one->key_length);
	if (equal == 0) return equal;
	equal = are_arrays_equal_uint(entry_one->key_uint, entry_two->key_uint, entry_one->key_length);

	return equal;
}

unsigned int are_dictionary_entry_keys_equal_char( dictionary_entry_t* entry_one,
											  	   dictionary_entry_t* entry_two )
{

	unsigned int equal;

	//===Sanity Check===//
	if (entry_one == NULL){
		fprintf(stderr, "Error:: Entry One is NULL! In Function -- are_dictionary_entry_keys_equal_char!\n");
		quit();
	}
	if (entry_two == NULL){
		fprintf(stderr, "Error:: Entry Two is NULL! In Function -- are_dictionary_entry_keys_equal_char!\n");
		quit();
	}

	//===Try To Prove Unequality===//
	equal = 1;

	//===Compare Key Lengths===//
	if (entry_one->key_length != entry_two->key_length){
		equal = 0;
		return equal;
	}

	//===Check Key Equality===//	
	equal = are_arrays_equal_char(entry_one->key_char, entry_two->key_char, entry_one->key_length);

	return equal;
}	

unsigned int are_dictionary_entry_keys_equal_uint( dictionary_entry_t* entry_one,
											  	   dictionary_entry_t* entry_two )
{

	unsigned int equal;

	//===Sanity Check===//
	if (entry_one == NULL){
		fprintf(stderr, "Error:: Entry One is NULL! In Function -- are_dictionary_entry_keys_equal_uint!\n");
		quit();
	}
	if (entry_two == NULL){
		fprintf(stderr, "Error:: Entry Two is NULL! In Function -- are_dictionary_entry_keys_equal_uint!\n");
		quit();
	}

	//===Try To Prove Unequality===//
	equal = 1;

	//===Compare Key Lengths===//
	if (entry_one->key_length != entry_two->key_length){
		equal = 0;
		return equal;
	}

	//===Check Key Equality===//	
	equal = are_arrays_equal_uint(entry_one->key_uint, entry_two->key_uint, entry_one->key_length);

	return equal;
}	

unsigned int get_dictionary_entry_minimum_value_uint( dictionary_entry_t* self )
{	
	return find_minimum_uint(self->value_uint, self->value_length);
}

void sort_dictionary_entry_values_uint( dictionary_entry_t* self )
{
	sort_array_uint(self->value_uint, self->value_length);
	return;
}

//================================================================================================//
//======================================DICTIONARY METHODS========================================//
//================================================================================================//

void initialize_dictionary( dictionary_t* self,
							data_type key_type,
							data_type value_type )
{
	unsigned int i;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self is NULL! In Function -- initialize_dictionary!\n");
		quit();
	}

	//===Initialize Each Entry===//
	for (i=0; i<MAX_DICTIONARY_ENTRIES; i++){
		initialize_dictionary_entry(&(self->entries[i]));
	}
	self->num_entries = 0;
	self->key_type = key_type;
	self->value_type = value_type;

	return;
}

void print_dictionary( dictionary_t* self, 
					   FILE* file )
{
	unsigned int i;
	for (i=0; i<MAX_DICTIONARY_ENTRIES; i++){
		if (!self->entries[i].available){
			print_dictionary_entry(&(self->entries[i]), self->key_type, self->value_type,file);
		}
	}
	return;
}

void print_dictionary_keys( dictionary_t* self )
{
	unsigned int i, j;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self is NULL! In Function -- print_dictionary_keys!\n");
		quit();
	}

	//===Print Keys===//
	for (j=0; j<MAX_DICTIONARY_ENTRIES; j++){
		if (!self->entries[j].available){

			fprintf(stdout, "Key: ");
			if (self->key_type == UINT){
				for (i=0; i<self->entries[j].key_length; i++){
					fprintf(stdout, "%d ", self->entries[j].key_uint[i]);
				}	
			}
			else if (self->key_type == CHAR){
				for (i=0; i<self->entries[j].key_length; i++){
					fprintf(stdout, "%c", self->entries[j].key_char[i]);
				}	
			}
			else{
				fprintf(stderr, "Error:: Key Type is Invalid! In Function -- print_dictionary_keys!\n");
				quit();
			}
			fprintf(stdout, "\n");

		}
	}


	return;
}

void print_dictionary_by_level( dictionary_t* self )
{

	unsigned int i, j, key_length;
	//===Find Longest Key Length===//
	key_length = 0;
	for (i=0; i<MAX_DICTIONARY_ENTRIES; i++){
		if (!self->entries[i].available){
			if (self->entries[i].key_length > key_length){
				key_length = self->entries[i].key_length;
			}
		}
	}

	//===Print Each Key Length===//
	for (j=key_length; j>0; j--){
		for (i=0; i<MAX_DICTIONARY_ENTRIES; i++){
			if (!self->entries[i].available && self->entries[i].key_length == j){
				print_dictionary_entry(&(self->entries[i]), self->key_type, self->value_type,stdout);
			}
		}
		newline();
	}

	return;
}

void print_dictionary_by_key_length( dictionary_t* self,
									 unsigned int key_length )
{

	unsigned int i;
	for (i=0; i<MAX_DICTIONARY_ENTRIES; i++){
		if (!self->entries[i].available && self->entries[i].key_length == key_length){
			print_dictionary_entry(&(self->entries[i]), self->key_type, self->value_type,stdout);
		}
	}

	return;
}

void get_dictionary_best_entry_by_key_length( dictionary_t* self,
											  unsigned int key_length,
											  dictionary_entry_t* entry )
{

	unsigned int i;
	double best_value;

	best_value = -DBL_MAX;
	for (i=0; i<MAX_DICTIONARY_ENTRIES; i++){
		if (!self->entries[i].available && self->entries[i].key_length == key_length){
			if (self->entries[i].value_dbl[0] > best_value){
				best_value = self->entries[i].value_dbl[0];
				copy_dictionary_entry(&(self->entries[i]), entry);
			}
		}
	}
	return;
}

long long int hash_array_char( char* key,
							   unsigned int key_length )
{
	unsigned int i;
	long long int hash;

	//===Sanity Check===//
	if (key == NULL){
		fprintf(stderr, "Error:: Key is NULL! In Function -- hash_array_char!\n");
		quit();
	}
	if (key_length == 0){
		fprintf(stderr, "Error:: Length is Invalid! In Function -- hash_array_char!\n");
		quit();
	}

	//===Make Key===//
	hash = 0;
	for (i=0; i<key_length; i++){
		hash=hash*DICTIONARY_HASH_MULTIPLIER+pow(((long long int)key[i])*DICTIONARY_KEY_STRING_BASE,i+1);
	}
	hash = hash % MAX_DICTIONARY_ENTRIES;
	return hash;
}

long long int hash_array_uint( unsigned int* key,
							   unsigned int key_length )
{
	unsigned int i;
	long long int hash;

	//===Sanity Check===//
	if (key == NULL){
		fprintf(stderr, "Error:: Key is NULL! In Function -- hash_array_uint!\n");
		quit();
	}
	if (key_length == 0){
		fprintf(stderr, "Error:: Length is Invalid! In Function -- hash_array_uint!\n");
		quit();
	}

	//===Make Key===//
	hash = 0;
	for (i=0; i<key_length; i++){
		hash = hash * DICTIONARY_HASH_MULTIPLIER + (long long int)key[i];
	}
	hash = hash % MAX_DICTIONARY_ENTRIES;
	return hash;
}

unsigned int has_dictionary_entry( dictionary_t* self,
								   dictionary_entry_t* entry )
{
	unsigned int has_entry;
	long long int hash, starting_hash;

	//===Get The Hash===//
	if (self->key_type == UINT){
		hash = hash_array_uint(entry->key_uint,entry->key_length);
	}
	else if (self->key_type == CHAR){
		hash = hash_array_char(entry->key_char,entry->key_length);
	}
	else{
		fprintf(stderr, "Error:: Key Type Is Invalid! In Function -- has_dictionary_entry!\n");
		quit();
	}
	starting_hash = hash;

	//===Keep Looking If Not Available==//	
	has_entry = 0;	
	while(1){		

		//===Check If Equal===//
		if (self->key_type == UINT){
			if (are_dictionary_entry_keys_equal_uint(&(self->entries[hash]), entry)){
				has_entry = 1;
				break;
			}
		}
		else if (self->key_type == CHAR){
			if (are_dictionary_entry_keys_equal_char(&(self->entries[hash]), entry)){
				has_entry = 1;
				break;
			}
		}

		//===Else Continue===//
		hash = (hash + 1) % (MAX_DICTIONARY_ENTRIES);
		if (hash == starting_hash){
			has_entry = 0;
			break;
		}
	}

	return has_entry;
}

void get_dictionary_entry_by_key_char( dictionary_t* self, 
								  	   char* key,
								  	   unsigned int key_length,
								  	   dictionary_entry_t** entry )
{

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dictionary is NULL! In Function -- get_dictionary_entry_by_key_char!\n");
		quit();
	}
	if (key == NULL){
		fprintf(stderr, "Error:: Key is NULL! In Function -- get_dictionary_entry_by_key_char!\n");
		quit();
	}
	if (key_length == 0){
		fprintf(stderr, "Error:: Key length is Invalid! In Function -- get_dictionary_entry_by_key_char!\n");
		quit();
	}

	long long int hash, starting_hash;

	//===Get The Hash===//
	hash = hash_array_char(key,key_length);
	starting_hash = hash;

	//===Keep Looking If Not Available==//		
	while(!self->entries[hash].available){		

		//===Check If Equal===//
		if (are_arrays_equal_char(self->entries[hash].key_char, key, key_length)){
			*entry = &(self->entries[hash]);			
			break;
		}

		//===Else Continue===//
		hash = (hash + 1) % (MAX_DICTIONARY_ENTRIES);
		if (hash == starting_hash){
			entry = NULL;
			if (entry == NULL){
				fprintf(stderr, "Error:: Entry Was Not Found! In Function -- get_dictionary_entry_by_key_char!\n");
				quit();
			}
		}
	}
	return;
}
	
//use this to get the value in the entry by getting the entry itself
//because we get the entry, and then we can delete it.
void get_dictionary_entry_by_key_uint( dictionary_t* self, 
								  	   unsigned int* key,
								  	   unsigned int key_length,
								  	   dictionary_entry_t** entry )
{

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dictionary is NULL! In Function -- get_dictionary_entry_by_key_uint!\n");
		quit();
	}
	if (key == NULL){
		fprintf(stderr, "Error:: Key is NULL! In Function -- get_dictionary_entry_by_key!_uint\n");
		quit();
	}
	if (key_length == 0){
		fprintf(stderr, "Error:: Key length is Invalid! In Function -- get_dictionary_entry_by_key_uint!\n");
		quit();
	}

	long long int hash, starting_hash;

	//===Get The Hash===//
	hash = hash_array_uint(key,key_length);
	starting_hash = hash;

	//===Keep Looking If Not Available==//		
	while(!self->entries[hash].available){		

		//===Check If Equal===//
		if (are_arrays_equal_uint(self->entries[hash].key_uint, key, key_length)){
			*entry = &(self->entries[hash]);			
			break;
		}

		//===Else Continue===//
		hash = (hash + 1) % (MAX_DICTIONARY_ENTRIES);
		if (hash == starting_hash){
			entry = NULL;
			if (entry == NULL){
				fprintf(stderr, "Error:: Entry Was Not Found! In Function -- get_dictionary_entry_by_key_uint!\n");
				quit();
			}
		}
	}
	return;
}

void add_dictionary_entry( dictionary_t* self,
						   dictionary_entry_t* entry )
{
	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dictionary is NULL! In Function -- add_dictionary_entry!\n");
		quit();
	}
	if (entry == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- add_dictionary_entry!\n");
		quit();
	}

	long long int hash, starting_hash;

	//===Check If Full===//
	if (self->num_entries == MAX_DICTIONARY_ENTRIES){
		fprintf(stderr, "Error:: Dictionary Is Full! In Function -- add_dictionary_entry!\n"); 		
		quit();
	}
	
	//===Get The Hash===//
	if (self->key_type == UINT){
		hash = hash_array_uint(entry->key_uint,entry->key_length);
	}
	else if (self->key_type == CHAR){
		hash = hash_array_char(entry->key_char,entry->key_length);
	}
	else{
		fprintf(stderr, "Error:: Key Type Is Invalid! In Function -- add_dictionary_entry!\n");
		quit();
	}
	starting_hash = hash;

	//===Search For A Spot===//
	while(!self->entries[hash].available){		

		//===Else Continue===//
		hash = (hash + 1) % (MAX_DICTIONARY_ENTRIES);

		//===Full So Quit===//
		if (hash == starting_hash){
			fprintf(stderr, "Error:: Dictionary Is Full! In Function -- add_dictionary_entry!\n"); 		
			quit();
		}
	}

	//===Add In===//
	copy_dictionary_entry(entry,&(self->entries[hash]));
	self->entries[hash].available = 0;
	self->num_entries++;

	return;
}

void append_dictionary_entry( dictionary_t* self,
							  dictionary_entry_t* entry )
{

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dictionary is NULL! In Function -- append_dictionary_entry!\n");
		quit();
	}
	if (entry == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- append_dictionary_entry!\n");
		quit();
	}
		
	unsigned int i, k, equal;
	long long int hash, starting_hash;

	//===Check If Full===//
	if (self->num_entries == MAX_DICTIONARY_ENTRIES){
		fprintf(stderr, "Error:: Dictionary Is Full! In Function -- append_dictionary_entry!\n"); 		
		quit();
	}
	
	//===Get The Hash===//
	if (self->key_type == UINT){
		hash = hash_array_uint(entry->key_uint,entry->key_length);
	}
	else if (self->key_type == CHAR){
		hash = hash_array_char(entry->key_char,entry->key_length);
	}
	else{
		fprintf(stderr, "Error:: Key Type Is Invalid! In Function -- append_dictionary_entry!\n");
		quit();
	}
	starting_hash = hash;

	//===Search For Same Key===//
	equal = 0;
	while(1){
	
		//===Check If Keys Are Equal===//
		if (self->key_type == UINT){
			equal = are_dictionary_entry_keys_equal_uint(&(self->entries[hash]), entry);
			if (equal) break;
		}
		else if (self->key_type == CHAR){
			equal = are_dictionary_entry_keys_equal_char(&(self->entries[hash]), entry);
			if (equal) break;
		}

		//===Else Continue===//
		hash = (hash + 1) % (MAX_DICTIONARY_ENTRIES);

		//===Full So Quit===//
		if (hash == starting_hash){
			fprintf(stderr, "Error:: Entry Key Not Found! In Function -- append_dictionary_entry!\n"); 		
			quit();
		}
	}

	//===Append To Entry===//
	k = 0;
	for (i=self->entries[hash].value_length; i<self->entries[hash].value_length+entry->value_length; i++){
		if (self->value_type == UINT){
			self->entries[hash].value_uint[i] = entry->value_uint[k++];
		}
		else if (self->value_type == DOUBLE){
			self->entries[hash].value_dbl[i] = entry->value_dbl[k++];
		}
	}
	self->entries[hash].value_length += entry->value_length;

	return;
}

void delete_dictionary_entry( dictionary_t* self,
						      dictionary_entry_t* entry )
{

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dictionary is NULL! In Function -- delete_dictionary_entry!\n");
		quit();
	}
	if (entry == NULL){
		fprintf(stderr, "Error:: Entry is NULL! In Function -- delete_dictionary_entry!\n");
		quit();
	}

	unsigned int found;
	long long int hash, starting_hash;

	//===Get The Hash===//
	if (self->key_type == UINT){
		hash = hash_array_uint(entry->key_uint,entry->key_length);
	}
	else if (self->key_type == CHAR){
		hash = hash_array_char(entry->key_char,entry->key_length);
	}
	else{
		fprintf(stderr, "Error:: Key Type Is Invalid! In Function -- delete_dictionary_entry!\n");
		quit();
	}
	starting_hash = hash;

	//===Search Until A Spot Opens===//
	while(self->entries[hash].available){		

		//===Check If Key Matches===//
		found = are_dictionary_entry_keys_equal_uint(&(self->entries[hash]), entry);
		if (found){
			break;
		}
		else{
			//===Else Continue===//
			hash = (hash + 1) % (MAX_DICTIONARY_ENTRIES);

			//===Full So Quit===//
			if (hash == starting_hash){
				fprintf(stderr, "Error:: Dictionary Is Full! In Function -- delete_dictionary_entry!\n"); 		
				quit();
			}

		}
	}

	//===Reset===//
	reset_dictionary_entry(&(self->entries[hash]));
	self->num_entries--;

	return;
}

void generate_dictionary_power_set( dictionary_t* self,
									unsigned int num_features,
									random_number_generator_t* rng )
{
	unsigned int t, j;
	unsigned int c[MAX_POLYNOMIAL_ORDER+2];
	unsigned int features[MAX_POLYNOMIAL_ORDER+2];
	double value;
	dictionary_entry_t entry;

	//===Run Through All num_features Choose t===//
	for (t=1; t<=num_features; t++){

		//===Initialize===//
		for (j=1; j<=t; j++){
			c[j] = j-1;
		}
		c[t+1] = num_features;
		c[t+2] = 0;

		j = 1;
		while(j<=t){

			//===Create This Subsets Value That Satisfies Monotonicity===//
			value = get_random_uniform_dbl((t+1)*100.0, (t+2)*100.0, rng);
			if (t == num_features) value = 1000.0;

			//===Create This Subsets Key===//
			copy_array_uint(c+1, t, features);
			add_array_constant_uint(features, t, 1);

			//===Create Entry===//
			set_dictionary_entry_key_uint(&entry, features, t);
			set_dictionary_entry_value_dbl(&entry, &value, 1);

			//===Add To Dictionary===//
			add_dictionary_entry(self, &entry);
		
			//===Create Next Subset===//
			j = 1;
			while(c[j] +1 == c[j+1]){
				c[j] = j-1;
				j = j+1;
			}
			c[j] = c[j] + 1;

		}
	}	

	return;
}


//================================================================================================//
//=========================================TEST METHODS===========================================//
//================================================================================================//

void test_dictionary()
{
	unsigned int key_length;
	unsigned int key[100];
	double value;
	dictionary_t* dictionary;
	dictionary_entry_t* entry;

	//===Mallocs===//
	dictionary = malloc(sizeof(dictionary_t));
	entry = malloc(sizeof(dictionary_entry_t));

	//===Init===//
	initialize_dictionary(dictionary, UINT, DOUBLE);

	//===Set Entry===//
	key_length = 4; value = 1000;
	key[0] = 1; key[1] = 2; key[2] = 3; key[3] = 4;
	initialize_dictionary_entry(entry);
	set_dictionary_entry_key_uint(entry, key, key_length);
	set_dictionary_entry_value_dbl(entry, &value, 1);
	add_dictionary_entry(dictionary, entry);
	
	//===Print===//
	fprintf(stdout, "Dictionary After Addition:\n");
	print_dictionary(dictionary,stdout);
	newline();

	//===Get Entry===//
	get_dictionary_entry_by_key_uint(dictionary, key, key_length, &entry);

	//===Delete Entry===//
	delete_dictionary_entry(dictionary, entry);

	//===Print===//
	fprintf(stdout, "Dictionary After Deletion:\n");
	print_dictionary(dictionary,stdout);
	newline();

	//===Clean Up===//
	free(entry);
	free(dictionary);

	return;
}	

void test_dictionary_power_set()
{
	unsigned int num_features;
	dictionary_t* dictionary;
	random_number_generator_t* rng;

	//===Mallocs===//
	rng = malloc(sizeof(random_number_generator_t));
	dictionary = malloc(sizeof(dictionary_t));

	//===Init===//
	initialize_random_number_generator(rng);
	initialize_dictionary(dictionary, UINT, DOUBLE);

	//===Generate===//
	num_features = 5;
	generate_dictionary_power_set(dictionary, num_features, rng);

	//===Print===//
	fprintf(stdout, "Dictionary Of Power Set:\n");
	print_dictionary(dictionary,stdout);
	newline();

	//===Clean Up===//
	free(rng);
	free(dictionary);

	return;
}

void test_dictionary_string_keys()
{
	unsigned int key_length;
	unsigned int values[100];
	char key[100];
	dictionary_t* dictionary;
	dictionary_entry_t* entry;

	//===Mallocs===//
	dictionary = malloc(sizeof(dictionary_t));
	entry = malloc(sizeof(dictionary_entry_t));

	//===Init===//
	initialize_dictionary(dictionary, CHAR, UINT);

	//===Set Up Entry===//
	key_length = 3;
	values[0] = 100; values[1] = 345; values[2] = 888; values[3] = 1042;
	strcpy(key, "M01");
	initialize_dictionary_entry(entry);
	set_dictionary_entry_key_char(entry, key, key_length);
	set_dictionary_entry_value_uint(entry, values, 4);
	add_dictionary_entry(dictionary, entry);

	//===Print===//
	fprintf(stdout, "Dictionary:\n");
	print_dictionary(dictionary,stdout);
	newline();


	//===Clean Up===//
	free(entry);
	free(dictionary);

	return;
}
