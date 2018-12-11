#include "branch_and_bound.h"

//================================================================================================//
//=========================================NODE METHODS===========================================//
//================================================================================================//

void initialize_bb_node( bb_node_t* self )
{
	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- initialize_bb_node!\n");
		quit();
	}

	//===Initialize===//
	self->level = MAX_NUM_BB_LEVELS;
	self->num_descendants = 0;
	self->sibling_rank = 0;
	self->feature = MAX_NUM_FEATURES;
	self->value = -DBL_MAX;
	self->parent = NULL;
	for (i=0; i<MAX_NUM_FEATURES; i++){
		self->descendants[i] = NULL;
	}	

	return;
}

void delete_bb_node( bb_node_t* self )
{
	if (0){
		unsigned int i;
		for (i=0; i<MAX_NUM_FEATURES; i++){
			if (self->descendants[i] != NULL){
				delete_bb_node(self->descendants[i]);
			}
		}
	}

	//===Clean Up===//
	free(self);
	self = NULL;

	return;
}

void print_bb_node( bb_node_t* self )
{

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- print_bb_node!\n");
		quit();
	}

	//===Print===//
	fprintf(stdout, "---------------------------------\n");
	fprintf(stdout, "Level: %d ", self->level);
	fprintf(stdout, "Feature: %d ", self->feature);
	fprintf(stdout, "Value: %lf ", self->value);
	fprintf(stdout, "Sibling Rank: %d ", self->sibling_rank);
	fprintf(stdout, "Num Descendants: %d ", self->num_descendants);
	newline();

	return;
}


//================================================================================================//
//=========================================TREE METHODS===========================================//
//================================================================================================//

void initialize_bb_tree( bb_tree_t* self, 
						 unsigned int* features,
						 unsigned int num_features,
						 unsigned int subset_size )
{

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- initialize_bb_tree!\n");
		quit();
	}
	if (features == NULL){
		fprintf(stderr, "Error:: Features Is NULL! In Function -- initialize_bb_tree!\n");
		quit();
	}
	if (num_features == 0 || num_features > MAX_NUM_FEATURES){
		fprintf(stderr, "Error:: Num_features Is Invalid! In Function -- initialize_bb_tree!\n");
		quit();
	}
	if (subset_size == 0 || subset_size > MAX_NUM_BB_LEVELS){
		fprintf(stderr, "Error:: Subset_size Is Invalid! In Function -- initialize_bb_tree!\n");
		quit();
	}

	//===Init Singles===//
	self->num_features = num_features;
	self->num_available_features = num_features;
	self->subset_size = subset_size;
	self->num_levels = num_features - subset_size;
	self->loop_count = 0;
	self->current_best_objective = 2*EPS;

	//===Init Arrays===//
	copy_array_uint(features, num_features, self->features);
	copy_array_uint(features, num_features, self->available_features);
	initialize_array_uint(self->current_feature_path, MAX_NUM_FEATURES);
	initialize_array_uint(self->optimal_features, MAX_NUM_FEATURES);

	//===Init Nodes===//
	initialize_bb_node(&(self->root));
	self->current_node = &(self->root);

	//===Set Up Root Node===//
	self->root.num_descendants = subset_size + 1;
	self->root.level = 0;
	self->root.feature = 0;
	self->root.sibling_rank = subset_size + 1;
	self->root.value = 0;
	self->root.parent = NULL;

	//===Init Dictionary===//
	initialize_dictionary(&(self->test_feature_dictionary), UINT, DOUBLE);

	return;
}

void run_bb_tree( bb_tree_t* self )
{
	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- run_bb_tree!\n");
		quit();
	}

	//===Run===//	
	initialize_bb_tree_next_node(self);

	return;	
}

void print_bb_tree_optimal_features( bb_tree_t* self,
									 FILE* file )
{
	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- print_bb_tree_optimal_features!\n");
		quit();
	}
	if (file == NULL){
		fprintf(stderr, "Error:: File Is NULL! In Function -- print_bb_tree_optimal_features!\n");
		quit();
	}

	//===Print Features===//
	if (isatty(fileno(file))){
		fprintf(stdout, "Optimal Features: ");
	}
	for (i=0; i<self->subset_size; i++){
		fprintf(file, "%d ", self->optimal_features[i]);
	}
	fprintf(file, "\n");

	return;
}

unsigned int should_bb_tree_unwind( bb_tree_t* self)
{
	unsigned int unwind;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- should_bb_tree_unwind!\n");
		quit();
	}

	//===Check Unwind===//
	unwind = 0;
	if (self->current_node->parent == NULL){
		if (self->current_node->level == 0 && self->current_node->num_descendants == 0){
			unwind = 1;
		}
	}
	return unwind;
}

void initialize_bb_tree_next_node( bb_tree_t* self )
{
	unsigned int i, j, k, key_length;
	unsigned int indices[MAX_NUM_FEATURES];
	unsigned int feature_key[MAX_NUM_FEATURES];
	double j_values[MAX_NUM_FEATURES];
	#if USE_TEST_DICTIONARY
		dictionary_entry_t* entry;
	#else
		unsigned int feature_indicator[MAX_NUM_FEATURES];
	#endif
	bb_node_t* track;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- initialize_bb_tree_next_node!\n");
		quit();
	}

	#if DEBUG_BB
		fprintf(stdout,"Press ENTER key to Continue.\n");  
		getchar();  
	#endif


	#if DEBUG_BB
		newline();
		fprintf(stdout, "#===============Creating A New Node===============#\n");	
	#endif

	//===Check Unwind===//
	if (should_bb_tree_unwind(self)){
		return;
	}

	//===Rank Available Features===//
	for (i=0; i<self->num_available_features; i++){

		//===Create Current Path===//
		k = 0;
		track = self->current_node;
		for (j=self->current_node->level; j>0; j--){
			self->current_feature_path[k++] = track->feature;
			track = track->parent;
		}
		
		//===Add Current Feature To Current Path===//
		self->current_feature_path[self->current_node->level] = self->available_features[i];
				
		//===Remove Amended Current Path From Total Feature List===// (to get key)
		key_length = self->num_features;
		copy_array_uint(self->features, self->num_features, feature_key);
		for (j=0; j<=self->current_node->level; j++){
			remove_array_element_uint(feature_key,&key_length,self->current_feature_path[j]);
		}
		
		#if DEBUG_BB
			fprintf(stdout, "Testing Feature: %d\n", self->available_features[i]);
			fprintf(stdout, "Path: ");
			for (j=0; j<=self->current_node->level; j++){
				fprintf(stdout, "%d ", self->current_feature_path[j]);
			}
			fprintf(stdout, "Key: ");
			for (j=0; j<key_length; j++){
				fprintf(stdout, "%d ", feature_key[j]);
			}
			newline();
		#endif

		//===Remove Current Feature From Current Path===// (for next round)
		self->current_feature_path[self->current_node->level] = 0;
			

		#if USE_TEST_DICTIONARY
			//===Get Entry For Potential New Path===//
			get_dictionary_entry_by_key_uint(&(self->test_feature_dictionary), feature_key, key_length, &entry);

			//===Print Entry===//
			#if DEBUG_BB
				fprintf(stdout, "Feature: %d	", self->available_features[i]);
				print_dictionary_entry(entry, self->key_type, self->value_type,stdout);
			#endif

			//===Copy Over J Value===//
			j_values[i] = entry->value_dbl[0];		
		#else
			//===Make Subset From Feature Keys===//
			initialize_array_uint(feature_indicator, MAX_NUM_FEATURES);
			for (j=0; j<key_length; j++){
				feature_indicator[feature_key[j]-1] = 1;
			}			
			subset_dataset_by_feature_dbl(&(self->full_dataset), feature_indicator, &(self->subset));
			
			//===Calculate J Value Of Subset===//
			j_values[i] = compute_dataset_seperability_criterion_1(&(self->subset));
		#endif
	}

	//===Check If No More Descendants===//
	if (self->current_node->num_descendants == 0){
		fprintf(stdout, "OoOps! Ran Out Of Descendants!\n");
		quit();
	}


	//===Sort J Values===//
	sort_array_indices_dbl(j_values, self->num_available_features, indices);
	#if DEBUG_BB
		newline();
		fprintf(stdout, "Sorted J Values: ");
		for (i=0; i<self->current_node->num_descendants; i++){
			fprintf(stdout, "%d ", self->available_features[indices[i]]);
		}
		newline(); 
		newline();
	#endif

	//===Create Descendants===//
	for (i=0; i<self->current_node->num_descendants; i++){
	
		//===Allocate Space===//
		self->current_node->descendants[i] = malloc(sizeof(bb_node_t));	

		//===Initialize===//
		initialize_bb_node(self->current_node->descendants[i]);

		//===Set Parent===//
		self->current_node->descendants[i]->parent = self->current_node;

		//===Create Sibling Ranks===//
		self->current_node->descendants[i]->sibling_rank = self->current_node->sibling_rank - i;
		self->current_node->descendants[i]->num_descendants = self->current_node->sibling_rank - i;
		
		//===Add Feature===//
		self->current_node->descendants[i]->feature = self->available_features[indices[i]];
		
		//===Add Value===//
		self->current_node->descendants[i]->value = j_values[indices[i]];

		//===Add Level===//
		self->current_node->descendants[i]->level = self->current_node->level + 1;

		//===Copy Over===//
		feature_key[i] = self->available_features[indices[i]];
	}

	//===Remove Descendant Features From Available Set===//
	for (i=0; i<self->current_node->num_descendants; i++){
		remove_array_element_uint(self->available_features,&(self->num_available_features),feature_key[i]);
	}

	#if DEBUG_BB
		fprintf(stdout, "Current Node: \n");
		print_bb_node(self->current_node);
		newline();
	#endif

	//===Select Next Node===//
	self->current_node = self->current_node->descendants[self->current_node->num_descendants-1];

	#if DEBUG_BB
		fprintf(stdout, "Next Node: \n");
		print_bb_node(self->current_node);
		newline();
		fprintf(stdout, "Available Features: \n");
		print_vector_uint(self->available_features, self->num_available_features, stdout);
		newline();
	#endif

	//===Increment Loop Count===//
	self->loop_count += 1;

	//===Testing Current Node===//
	test_bb_tree_current_node(self);

	return;
}

void test_bb_tree_current_node( bb_tree_t* self )
{

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- test_bb_tree_current_node!\n");
		quit();
	}

	#if DEBUG_BB
		fprintf(stdout,"Press ENTER key to Continue.\n");  
		getchar();  
	#endif

	//===Check Unwind===//
	if (should_bb_tree_unwind(self)){
		return;
	}

	#if DEBUG_BB
		newline();
		fprintf(stdout, "#===============Testing The New Node===============#\n");
		
		//===Print Current Path===//
		unsigned int k, j;
		unsigned int path[100], key[100];
		double j_value;
		unsigned int feature_indicator[MAX_NUM_FEATURES];
		bb_node_t* track;		
		#if USE_TEST_DICTIONARY
				dictionary_entry_t* entry;
		#endif

		//===Make Current Path===//
		k = 0;
		track = self->current_node;
		fprintf(stdout, "Current Path: ");
		for (j=self->current_node->level; j>0; j--){
			path[k] = track->feature;
			track = track->parent;
			fprintf(stdout, "%d ", path[k]);
			k++;
		}

		//===Create Features To Save===//
		fprintf(stdout, " Key: ");
		k = self->num_features;
		copy_array_uint(self->features, self->num_features, key);
		for (j=0; j<self->current_node->level; j++){
			remove_array_element_uint(key,&k,path[j]);
		}
		for (j=0; j<k; j++){			
			fprintf(stdout, "%d ", key[j]);
		}


		//===Print Entry Value===//
		#if USE_TEST_DICTIONARY
			get_dictionary_entry_by_key_uint(&(self->test_feature_dictionary), key, k, &entry);
			fprintf(stdout, " Value: %lf", entry->value_dbl[0]);
			newline();
		#else
			//===Make Subset From Feature Keys===//
			initialize_array_uint(feature_indicator, MAX_NUM_FEATURES);
			for (j=0; j<k; j++){
				feature_indicator[key[j]-1] = 1;
			}			
			subset_dataset_by_feature_dbl(&(self->full_dataset), feature_indicator, &(self->subset));
		
			//===Calculate J Value Of Subset===//
			j_value = compute_dataset_seperability_criterion_1(&(self->subset));
			fprintf(stdout, " Value: %lf", j_value);
			newline();
		#endif		

	#endif

	//===Check To See If Node Violates Bound===//
	if (self->current_node->value < self->current_best_objective){

		#if DEBUG_BB			
			fprintf(stdout, "Node Is NOT Optimal!\n");
		#endif

		//===Delete===//
		backtrack_bb_tree(self);

		//===Return===//
		return;
		
	}

	//===Check Current Level===//
	if (self->current_node->level == self->num_levels){

		#if DEBUG_BB
			fprintf(stdout, "Node Is LAST Node!\n");
		#endif
		
		//===Save===//
		update_bb_bound(self);

		//===Return===//
		return;
	}

	//===Create Its Next Level And Go There===//
	initialize_bb_tree_next_node(self);


	return;
}

void backtrack_bb_tree( bb_tree_t* self )
{

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- backtrack_bb_tree!\n");
		quit();
	}

	#if DEBUG_BB
		fprintf(stdout,"Press ENTER key to Continue.\n");  
		getchar();  
	#endif

	//===Check Unwind===//
	if (should_bb_tree_unwind(self)){
		return;
	}

	#if DEBUG_BB
		newline();
		fprintf(stdout, "#===============Backtracking===============#\n");
	#endif

	//===Add Feature Back===//
	self->available_features[(self->num_available_features)++] = self->current_node->feature;

	//===Check If Done===//
	if (self->current_node->parent == NULL){
		if (self->current_node->level == 0 && self->current_node->num_descendants == 0){
			//===Print Optimal Features===//
			print_bb_tree_optimal_features(self,stdout);
			return;
		}
	}

	//===Backtrack===//
	self->current_node = self->current_node->parent;

	#if DEBUG_BB
		fprintf(stdout, "New Current Backtracked Node: \n");
		print_bb_node(self->current_node);
	#endif	

	//===Delete Previous Node===//	
	self->current_node->num_descendants = self->current_node->num_descendants-1;
	#if DEBUG_BB
		fprintf(stdout, "New Current Backtracked Node: \n");
		print_bb_node(self->current_node);
	#endif
	if (self->current_node->descendants[self->current_node->num_descendants] != NULL){
		//===Clean Up===//
		free(self->current_node->descendants[self->current_node->num_descendants]);
		self->current_node->descendants[self->current_node->num_descendants] = NULL;
	}

	//===Select Next Node===//
	if (self->current_node->num_descendants == 0){
		//===Backtrack===//
		backtrack_bb_tree(self);
	}
	else{
		//===Select Sibling Node===//
		self->current_node = self->current_node->descendants[self->current_node->num_descendants-1];

		#if DEBUG_BB
			fprintf(stdout, "New Current Sibling Node: \n");
			print_bb_node(self->current_node);
		#endif

		//===Test Node===//
		test_bb_tree_current_node(self);
	}

	return;
}

void update_bb_bound( bb_tree_t* self )
{
	unsigned int i, j, key_length;
	unsigned int feature_key[100], feature_track[100];
	bb_node_t* track;
	#if USE_TEST_DICTIONARY
		dictionary_entry_t* entry;
	#else
		double j_value;
		unsigned int feature_indicator[MAX_NUM_FEATURES];
	#endif

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- update_bb_bound!\n");
		quit();
	}

	#if DEBUG_BB
		fprintf(stdout,"Press ENTER key to Continue.\n");  
		getchar();  
	#endif

	#if DEBUG_BB
		newline();
		fprintf(stdout, "#===============Updating BB Bound===============#\n");
	#endif

	//===Track Features===//
	track = self->current_node;
	for (i=0; i<self->num_levels; i++){
		feature_track[i] = track->feature;
		track = track->parent;
	}
	
	//===Get Key===//
	key_length = self->num_features;
	copy_array_uint(self->features, self->num_features, feature_key);
	for (j=0; j<self->num_levels; j++){
		remove_array_element_uint(feature_key,&key_length,feature_track[j]);
	}

	//===Save Optimal Features===//
	copy_array_uint(feature_key, key_length, self->optimal_features);

	//===Get Optimal Entry===//
	#if USE_TEST_DICTIONARY
		get_dictionary_entry_by_key_uint(&(self->test_feature_dictionary), feature_key, key_length, &entry);
	#else
		//===Make Subset From Feature Keys===//
		initialize_array_uint(feature_indicator, MAX_NUM_FEATURES);
		for (j=0; j<key_length; j++){
			feature_indicator[feature_key[j]-1] = 1;
		}			
		subset_dataset_by_feature_dbl(&(self->full_dataset), feature_indicator, &(self->subset));
	
		//===Calculate J Value Of Subset===//
		j_value = compute_dataset_seperability_criterion_1(&(self->subset));
	#endif

	//===Update Bound===//
	#if USE_TEST_DICTIONARY	
		self->current_best_objective = entry->value_dbl[0];
	#else
		self->current_best_objective = j_value;
	#endif

	#if DEBUG_BB
		fprintf(stdout, "Current Best Bound: %lf\n", self->current_best_objective);
		newline();
	#endif

	#if DEBUG_BB
		fprintf(stdout, "Optimal Features: ");
		for (i=0; i<self->subset_size; i++){
			fprintf(stdout, "%d ", self->optimal_features[i]);
		}
		newline();
		newline();
	#endif

	//===Backtrack===//
	backtrack_bb_tree(self);

	return;
}	


//================================================================================================//
//=========================================TEST METHODS===========================================//
//================================================================================================//


void setup_bb_test_dictionary( bb_tree_t* self )
{
	unsigned int key_length;
	double value;
	unsigned int key[100];
	dictionary_entry_t entry;

	if (self->num_features == 4){

		//===Set Entry===//
		key_length = 3; value = 1.0;
		key[0] = 2; key[1] = 3; key[2] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 3; value = 2.0;
		key[0] = 1; key[1] = 3; key[2] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 3; value = 3.0;
		key[0] = 1; key[1] = 2; key[2] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 3; value = 6.0;
		key[0] = 1; key[1] = 2; key[2] = 3;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 2; value = 0.5;
		key[0] = 1; key[1] = 2;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 2; value = 0.33;
		key[0] = 1; key[1] = 3;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 2; value = 0.25;
		key[0] = 1; key[1] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 2; value = 4.25;
		key[0] = 2; key[1] = 3;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 2; value = 4.0;
		key[0] = 2; key[1] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Set Entry===//
		key_length = 2; value = 5.0;
		key[0] = 3; key[1] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//===Print===//
		fprintf(stdout, "Test Dictionary:\n");
		print_dictionary(&(self->test_feature_dictionary),stdout);
		newline();
	}
	
	if (self->num_features == 5){


		//===Set Entry===// (take out 5)
		key_length = 4; value = 11.0;
		key[0] = 1; key[1] = 2; key[2] = 3; key[3] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 4)
		key_length = 4; value = 12.0;
		key[0] = 1; key[1] = 2; key[2] = 3; key[3] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===//(take out 1)
		key_length = 4; value = 3.0;
		key[0] = 2; key[1] = 3; key[2] = 4; key[3] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 2)
		key_length = 4; value = 4.0;
		key[0] = 1; key[1] = 3; key[2] = 4; key[3] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 3)
		key_length = 4; value = 5.0;
		key[0] = 1; key[1] = 2; key[2] = 4; key[3] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//=============//
		//===Level 2===//
		//=============//

		//===Set Entry===// (take out 2 and 5)
		key_length = 3; value = 11.1; 
		key[0] = 1; key[1] = 3; key[2] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);


		//===Set Entry===// (take out 3 and 4)
		key_length = 3; value = 4.0;
		key[0] = 1; key[1] = 2; key[2] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 2 and 3)
		key_length = 3; value = 0.333;
		key[0] = 1; key[1] = 4; key[2] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 2 and 4)
		key_length = 3; value = 1.0;
		key[0] = 1; key[1] = 3; key[2] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 and 4)
		key_length = 3; value = 7.0; 
		key[0] = 2; key[1] = 3; key[2] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 and 3)
		key_length = 3; value = 6.0;
		key[0] = 2; key[1] = 4; key[2] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 and 2)
		key_length = 3; value = 5.0;
		key[0] = 3; key[1] = 4; key[2] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 and 5)
		key_length = 3; value = 19.0;
		key[0] = 2; key[1] = 3; key[2] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//==============//
		//===Level 3===//
		//==============//
		
		//===Set Entry===// (take out 3 4 5)
		key_length = 2; value = 0.5; 
		key[0] = 1; key[1] = 2;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 2 4 5)
		key_length = 2; value = 0.1;
		key[0] = 1; key[1] = 3;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 2 3 4)
		key_length = 2; value = 1.5;
		key[0] = 1; key[1] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 2 3 5)
		key_length = 2; value = 0.125;
		key[0] = 1; key[1] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (tkae out 2 4 5)
		key_length = 2; value = 6.0;
		key[0] = 3; key[1] = 2;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 3 4)
		key_length = 2; value = 0.3;
		key[0] = 2; key[1] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 4 5)
		key_length = 2; value = 4.0;
		key[0] = 2; key[1] = 3;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//


		//===Set Entry===// (take out 1 3 5)
		key_length = 2; value = 0.5;
		key[0] = 2; key[1] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 2 3)
		key_length = 2; value = 15.0;
		key[0] = 4; key[1] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 2 4)
		key_length = 2; value = 12.0;
		key[0] = 3; key[1] = 5;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

		//----//

		//===Set Entry===// (take out 1 2 5)
		key_length = 2; value = 7.0;
		key[0] = 3; key[1] = 4;
		initialize_dictionary_entry(&entry);
		set_dictionary_entry_key_uint(&entry, key, key_length);
		set_dictionary_entry_value_dbl(&entry, &value, 1);

		//===Add To Dictionary===//
		add_dictionary_entry(&(self->test_feature_dictionary), &entry);

	}

	return;
}

void make_bb_test_dataset( bb_tree_t* self )
{
	unsigned int num_records, num_classes;
	random_number_generator_t* rng;

	//===Mallocs===//
	rng = malloc(sizeof(random_number_generator_t));

	//===Initializations===//
	initialize_random_number_generator(rng);	

	//===Make Test Database===//
	num_records = 20000; num_classes = 2;
	generate_seperable_dataset_dbl(num_classes, num_records, self->num_features, &(self->full_dataset), rng);

	//===Clean Up===//
	free(rng);

	return;
}

void test_bb()
{
	int status;
	unsigned int i, num_features, subset_size;
	unsigned int features[100];
	bb_tree_t* tree;
	random_number_generator_t* rng;
	#if USE_TEST_DICTIONARY
		dictionary_entry_t entry;
	#endif

	//===Mallocs===//
	tree = malloc(sizeof(bb_tree_t));
	rng = malloc(sizeof(random_number_generator_t));

	//===Set Unlimited Stack Size===//	
	status = system("ulimit -s unlimited");
	if (status){
		fprintf(stderr, "Error:: Stack Size Must Be Set To Unlimited!\n");
		quit();
	}

	//===Init Features===//
	num_features = 5;
	for (i=0; i<num_features; i++){
		features[i] = i+1;
	}

	//===Initializations===//	
	subset_size = 2;
	initialize_bb_tree(tree, features, num_features, subset_size);
	initialize_random_number_generator(rng);

	#if USE_TEST_DICTIONARY
		//===Make Dictionary===//
		//seed_random_number_generator(rng, 42);
		generate_dictionary_power_set(&(tree->test_feature_dictionary), num_features, rng);
	#else
		//===Make Dataset===//
		make_bb_test_dataset(tree);
	#endif

	//===Run===//
	run_bb_tree(tree);

	//===Check Unwind===//
	if (should_bb_tree_unwind(tree)){
		fprintf(stdout, "Done!\n");
		print_bb_tree_optimal_features(tree,stdout);
	}

	//===Print===//
	#if USE_TEST_DICTIONARY
		newline();
		fprintf(stdout, "Best Subset:\n");
		get_dictionary_best_entry_by_key_length(&(tree->test_feature_dictionary), subset_size, &entry);
		print_dictionary_entry(&entry, tree->test_feature_dictionary.key_type, tree->test_feature_dictionary.value_type,stdout);
		newline();
		fprintf(stdout, "Dictionary With Subset Size Key Lengths:\n");
		print_dictionary_by_key_length(&(tree->test_feature_dictionary), subset_size);
		newline();
	#endif

	//===Clean Up===//
	free(rng);
	free(tree);

	return;
}


