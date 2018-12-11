#include "machine_learning.h"

//================================================================================================//
//===================================SELF ORGANIZING MAP METHODS==================================//
//================================================================================================//

void initialize_som_node( som_node_t* self,
						  unsigned int x_location,
						  unsigned int y_location,
						  unsigned int* num_features,
						  random_number_generator_t* rng )
{

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: som_node_t Is NULL! In Function -- initialize_som_node!\n");
		quit();
	}
	if (rng == NULL){
		fprintf(stderr, "Error:: random_number_generator_t Is NULL! In Function -- initialize_som_node!\n");
		quit();
	}
	if (x_location > MAX_NUM_SOM_NODES/2){
		fprintf(stderr, "Error:: x_location Is Too Invalid! In Function -- initialize_som_node!\n");
		quit();
	}
	if (y_location > MAX_NUM_SOM_NODES/2){
		fprintf(stderr, "Error:: y_location Is Too Invalid! In Function -- initialize_som_node!\n");
		quit();
	}
	if (*num_features == 0 || *num_features > MAX_NUM_FEATURES){
		fprintf(stderr, "Error:: num_features Is Invalid! In Function -- initialize_som_node!\n");
		quit();
	}

	//===Set Locals===//
	self->location[0] = x_location; self->location[1] = y_location;
	self->class_total = 0; 
	self->num_times_bmu = 0;
	self->num_features = num_features;
	self->distance_factor = 1.0;

	//===Init Weights===//
	generate_uniform_process_dbl(0.0, 1.0, *num_features, self->weights, rng);

	return;
}

void calculate_som_node_distance( som_node_t* self,
								  som_node_t* bmu )
{
	double self_location[2], bmu_location[2];

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: som_node_t Is NULL! In Function -- calculate_som_node_distance!\n");
		quit();
	}
	if (bmu==NULL){
		fprintf(stderr, "Error:: bmu Is NULL! In Function -- calculate_som_node_distance!\n");
		quit();
	}

	//===Convert To Dbl===//
	convert_vector_uint_to_dbl(self->location, 2, self_location);
	convert_vector_uint_to_dbl(bmu->location, 2, bmu_location);

	//===Calculate Distance===//
	self->distance = compute_minkowski_distance_dbl(self_location, bmu_location, 2, 2);

	return;
}

void update_som_node_weights( som_node_t* self,
							  som_t* som )
{
	double influence;
	double correction[MAX_NUM_FEATURES];

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: som_node_t Is NULL! In Function -- update_som_node_weights!\n");
		quit();
	}
	if (som==NULL){
		fprintf(stderr, "Error:: som Is NULL! In Function -- update_som_node_weights!\n");
		quit();
	}

	//===Calculate Influence Function===//
	calculate_som_node_distance(self, som->bmu);
	influence = exp(-pow(self->distance, 2.0)/(2.0*pow(som->radius,2.0)));

	//===Calculate Correction===//
	subtract_vectors_dbl(som->input, self->weights, som->num_features, correction);
	gain_array_constant_dbl(correction, som->num_features, som->learning_rate);
	gain_array_constant_dbl(correction, som->num_features, influence);

	//===Add Correction===//
	add_vectors_dbl(self->weights, correction, som->num_features, self->weights);

	return;
}							
							  
void initialize_som( som_t* self,
					 unsigned int num_x_nodes,
					 unsigned int num_y_nodes,
					 unsigned int num_features,
					 unsigned int num_iterations,
					 double initial_learning_rate,
					 random_number_generator_t* rng  )
{
	unsigned int x, y;

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: SOM Is NULL! In Function -- initialize_som!\n");
		quit();
	}
	if (num_x_nodes > MAX_NUM_SOM_NODES/2){
		fprintf(stderr, "Error:: num_x_nodes Is Too Large! In Function -- initialize_som!\n");
		quit();
	}
	if (num_y_nodes > MAX_NUM_SOM_NODES/2){
		fprintf(stderr, "Error:: num_y_nodes Is Too Large! In Function -- initialize_som!\n");
		quit();
	}
	if (num_features == 0 || num_features > MAX_NUM_FEATURES){
		fprintf(stderr, "Error:: num_features Is Invalid! In Function -- initialize_som!\n");
		quit();
	}
	if (num_iterations == 0 || num_iterations > MAX_NUM_SOM_ITERATIONS){
		fprintf(stderr, "Error:: num_iterations Is Invalid! In Function -- initialize_som!\n");
		quit();
	}
	if (!isnormal((double)initial_learning_rate)){
		fprintf(stderr, "Error:: initial_learning_rate Is Invalid! In Function -- initialize_som!\n");
		quit();
	}
	if (rng == NULL){
		fprintf(stderr, "Error:: random_number_generator_t Is NULL! In Function -- initialize_som!\n");
		quit();
	}

	//===Set Up Grid===//
	self->num_nodes = num_x_nodes + num_y_nodes;
	self->num_x_nodes = num_x_nodes; self->num_y_nodes = num_y_nodes;

	//===Set Learning Rate===//
	self->initial_learning_rate = initial_learning_rate;
	self->learning_rate = initial_learning_rate;

	//===Set Input And BMU===//
	initialize_array_dbl(self->input, MAX_NUM_FEATURES);
	self->bmu = NULL; 

	//===Set Radius Decay Constant===//
	self->num_iterations = num_iterations;
	self->radius_decay_constant = ((double)(num_iterations))/(log(((double)self->num_nodes)/2.0));

	//===Set Initial Radius===//
	self->initial_radius = ((double)(MAX(num_x_nodes,num_y_nodes)))/2.0; 
	self->radius = self->initial_radius;

	//===Initialize Nodes===//
	self->num_features = num_features;
	for (x=0; x<num_x_nodes; x++){
		for (y=0; y<num_y_nodes; y++){
			initialize_som_node(&(self->nodes[x][y]), x, y, &(self->num_features), rng);
		}
	}

	return;
}

void set_som_input( som_t* self,
					double* input,
					unsigned int num_features,
					unsigned int input_class )
{

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: SOM Is NULL! In Function -- set_som_input!\n");
		quit();
	}
	if (input==NULL){
		fprintf(stderr, "Error:: input Is NULL! In Function -- set_som_input!\n");
		quit();
	}
	if (num_features == 0 || num_features != self->num_features){
		fprintf(stderr, "Error:: num_features Is Invalid! In Function -- set_som_input!\n");
		quit();
	}

	//===Copy Over===//
	copy_array_dbl(input, num_features, self->input);
	self->input_class = input_class;

	return;
}

void update_som_radius( som_t* self,
					    unsigned int iteration )
{

	double t;

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- update_som_radius!\n");
		quit();
	}

	//===Update Iteration===//
	t = iteration;	
	self->radius = self->initial_radius * exp(-t/self->radius_decay_constant);
	
	return;
}

void update_som_learning_rate( som_t* self,
							   unsigned int iteration )
{

	double t;

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- update_som_learning_rate!\n");
		quit();
	}

	//===Update Iteration===//
	t = iteration;	
	self->learning_rate = self->initial_learning_rate * exp(-t/self->radius_decay_constant);

	return;
}

void find_som_best_matching_unit( som_t* self )
{
	unsigned int x, y;
	double distance, min_distance;

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- find_som_best_matching_unit!\n");
		quit();
	}

	//===Check Every Node===//
	min_distance = DBL_MAX;
	for (x=0; x<self->num_x_nodes; x++){
		for (y=0; y<self->num_y_nodes; y++){
			//===Compute Distance===//
			distance = compute_minkowski_distance_dbl(self->nodes[x][y].weights, self->input, 
					   self->num_features, 2);
			//===Update BMU===//
			if (distance < min_distance){
				min_distance = distance;
				self->bmu = &(self->nodes[x][y]);
				self->bmu->num_times_bmu += 1;
				self->bmu->class_total += self->input_class;
			}
		}
	}	
	return;
}

/*
	To determine class clusters:
		After training, present each input to the SOM
		Add class of input to BMU's total class, keep track of how many times node is BMU
		After all inputs are presented, average each unit's class total class
		Round total class to either 0 or 1 (or 2 or etc)
*/

void train_som( som_t* self,
			    dataset_t* dataset )
{
	unsigned int iteration, input_index, x, y;
	double input[MAX_NUM_FEATURES];

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- train_som!\n");
		quit();
	}
	if (dataset==NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- train_som!\n");
		quit();
	}
	
	iteration = 0;
	input_index = 0;
	while (iteration < self->num_iterations){

		//===Set Input===//
		get_dataset_record_dbl(dataset, input_index, input);
		set_som_input(self, input, dataset->num_features, dataset->classes[input_index]);

		//===Find BMU===//
		find_som_best_matching_unit(self);

		//===Update Weights===//
		for (x=0; x<self->num_x_nodes; x++){
			for (y=0; y<self->num_y_nodes; y++){

				//===Calculate Distance To BMU===//		
				calculate_som_node_distance(&(self->nodes[x][y]), self->bmu);
				
				//===Update Weights===//
				if (self->nodes[x][y].distance < self->radius){
					update_som_node_weights(&(self->nodes[x][y]), self);
				}

			}
		}

		//===Update Iteration And Input Index===//
		iteration++;
		input_index = (input_index + 1) % (dataset->num_records);

		//===Update Radius And Learning Rate===//		
		update_som_radius(self, iteration);
		update_som_learning_rate(self, iteration);

	}

	return;
}

void make_som_class_map( som_t* self,
						 dataset_t* dataset )
{

	unsigned int r, x, y;
	double input[MAX_NUM_FEATURES];
	FILE* fout;

	//===Sanity Check===//
	if (self==NULL){
		fprintf(stderr, "Error:: Self Is NULL! In Function -- make_som_class_map!\n");
		quit();
	}
	if (dataset==NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- make_som_class_map!\n");
		quit();
	}

	//===Loop Through All Records===//
	for (r=0; r<dataset->num_records; r++){

		//===Set Input===//
		get_dataset_record_dbl(dataset, r, input);
		set_som_input(self, input, dataset->num_features, dataset->classes[r]);

		//===Find BMU===//
		find_som_best_matching_unit(self);		

	}	

	//===Average Class===//
	fout = fopen("som_class_map.dat", "w");
	for (x=0; x<self->num_x_nodes; x++){
		for (y=0; y<self->num_y_nodes; y++){

			//===Compute Average Class===//
			self->nodes[x][y].average_class = ((double)(self->nodes[x][y].class_total));
			self->nodes[x][y].average_class /= ((double)(self->nodes[x][y].num_times_bmu));

			if (0){
				//===Set To Integer===//
				self->nodes[x][y].average_class = round(self->nodes[x][y].average_class);
				if (self->nodes[x][y].average_class >= dataset->num_classes){
					self->nodes[x][y].average_class = dataset->num_classes-1;
				}
			}

			//===Print To File===//
			fprintf(fout, "%d %d %lf\n", self->nodes[x][y].location[0], self->nodes[x][y].location[1],
										 self->nodes[x][y].average_class);
		}
	}
	fclose(fout);

	return;
}

//================================================================================================//
//======================================OPTIMIZATION METHODS======================================//
//================================================================================================//

void run_conjugate_gradient_dbl( double* A,
								 double* b,
								 double* M1,
								 unsigned int dimension,
								 double* x )
{
	//===Sanity Checks===//
	if (A == NULL){
		fprintf(stderr, "Error:: A Is NULL! In Function -- run_conjugate_gradient_dbl!\n");
		quit();
	}
	if (b == NULL){
		fprintf(stderr, "Error:: b Is NULL! In Function -- run_conjugate_gradient_dbl!\n");
		quit();
	}
	if (x == NULL){
		fprintf(stderr, "Error:: x Is NULL! In Function -- run_conjugate_gradient_dbl!\n");
		quit();
	}
	if (M1 == NULL){
		fprintf(stderr, "Error:: Inverse Preconditioner M1 Is NULL! In Function -- run_conjugate_gradient_dbl!\n");
		quit();
	}
	if (dimension == 0 || dimension > MAX_POLYNOMIAL_ORDER+1){
		fprintf(stderr, "Error:: Dimension Is Invalid! In Function -- run_conjugate_gradient_dbl!\n");
		quit();
	}

	unsigned int t, i;
	double delta, delta0, beta, nt;
	double temp[MAX_POLYNOMIAL_ORDER+1];
	double r_prev[MAX_POLYNOMIAL_ORDER+1];
	double r[MAX_POLYNOMIAL_ORDER+1];
	double d[MAX_POLYNOMIAL_ORDER+1];
	
	
	//===Form x0===//
	initialize_array_constant_dbl(x, dimension, 10e-4);

	//===Form r0===//
	matrix_vector_multiply_dbl(x, dimension, A, dimension, dimension, temp);
	subtract_vectors_dbl(b, temp, dimension, r);

	//===Form d0===//
	matrix_vector_multiply_dbl(r, dimension, M1, dimension, dimension, d);
		
	//===Form delta0===//
	delta0 = compute_inner_product_dbl(r, r, dimension);
	delta = delta0;
	
	//Note: If matrix A is NOT positive definite, this technique WILL NOT WORK

	//===Run Algorithm===//
	t = 0;
	while(t < MAX_CG_ITERATIONS && delta > MAX_CG_ERROR_TOLERANCE*delta0){

		//fprintf(stdout, "Iteration: %d\n", t);
	
		//===Compute Step Size nt===//
		nt = compute_quadratic_form_dbl(M1, dimension, dimension, r, r);
		temp[0] = compute_quadratic_form_dbl(A, dimension, dimension, d, d);
		nt /= temp[0];
			

		//===Update Solution xt+1===//
		for (i=0; i<dimension; i++){
			x[i] = x[i] + nt * d[i];
		}	

		//===Update Residual rt+1===//
		copy_array_dbl(r, dimension, r_prev);
		if ((t+1) % 50 == 0){
			matrix_vector_multiply_dbl(x, dimension, A, dimension, dimension, temp);
			subtract_vectors_dbl(b, temp, dimension, r);
		}
		else{
			matrix_vector_multiply_dbl(d, dimension, A, dimension, dimension, temp);
			for (i=0; i<dimension; i++){
				r[i] = r[i] - nt * temp[i];
			}	
		}

		//===Update Beta_t+1===//
		beta = compute_quadratic_form_dbl(M1, dimension, dimension, r, r);
		temp[0] = compute_quadratic_form_dbl(M1, dimension, dimension, r_prev, r_prev);
		beta /= temp[0];
	
		//===Update d_t+1===//
		matrix_vector_multiply_dbl(r, dimension, M1, dimension, dimension, temp);
		for (i=0; i<dimension; i++){
			d[i] = temp[i] + beta * d[i];
		}	

		//===Update delta===//
		delta = compute_inner_product_dbl(r, r, dimension);

		//===Update Iteration===//
		t = t + 1;

	}

	if (0){
		if (dimension > 10){
			fprintf(stdout, "Residual: \n");
			print_vector_dbl(r, 10, stdout);
		}
		else{
			fprintf(stdout, "Residual: \n");
			print_vector_dbl(r, dimension, stdout);
		}
	}

	return;
}


//================================================================================================//
//========================================DATASET METHODS=========================================//
//================================================================================================//

void initialize_dataset( dataset_t* self,
						 unsigned int num_features,
						 unsigned int num_attributes,
						 unsigned int num_classes,
						 data_type type )
{
	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- initialize_dataset!\n");
		quit();
	}
	if (num_features > MAX_NUM_FEATURES){
		fprintf(stderr, "Error:: num_features Is Too Large! In Function -- initialize_dataset!\n");
		quit();
	}
	if (num_attributes > MAX_NUM_ATTRIBUTES){
		fprintf(stderr, "Error:: num_attributes Is Too Large! In Function -- initialize_dataset!\n");
		quit();
	}
	if (num_classes > MAX_NUM_CLASSES){
		fprintf(stderr, "Error:: num_classes Is Too Large! In Function -- initialize_dataset!\n");
		quit();
	}

	//===Set Locals===//
	self->num_records = 0;
	self->num_features = num_features;
	self->num_attributes = num_attributes;
	self->num_classes = num_classes;
	self->type = type;

	//===Initialize===//
	for (i=0; i<MAX_NUM_RECORDS; i++){
		initialize_array_cmplx(self->record.cmplx[i], MAX_NUM_FEATURES);
		initialize_array_dbl(self->record_attributes[i], MAX_NUM_ATTRIBUTES);
		self->classes[i] = UINT_MAX;
	}
	initialize_array_uint(self->class_sizes,MAX_NUM_CLASSES);
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_ATTRIBUTES; i++){
		self->record_attribute_names[i][0] = '\0';
	}

	//===Create Initial Stats===//
	if (type == DOUBLE){
		initialize_identity_matrix_dbl(self->covariance_matrix.dbl, num_features, num_features);
		for (i=0; i<MAX_NUM_CLASSES; i++){
			initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], num_features, num_features);
		}
		initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, num_features, num_features);
		initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, num_features, num_features);	
	}
	else if (type == COMPLEX){
		initialize_identity_matrix_cmplx(self->covariance_matrix.cmplx, num_features, num_features);
		for (i=0; i<MAX_NUM_CLASSES; i++){
			initialize_identity_matrix_cmplx(self->class_covariance_matrix.cmplx[i], num_features, num_features);
		}
		initialize_identity_matrix_cmplx(self->within_class_scatter_matrix.cmplx, num_features, num_features);
		initialize_identity_matrix_cmplx(self->between_class_scatter_matrix.cmplx, num_features, num_features);	
	}
	else{
		fprintf(stderr, "Error:: Data Type Is Invalid! In Function -- initialize_dataset!\n");
		quit(); 
	}

	return;
}	

void reset_dataset( dataset_t* self )
{
	unsigned int i;

	//===Reset Locals===//
	self->num_records = 0;
	self->num_features = 0;
	self->num_attributes = 0;
	self->num_classes = 0;

	//===Reinitialize===//
	for (i=0; i<MAX_NUM_RECORDS; i++){
		initialize_array_cmplx(self->record.cmplx[i], MAX_NUM_FEATURES);
		initialize_array_dbl(self->record_attributes[i], MAX_NUM_ATTRIBUTES);
		self->classes[i] = UINT_MAX;
	}
	initialize_array_uint(self->class_sizes,MAX_NUM_CLASSES);
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	for (i=0; i<MAX_NUM_ATTRIBUTES; i++){
		self->record_attribute_names[i][0] = '\0';
	}

	//===Recreate Initial Stats===//
	if (self->type == DOUBLE){
		initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);
		for (i=0; i<MAX_NUM_CLASSES; i++){
			initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
		}
		initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);
		initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);	
	}
	else if (self->type == COMPLEX){
		initialize_identity_matrix_cmplx(self->covariance_matrix.cmplx, self->num_features, self->num_features);
		for (i=0; i<MAX_NUM_CLASSES; i++){
			initialize_identity_matrix_cmplx(self->class_covariance_matrix.cmplx[i], self->num_features, self->num_features);
		}
		initialize_identity_matrix_cmplx(self->within_class_scatter_matrix.cmplx, self->num_features, self->num_features);
		initialize_identity_matrix_cmplx(self->between_class_scatter_matrix.cmplx, self->num_features, self->num_features);	
	}
	else{
		fprintf(stderr, "Error:: Data Type Is Invalid! In Function -- reset_dataset!\n");
		quit(); 
	}
	

	return;
}

void copy_dataset_dbl( dataset_t* original,
					   dataset_t* copy )
{
	
	unsigned int i;

	//===Sanity Checks===//
	if (original == NULL){
		fprintf(stderr, "Error:: Original Is NULL! In Function -- copy_dataset_dbl!\n");
		quit();
	}
	if (copy == NULL){
		fprintf(stderr, "Error:: Copy Is NULL! In Function -- copy_dataset_dbl!\n");
		quit();
	}
	
	//===Copy Members===//
	copy->num_records = original->num_records;
	copy->num_features = original->num_features;
	copy->num_attributes = original->num_attributes;
	copy->num_classes = original->num_classes;
	copy_array_uint(original->classes, MAX_NUM_RECORDS, copy->classes);
	copy_array_uint(original->class_sizes, MAX_NUM_CLASSES, copy->class_sizes);
	for (i=0; i<MAX_NUM_ATTRIBUTES; i++){
		copy_array_char(original->record_attribute_names[i], 80, copy->record_attribute_names[i]);
	}
	copy_array_dbl(original->class_probabilities, MAX_NUM_CLASSES, copy->class_probabilities);
	for (i=0; i<MAX_NUM_RECORDS; i++){
		copy_array_dbl(original->record_attributes[i], MAX_NUM_ATTRIBUTES, copy->record_attributes[i]);	
	}
	copy->type = original->type;

	//===Copy Data===//
	for (i=0; i<MAX_NUM_RECORDS; i++){
		copy_array_dbl(original->record.dbl[i], MAX_NUM_FEATURES, copy->record.dbl[i]);
	}
	copy_array_dbl(original->covariance_matrix.dbl, MAX_NUM_FEATURES*MAX_NUM_FEATURES, copy->covariance_matrix.dbl);
	copy_array_dbl(original->record_centroid.dbl, MAX_NUM_FEATURES, copy->record_centroid.dbl);
	for (i=0; i<MAX_NUM_CLASSES; i++){
		copy_array_dbl(original->class_centroid.dbl[i], MAX_NUM_FEATURES, copy->class_centroid.dbl[i]);
	}
	copy_array_dbl(original->within_class_scatter_matrix.dbl, MAX_NUM_FEATURES*MAX_NUM_FEATURES, copy->within_class_scatter_matrix.dbl);
	copy_array_dbl(original->between_class_scatter_matrix.dbl, MAX_NUM_FEATURES*MAX_NUM_FEATURES, copy->between_class_scatter_matrix.dbl);
	return;
}

//this could be used to load in partial results before setting up online learning
void load_dataset_dbl( char* filename,
					   dataset_t* self )
{
	int item, done;
	unsigned int i, line, class;
	double real;
	double attributes[MAX_NUM_ATTRIBUTES];
	double record[MAX_NUM_FEATURES];
	FILE* fin;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- load_dataset_dbl!\n");
		quit();
	}
	if (filename == NULL){
		fprintf(stderr, "Error:: filename Is NULL! In Function -- load_dataset_dbl!\n");
		quit();
	}

	//===Open And Read===//
	fin = fopen(filename, "r");
	line = 0; done = 0;
	while (!done){ 
		
		//===Read Class===//
		item = fscanf(fin,"%d",&class);
		if (item == EOF){
			done = 1;
			break;
		}

		//===Read Attributes===//
		for (i=0; i<self->num_attributes; i++){
			item = fscanf(fin,"%lf",&real);
			if (item == EOF){
				done = 1;
				break;
			}
			attributes[i] = real;
		}

		//===Read Features===//
		for (i=0; i<self->num_features; i++){
			item = fscanf(fin,"%lf",&real);
			if (item == EOF){
				done = 1;
				break;
			}
			record[i] = real;
		}

		//===Add Recrod===//
		add_dataset_record_dbl(self, record, attributes, &class);

		//===Increment Line===//
		line++;
	}

	//===Clean Up===//
	fclose(fin);

	return;
}

void get_dataset_record_indices_by_attribute_dbl( dataset_t* self,
									     		  unsigned int attribute_index,
							   				   	  double attribute_level,
											      unsigned int* record_indices,
											      unsigned int* num_records )
{
	unsigned int i, record_count;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 get_dataset_record_indices_by_attribute_dbl!\n");
		quit();
	}
	if (attribute_index > self->num_attributes || attribute_index > MAX_NUM_ATTRIBUTES){
		fprintf(stderr, "Error:: attribute_index Is Invalid! In Function --  \
						 get_dataset_record_indices_by_attribute_dbl!\n");
		quit();
	}
	if (record_indices == NULL){
		fprintf(stderr, "Error:: record_indices Is NULL! In Function --\
						 get_dataset_record_indices_by_attribute_dbl!\n");
		quit();
	}
	if (num_records == NULL){
		fprintf(stderr, "Error:: num_records Is NULL! In Function --  \
						 get_dataset_record_indices_by_attribute_dbl!\n");
		quit();
	}

	//===Find Records===//
	record_count = 0;
	for (i=0; i<self->num_records; i++){
		//===Copy Index===//		
		if (fabs(self->record_attributes[i][attribute_index] - attribute_level) < 10.0*EPS){
			record_indices[record_count] = i;
			record_count++;
		}
	}

	//===Set Count===//
	*num_records = record_count;

	return;
}

//subset_dataset_by_attribute
//so needs to create a new dataset with attributes[SNR, Poop_color] ie [0, 1]
//then num_attribute_levels[3, 2]
//attribute_levels[0, -3, 6, Brown, Green]

//so first, find all records with SNR = 0, copy them over to new dataset
// SNR = -3, copy them over
// SNR = 6, copy them over
// Color = Brown, copy them over
// Color = Green, copy them over

//what's the point?
//for equalization:
/*
	I can subset based on -3, find num records
	I can subset based on 0, find num records
		then find smallest number of records

	maybe I need:
		find_smallest_attribute_level //returns what the attribute is, and what is it's num_records
		find_attribute_level_sizes ?? 


why do i need to equalize?
	-- because i want to be able to run SVMs in C
		-- it needs to take a dataset, and train an SVM
		-- in order for the SVM to be most effective, it needs to have equalized classes and attributes
	-- because i want to be able to visualize class distributions
		-- in order for the visualization to be most effective, classes need to be balanced

why do i want to run SVMs in C?
	-- this way, I can hook it up with the log filter
		-- i can gather data, add it to the dataset, retrain the SVM, and apply the retrained SVM

	balance_dataset
		-- for each attribute
			-- find smallest level
			-- resample each other level to be of the same size

*/

void subset_dataset_by_feature_dbl( dataset_t* self,
									unsigned int* feature_indices,
									dataset_t* subset )
{
	unsigned int i, j, k, num_features;
	double record[MAX_NUM_FEATURES];

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 subset_dataset_by_feature_dbl!\n");
		quit();
	}
	if (feature_indices == NULL){
		fprintf(stderr, "Error:: attribute_indices Is NULL! In Function --  \
						 subset_dataset_by_feature_dbl!\n");
		quit();
	}
	if (subset == NULL){
		fprintf(stderr, "Error:: subset Is NULL! In Function --  \
						 subset_dataset_by_feature_dbl!\n");
		quit();
	}

	//===Count Features===//
	num_features = 0;
	for (i=0; i<self->num_features; i++){
		if (feature_indices[i] != 0){
			num_features++;
		}
	}

	//===Init Subset===//
	initialize_dataset(subset, num_features, self->num_attributes, self->num_classes, DOUBLE);

	//===Sweep Through Records===//
	for (i=0; i<self->num_records; i++){

		//===Copy Over Features===//
		k = 0;
		for (j=0; j<num_features; j++){
			if (feature_indices[j] != 0){
				record[k++] = self->record.dbl[i][j];
			}
		}

		//===Add Record===//
		add_dataset_record_dbl(subset, record, self->record_attributes[i], &(self->classes[i]));
	}

	return;
}	

void subset_dataset_by_attribute_dbl( dataset_t* self,
								  	  unsigned int attribute_index,
								  	  double* attribute_levels,
								  	  unsigned int num_attribute_levels,
								  	  dataset_t* subset )
{
	unsigned int i,j;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 subset_dataset_by_attribute_dbl!\n");
		quit();
	}
	if (attribute_index >= self->num_attributes || attribute_index >= MAX_NUM_ATTRIBUTES){
		fprintf(stderr, "Error:: attribute_index Is Invalid! In Function --  \
						 subset_dataset_by_attribute_dbl!\n");
		quit();
	}
	if (attribute_levels == NULL){
		fprintf(stderr, "Error:: attribute_levels Is NULL! In Function --  \
						 subset_dataset_by_attribute_dbl!\n");
		quit();
	}
	if (num_attribute_levels == 0 || num_attribute_levels > MAX_NUM_RECORDS){
		fprintf(stderr, "Error:: num_attribute_levels Is Invalid! In Function --  \
						 subset_dataset_by_attribute_dbl!\n");
		quit();
	}
	if (subset == NULL){
		fprintf(stderr, "Error:: subset Is NULL! In Function --  \
						 subset_dataset_by_attribute_dbl!\n");
		quit();
	}

	//===Init Subset===//
	initialize_dataset(subset, self->num_features, self->num_attributes, self->num_classes, DOUBLE);

	//===Sweep Through Records===//
	for (i=0; i<self->num_records; i++){
		//===Check Record's Attribute===//
		for (j=0; j<num_attribute_levels; j++){
			if (are_equal_dbl(self->record_attributes[i][attribute_index], attribute_levels[j])){
				add_dataset_record_dbl(subset, self->record.dbl[i], 
									   self->record_attributes[i], &(self->classes[i]));
				break;
			}
		}
	}
	
	return;
}

void get_dataset_indices_by_class( dataset_t* self,
								   unsigned int class_level,
								   unsigned int* indices,
								   unsigned int* num_indices )
{
	unsigned int i, num_idx;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 get_dataset_indices_by_class!\n");
		quit();
	}
	if (indices == NULL){
		fprintf(stderr, "Error:: indices Is NULL! In Function --  \
						 get_dataset_indices_by_class!\n");
		quit();
	}
	if (num_indices == NULL){
		fprintf(stderr, "Error:: num_indices Is NULL! In Function --  \
						 get_dataset_indices_by_class!\n");
		quit();
	}

	num_idx = 0;		
	//===Sweep Through Records===//
	for (i=0; i<self->num_records; i++){
		if (self->classes[i] == class_level){
			indices[num_idx++] = i;
		}
	}
	*num_indices = num_idx;

	return;
}

void get_dataset_indices_by_attribute_dbl( dataset_t* self,
								  	   	   unsigned int attribute_index,
								  	   	   double* attribute_levels,
								  	   	   unsigned int num_attribute_levels,
								  	   	   unsigned int* indices,
									   	   unsigned int* num_indices )
{
	unsigned int i,j,num_idx;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 get_dataset_indices_by_attribute_dbl!\n");
		quit();
	}
	if (attribute_index >= self->num_attributes || attribute_index >= MAX_NUM_ATTRIBUTES){
		fprintf(stderr, "Error:: attribute_index Is Invalid! In Function --  \
						 get_dataset_indices_by_attribute_dbl!\n");
		quit();
	}
	if (attribute_levels == NULL){
		fprintf(stderr, "Error:: attribute_levels Is NULL! In Function --  \
						 get_dataset_indices_by_attribute_dbl!\n");
		quit();
	}
	if (num_attribute_levels == 0 || num_attribute_levels > MAX_NUM_RECORDS){
		fprintf(stderr, "Error:: num_attribute_levels Is Invalid! In Function --  \
						 get_dataset_indices_by_attribute_dbl!\n");
		quit();
	}
	if (indices == NULL){
		fprintf(stderr, "Error:: indices Is NULL! In Function --  \
						 get_dataset_indices_by_attribute_dbl!\n");
		quit();
	}
	if (num_indices == NULL){
		fprintf(stderr, "Error:: num_indices Is NULL! In Function --  \
						 get_dataset_indices_by_attribute_dbl!\n");
		quit();
	}

	num_idx = 0;		
	//===Sweep Through Records===//
	for (i=0; i<self->num_records; i++){
		//===Check Record's Attribute===//
		for (j=0; j<num_attribute_levels; j++){
			if (are_equal_dbl(self->record_attributes[i][attribute_index], attribute_levels[j])){
				indices[num_idx++] = i;
				break;
			}
		}
	}
	*num_indices = num_idx;
	return;
}
										

void subset_dataset_by_class_dbl( dataset_t* self,
								  unsigned int class,
							  	  dataset_t* subset )
{
	unsigned int i;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 subset_dataset_by_class_dbl!\n");
		quit();
	}
	if (class > self->num_classes || class >= MAX_NUM_CLASSES){
		fprintf(stderr, "Error:: class Is Invalid! In Function --  \
						 subset_dataset_by_class_dbl!\n");
		quit();
	}
	if (subset == NULL){
		fprintf(stderr, "Error:: subset Is NULL! In Function --  \
						 subset_dataset_by_class_dbl!\n");
		quit();
	}

	//===Init Subset===//
	initialize_dataset(subset, self->num_features, self->num_attributes, 1, DOUBLE);

	//===Sweep Through Records===//
	for (i=0; i<self->num_records; i++){
		if (self->classes[i] == class){
			add_dataset_record_dbl(subset,self->record.dbl[i],self->record_attributes[i],&(class));
		}
	}

	return;
}

void get_dataset_unique_classes( dataset_t* self,
								 unsigned int* unique_classes )
{
	find_unique_uint(self->classes, self->num_records, unique_classes, &(self->num_classes));
	return;
}

void make_dataset_test_indices( dataset_t* self,
						 		unsigned int num_folds,
						 		unsigned int* test_indices,
						 		random_number_generator_t* rng )
{

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 make_dataset_test_indices!\n");
		quit();
	}
	if (!isnormal((double)num_folds)){
		fprintf(stderr, "Error:: num_folds Is Invalid! In Function --  \
						 make_dataset_test_indices!\n");
		quit();
	}
	if (test_indices == NULL){
		fprintf(stderr, "Error:: test_indices Is NULL! In Function --  \
						 make_dataset_test_indices!\n");
		quit();
	}
	if (rng == NULL){
		fprintf(stderr, "Error:: rng Is NULL! In Function --  \
						 make_dataset_test_indices!\n");
		quit();
	}

	unsigned int i, j, length, num_indices, good_sample;
	unsigned int idx, all_indices_length, fold_length, sample;
	unsigned int all_indices[MAX_NUM_RECORDS];
	
	//===Initialize With Unattainable Sample Number===//
	initialize_range_uint(all_indices, 0, self->num_records);
	all_indices_length = self->num_records;

	//===Make Folds===//
	fold_length = self->num_records/num_folds;
 	length = 0;
	for (i=0; i<num_folds; i++){

		//===Make Last Fold===//		
		if (i == num_folds-1){
			//===Copy Over===//
			copy_array_uint(all_indices, all_indices_length, test_indices + length);
		}
		else{
			//===Make Current Fold===//
			for (j=0; j<fold_length; j++){

				good_sample = 0;

				while(good_sample == 0){
					//===Generate Sample===//
					sample = get_random_uniform(0, self->num_records-1, rng);

					//===Check Sample===//
					num_indices = 0;
					find_indices_where_equal_uint(all_indices, all_indices_length,
									   			  sample, &num_indices, &idx);
					if (num_indices == 0){
						good_sample = 0;
					}
					else{
						remove_from_array_uint(all_indices, &all_indices_length, idx);
						good_sample = 1;
					}
				}

				//===Copy Over Sample===//
				*(test_indices + length) = sample;
				length++;

			}
		}
	}

	return;
}

void find_attribute_levels_dbl( dataset_t* self,
								unsigned int attribute_index,
								double* attribute_levels,
								unsigned int* num_levels )
{
	unsigned int i, attribute_count;
	double attributes[MAX_NUM_RECORDS];

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function --  \
						 find_attribute_levels_dbl!\n");
		quit();
	}
	if (attribute_levels == NULL){
		fprintf(stderr, "Error:: attribute_levels Is NULL! In Function --  \
						 find_attribute_levels_dbl!\n");
		quit();
	}
	if (num_levels == NULL){
		fprintf(stderr, "Error:: num_levels Is NULL! In Function --  \
						 find_attribute_levels_dbl!\n");
		quit();
	}

	//===Collect Attributes===//
	attribute_count = 0;
	for (i=0; i<self->num_records; i++){
		attributes[attribute_count++] = self->record_attributes[i][attribute_index];
	}

	//===Collect Unique Values===//
	find_unique_dbl(attributes, attribute_count, attribute_levels, num_levels);

	return;
}

void balance_dataset_dbl( dataset_t* self )
{

	unsigned int k, m, num_attribute_levels;
	unsigned int i, j, idx, num_class_levels, num_records, min_num_records;
	unsigned int class_levels[MAX_NUM_RECORDS];
	unsigned int resampled_record_indices[MAX_NUM_RECORDS];
	unsigned int class_record_indices[MAX_NUM_RECORDS];
	double attribute_levels[MAX_NUM_RECORDS];
	dataset_t *subset, *subsubset;
	random_number_generator_t* rng;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- balance_dataset_dbl!\n");
		quit();
	}

	//===Mallocs===//
	subset = malloc(sizeof(dataset_t));
	subsubset = malloc(sizeof(dataset_t));
	rng = malloc(sizeof(random_number_generator_t));

	//===Extract All Classes===//
	find_dataset_class_levels_dbl(self, class_levels, &num_class_levels);

	//===Find Min Num Class Records Across Attributes===//
	min_num_records = UINT_MAX;
	for (k=0; k<self->num_attributes; k++){

		//===Get Attribute Levels===//
		find_attribute_levels_dbl(self, k, attribute_levels, &num_attribute_levels);

		//===Subset Based On Current Factor Level===//
		for (m=0; m<num_attribute_levels; m++){

			//===Init Subset===//
			initialize_dataset(subset, self->num_features, 1, self->num_classes, DOUBLE);
			subset_dataset_by_attribute_dbl(self, 0, attribute_levels + m, 1, subset);

			//===Find Smallest Class Per Attribute===//
			for (i=0; i<num_class_levels; i++){
				get_dataset_indices_by_class(subset, class_levels[i], class_record_indices, &num_records);
				if (num_records < min_num_records){
					min_num_records = num_records;
				}
			}
		}
	}

	//===Threshold===//
	min_num_records = MIN(min_num_records, MAX_NUM_BALANCED_RECORDS);

	//===Make Sure There Are Records To Balance===//
	if (min_num_records == 0){
		fprintf(stderr, "Error:: Dataset Is Missing Attribute Data Or Class Data! \
						 In Function -- balance_dataset_dbl!\n");
		quit();
	}

	//===Copy All Back To Subset===//
	initialize_random_number_generator(rng);
	initialize_dataset(subsubset, self->num_features, self->num_attributes, self->num_classes, DOUBLE);
	for (k=0; k<self->num_attributes; k++){
	
		//===Find Current Attribute Levels===//
		find_attribute_levels_dbl(self, k, attribute_levels, &num_attribute_levels);

		//===Subset Based On Attribute Level===//
		for (m=0; m<num_attribute_levels; m++){

			//===Init Subset===//
			initialize_dataset(subset, self->num_features, 1, self->num_classes, DOUBLE);
			subset_dataset_by_attribute_dbl(self, k, attribute_levels + m, 1, subset);

			//===Resample Smallest Class===//
			for (i=0; i<num_class_levels; i++){

				//===Get Record Indices Of Current Class Level===//
				get_dataset_indices_by_class(subset, class_levels[i], class_record_indices, &num_records);

				//===Resample Indices Based On Min Record Number===//
				randomly_sample_array_uint(class_record_indices, num_records, min_num_records,
								 		   resampled_record_indices, rng);

				//===Copy Over These Indices To The SubSubset===//
				for (j=0; j<min_num_records; j++){
					idx = resampled_record_indices[j];
					add_dataset_record_dbl(subsubset,subset->record.dbl[idx],subset->record_attributes[idx],
										  &(subset->classes[idx]));
				}

			}

		}
	}			

	//===Copy Over Newly Equalized===//
	initialize_dataset(self, self->num_features, self->num_attributes, self->num_classes, DOUBLE);
	copy_dataset_dbl(subsubset, self);

	fprintf(stdout, "Min Records: %d\n", min_num_records);
	fprintf(stdout, "Num Records: %d\n", self->num_records);

	
	//===Clean Up===//
	free(rng);
	free(subsubset);
	free(subset);

	return;
}

void balance_dataset_classes_dbl( dataset_t* self )
{
	unsigned int i, j, idx, num_class_levels, num_records, min_num_records;
	unsigned int class_levels[MAX_NUM_RECORDS];
	unsigned int resampled_record_indices[MAX_NUM_RECORDS];
	unsigned int class_record_indices[MAX_NUM_RECORDS];
	dataset_t *subset;
	random_number_generator_t* rng;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- balance_dataset_classes_dbl!\n");
		quit();
	}

	//===Mallocs===//
	subset = malloc(sizeof(dataset_t));
	rng = malloc(sizeof(random_number_generator_t));

	//===Extract All Classes===//
	find_dataset_class_levels_dbl(self, class_levels, &num_class_levels);

	//===Find Min Num Records===//
	min_num_records = UINT_MAX;
	for (i=0; i<num_class_levels; i++){
		get_dataset_indices_by_class(self, class_levels[i], class_record_indices, &num_records);
		if (num_records < min_num_records){
			min_num_records = num_records;
		}
	}

	//===Init Subset===//
	initialize_dataset(subset, self->num_features, self->num_attributes, self->num_classes, DOUBLE);

	//===Copy All Back To Subset===//
	initialize_random_number_generator(rng);
	for (i=0; i<num_class_levels; i++){

		//===Get Record Indices Of Current Class Level===//
		get_dataset_indices_by_class(self, class_levels[i], class_record_indices, &num_records);

		//===Resample Indices Based On Min Record Number===//
		randomly_sample_array_uint(class_record_indices, num_records, min_num_records,
						 		   resampled_record_indices, rng);

		//===Copy Over These Indices To The Subset===//
		for (j=0; j<min_num_records; j++){
			idx = resampled_record_indices[j];
			add_dataset_record_dbl(subset,self->record.dbl[idx],self->record_attributes[idx],
								  &(self->classes[idx]));
		}
	}

	//===Copy Over Newly Equalized===//
	initialize_dataset(self, self->num_features, self->num_attributes, self->num_classes, DOUBLE);
	copy_dataset_dbl(subset, self);

	//===Clean Up===//
	free(rng);
	free(subset);

	return;
}

void balance_dataset_attribute_dbl( dataset_t* self,
									unsigned int attribute_index )
{

	unsigned int i, j, idx, num_attribute_levels;
	unsigned int min_num_records, num_records;
	unsigned int resampled_record_indices[MAX_NUM_RECORDS];
	unsigned int attribute_record_indices[MAX_NUM_RECORDS];
	double attribute_levels[MAX_NUM_RECORDS];
	dataset_t *subset;
	random_number_generator_t* rng;

	//===Sanity Check===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- balance_dataset_attribute_dbl!\n");
		quit();
	}
	if (attribute_index >= self->num_attributes || attribute_index >= MAX_NUM_ATTRIBUTES){
		fprintf(stderr, "Error:: attribute_index Is Invalid! In Function --  \
						 balance_dataset_attribute_dbl!\n");
		quit();
	}

	//===Mallocs===//
	subset = malloc(sizeof(dataset_t));
	rng = malloc(sizeof(random_number_generator_t));
		
	//===Extract All Attribute Levels===//
	find_attribute_levels_dbl(self, attribute_index, attribute_levels, &num_attribute_levels);

	//===Find Smallest Subset===//
	min_num_records = UINT_MAX;
	for (i=0; i<num_attribute_levels; i++){
		get_dataset_indices_by_attribute_dbl(self, attribute_index, attribute_levels + i, 1, 
											 attribute_record_indices, &num_records);
		if (num_records < min_num_records){
			min_num_records = num_records;
		}
	}

	//===Init Subset===//
	initialize_dataset(subset, self->num_features, self->num_attributes, self->num_classes, DOUBLE);

	//===Copy All Back To Subset===//
	initialize_random_number_generator(rng);
	for (i=0; i<num_attribute_levels; i++){

		//===Get Record Indices Of Current Attribute Level===//
		get_dataset_indices_by_attribute_dbl(self, attribute_index, attribute_levels + i, 1, 
											 attribute_record_indices, &num_records);

		//===Resample Indices Based On Min Record Number===//
		randomly_sample_array_uint(attribute_record_indices, num_records, min_num_records,
						 		   resampled_record_indices, rng);

		//===Copy Over These Indices To The Subset===//
		for (j=0; j<min_num_records; j++){
			idx = resampled_record_indices[j];
			add_dataset_record_dbl(subset,self->record.dbl[idx],self->record_attributes[idx],
								  &(self->classes[idx]));
		}
	}	
	
	//===Copy Over Newly Equalized===//
	initialize_dataset(self, self->num_features, self->num_attributes, self->num_classes, DOUBLE);
	copy_dataset_dbl(subset, self);
	
	//===Clean Up===//
	free(rng);
	free(subset);

	return;
}

void get_dataset_record_dbl( dataset_t* self,
							 unsigned int index,
							 double* record )
{
	copy_array_dbl(self->record.dbl[index], self->num_features, record); 
	return;
}

void remove_dataset_record_dbl( dataset_t* self,
								unsigned int index )
{
	unsigned int i;
	//===Sanity Checks===//
	if (index >= self->num_records){
		fprintf(stderr, "Error:: index Is Too Large! In Function -- remove_dataset_record_dbl!\n");
		quit();	
	}

	//===Remove Record===//
	initialize_array_dbl(self->record.dbl[index], MAX_NUM_FEATURES);
	for (i=index+1; i<self->num_records; i++){
		copy_array_dbl(self->record.dbl[i], self->num_features, self->record.dbl[i-1]); 
	}
	initialize_array_dbl(self->record.dbl[self->num_records-1], MAX_NUM_FEATURES);
	//===Remove Class===//
	for (i=index+1; i<self->num_records; i++){
		self->classes[i-1] = self->classes[i];
	}	
	self->classes[self->num_records - 1] = UINT_MAX;
	//===Remove Attributes===//
	if (self->num_attributes != 0){
		for (i=index+1; i<self->num_records; i++){
			copy_array_dbl(self->record_attributes[i], self->num_attributes, self->record_attributes[i-1]); 
		}	
		initialize_array_dbl(self->record_attributes[self->num_records-1], MAX_NUM_ATTRIBUTES);
	}
	//===Decrement Stats===//
	self->num_records -= 1;
	self->class_sizes[self->classes[index]] -= 1;
	
	return;
}

void add_dataset_record_dbl( dataset_t* self,
							 double* record,
							 double* attributes,
							 unsigned int* class )
{

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- add_dataset_record_dbl!\n");
		quit();
	}
	if (record == NULL){
		fprintf(stderr, "Error:: Record Is NULL! In Function -- add_dataset_record_dbl!\n");
		quit();
	}
	if (class == NULL){
		fprintf(stderr, "Error:: Class Is NULL! In Function -- add_dataset_record_dbl!\n");
		quit();
	}
	if (self->num_records >= MAX_NUM_RECORDS){
		fprintf(stderr, "Error:: Dataset Is full! In Function -- add_dataset_record_dbl!\n");
		quit();
	}

	//===Add To Correct Place===//
	copy_array_dbl(record, self->num_features, self->record.dbl[self->num_records]); 
	if (attributes != NULL && self->num_attributes != 0){
		copy_array_dbl(attributes, self->num_attributes, self->record_attributes[self->num_records]); 
	}
	self->classes[self->num_records++] = *class;
	self->class_sizes[*class] += 1;	
	
	return;
}

void copy_dataset_feature_dbl( dataset_t* self,
							   unsigned int feature_index,
							   double* feature_records )
{
	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- copy_dataset_feature_dbl!\n");
		quit();
	}
	if (feature_index >= self->num_features){
		fprintf(stderr, "Error:: feature_index Is Too Large! In Function -- copy_dataset_feature_dbl!\n");
		quit();	
	}
	if (feature_records == NULL){
		fprintf(stderr, "Error:: feature_records Is NULL! In Function -- copy_dataset_feature_dbl!\n");
		quit();	
	}

	//===Copy===//
	for (i=0; i<self->num_records; i++){
		feature_records[i] = self->record.dbl[i][feature_index];
	}

	return;
}

void set_dataset_feature_dbl( dataset_t* self,
							  double* feature_records,
							  unsigned int feature_index )
{

	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- set_dataset_feature_dbl!\n");
		quit();
	}
	if (feature_index >= self->num_features){
		fprintf(stderr, "Error:: feature_index Is Too Large! In Function -- set_dataset_feature_dbl!\n");
		quit();	
	}
	if (feature_records == NULL){
		fprintf(stderr, "Error:: feature_records Is NULL! In Function -- set_dataset_feature_dbl!\n");
		quit();	
	}

	//===Copy Over===//
	for (i=0; i<self->num_records; i++){
		self->record.dbl[i][feature_index] = feature_records[i];
	}

	return;
}

void remove_dataset_feature_dbl( dataset_t* self,
								 unsigned int feature_index )
{
	unsigned int i;
	double feature_records[MAX_NUM_RECORDS];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- remove_dataset_feature_dbl!\n");
		quit();
	}
	if (feature_index >= self->num_features){
		fprintf(stderr, "Error:: feature_index Is Too Large! In Function -- remove_dataset_feature_dbl!\n");
		quit();	
	}

	//===Zero Out Record===//
	for (i=0; i<self->num_records; i++){
		self->record.dbl[i][feature_index] = 0;
	}

	//===Copy Features Over===//
	for (i=feature_index+1; i<self->num_features; i++){
		//===Copy Next Feature===//
		copy_dataset_feature_dbl(self,i,feature_records);
		//===Set Next Feature Back===//
		set_dataset_feature_dbl(self, feature_records, i-1);
	}	

	//===Zero Out Last Feature===//
	for (i=0; i<self->num_records; i++){
		self->record.dbl[i][self->num_features-1] = 0;
	}

	//===Set Num Features===//
	self->num_features -= 1;

	return;
}


void set_dataset_classes_dbl( dataset_t* self,
							  unsigned int* classes )
{
	unsigned int i;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- set_dataset_classes_dbl!\n");
		quit();
	}
	if (classes == NULL){
		fprintf(stderr, "Error:: feature_records Is NULL! In Function -- set_dataset_classes_dbl!\n");
		quit();	
	}

	//===Copy Over===//
	for (i=0; i<self->num_records; i++){
		self->classes[i] = classes[i];
	}
	
	return;
}	

void find_dataset_class_levels_dbl( dataset_t* self,
							   		unsigned int* unique_classes,
							   		unsigned int* num_classes )
{
	find_unique_uint(self->classes, self->num_records, unique_classes, num_classes);
	return;
}

void print_dataset_records_dbl( dataset_t* self,
								FILE* file )
{
	unsigned int i, j;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- print_dataset_records_dbl!\n");
		quit();
	}
	if (file == NULL){
		fprintf(stderr, "Error:: File Is NULL! In Function -- print_dataset_records_dbl!\n");
		quit();
	}

	for (i=0; i<self->num_records; i++){
		//===Print Class===//
		fprintf(file, "%d ", self->classes[i]);
		//===Print Attributes===//
		for (j=0; j<self->num_attributes; j++){
			fprintf(file, "%+lf ", self->record_attributes[i][j]);
		}
		//===Print Record===//
		for (j=0; j<self->num_features; j++){
			fprintf(file, "%+16.16e ", self->record.dbl[i][j]);
		}
		fprintf(file, "\n");
	}
	return;
}

void compute_dataset_feature_means_dbl( dataset_t* self,
										double* feature_means )
{

	unsigned int i, j;
	double mean;
	for (i=0; i<self->num_features; i++){
		mean = 0;
		for (j=0; j<self->num_records; j++){
			mean += self->record.dbl[j][i];
		}
		mean /= ((double)(self->num_records));
		feature_means[i] = mean;
	}
	return;
}

void compute_dataset_feature_variances_dbl( dataset_t* self,
										    double* feature_variances )
{

	unsigned int i, j;
	double variance;
	double feature_means[MAX_NUM_FEATURES];

	//===Compute Means===//
	compute_dataset_feature_means_dbl(self, feature_means);

	//===Compute Variances===//
	for (i=0; i<self->num_features; i++){
		variance = 0;
		for (j=0; j<self->num_records; j++){
			variance += pow((self->record.dbl[j][i] - feature_means[i]),2.0);
		}
		variance /= (((double)(self->num_records)) - 1.0);
		feature_variances[i] = variance;
	}

	return;
}


void compute_dataset_records_zscores_dbl( dataset_t* self )
{
	unsigned int i, j;
	double feature_means[MAX_NUM_FEATURES];
	double feature_variances[MAX_NUM_FEATURES];

	//===Compute Feature Means===//
	compute_dataset_feature_means_dbl(self, feature_means);

	//===Compute Feature Variances===//
	compute_dataset_feature_variances_dbl(self,feature_variances);

	//===Center===//
	for (i=0; i<self->num_features; i++){
		for (j=0; j<self->num_records; j++){
			self->record.dbl[j][i] = (self->record.dbl[i][j] - feature_means[i])/sqrt(feature_variances[i]);			
		}
	}
	
	return;
}


void compress_dataset_records_dbl( dataset_t* self )
{
	unsigned int i, j;
	double min_feature, max_feature;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compress_dataset_records_dbl!\n");
		quit();
	}

	//===Run The Compression==//
	for (i=0; i<self->num_features; i++){

		//===Find Max And Min Of Feature===//
		min_feature = DBL_MAX; max_feature = -DBL_MAX;
		for (j=0; j<self->num_records; j++){
			if (self->record.dbl[j][i] > max_feature){
				max_feature = self->record.dbl[j][i];
			}
			if (self->record.dbl[j][i] < min_feature){
				min_feature = self->record.dbl[j][i];
			}
		}

		//===Compress to [0,1]===//
		for (j=0; j<self->num_records; j++){
			self->record.dbl[j][i] = (self->record.dbl[j][i] - min_feature)/(max_feature - min_feature + 2.0*EPS);
		}
	}
	return;
}

void compute_dataset_class_sizes( dataset_t* self )
{
	unsigned int i, j, n_class;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_class_sizes!\n");
		quit();
	}
	//===Compute Feature Numbers Per Class===//
	for (i=0; i<self->num_classes; i++){
		n_class = 0;
		for (j=0; j<self->num_records; j++){
			if (self->classes[j] == i){
				n_class++;
			}
		}
		self->class_sizes[i] = n_class;
	}
	return;
}	

void compute_dataset_class_probabilities( dataset_t* self )
{

	unsigned int i, j;
	double n_class;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_class_probabilities!\n");
		quit();
	}
	//===Compute Feature Numbers Per Class===//
	for (i=0; i<self->num_classes; i++){
		n_class = 0;
		for (j=0; j<self->num_records; j++){
			if (self->classes[j] == i){
				n_class += 1;
			}
		}
		self->class_probabilities[i] = n_class/((double)(self->num_records)) + EPS;
	}
	return;
}


void compute_dataset_record_centroid_dbl( dataset_t* self )
{
	unsigned int j;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_record_centroid_dbl!\n");
		quit();
	}

	//===Initialize Dataset Centroids===//
	initialize_array_dbl(self->record_centroid.dbl, MAX_NUM_FEATURES);

	//===Update Class Centroids===//
	for (j=0; j<self->num_records; j++){
		add_vectors_dbl(self->record_centroid.dbl, self->record.dbl[j], self->num_features, self->record_centroid.dbl);
	}
	gain_array_constant_dbl(self->record_centroid.dbl, self->num_features, 1.0/((double)(self->num_records)));
	
	return;
}


void compute_dataset_class_centroids_dbl( dataset_t* self )
{

	unsigned int i, j;
	double num_records;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_class_centroids_dbl!\n");
		quit();
	}

	//===Initialize Class Centroids===//
	for (i=0; i<self->num_classes; i++){
		initialize_array_dbl(self->class_centroid.dbl[i], MAX_NUM_FEATURES);
	}

	//===Update Class Centroids===//
	for (i=0; i<self->num_classes; i++){
		num_records = 0;
		for (j=0; j<self->num_records; j++){
			if (self->classes[j] == i){
				add_vectors_dbl(self->class_centroid.dbl[i], self->record.dbl[j], self->num_features, self->class_centroid.dbl[i]);
				num_records += 1;
			}
		}		
		if (num_records > 0){
			gain_array_constant_dbl(self->class_centroid.dbl[i], self->num_features, 1.0/num_records);
		}
	}
	return;
}

void compute_dataset_covariance_dbl( dataset_t* self )
{

	unsigned int j;
	double temp;
	double record[MAX_NUM_FEATURES];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_covariance_dbl!\n");
		quit();
	}

	//===Compute Mean===//
	compute_dataset_record_centroid_dbl(self);

	//===Initialize===//
	initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);

	//===Iteratively Update===//
	for (j=0; j<self->num_records; j++){
		//===Remove Mean===//
		subtract_vectors_dbl(self->record.dbl[j], self->record_centroid.dbl, self->num_features, record);
		//===Update Scatter With Rank1 Update===//
		rank1_matrix_update_dbl(self->covariance_matrix.dbl, 1.0, record, self->num_features, 1.0);
	}
	temp = 1.0/((double)((((int)(self->num_records)) - 1)));
	gain_array_constant_dbl(self->covariance_matrix.dbl, self->num_features*self->num_features, temp);

	return;
}

void compute_dataset_class_covariances_dbl( dataset_t* self )
{
	unsigned int i, j;
	double n_class;
	double record[MAX_NUM_FEATURES];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_class_covariances_dbl!\n");
		quit();
	}

	//===Compute Class Covariances===//
	compute_dataset_class_centroids_dbl(self);

	//===Update Covariance Matrices===//
	for (i=0; i<self->num_classes; i++){
		//===Initialize Covariance Matrix===//
		n_class = 0;
		initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
		for (j=0; j<self->num_records; j++){
			if (self->classes[j] == i){
				//===Remove Mean===//
				subtract_vectors_dbl(self->record.dbl[j], self->class_centroid.dbl[i], self->num_features, record);
				//===Update Scatter With Rank1 Update===//
				rank1_matrix_update_dbl(self->class_covariance_matrix.dbl[i], 1.0, record, self->num_features, 1.0);
				n_class += 1;
			}
		}
		//===Normalize===//
		if (n_class > 0){
			gain_array_constant_dbl(self->class_covariance_matrix.dbl[i], self->num_features*self->num_features, 1.0/(n_class-1.0));
		}
	}

	return;
}	

void compute_dataset_within_class_scatters_dbl( dataset_t* self )
{

	unsigned int i;
	double mean;

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_dataset_within_class_scatter_dbl!\n");
		quit();
	}

	//===Compute Class Covariances===//
	compute_dataset_class_centroids_dbl(self);
	compute_dataset_class_covariances_dbl(self);

	//===Initialize Scatter Matrix===//
	initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);

	//===Update Within Class Scatter===//
	for (i=0; i<self->num_classes; i++){
		//===Add In Class Covariance===//
		add_matrices_dbl(self->within_class_scatter_matrix.dbl, self->class_covariance_matrix.dbl[i],
					   	 self->num_features, self->num_features, self->within_class_scatter_matrix.dbl);
	}
	//===Compute Mean Number Of Points Per Class===//
	mean = ((double)compute_mean_uint(self->class_sizes, self->num_classes));

	//===Gain Within Class Scatter===//
	gain_array_constant_dbl(self->within_class_scatter_matrix.dbl, self->num_features*self->num_features, mean);
	
	return;
}


void compute_between_class_scatters_dbl( dataset_t* self )
{
	unsigned int i;
	double mean;
	double class_means[MAX_NUM_FEATURES];
	double temp[MAX_NUM_FEATURES];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- compute_between_class_scatters_dbl!\n");
		quit();
	}

	//(mean of classes - mean of all record)(mean of classes - mean of all record)^T
	//(mean of classes - mean of all class means)(mean of classes - mean of all class means)^T

	//===Compute Class Means===//
	compute_dataset_class_centroids_dbl(self);

	//===Compute Mean Of All Class Means===//
	initialize_array_dbl(class_means, MAX_NUM_FEATURES);
	for (i=0; i<self->num_classes; i++){
		add_vectors_dbl(class_means, self->class_centroid.dbl[i], self->num_features, class_means);
	}
	gain_array_constant_dbl(class_means, self->num_features, 1.0/((double)(self->num_classes)));

	//===Initialize Scatter Matrix===//
	initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);
	
	//===Update Between Class Scatter===//
	for (i=0; i<self->num_classes; i++){
		//===Subtract Mean Of All Classes From Current Class Mean===//		
		subtract_vectors_dbl(self->class_centroid.dbl[i], class_means, self->num_features, temp);
		//===Update Scatter With Rank1 Update===//
		rank1_matrix_update_dbl(self->between_class_scatter_matrix.dbl, 1.0, temp, self->num_features, 1.0);
	}
	mean = ((double)compute_mean_uint(self->class_sizes, self->num_classes));

	//===Gain Between Class Scatter===//
	gain_array_constant_dbl(self->between_class_scatter_matrix.dbl, self->num_features*self->num_features, mean);
	
	return;
}

void generate_random_dataset_dbl( unsigned int num_classes,
								  unsigned int num_records,
								  unsigned int num_features,
								  dataset_t* dataset,
								  random_number_generator_t* rng )
{

	unsigned int c, r, num_records_per_class;
	double record[MAX_NUM_FEATURES];
	double class_means[MAX_NUM_CLASSES];
	double class_variances[MAX_NUM_CLASSES];

	//===Sanity Check===//
	if (dataset == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- generate_random_dataset_dbl!\n");
		quit();
	}
	if (num_classes == 0){
		fprintf(stderr, "Error:: num_classes Is Invalid! In Function -- generate_random_dataset_dbl!\n");
		quit();
	}
	if (num_records == 0){
		fprintf(stderr, "Error:: num_records Is Invalid! In Function -- generate_random_dataset_dbl!\n");
		quit();
	}
	if (num_features == 0){
		fprintf(stderr, "Error:: num_features Is Invalid! In Function -- generate_random_dataset_dbl!\n");
		quit();
	}
	if (rng == NULL){
		fprintf(stderr, "Error:: random_number_generator Is NULL! In Function -- generate_random_dataset_dbl!\n");
		quit();
	}

	//===Initialize Dataset===//
	initialize_dataset(dataset, num_features, 0, num_classes, DOUBLE);

	//===Generate Class Parameters===//
	generate_uniform_process_dbl(-22, 33, num_classes, class_means, rng);
	generate_uniform_process_dbl(1, 12, num_classes, class_variances, rng);

	//===Generate All Records===//
	num_records_per_class = num_records/num_classes;
	for (c=0; c<num_classes; c++){
		for (r=0; r<num_records_per_class; r++){
			generate_gaussian_process_dbl(class_means[c], class_variances[c], num_features, record, rng);
			add_dataset_record_dbl(dataset, record, NULL, &c);
		}
	}

	return;
}


//================================================================================================//
//======================================SEPERABILITY METHODS======================================//
//================================================================================================//

double compute_dataset_seperability_criterion_1( dataset_t* self )
{

	double J, Sbw, Sw;
	double matrix[MAX_NUM_FEATURES*MAX_NUM_FEATURES];

	//===Compute Matrices===//
	compute_dataset_within_class_scatters_dbl(self);
	compute_between_class_scatters_dbl(self);

	if (0){
		//===Add==//
		add_matrices_dbl(self->between_class_scatter_matrix.dbl,
						 self->within_class_scatter_matrix.dbl,
						 self->num_features, self->num_features, matrix);
	}
	else{
		//===Copy===//
		copy_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features, matrix);
	}
	//===Take Determinants===//
	Sbw = compute_determinant_dbl(matrix, self->num_features, self->num_features);
	Sw = compute_determinant_dbl(self->within_class_scatter_matrix.dbl,
						  		 self->num_features, self->num_features);
	
	//===Compute===//
	J = fabs(Sbw)/fabs(Sw);
	J = 1.0/J;

	return J;
}



//================================================================================================//
//==========================================PCA METHODS===========================================//
//================================================================================================//

//KL1 Standard PCA
void run_dataset_global_max_variance_pca_dbl( dataset_t* self,
							   		  		  double percent_variance )
{
	int rows_left;
	unsigned int i, num_features, original_num_features;
	double total_variance, current_variance;
	double temp[MAX_NUM_FEATURES], record[MAX_NUM_FEATURES];
	double eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double eigenvalues[MAX_NUM_FEATURES];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- run_dataset_global_max_variance_pca_dbl!\n");
		quit();
	}
	if (!isnormal((double)percent_variance)){
		fprintf(stderr, "Error:: percent_variance Is Invalid! In Function -- run_dataset_global_max_variance_pca_dbl!\n");
		quit();
	}
	
	//===Compute Global Centroid And Covariance===//
	compute_dataset_record_centroid_dbl(self);
	compute_dataset_covariance_dbl(self);

	//===Get Sorted Eigenspectrum Of Feature Covariance Matrix===//
	compute_symmetric_eigensystem_dbl(self->covariance_matrix.dbl, self->num_features, 1, eigenvalues, eigenvectors);

	//===Compute Total Variance===//
	total_variance = compute_sum_dbl(eigenvalues,self->num_features);

	//===Compute Num Features To Keep===//
	current_variance = 0; num_features = 0;
	while((current_variance/total_variance) * 100.0 < percent_variance){
		current_variance += eigenvalues[num_features++];
	}

	//===Eliminate Rows From Eigenvector Matrix===//
	original_num_features = self->num_features;
	rows_left = (int)(self->num_features) - (int)(num_features);
	while(rows_left > 0){
		remove_matrix_row_dbl(eigenvectors, &self->num_features, original_num_features, self->num_features - 1);
		rows_left -= 1;
	}

	//===Run Projection===//
	for (i=0; i<self->num_records; i++){
		//===Global Mean Center The Records===//
		subtract_vectors_dbl(self->record.dbl[i], self->record_centroid.dbl, original_num_features, temp);

		//===Send Each Global Mean Centered Record Through The Tranposed Matrix To Create New Record Set===//
		matrix_vector_multiply_dbl(temp, original_num_features, eigenvectors, 
								   self->num_features, original_num_features, record);
		initialize_array_dbl(self->record.dbl[i], original_num_features);
		copy_array_dbl(record, self->num_features, self->record.dbl[i]);
	}

	//===Re-Init All Other Statistics===//
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);	

	//===Recreate Initial Stats===//
	initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
	}
	initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);
	initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);	

	return;
}

//KL2
void run_dataset_class_max_variance_pca_dbl( dataset_t* self,
							   				 double percent_variance )
{

	int rows_left;
	unsigned int i, num_features, original_num_features;
	double total_variance, current_variance;
	double temp[MAX_NUM_FEATURES], record[MAX_NUM_FEATURES];
	double eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double eigenvalues[MAX_NUM_FEATURES];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- run_dataset_class_max_variance_pca_dbl!\n");
		quit();
	}
	if (!isnormal((double)percent_variance)){
		fprintf(stderr, "Error:: percent_variance Is Invalid! In Function -- run_dataset_class_max_variance_pca_dbl!\n");
		quit();
	}

	//===Compute Class Centroids===//
	compute_dataset_class_centroids_dbl(self);
	compute_dataset_within_class_scatters_dbl(self);

	//===Get Sorted Eigenspectrum Of Within Class Scatter Matrix===//
	compute_symmetric_eigensystem_dbl(self->within_class_scatter_matrix.dbl, self->num_features, 1, eigenvalues, eigenvectors);

	//===Compute Total Variance===//
	total_variance = compute_sum_dbl(eigenvalues,self->num_features);

	//===Compute Num Features To Keep===//
	current_variance = 0; num_features = 0;
	while((current_variance/total_variance)*100.0 < percent_variance){
		current_variance += eigenvalues[num_features++];
	}

	//===Eliminate Rows From Eigenvector Matrix===//
	original_num_features = self->num_features;
	rows_left = (int)(self->num_features) - (int)(num_features);
	while(rows_left > 0){
		remove_matrix_row_dbl(eigenvectors, &self->num_features, original_num_features, self->num_features - 1);
		rows_left -= 1;
	}
	
	//===Run Projection===//
	for (i=0; i<self->num_records; i++){
		//===Class Mean Center The Records===//
		subtract_vectors_dbl(self->record.dbl[i], self->class_centroid.dbl[self->classes[i]], original_num_features, temp);

		//===Send Each Class Mean Centered Record Through The Tranposed Matrix To Create New Record Set===//
		matrix_vector_multiply_dbl(temp, original_num_features, eigenvectors, 
								   self->num_features, original_num_features, record);
		initialize_array_dbl(self->record.dbl[i], original_num_features);
		copy_array_dbl(record, self->num_features, self->record.dbl[i]);
	}


	//===Re-Init All Other Statistics===//
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);	

	//===Recreate Initial Stats===//
	initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
	}
	initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);
	initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);	

	return;
}

//use between class scatter matrix KL3
void run_dataset_class_mean_variance_pca_dbl( dataset_t* self,
							   				  double percent_variance )
{

	int rows_left;
	unsigned int i, num_features, original_num_features;
	unsigned int indices[MAX_NUM_FEATURES];
	double total_variance, current_variance;
	double eigenvalue_variances[MAX_NUM_FEATURES];
	double objective_function[MAX_NUM_FEATURES];
	double temp[MAX_NUM_FEATURES], record[MAX_NUM_FEATURES];
	double temp_eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double eigenvalues[MAX_NUM_FEATURES];


	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- run_dataset_class_mean_variance_pca_dbl!\n");
		quit();
	}
	if (!isnormal((double)percent_variance)){
		fprintf(stderr, "Error:: percent_variance Is Invalid! In Function -- run_dataset_class_mean_variance_pca_dbl!\n");
		quit();
	}

	//===Compute Matrices===//
	compute_dataset_within_class_scatters_dbl(self);
	compute_between_class_scatters_dbl(self);

	//===Get Sorted Eigenspectrum Of Within Class Scatter Matrix===//
	compute_symmetric_eigensystem_dbl(self->within_class_scatter_matrix.dbl, self->num_features, 1, eigenvalues, eigenvectors);

	fprintf(stdout, "Eigenvalues: %d\n", self->num_features);
	print_vector_dbl(eigenvalues, self->num_features, stdout);
	newline();
	fprintf(stdout, "Eigenvectors: %d\n", self->num_features);
	print_matrix_dbl(eigenvectors, self->num_features, self->num_features, stdout);
	newline();

	//===Compute Measure For Each Eigenvalue===//
	for (i=0; i<self->num_features; i++){
		matrix_vector_multiply_dbl(eigenvectors + i*self->num_features, self->num_features, self->between_class_scatter_matrix.dbl, 
								   self->num_features, self->num_features, record);
		objective_function[i] = compute_inner_product_dbl(eigenvectors + i*self->num_features,record,self->num_features)/eigenvalues[i];
	}

	//===Rearrange Eigenvalues Based On Smallest Measure To Largest Measure===//
	sort_array_indices_dbl(objective_function, self->num_features, indices);
	for (i=0; i<self->num_features; i++){
		temp[i] = eigenvalues[indices[i]];
		copy_array_dbl(eigenvectors + indices[i]*self->num_features, self->num_features, temp_eigenvectors + i*self->num_features);
	}
	copy_array_dbl(temp, self->num_features, eigenvalues);
	copy_matrix_dbl(temp_eigenvectors, self->num_features, self->num_features, eigenvectors);
	
	//===Compute Total Variance Measure===//
	total_variance = 0;
	for (i=0; i<self->num_features; i++){
		matrix_vector_multiply_dbl(eigenvectors + i*self->num_features, self->num_features, self->between_class_scatter_matrix.dbl, 
								   self->num_features, self->num_features, record);
		eigenvalue_variances[i] = compute_inner_product_dbl(eigenvectors + i*self->num_features,record,self->num_features)/eigenvalues[i]; 
		total_variance += eigenvalue_variances[i];
	}

	//===Compute Num Features To Keep===//
	current_variance = 0; num_features = 0;
	while((current_variance/total_variance)*100.0 < percent_variance){
		current_variance += eigenvalue_variances[num_features++];
	}

	//===Eliminate Rows From Eigenvector Matrix===//
	original_num_features = self->num_features;
	rows_left = (int)(self->num_features) - (int)(num_features);
	while(rows_left > 0){
		remove_matrix_row_dbl(eigenvectors, &self->num_features, original_num_features, self->num_features - 1);
		rows_left -= 1;
	}
	fprintf(stdout, "Eigenvectors: %d\n", self->num_features);
	print_matrix_dbl(eigenvectors, self->num_features, original_num_features, stdout);
	newline();
	
	//===Run Projection===//
	for (i=0; i<self->num_records; i++){
		//===Class Mean Center The Records===//
		subtract_vectors_dbl(self->record.dbl[i], self->class_centroid.dbl[self->classes[i]], original_num_features, temp);

		//===Send Each Class Mean Centered Record Through The Tranposed Matrix To Create New Record Set===//
		matrix_vector_multiply_dbl(temp, original_num_features, eigenvectors, 
								   self->num_features, original_num_features, record);
		initialize_array_dbl(self->record.dbl[i], original_num_features);
		copy_array_dbl(record, self->num_features, self->record.dbl[i]);
		fprintf(stdout, "Record %d: \n", i);
		print_vector_dbl(self->record.dbl[i], self->num_features, stdout);
	}

	//===Re-Init All Other Statistics===//
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);	

	//===Recreate Initial Stats===//
	initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
	}
	initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);
	initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);	

	return;


}

//KL4
void run_dataset_class_min_entropy_pca_dbl( dataset_t* self,
							   				double percent_entropy )
{
	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- run_dataset_class_min_entropy_pca_dbl!\n");
		quit();
	}
	if (!isnormal((double)percent_entropy)){
		fprintf(stderr, "Error:: percent_entropy Is Invalid! In Function -- run_dataset_class_min_entropy_pca_dbl!\n");
		quit();
	}

	int rows_left;
	unsigned int i, j, num_features, original_num_features;
	unsigned int indices[MAX_NUM_FEATURES];
	double total_entropy, current_entropy;
	double class_eigenvalues[MAX_NUM_CLASSES][MAX_NUM_FEATURES];
	double temp[MAX_NUM_FEATURES], record[MAX_NUM_FEATURES];
	double temp_eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double entropy[MAX_NUM_FEATURES];
	double eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double eigenvalues[MAX_NUM_FEATURES];

	//===Compute Matrices===//
	compute_dataset_class_centroids_dbl(self);
	compute_dataset_within_class_scatters_dbl(self);

	//===Get Sorted Eigenspectrum Of Within Class Scatter Matrix===//
	compute_symmetric_eigensystem_dbl(self->within_class_scatter_matrix.dbl, self->num_features, 1, eigenvalues, eigenvectors);

	//===Compute Each Class Eigenvalue===//
	compute_dataset_class_probabilities(self);
	for (i=0; i<self->num_classes; i++){
		for (j=0; j<self->num_features; j++){
			matrix_vector_multiply_dbl(eigenvectors + j*self->num_features, self->num_features, self->class_covariance_matrix.dbl[i], 
								   self->num_features, self->num_features, record);
			class_eigenvalues[i][j] = compute_inner_product_dbl(eigenvectors + j*self->num_features,record,self->num_features);
			class_eigenvalues[i][j] *= self->class_probabilities[i];
		}
	}

	//===Compute Entropy For Each Feature===//
	for (j=0; j<self->num_features; j++){
		entropy[j] = 0;
		for (i=0; i<self->num_classes; i++){
			entropy[j] += (class_eigenvalues[i][j]/eigenvalues[j]) * log((class_eigenvalues[i][j]/eigenvalues[j]));
		}
		entropy[j] *= -1.0;
	}

	//===Rearrange Eigenvalues Based On Smallest Measure To Largest Entropy===//
	sort_array_indices_dbl(entropy, self->num_features, indices);
	for (i=0; i<self->num_features; i++){
		temp[i] = eigenvalues[indices[i]];
		copy_array_dbl(eigenvectors + indices[i]*self->num_features, self->num_features, temp_eigenvectors + i*self->num_features);
	}
	copy_array_dbl(temp, self->num_features, eigenvalues);
	copy_matrix_dbl(temp_eigenvectors, self->num_features, self->num_features, eigenvectors);

	//===Compute Total Variance Measure===//
	total_entropy = 0;
	for (i=0; i<self->num_features; i++){
		total_entropy += entropy[i];
	}

	//===Compute Num Features To Keep===//
	current_entropy = 0; num_features = 0;
	while((current_entropy/total_entropy)*100.0 < percent_entropy){
		current_entropy += entropy[num_features++];
	}

	//===Eliminate Rows From Eigenvector Matrix===//
	original_num_features = self->num_features;
	rows_left = (int)(self->num_features) - (int)(num_features);
	while(rows_left > 0){
		remove_matrix_row_dbl(eigenvectors, &self->num_features, original_num_features, self->num_features - 1);
		rows_left -= 1;
	}

	//===Run Projection===//
	for (i=0; i<self->num_records; i++){
		//===Class Mean Center The Records===//
		subtract_vectors_dbl(self->record.dbl[i], self->class_centroid.dbl[self->classes[i]], original_num_features, temp);

		//===Send Each Class Mean Centered Record Through The Tranposed Matrix To Create New Record Set===//
		matrix_vector_multiply_dbl(temp, original_num_features, eigenvectors, 
								   self->num_features, original_num_features, record);
		initialize_array_dbl(self->record.dbl[i], original_num_features);
		copy_array_dbl(record, self->num_features, self->record.dbl[i]);
	}

	//===Re-Init All Other Statistics===//
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);	

	//===Recreate Initial Stats===//
	initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
	}
	initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);
	initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);	

	return;
}

//KL5
void run_dataset_class_compressed_variance_pca_dbl( dataset_t* self,
							   						double within_percent_variance,
													double between_percent_variance )
{
	int rows_left;
	unsigned int i, within_num_features, between_num_features, num_features;
	double total_variance, current_variance;
	double record[MAX_NUM_FEATURES];
	double within_eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double new_between_class_scatter[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double between_eigenvectors[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double eigenvalues[MAX_NUM_FEATURES];
	double between_eigenvalues[MAX_NUM_FEATURES];
	double within_eigenvalue_matrix[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double temp_matrix_1[MAX_NUM_FEATURES*MAX_NUM_FEATURES];
	double temp_matrix_2[MAX_NUM_FEATURES*MAX_NUM_FEATURES];

	//===Sanity Checks===//
	if (self == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- run_dataset_class_compressed_variance_pca_dbl!\n");
		quit();
	}
	if (!isnormal((double)within_percent_variance)){
		fprintf(stderr, "Error:: within_percent_variance Is Invalid! In Function -- run_dataset_class_compressed_variance_pca_dbl!\n");
		quit();
	}
	if (!isnormal((double)between_percent_variance)){
		fprintf(stderr, "Error:: between_percent_variance Is Invalid! In Function -- run_dataset_class_compressed_variance_pca_dbl!\n");
		quit();
	}

	//===Compute Matrices===//
	compute_dataset_within_class_scatters_dbl(self);
	compute_between_class_scatters_dbl(self);

	//===Get Sorted Eigenspectrum Of Within Class Scatter Matrix===//
	compute_symmetric_eigensystem_dbl(self->within_class_scatter_matrix.dbl, self->num_features, 1, eigenvalues, within_eigenvectors);
	
	//===Compute Total Variance===//
	total_variance = compute_sum_dbl(eigenvalues,self->num_features);

	//===Compute Num Features To Keep===//
	current_variance = 0; num_features = 0;
	while((current_variance/total_variance)*100.0 < within_percent_variance){
		current_variance += eigenvalues[num_features++];
	}

	//===Eliminate Rows From Eigenvector Matrix===//
	within_num_features = self->num_features;
	rows_left = (int)(self->num_features) - (int)(num_features);
	while(rows_left > 0){
		remove_matrix_row_dbl(within_eigenvectors, &within_num_features, self->num_features, within_num_features - 1);
		rows_left -= 1;
	}

	//===Form Within Eigenvector Diagonal Matrix===//
	initialize_array_dbl(within_eigenvalue_matrix, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	set_matrix_diagonal_dbl(within_eigenvalue_matrix, within_num_features, eigenvalues, within_num_features);
	pow_array_dbl(within_eigenvalue_matrix, within_num_features*within_num_features, 0.5);

	//===Form Projected Between Class Scatter Matrix===//
	transpose_matrix_dbl(within_eigenvectors, within_num_features, self->num_features);

	//===Ur L^-1/2===//
	matrix_matrix_multiply_dbl(within_eigenvectors, self->num_features, within_num_features,
							   within_eigenvalue_matrix, within_num_features, within_num_features,
							   temp_matrix_1);
	//===SB * (Ur L^-1/2)===//
	matrix_matrix_multiply_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features,
							   temp_matrix_1, self->num_features, within_num_features,
							   temp_matrix_2);
	//===(L^-1/2 Ur^T)===//
	transpose_matrix_dbl(within_eigenvectors, self->num_features, within_num_features);
	matrix_matrix_multiply_dbl(within_eigenvalue_matrix, within_num_features, within_num_features,
							   within_eigenvectors, within_num_features, self->num_features,
							   temp_matrix_1);
	//===(L^-1/2 Ur^T) * SB * (Ur L^-1/2)===//
	matrix_matrix_multiply_dbl(temp_matrix_1, within_num_features, self->num_features,
							   temp_matrix_2, self->num_features, within_num_features,
							   new_between_class_scatter);

	//===Get Sorted Eigenspectrum Of New Between Class Scatter Matrix===//
	compute_symmetric_eigensystem_dbl(new_between_class_scatter, within_num_features, 1, between_eigenvalues, between_eigenvectors);

	//===Compute Total Variance===//
	total_variance = compute_sum_dbl(between_eigenvalues, within_num_features);

	//===Compute Num Features To Keep===//
	current_variance = 0; num_features = 0;
	while((current_variance/total_variance)*100.0 < between_percent_variance){
		current_variance += between_eigenvalues[num_features++];
	}

	//===Eliminate Rows From Eigenvector Matrix===//
	between_num_features = within_num_features;
	rows_left = (int)(within_num_features) - (int)(num_features);
	while(rows_left > 0){
		remove_matrix_row_dbl(between_eigenvectors, &between_num_features, self->num_features, between_num_features - 1);
		rows_left -= 1;
	}

	//===Form Transform Matrix===//
	matrix_matrix_multiply_dbl(within_eigenvalue_matrix, within_num_features, within_num_features,
							   within_eigenvectors, within_num_features, self->num_features,
							   temp_matrix_1);
	matrix_matrix_multiply_dbl(between_eigenvectors, between_num_features, within_num_features,
							   temp_matrix_1, within_num_features, self->num_features,
							   temp_matrix_2);

	//===Run Projection===//
	for (i=0; i<self->num_records; i++){
		//===Send Each Record Through The Tranposed Matrix To Create New Record Set===//
		matrix_vector_multiply_dbl(self->record.dbl[i], self->num_features, temp_matrix_2, 
								   between_num_features, self->num_features, record);
		initialize_array_dbl(self->record.dbl[i], self->num_features);
		copy_array_dbl(record, between_num_features, self->record.dbl[i]);
	}

	//===Set Num Features===//
	self->num_features = between_num_features;

	//===Re-Init All Other Statistics===//
	initialize_array_cmplx(self->covariance_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->record_centroid.cmplx, MAX_NUM_FEATURES);	
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_array_cmplx(self->class_covariance_matrix.cmplx[i], MAX_NUM_FEATURES*MAX_NUM_FEATURES);
		initialize_array_cmplx(self->class_centroid.cmplx[i], MAX_NUM_FEATURES);
	}
	initialize_array_cmplx(self->within_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);
	initialize_array_cmplx(self->between_class_scatter_matrix.cmplx, MAX_NUM_FEATURES*MAX_NUM_FEATURES);	

	//===Recreate Initial Stats===//
	initialize_identity_matrix_dbl(self->covariance_matrix.dbl, self->num_features, self->num_features);
	for (i=0; i<MAX_NUM_CLASSES; i++){
		initialize_identity_matrix_dbl(self->class_covariance_matrix.dbl[i], self->num_features, self->num_features);
	}
	initialize_identity_matrix_dbl(self->within_class_scatter_matrix.dbl, self->num_features, self->num_features);
	initialize_identity_matrix_dbl(self->between_class_scatter_matrix.dbl, self->num_features, self->num_features);	


	return;
}

void generate_seperable_dataset_dbl( unsigned int num_classes,
								  	 unsigned int num_records,
								  	 unsigned int num_features,
								  	 dataset_t* dataset,
								  	 random_number_generator_t* rng )
{
	unsigned int i,j,k,class;
	double means[3][3], variances[3][3];
	double record[MAX_NUM_FEATURES];

	//===Sanity Check===//
	if (dataset == NULL){
		fprintf(stderr, "Error:: Dataset Is NULL! In Function -- generate_seperable_dataset_dbl!\n");
		quit();
	}
	if (rng == NULL){
		fprintf(stderr, "Error:: Rng Is NULL! In Function -- generate_seperable_dataset_dbl!\n");
		quit();
	}

	//===Initialize===//
	initialize_dataset(dataset, num_features, 0, num_classes, DOUBLE);
	
	//===Class 1===//
	means[0][0] = 10.0; means[0][1] = 5.0; means[0][2] = 7.5;
	variances[0][0] = 1.0; variances[0][1] = 0.5; variances[0][2] = 2.0;

	//===Class 2===//
	means[1][0] = -10.0; means[1][1] = 1.0; means[1][2] = -15.0;
	variances[1][0] = 3.0; variances[1][1] = 2.0; variances[2][2] = 3.25;

	//===Commons===//
	means[2][0] = 3.0; means[2][1] = 21.0;
	variances[2][0] = 0.10; variances[2][1] = 1.0;

	//===Generate Random Vectors===//
	for (i=0; i<num_records; i++){

		//===Class 0===//
		if (i%2){
			for (j=0; j<3; j++){
				record[j] = get_random_gaussian(means[0][j], variances[0][j], rng);
			}		
		}
		else{ //===Class 1===//
			for (j=0; j<3; j++){
				record[j] = get_random_gaussian(means[1][j], variances[1][j], rng);
			}		
		}

		//===Common===//
		k = 0;
		for (j=3; j<num_features; j++){
			record[j] = get_random_gaussian(means[2][k], variances[2][k], rng);
			k++;
		}

		//===Add To Dataset===//
		class = i%2;
		add_dataset_record_dbl(dataset, record, NULL, &class);
	}


	return;
}


//================================================================================================//
//=========================================TEST METHODS===========================================//
//================================================================================================//

void test_dataset_max_variance_pca_2()
{

	unsigned int num_features, num_classes, num_attributes, class;
	double record[MAX_NUM_FEATURES];
	dataset_t* dataset;

	//===Mallocs===//
	dataset = malloc(sizeof(dataset_t));

	//===Initialize Dataset===//
	num_features = 3; num_classes = 1; num_attributes = 0;
	initialize_dataset(dataset, num_features, num_attributes, num_classes, DOUBLE);

	//===Form Records===//
	if (1){
		record[0] = 7; record[1] = 4; record[2] = 3; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 4; record[1] = 1; record[2] = 8; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 6; record[1] = 3; record[2] = 5; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 8; record[1] = 6; record[2] = 1; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 8; record[1] = 5; record[2] = 7; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 7; record[1] = 2; record[2] = 9; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 5; record[1] = 3; record[2] = 3; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 9; record[1] = 5; record[2] = 8; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 7; record[1] = 4; record[2] = 5; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 8; record[1] = 2; record[2] = 2; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
	}
	else{
		record[0] = 7; record[1] = 4; record[2] = 3; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 4; record[1] = 1; record[2] = 8; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 6; record[1] = 3; record[2] = 5; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 8; record[1] = 6; record[2] = 1; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 8; record[1] = 5; record[2] = 7; class = 0;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 7; record[1] = 2; record[2] = 9; class = 1;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 5; record[1] = 3; record[2] = 3; class = 1;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 9; record[1] = 5; record[2] = 8; class = 1;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 7; record[1] = 4; record[2] = 5; class = 1;
		add_dataset_record_dbl(dataset, record, NULL, &class);
		record[0] = 8; record[1] = 2; record[2] = 2; class = 1;
		add_dataset_record_dbl(dataset, record, NULL, &class);
	}
	//===Compute Covariance Matrix===//
	compute_dataset_record_centroid_dbl(dataset);
	compute_dataset_covariance_dbl(dataset);
	fprintf(stdout, "Record Centroid: \n");
	print_vector_dbl(dataset->record_centroid.dbl, num_features, stdout);
	newline();
	fprintf(stdout, "Covariance:\n");
	print_matrix_dbl(dataset->covariance_matrix.dbl, num_features, num_features, stdout);
	newline();



	//===Compute Class Covariances===//
	compute_dataset_class_centroids_dbl(dataset);
	compute_dataset_class_covariances_dbl(dataset);
	compute_dataset_within_class_scatters_dbl(dataset);
	compute_dataset_class_probabilities(dataset);
	fprintf(stdout, "Class Centroid: \n");
	print_vector_dbl(dataset->class_centroid.dbl[0], num_features, stdout);
	newline();
	fprintf(stdout, "Class Covariance:\n");
	print_matrix_dbl(dataset->class_covariance_matrix.dbl[0], num_features, num_features, stdout);
	newline();
	fprintf(stdout, "Within Class Scatter:\n");
	print_matrix_dbl(dataset->within_class_scatter_matrix.dbl, num_features, num_features, stdout);
	newline();


	//===Run PCA===//
	//run_dataset_global_max_variance_pca_dbl(dataset, 85); checked!
	//run_dataset_class_max_variance_pca_dbl(dataset, 70); checked!
	//run_dataset_class_mean_variance_pca_dbl(dataset, 40); checked! want small percent here
	//run_dataset_class_min_entropy_pca_dbl(dataset, 70); checked! want small percent here
	//run_dataset_class_compressed_variance_pca_dbl(dataset, 70, 70); //checked! i think

	return;
}


void test_dataset_max_variance_pca()
{
	unsigned int i, num_features, num_classes, class, num_attributes;
	double record[MAX_NUM_FEATURES];
	double true_records[10][MAX_NUM_FEATURES];
	dataset_t* dataset;

	//===Mallocs===//
	dataset = malloc(sizeof(dataset_t));

	//===Initialize Dataset===//
	num_features = 2; num_classes = 2; num_attributes = 0;
	initialize_dataset(dataset, num_features, num_attributes, num_classes, DOUBLE);

	//===Form Records===//
	record[0] = 1; record[1] = -1; class = 0;
	add_dataset_record_dbl(dataset, record, NULL, &class);
	record[0] = 0; record[1] = 1; class = 0;
	add_dataset_record_dbl(dataset, record, NULL, &class);
	record[0] = -1; record[1] = 0; class = 0;
	add_dataset_record_dbl(dataset, record, NULL, &class);

	//===Compute Covariance Matrix===//
	compute_dataset_record_centroid_dbl(dataset);
	compute_dataset_covariance_dbl(dataset);
	fprintf(stdout, "Covariance:\n");
	print_matrix_dbl(dataset->covariance_matrix.dbl, num_features, num_features, stdout);
	newline();

	//===Run PCA===//
	run_dataset_global_max_variance_pca_dbl(dataset, 100);

	//===Set True Records===//
	true_records[0][0] = 2.0/sqrt(2.0); true_records[0][1] = 0;
	true_records[1][0] = -1.0/sqrt(2.0); true_records[1][1] = 1.0/sqrt(2.0);
	true_records[2][0] = -1.0/sqrt(2.0); true_records[2][1] = -1.0/sqrt(2.0);

	//===Check Against True Values===//
	for (i=0; i<dataset->num_records; i++){
		fprintf(stdout, "Projected Record: %d\n", i);
		print_vector_dbl(dataset->record.dbl[i], dataset->num_features, stdout);
		fprintf(stdout, "True Projected Record: %d\n", i);
		print_vector_dbl(true_records[i], dataset->num_features, stdout);
		newline();
	}

	//===Clean Up===//
	free(dataset);

	return;
}

void test_load_dataset()
{
	unsigned int num_features, num_classes, num_attributes;
	char* filename;
	dataset_t* dataset;
	FILE* fout;

	//===Mallocs===//
	dataset = malloc(sizeof(dataset_t));

	//===Initialize Dataset===//
	num_features = 12; num_classes = 2; num_attributes = 1;
	initialize_dataset(dataset, num_features, num_attributes, num_classes, DOUBLE);

	//===Load Data===//
	filename = "OldR/Data/logfilter_vad_features.dat";
	load_dataset_dbl(filename, dataset);

	//===Save Data===//
	fout = fopen("test_features.dat", "w");
	print_dataset_records_dbl(dataset, fout);
	fclose(fout);

	//===Clean Up===//
	free(dataset);

	return;
}

void test_som()
{
	unsigned int num_classes, num_records, num_features, num_attributes;
	unsigned int num_x_nodes, num_y_nodes, num_iterations;
	unsigned int test_indices[MAX_NUM_RECORDS];
	double learning_rate;
	char* filename;
	char* outname;
	dataset_t* dataset;
	som_t* som;
	random_number_generator_t* rng;
	FILE* fout;

	//===Mallocs===//
	dataset = malloc(sizeof(dataset_t));
	som = malloc(sizeof(som_t));
	rng = malloc(sizeof(random_number_generator_t));

	//===Init RNG===//
	initialize_random_number_generator(rng);

	//===Initialize Dataset===//
	if (0){
		num_classes = 2; num_records = 100; num_features = 5;
		generate_random_dataset_dbl(num_classes, num_records, num_features, dataset, rng);
		compress_dataset_records_dbl(dataset);
	}
	else{

		//===Load Data===//
		num_classes = 2; num_attributes = 1; num_features = 16;
		initialize_dataset(dataset, num_features, num_attributes, num_classes, DOUBLE);
		fprintf(stdout, "Loading Dataset...\n");
		filename = "OldR/Data/logfilter_vad_model_features.dat";
		//filename = "OldR/Data/logfilter_vad_mfcc_features.dat";
		load_dataset_dbl(filename, dataset);

		//===Balance Dataset===//
		fprintf(stdout, "Balancing Dataset...\n");
		balance_dataset_dbl(dataset);

		//===Print Records===//
		fprintf(stdout, "Printing Dataset...\n");
		outname = "OldR/Data/logfilter_vad_model_balanced.dat";
		//outname = "OldR/Data/logfilter_vad_mfcc_balanced.dat";
		fout = fopen(outname, "w");
		print_dataset_records_dbl(dataset, fout);
		fclose(fout);

	}

	//===Perform Fold Test===//
	fprintf(stdout, "Making Test Indices...\n");
	make_dataset_test_indices(dataset, 10, test_indices, rng);
	
	//===Initialize SOM===//
	learning_rate = 0.25; //works fairly well to show evenness
	num_x_nodes = MAX_NUM_SOM_NODES/2; num_y_nodes = MAX_NUM_SOM_NODES/2; num_iterations = MAX_NUM_SOM_ITERATIONS;
	fprintf(stdout, "Initialzing SOM...\n");
	initialize_som(som, num_x_nodes, num_y_nodes, dataset->num_features, num_iterations, learning_rate, rng);

	//===Train SOM===//
	fprintf(stdout, "Training SOM...\n");
	train_som(som, dataset);

	//===Make Map===//
	fprintf(stdout, "Mapping SOM...\n");
	make_som_class_map(som,dataset);

	//===Clean Up===//
	free(rng);
	free(som);
	free(dataset);

	return;
}


void test_conjuate_gradient_dbl()
{
	unsigned int dimension;
	double error;
	double A[100];
	double M1[100];
	double b[100];
	double x_true[100];
	double x_hat[100];

	//===Set Up System===//
	dimension = 3;
	A[0] = 3; A[1] = -2; A[2] = 5;
	A[3] = 4; A[4] = -7; A[5] = -1;
	A[6] = 5; A[7] = -6; A[8] = 4;

	b[0] = 2;	x_true[0] = 1.0; 
	b[1] = 19;  x_true[1] = -2.0;
	b[2] = 13;	x_true[2] = -1.0;

	initialize_identity_matrix_dbl(M1, dimension, dimension);

	//===Run CG===//
	initialize_array_constant_dbl(x_hat, dimension, 10e-4);
	run_conjugate_gradient_dbl(A, b, M1, dimension, x_hat);

	//===Print Results===//
	fprintf(stdout, "Found Answer:\n");
	print_vector_dbl(x_hat, dimension, stdout);
	newline();

	fprintf(stdout, "True Answer:\n");
	print_vector_dbl(x_true, dimension, stdout);
	newline();
	
	fprintf(stdout, "Error:\n");
	error = compute_mean_squared_error_dbl(x_hat, x_true, dimension);
	fprintf(stdout, "%+lf\n", error);
	newline();

	return;
}


void test_bb_dataset()
{
	unsigned int i, j, k, class, num_records,num_features, num_attributes, num_classes;
	double j_value;
	unsigned int feature_indicator[MAX_NUM_FEATURES];
	double means[MAX_NUM_CLASSES][MAX_NUM_FEATURES], variances[MAX_NUM_CLASSES][MAX_NUM_FEATURES];
	double record[MAX_NUM_FEATURES];
	dataset_t* dataset, *subset;
	random_number_generator_t* rng;

	//===Mallocs===//
	dataset = malloc(sizeof(dataset_t));
	subset = malloc(sizeof(dataset_t));
	rng = malloc(sizeof(random_number_generator_t));

	//===Initializations===//
	num_features = 5; num_attributes = 0; num_classes = 2;
	initialize_random_number_generator(rng);	

	num_records = 10000;
	generate_seperable_dataset_dbl(num_classes, num_records, num_features, dataset, rng);
	quit();

	initialize_dataset(dataset, num_features, num_attributes, num_classes, DOUBLE);

	//===Class 1===//
	means[0][0] = 10.0; means[0][1] = 5.0; means[0][2] = 7.5;
	variances[0][0] = 1.0; variances[0][1] = 0.5; variances[0][2] = 2.0;

	//===Class 2===//
	means[1][0] = -10.0; means[1][1] = 1.0; means[1][2] = -15.0;
	variances[1][0] = 3.0; variances[1][1] = 2.0; variances[2][2] = 3.25;

	//===Commons===//
	means[2][0] = 3.0; means[2][1] = 21.0;
	variances[2][0] = 0.10; variances[2][1] = 1.0;

	//===Generate Random Vectors===//
	num_records = 10000;
	for (i=0; i<num_records; i++){

		//===Class 0===//
		if (i%2){
			for (j=0; j<3; j++){
				record[j] = get_random_gaussian(means[0][j], variances[0][j], rng);
			}		
		}
		else{ //===Class 1===//
			for (j=0; j<3; j++){
				record[j] = get_random_gaussian(means[1][j], variances[1][j], rng);
			}		
		}

		//===Common===//
		k = 0;
		for (j=3; j<num_features; j++){
			record[j] = get_random_gaussian(means[2][k], variances[2][k], rng);
			k++;
		}

		//===Add To Dataset===//
		class = i%2;
		add_dataset_record_dbl(dataset, record, NULL, &class);
	}

	//===Partition And Compute J===//
	initialize_array_uint(feature_indicator, MAX_NUM_FEATURES);
	for (i=0; i<num_features-1; i++){

		//===Init Subset===//
		initialize_dataset(subset, i+1, num_attributes, num_classes, DOUBLE);

		//===Make Subset===//
		feature_indicator[i] = 1;
		subset_dataset_by_feature_dbl(dataset, feature_indicator, subset);

		//===Compute J===//
		j_value = compute_dataset_seperability_criterion_1(subset);
		fprintf(stdout, "J %d: %lf\n", i+1, j_value);

	}
	//===Compute Full J===//
	j_value = compute_dataset_seperability_criterion_1(dataset);
	fprintf(stdout, "J Full: %lf\n", j_value);

	//===Clean Up===//
	free(rng);
	free(dataset);

	return;
}
