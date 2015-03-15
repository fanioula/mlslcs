mlaslcs
=======

The mlaslcs source code.


Regarding the defaultLcs.properties file, 
there are several parameters to be considered about or altered.


	filename: 
		The (absolute) path to the .arff file that the LCS will be trained with.
	
	testFile (optional): 
		Include a (absolute) path to the .arff file you want the LCS to be evaluated under.
		use a # to comment the line if n-fold validation is desired.
						 
	numberOfLabels: 
		The number of labels of the data set used to train the LCS. 
	
	numOfFoldRepetitions:
		If the testFile line is commented, set numOfFoldRepetitions to X, 
		where X is the number of times the LCS will be trained with each fold.
		Default value is numOfFoldRepetitions = 3.
		
	trainIterations:
		The number of times the LCS will be trained with each data set instance.
		
	callbackRate:
		At every callbackRate iterations a function is called to store information about the progress 
		of the training process.
		
	UpdateOnlyPercentage:
		A (float) number. UpdateOnlyPercentage * trainIterations provides the number of iterations 
		that the LCS is being trained after trainIterations iterations, with the genetic algorithm deactivated.
		
	populationSize:
		The upper limit of the number of rules the LCS will hold.
		
	crossoverRate, mutationRate, thetaGA: 
		GA-specific parameters. Usually only the thetaGA should be adjusted.
		
	AttributeGeneralizationRate:
		The rate by which each instance attribute will be turned into a don't care (#)
		in rules created by the covering component.
	
	LabelGeneralizationRate:
		The rate by which each instance label will be turned into a don't care (#) 
		in rules created by the covering component.
		
	ClusteringAttributeGeneralizationRate:
		The rate by which each instance attribute will be turned into a don't care (#)
		in rules created by the clustering component.
		
	ClusteringLabelGeneralizationRate:
		The rate by which each instance label will be turned into a don't care (#) 
		in rules created by the clustering component.
		
	precisionBits:
		The number of bits used in the numerical attributes representation.
		
	LearningRate:
		The learning rate used to calculate the niche size estimate.
	
	ASLCS_N:
		fitness = (accuracy) ^ ASLCS_N
		
	ASLCS_THETA_DEL:
		The experience threshold for deletion.
	
	ASLCS_OMEGA, ASLCS_PHI:
		When a rule doesn't care about a label the accuracy of the rule is updated as follows:
		accuracy = (true_positives + ASLCS_OMEGA) / (match_set_appearances + ASLCS_PHI)
	
	CLUSTER_GAMMA:
		The γ parameter used in the clustering component.
		
	initializePopulation:
		Boolean. Used to activate or deactivate the clustering component.
		
	
	
