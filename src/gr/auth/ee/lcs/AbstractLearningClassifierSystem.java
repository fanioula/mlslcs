/*
 *	Copyright (C) 2011 by F. Tzima and M. Allamanis
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in
 *	all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *	THE SOFTWARE.
 */
/**
 * 
 */
package gr.auth.ee.lcs;
//comment

import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.classifiers.populationcontrol.FixedSizeSetWorstFitnessDeletion;
import gr.auth.ee.lcs.classifiers.statistics.MeanAttributeSpecificityStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanCoverageStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanFitnessStatistic;
import gr.auth.ee.lcs.classifiers.statistics.MeanLabelSpecificity;
import gr.auth.ee.lcs.classifiers.statistics.WeightedMeanAttributeSpecificityStatistic;
import gr.auth.ee.lcs.classifiers.statistics.WeightedMeanCoverageStatistic;
import gr.auth.ee.lcs.classifiers.statistics.WeightedMeanLabelSpecificity;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.data.representations.complex.GenericMultiLabelRepresentation;
import gr.auth.ee.lcs.data.representations.complex.GenericMultiLabelRepresentation.VotingClassificationStrategy;
import gr.auth.ee.lcs.evaluators.AccuracyRecallEvaluator;
import gr.auth.ee.lcs.evaluators.ExactMatchEvalutor;
import gr.auth.ee.lcs.evaluators.FileLogger;
import gr.auth.ee.lcs.evaluators.HammingLossEvaluator;
import gr.auth.ee.lcs.geneticalgorithm.selectors.RouletteWheelSelector;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.InstancesUtility;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


/**
 * An abstract LCS class to be implemented by all LCSs.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public abstract class AbstractLearningClassifierSystem {
	
	
	public String hookedMetricsFileDirectory;
	
	public final int UPDATE_MODE = (int) SettingsLoader.getNumericSetting("UPDATE_MODE", 0);
	
	/**
	 * Selection of the update mode, which adds offsprings to the population
	 * as soon as they are created.
	 * */
	public static final int UPDATE_MODE_IMMEDIATE = 0;
	
	
	/**
	 * Selection of the update mode, which adds the total number 
	 * of produced offsprings en masse to the population.
	 * */
	public static final int UPDATE_MODE_HOLD = 1;
	
	
	/**
	 * The mean correct set numerosity (in miscroclassifiers) of the population.
	 * */
	public double meanCorrectSetNumerosity = 0;
	
	/**
	 * Its value represents the current learning iteration (iterations * datasetInstanceIndex).
	 * */
	private int cummulativeCurrentInstanceIndex = 0;
	


	/**
	 * The train set.
	 * @uml.property  name="instances" multiplicity="(0 -1)" dimension="2"
	 */
	public double[][] instances;
	public double[][] testInstances;

	public Instances trainSet;
	public Instances testSet;
	

	/**
	 * The LCS instance transform bridge.
	 * @uml.property  name="transformBridge"
	 * @uml.associationEnd  
	 */
	
	public double labelCardinality = 1;
	
	public int numberOfCoversOccured = 0;

	
	
	private ClassifierTransformBridge transformBridge;

	/**
	 * The Abstract Update Algorithm Strategy of the LCS.
	 * @uml.property  name="updateStrategy"
	 * @uml.associationEnd  
	 */
	protected AbstractUpdateStrategy updateStrategy;

	/**
	 * The rule population.
	 * @uml.property  name="rulePopulation"
	 * @uml.associationEnd  
	 */
	protected ClassifierSet rulePopulation;

	/**
	 * A vector of all evaluator hooks.
	 * @uml.property  name="hooks"
	 * @uml.associationEnd  multiplicity="(0 -1)" elementType="gr.auth.ee.lcs.data.ILCSMetric"
	 */
	private final Vector<ILCSMetric> hooks;

	/**
	 * Frequency of the hook callback execution.
	 * @uml.property  name="hookCallbackRate"
	 */
	private int hookCallbackRate;
	
	
	public int repetition;
	
	private final boolean thoroughlyCheckWIthPopulation = SettingsLoader.getStringSetting("thoroughlyCheckWIthPopulation", "true").equals("true");

	
//	/**
//	 * Matrix used to store the time measurements for different phases of the train procedure.
//	 */
//	public double[][] timeMeasurements;
	
	public double[][] systemAccuracy;
	
	public Vector<Float> 	qualityIndexOfDeleted = new Vector<Float>();
	public Vector<Float> 	qualityIndexOfClassifiersCoveredDeleted = new Vector<Float>();
	public Vector<Float> 	qualityIndexOfClassifiersGaedDeleted = new Vector<Float>();
	
	public Vector<Float> 	accuracyOfDeleted = new Vector<Float>();
	public Vector<Float> 	accuracyOfCoveredDeletion = new Vector<Float>();
	public Vector<Float> 	accuracyOfGaedDeletion = new Vector<Float>();
	
	public Vector<Integer> iteration = new Vector<Integer>();
	public Vector<Integer> originOfDeleted = new Vector<Integer>();

	public Vector<Float> 	systemAccuracyInTraining = new Vector<Float>();
	public Vector<Float> 	systemAccuracyInTestingWithPcut = new Vector<Float>();
	public Vector<Float> 	systemCoverage = new Vector<Float>();

	
	public int numberOfClassifiersDeletedInMatchSets;
	
	public int totalRepetition = 0;
		
	public final int iterations;
	
	/**
	 * Constructor.
	 */ 	
	protected AbstractLearningClassifierSystem() {
		try {
			SettingsLoader.loadSettings();
		} catch (IOException e) {
			e.printStackTrace();
		}
		hooks = new Vector<ILCSMetric>();
		hookCallbackRate = (int) SettingsLoader.getNumericSetting("callbackRate", 100);
		iterations = (int) SettingsLoader.getNumericSetting("trainIterations",1000);
	}
	
	
	public void assimilateDuplicateClassifiers(ClassifierSet rulePopulation, 
											final boolean evolve) {
		//if (evolve) {
			// if subsumption is only made by the parents and not the whole population, merge classifiers to avoid duplicates
		
				for (int j = 0; j < rulePopulation.getNumberOfMacroclassifiers() ; j++) {
				//for (int j = rulePopulation.getNumberOfMacroclassifiers() -1; j >= 0 ; j--) {

					Vector<Integer> indicesOfDuplicates    = new Vector<Integer>();
					Vector<Float> 	fitnessOfDuplicates    = new Vector<Float>();
					Vector<Integer> experienceOfDuplicates = new Vector<Integer>();

					final Classifier aClassifier = rulePopulation.getMacroclassifiersVector().get(j).myClassifier;
					
					for (int i = rulePopulation.getNumberOfMacroclassifiers() - 1; i >= 0 ; i--) {
					//for (int i = 0; i < rulePopulation.getNumberOfMacroclassifiers(); i++) {

						Classifier theClassifier = rulePopulation.getMacroclassifiersVector().get(i).myClassifier;
						
						if (theClassifier.equals(aClassifier)) { 
							indicesOfDuplicates.add(i);
							float theClassifierFitness = (float) (rulePopulation.getMacroclassifiersVector().get(i).numerosity 
									* getUpdateStrategy().getComparisonValue(theClassifier, AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
							fitnessOfDuplicates.add(theClassifierFitness);
							experienceOfDuplicates.add(theClassifier.experience);

						}
					} // exo brei ta indexes ton diplon kanonon sto vector myMacroclassifiers
					
					/*an bro enan mono, simainei oti aClassifier == theClassifier, opote den exei noima na ginei afomoiosi
					 * an bro duo i kai perissoterous simainei oti prepei na epilekso poios apo olous 9a afomoiosei olous tous allous.
					 * opoios exei megalutero fitness afomoionei tous upoloipous. an duo exoun to idio fitness, 9a afomoiosei autos me to megalutero experience
					 * */
					if (indicesOfDuplicates.size() >= 2) {
						
						int indexOfSurvivor = 0;
						float maxFitness = 0;
						for(int k = 0; k < indicesOfDuplicates.size(); k++) {
							if (fitnessOfDuplicates.elementAt(k) > maxFitness) {
								maxFitness = fitnessOfDuplicates.elementAt(k);
								indexOfSurvivor = k;
							}
							else if (fitnessOfDuplicates.elementAt(k) == maxFitness) {
								if (experienceOfDuplicates.elementAt(k) >= experienceOfDuplicates.elementAt(indexOfSurvivor)) {
									indexOfSurvivor = k;
								}
									
							}
						}
						// exo brei poios 9a einai o epizon classifier. initiate assimilation
						//for (int k = indicesOfDuplicates.size() -1; k >= 0 ; k--) {
						for (int k = 0; k < indicesOfDuplicates.size() ; k++) {

							if (k != indexOfSurvivor) {
								rulePopulation.getMacroclassifiersVector().get(indicesOfDuplicates.elementAt(indexOfSurvivor)).numerosity += 
									rulePopulation.getMacroclassifiersVector().get(indicesOfDuplicates.elementAt(k)).numerosity;
								rulePopulation.getMacroclassifiersVector().get(indicesOfDuplicates.elementAt(indexOfSurvivor)).numberOfSubsumptions++;
								rulePopulation.totalNumerosity += rulePopulation.getMacroclassifiersVector().get(indicesOfDuplicates.elementAt(k)).numerosity;
								rulePopulation.deleteMacroclassifier(indicesOfDuplicates.elementAt(k));
							}
						}
						
					}	
					
					//if (indicesOfDuplicates.size() != 0) {
						indicesOfDuplicates.clear();
						fitnessOfDuplicates.clear();
						experienceOfDuplicates.clear();
					//}
					
				}				
			}

	

	/**
	 * Classify a single instance.
	 * 
	 * @param instance
	 *            the instance to classify
	 * @return the labels the instance is classified in
	 */
	public abstract int[] classifyInstance(double[] instance);
	

	/**
	 * Creates a new instance of the actual implementation of the LCS.
	 * 
	 * @return a pointer to the new instance.
	 */
	public abstract AbstractLearningClassifierSystem createNew();

	/**
	 * Execute hooks.
	 * 
	 * @param aSet
	 *            the set on which to run the callbacks
	 */
	private void executeCallbacks(final ClassifierSet aSet, 
								    final int repetition, boolean evolve) {
		
		for (int i = 0; i < hooks.size(); i++) {
			hooks.elementAt(i).getMetric(this);
		}
		
		int numberOfClassifiersCovered = 0;
		int numberClassifiersGaed = 0;
		int numberOfSubsumptions = 0;
		double meanNs = 0;
		
		for (int i = 0; i < rulePopulation.getNumberOfMacroclassifiers(); i++) {
			if (this.getRulePopulation().getMacroclassifier(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER) {
				numberOfClassifiersCovered++;
			}
			else if (this.getRulePopulation().getMacroclassifier(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
				numberClassifiersGaed++;
			}
			numberOfSubsumptions += this.getRulePopulation().getMacroclassifier(i).numberOfSubsumptions;
			meanNs += this.getRulePopulation().getMacroclassifier(i).myClassifier.getNs();


		}
		
		int test = 0;
		int repetitionF = repetition;
		if (!evolve)
		{
			repetitionF += this.iterations;
			if (repetitionF >= (int)(iterations*(1+SettingsLoader.getNumericSetting("UpdateOnlyPercentage", .1))))
			{
				getRulePopulation().checkWholePopulationForPossibleSubsumptions();
				test = this.getRulePopulation().numberOfRulesForFullCoverageWithSpecificDecisions();
			}
		}
		
		System.out.println("\n");
		
		// rules for vallim4
		String[] BAMRules1V = {
				"00#### => 11##",
				"01#### => 10##",
				"1##### => 01##"
		};
		
		String[] BAMRules2aV = {
				"####00 => ##00",
				"####01 => ##01",
				"####10 => ##10",
				"####11 => ##11"
		};
		
		String[] BAMRules2bV = {
				"#####0 => ###0",
				"#####1 => ###1",
				"####0# => ##0#",
				"####1# => ##1#",
		};
		
		// rules for mlposition4
		String[] BAMRules1P = {
				"0000 => 0000",
				"0001 => 0001"
		};
		
		String[] BAMRules2aP = { };
		
		String[] BAMRules2bP = {
				"001# => 0010",
				"01## => 0100",
				"1### => 1000"
		};
		
		String trainFileName = SettingsLoader.getStringSetting("filename", "");
		if (trainFileName.contains("vallim4") || trainFileName.contains("mlposition4"))
		{
			String[] BAMrules1 = trainFileName.contains("vallim4") ? BAMRules1V : BAMRules1P;
			String[] BAMrules2a = trainFileName.contains("vallim4") ? BAMRules2aV : BAMRules2aP;
			String[] BAMrules2b = trainFileName.contains("vallim4") ? BAMRules2bV : BAMRules2bP;
	
			double temp1 = ClassifierSet.percentageOfBAMDiscovered(rulePopulation, BAMrules1);		
			double temp2a = ClassifierSet.percentageOfBAMDiscovered(rulePopulation, BAMrules2a);		
			double temp2b = ClassifierSet.percentageOfBAMDiscovered(rulePopulation, BAMrules2b);
			double temp2 = (temp2a > temp2b) ? temp2a : temp2b;
			
			try {
				final FileWriter fstream2 = new FileWriter(hookedMetricsFileDirectory + "/bam.txt", true);
				final BufferedWriter buffer2 = new BufferedWriter(fstream2);
				double totalPercentage = (temp1*BAMrules1.length/(BAMrules1.length+BAMrules2b.length)+temp2*BAMrules2b.length/(BAMrules1.length+BAMrules2b.length))*100;
				String testS = temp1 + "," +  temp2a + "," + temp2b+ "," + totalPercentage + "\n";
//				System.out.println(testS);
				buffer2.write(testS);	
				buffer2.flush();
				buffer2.close();
			} 
			catch (Exception e) {
				e.printStackTrace();
			}	
		}
		 
		
		meanNs /= this.getRulePopulation().getNumberOfMacroclassifiers();
		
		if (repetition % 100 == 0 || repetitionF >= (int)(iterations*(1+SettingsLoader.getNumericSetting("UpdateOnlyPercentage", .1))))
			try {
	
				// record the rule population and its metrics in population.txt
				final FileWriter fstream = new FileWriter(this.hookedMetricsFileDirectory + "/population_" + repetitionF +".txt", true);
				final BufferedWriter buffer = new BufferedWriter(fstream);
				buffer.write(					
						  String.valueOf(this.repetition) + "th repetition:"
						+ System.getProperty("line.separator")
						+ System.getProperty("line.separator")
						+ "Population size: " + rulePopulation.getNumberOfMacroclassifiers()
						+ System.getProperty("line.separator")
						+ "Timestamp: " + rulePopulation.totalGAInvocations
						+ System.getProperty("line.separator")
						+ "Classifiers in population covered :" + numberOfClassifiersCovered
						+ System.getProperty("line.separator")
						+ "Classifiers in population ga-ed :" 	+ numberClassifiersGaed
						+ System.getProperty("line.separator")
						+ "Covers occured: " + numberOfCoversOccured
						+ System.getProperty("line.separator")
						+ "Subsumptions: " + numberOfSubsumptions
						+ System.getProperty("line.separator")
						+ "Mean ns: " + meanNs
						+ System.getProperty("line.separator")
						+ "NumRules for full coverage with specific decisions: " +test
						+ System.getProperty("line.separator")
						+ rulePopulation
						+ System.getProperty("line.separator"));
				buffer.flush();
				buffer.close();
			} 
			catch (Exception e) {
				e.printStackTrace();
			}	
		
		this.numberOfCoversOccured = 0;
	
	}

	/**
	 * Return the LCS's classifier transform bridge.
	 * 
	 * @return the lcs's classifier transform bridge
	 */
	public final ClassifierTransformBridge getClassifierTransformBridge() {
		return transformBridge;
	}
	
	
	public int getCummulativeCurrentInstanceIndex() {
		return cummulativeCurrentInstanceIndex;
	}

	/**
	 * Returns a string array of the names of the evaluation metrics.
	 * 
	 * @return a string array containing the evaluation names.
	 */
	public abstract String[] getEvaluationNames();

	/**
	 * Returns the evaluation metrics for the given test set.
	 * 
	 * @param testSet
	 *            the test set on which to calculate the metrics
	 * @return a double array containing the metrics
	 */
	public abstract double[] getEvaluations(Instances testSet);
	

	/**
	 * Create a new classifier for the specific LCS.
	 * 
	 * @return the new classifier.
	 */
	public final Classifier getNewClassifier() {
		return Classifier.createNewClassifier(this);
	}

	/**
	 * Return a new classifier object for the specific LCS given a chromosome.
	 * 
	 * @param chromosome
	 *            the chromosome to be replicated
	 * @return a new classifier containing information about the LCS
	 */
	public final Classifier getNewClassifier(final ExtendedBitSet chromosome) {
		return Classifier.createNewClassifier(this, chromosome);
	}

	/**
	 * Getter for the rule population.
	 * @return  a ClassifierSet containing the LCSs population
	 * @uml.property  name="rulePopulation"
	 */
	public final ClassifierSet getRulePopulation() {
		return rulePopulation;
	}

	/**
	 * Returns the LCS's update strategy.
	 * @return  the update strategy
	 * @uml.property  name="updateStrategy"
	 */
	public final AbstractUpdateStrategy getUpdateStrategy() {
		return updateStrategy;
	}

	
	/**
	 * collect the system's multilabel accuracy per iteration, plus every classifier's accuracy per iteration(TODO)
	 * */
	public void harvestAccuracies(int iteration){
		
		final AccuracyRecallEvaluator trainingAccuracy = new AccuracyRecallEvaluator(trainSet, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY);

		final VotingClassificationStrategy str = ((GenericMultiLabelRepresentation) transformBridge).new VotingClassificationStrategy();
		((GenericMultiLabelRepresentation) transformBridge).setClassificationStrategy(str);
		str.proportionalCutCalibration(this.instances, rulePopulation);
		
		final AccuracyRecallEvaluator testingAccuracyWithPcut  = new AccuracyRecallEvaluator(testSet, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY);
		

		final MeanCoverageStatistic coverage = new MeanCoverageStatistic();

		double trainAcc = trainingAccuracy.getMetric(this);
		double testAccPcut = testingAccuracyWithPcut.getMetric(this);
		double cov = coverage.getMetric(this);

		systemAccuracyInTraining.add((float) trainAcc);
		systemAccuracyInTestingWithPcut.add((float) testAccPcut);
		systemCoverage.add((float) cov);	
		
	}
	
	
	/**
	 * Initialize the rule population by clustering the train set and producing rules based upon the clusters.
	 * The train set is initially divided in as many partitions as are the distinct label combinations.
	 * @throws Exception 
	 * 
	 * @param file
	 * 			the .arff file
	 * */
	public ClassifierSet initializePopulation (final String file) throws Exception {
		
		final double gamma = SettingsLoader.getNumericSetting("CLUSTER_GAMMA", .2);
		
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		
		final Instances set = InstancesUtility.openInstance(file);

		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setSeed(10);
		kmeans.setPreserveInstancesOrder(true);

		/*
		 * Table partitions will hold instances only with attributes.
		 * On the contrary, table partitionsWithCLasses will hold only the labels
		 */
		Instances[] partitions = InstancesUtility.partitionInstances(this, file);
		Instances[] partitionsWithCLasses = InstancesUtility.partitionInstances(this, file);
		
		
		/*
		 * Instead of having multiple positions for the same label combination, use only one.
		 * This is the one that will be used to "cover" the centroids.
		 */
		for (int i = 0; i <  partitionsWithCLasses.length; i++) {
			Instance temp = partitionsWithCLasses[i].instance(0);
			partitionsWithCLasses[i].delete();
			partitionsWithCLasses[i].add(temp);
		}
		
		
		/*
		 * Delete the labels from the partitions.
		 */
		String attributesIndicesForDeletion = "";
		
		for (int k = set.numAttributes() - numberOfLabels + 1; k <= set.numAttributes(); k++) {
			if (k != set.numAttributes())
				attributesIndicesForDeletion += k + ",";
			else
				attributesIndicesForDeletion += k;
		}
		
		/* 	attributesIncicesForDeletion = 8,9,10,11,12,13,14 e.g. for 7 attributes and 7 labels. 
		 * It does not start from 7 because it assumes that the user inputs the number. See the api.
		 */
		for (int i = 0; i < partitions.length; i++) {
		     Remove remove = new Remove();
		     remove.setAttributeIndices(attributesIndicesForDeletion);
		     remove.setInvertSelection(false);
		     remove.setInputFormat(partitions[i]);
		     partitions[i] = Filter.useFilter(partitions[i], remove);	
		     //System.out.println(partitions[i]);
		}
		// partitions now contains only attributes
		
		/*
		 * delete the attributes from partitionsWithCLasses
		 */
		String labelsIndicesForDeletion = "";

		for (int k = 1; k <= set.numAttributes() - numberOfLabels; k++) {
			if (k != set.numAttributes() - numberOfLabels)
				labelsIndicesForDeletion += k + ",";
			else
				labelsIndicesForDeletion += k;
		}
		
		/* 	attributesIncicesForDeletion = 8,9,10,11,12,13,14 e.g. for 7 attributes and 7 labels. 
		 * It does not start from 7 because it assumes that the user inputs the number. See the api.
		 */
		for (int i = 0; i < partitionsWithCLasses.length; i++) {
		     Remove remove = new Remove();
		     remove.setAttributeIndices(labelsIndicesForDeletion);
		     remove.setInvertSelection(false);
		     remove.setInputFormat(partitionsWithCLasses[i]);
		     partitionsWithCLasses[i] = Filter.useFilter(partitionsWithCLasses[i], remove);	
		     //System.out.println(partitionsWithCLasses[i]);
		}
		// partitionsWithCLasses now contains only labels
		
		int populationSize = (int) SettingsLoader.getNumericSetting("populationSize", 1500);
		
		// the set used to store the rules from all the clusters
		ClassifierSet initialClassifiers = new ClassifierSet(
															new FixedSizeSetWorstFitnessDeletion(this,
																	 populationSize,
																	 new RouletteWheelSelector(AbstractUpdateStrategy.COMPARISON_MODE_DELETION, true)));

		for (int i = 0; i < partitions.length; i++) {
			
			try {
				
				kmeans.setNumClusters((int) Math.ceil(gamma * partitions[i].numInstances()));
				kmeans.buildClusterer(partitions[i]);
//				int[] assignments = kmeans.getAssignments();

/*				int k=0;
				for (int j = 0; j < assignments.length; j++) {
					System.out.printf("Instance %d => Cluster %d ", k, assignments[j]);
					k++;
					System.out.println();

				}
				System.out.println();*/
					
				Instances centroids = kmeans.getClusterCentroids();
				int numOfCentroidAttributes = centroids.numAttributes();
				
				
				/*
				 * The centroids in this stage hold only attributes. To continue, we need to provide them the labels.
				 * These are the ones we removed earlier.
				 * But first, open up positions for attributes.
				 * */
				
				for (int j = 0; j < numberOfLabels; j++) {
					Attribute label = new Attribute("label" + j);
					centroids.insertAttributeAt(label, numOfCentroidAttributes + j);
				}
				
				
				for (int centroidInstances = 0; centroidInstances < centroids.numInstances(); centroidInstances++) {
					for (int labels = 0; labels < numberOfLabels; labels++) {
						centroids.instance(centroidInstances).setValue(numOfCentroidAttributes + labels, partitionsWithCLasses[i].instance(0).value(labels));
					}
				}

				double[][] centroidsArray = InstancesUtility.convertIntancesToDouble(centroids);

				for (int j = 0; j < centroidsArray.length; j++) {
					//System.out.printf("Instance %d => Cluster %d ", k, assignments[j]);
					final Classifier coveringClassifier = this.getClassifierTransformBridge().createRandomClusteringClassifier(centroidsArray[j]);
					
					
					coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_INIT); 
					initialClassifiers.addClassifier(new Macroclassifier(coveringClassifier, 1), false);	
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
//		System.out.println(initialClassifiers);
		return initialClassifiers;
	}
	
	
	
	/**
	 * Initialize the rule population by clustering the train set and producing rules based upon the clusters.
	 * The train set is initially divided in as many partitions as are the distinct label combinations.
	 * @throws Exception 
	 * 
	 * @param trainSet
	 * 				the type of Instances train set
	 * */
	
	public ClassifierSet initializePopulation (final Instances trainset) throws Exception {
		
		final double gamma = SettingsLoader.getNumericSetting("CLUSTER_GAMMA", .2);
		
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		
		final Instances set = trainset;

		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setSeed(10);
		kmeans.setPreserveInstancesOrder(true);
		
		/*
		 * Table partitions will hold instances only with attributes.
		 * On the contrary, table partitionsWithCLasses will hold only the labels
		 */
		Instances[] partitions = InstancesUtility.partitionInstances(this, trainset);
		Instances[] partitionsWithCLasses = InstancesUtility.partitionInstances(this, trainset);
		

		 /*
		 * Instead of having multiple positions for the same label combination, use only one.
		 * This is the one that will be used to "cover" the centroids.
		 */
		
		for (int i = 0; i <  partitionsWithCLasses.length; i++) {
			Instance temp = partitionsWithCLasses[i].instance(0);
			partitionsWithCLasses[i].delete();
			partitionsWithCLasses[i].add(temp);
		}
		
		
		 /*
		 * Delete the labels from the partitions.
		 */
		String attributesIndicesForDeletion = "";
		
		for (int k = set.numAttributes() - numberOfLabels + 1; k <= set.numAttributes(); k++) {
			if (k != set.numAttributes())
				attributesIndicesForDeletion += k + ",";
			else
				attributesIndicesForDeletion += k;
		}
		 /* 	attributesIncicesForDeletion = 8,9,10,11,12,13,14 e.g. for 7 attributes and 7 labels. 
		 * It does not start from 7 because it assumes that the user inputs the number. See the api.
		 */
		for (int i = 0; i < partitions.length; i++) {
		     Remove remove = new Remove();
		     remove.setAttributeIndices(attributesIndicesForDeletion);
		     remove.setInvertSelection(false);
		     remove.setInputFormat(partitions[i]);
		     partitions[i] = Filter.useFilter(partitions[i], remove);	
		}
		// partitions now contains only attributes
		
		 /*
		 * delete the attributes from partitionsWithCLasses
		 */
		String labelsIndicesForDeletion = "";

		for (int k = 1; k <= set.numAttributes() - numberOfLabels; k++) {
			if (k != set.numAttributes() - numberOfLabels)
				labelsIndicesForDeletion += k + ",";
			else
				labelsIndicesForDeletion += k;
		}
		 /* 	attributesIncicesForDeletion = 8,9,10,11,12,13,14 e.g. for 7 attributes and 7 labels. 
		 * It does not start from 7 because it assumes that the user inputs the number. See the api.
		 */		
		for (int i = 0; i < partitionsWithCLasses.length; i++) {
		     Remove remove = new Remove();
		     remove.setAttributeIndices(labelsIndicesForDeletion);
		     remove.setInvertSelection(false);
		     remove.setInputFormat(partitionsWithCLasses[i]);
		     partitionsWithCLasses[i] = Filter.useFilter(partitionsWithCLasses[i], remove);	
		     //System.out.println(partitionsWithCLasses[i]);
		}
		// partitionsWithCLasses now contains only labels
	
		int populationSize = (int) SettingsLoader.getNumericSetting("populationSize", 1500);
		
		 // the set used to store the rules from all the clusters
		ClassifierSet initialClassifiers = new ClassifierSet(
															new FixedSizeSetWorstFitnessDeletion(this,
																	 populationSize,
																	 new RouletteWheelSelector(AbstractUpdateStrategy.COMPARISON_MODE_DELETION, true)));

		for (int i = 0; i < partitions.length; i++) {
			
			
			try {
				
				kmeans.setNumClusters((int) Math.ceil(gamma * partitions[i].numInstances()));
				kmeans.buildClusterer(partitions[i]);
//				int[] assignments = kmeans.getAssignments();
				
/*				int k=0;
				for (int j = 0; j < assignments.length; j++) {
					System.out.printf("Instance %d => Cluster %d ", k, assignments[j]);
					k++;
					System.out.println();

				}
				System.out.println();*/
					
				Instances centroids = kmeans.getClusterCentroids();

				int numOfCentroidAttributes = centroids.numAttributes();
				
				

				/*
				 * The centroids in this stage hold only attributes. To continue, we need to provide them the labels.
				 * These are the ones we removed earlier.
				 * But first, open up positions for attributes.
				 * */
				
				for (int j = 0; j < numberOfLabels; j++) {
					Attribute label = new Attribute("label" + j);
					centroids.insertAttributeAt(label, numOfCentroidAttributes + j);
				}
				
				
				for (int centroidInstances = 0; centroidInstances < centroids.numInstances(); centroidInstances++) {
					for (int labels = 0; labels < numberOfLabels; labels++) {
						centroids.instance(centroidInstances).setValue(numOfCentroidAttributes + labels, partitionsWithCLasses[i].instance(0).value(labels));
					}
				}

				//System.out.println(centroids);
				double[][] centroidsArray = InstancesUtility.convertIntancesToDouble(centroids);

				for (int j = 0; j < centroidsArray.length; j++) {
					//System.out.printf("Instance %d => Cluster %d ", k, assignments[j]);
					final Classifier coveringClassifier = this.getClassifierTransformBridge().createRandomCoveringClassifier(centroidsArray[j]);
					
					
					coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_INIT); 
					initialClassifiers.addClassifier(new Macroclassifier(coveringClassifier, 1), false);	
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		//System.out.println(initialClassifiers);
		return initialClassifiers;
	}
	
	
	
	
	/**
	 * Prints the population classifiers of the LCS.
	 */
	public final void printSet() {
		rulePopulation.print();
	}
	
	

	/**
	 * Register an evaluator to be called during training.
	 * 
	 * @param evaluator
	 *            the evaluator to register
	 * @return true if the evaluator has been registered successfully
	 */
	public final boolean registerHook(final ILCSMetric evaluator) {
		return hooks.add(evaluator);
	}

	/**
	 * Registration of hooks to perform periodical inspection using metrics.
	 * 
	 * @param numberOfLabels 
	 *				the dataset's number of labels. 
	 *
	 *@param instances
	 *			the set of instances on which we will evaluate on. (train or test)
	 *
	 * @author alexandros filotheou
	 * 
	 * 
	 * */
	public void registerMultilabelHooks(double[][] instances, int numberOfLabels, int numOfFolds) {
		
		new FileLogger(this, numOfFolds);
				
		this.registerHook(new FileLogger("accuracy",
				new AccuracyRecallEvaluator(instances, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY)));
		
		this.registerHook(new FileLogger("recall",
				new AccuracyRecallEvaluator(instances, false, this, AccuracyRecallEvaluator.TYPE_RECALL)));
		
		this.registerHook(new FileLogger("exactMatch", 
				new ExactMatchEvalutor(instances, false, this)));
		
		this.registerHook(new FileLogger("hamming", 
				new HammingLossEvaluator(instances, false, numberOfLabels, this)));
		
		this.registerHook(new FileLogger("meanFitness",
				new MeanFitnessStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		this.registerHook(new FileLogger("meanCoverage",
				new MeanCoverageStatistic()));
		
		this.registerHook(new FileLogger("weightedMeanCoverage",
				new WeightedMeanCoverageStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		this.registerHook(new FileLogger("meanAttributeSpecificity",
				new MeanAttributeSpecificityStatistic()));
		
		this.registerHook(new FileLogger("weightedMeanAttributeSpecificity",
				new WeightedMeanAttributeSpecificityStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		this.registerHook(new FileLogger("meanLabelSpecificity",
				new MeanLabelSpecificity(numberOfLabels)));
		
		this.registerHook(new FileLogger("weightedMeanLabelSpecificity",
				new WeightedMeanLabelSpecificity(numberOfLabels, AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION)));
		
		
//		if (SettingsLoader.getStringSetting("filename", "").indexOf("position") != -1) {
//
//			this.registerHook(new FileLogger("BAM", new PositionBAMEvaluator
//															((int) SettingsLoader.getNumericSetting("numberOfLabels", 1), 
//																	PositionBAMEvaluator.GENERIC_REPRESENTATION, this))); 
//		}
//		
//		if (SettingsLoader.getStringSetting("filename", "").indexOf("identity") != -1) {
//			this.registerHook(new FileLogger("BAM", new IdentityBAMEvaluator
//															((int) SettingsLoader.getNumericSetting("numberOfLabels", 1), 
//																	IdentityBAMEvaluator.GENERIC_REPRESENTATION, this)));
//		}
		
	}
	
	
	
	/**
	 * Save the rules to the given filename.
	 * 
	 * @param filename
	 */
	public final void saveRules(String filename) {
		ClassifierSet.saveClassifierSet(rulePopulation, filename);
	}

	/**
	 * Constructor.
	 * 
	 * @param bridge
	 *            the classifier transform bridge
	 * @param update
	 *            the update strategy
	 */
	public final void setElements(final ClassifierTransformBridge bridge,
									final AbstractUpdateStrategy update) {
		transformBridge = bridge;
		updateStrategy = update;
	}

	/**
	 * @param rate
	 */
	public void setHookCallbackRate(int rate) {
		hookCallbackRate = rate;
	}

	
	public void setHookedMetricsFileDirectory(String file) {
		hookedMetricsFileDirectory = file;
	}
	
	/**
	 * Sets the LCS's population.
	 * @param population  the new LCS's population
	 */
	public final void setRulePopulation(ClassifierSet population) {
		rulePopulation = population;
	}

	/**
	 * Run the LCS and train it.
	 */
	public abstract void train();

	/**
	 * Train population with all train instances and perform evolution.
	 * 
	 * @param iterations
	 *            the number of full iterations (one iteration the LCS is
	 *            trained with all instances) to train the LCS
	 * @param population
	 *            the population of the classifiers to train.
	 */
	protected final void trainSet(final int iterations,
								    final ClassifierSet population) {
		
		trainSet(iterations, population, true); // evolve = true
	}

	/**
	 * Train a classifier set with all train instances.
	 * 
	 * @param iterations
	 *            the number of full iterations (one iteration the LCS is
	 *            trained with all instances) to train the LCS
	 * @param population
	 *            the population of the classifiers to train.
	 * @param evolve
	 *            set true to evolve population, false to only update it
	 *            
	 *            
	 *            ekteleitai gia iterations fores me evolve = true
	 *            kai (int) 0.1 * iterations fores me evolve = false
	 */
	public final void trainSet(final int iterations,
							     final ClassifierSet population, 
							     final boolean evolve) {

		
		
		final int numInstances = instances.length;

		repetition = 0;
		
		int trainsBeforeHook = 0;
		while (repetition < iterations) { 		
			System.out.print("[");

			while ((trainsBeforeHook < hookCallbackRate) && (repetition < iterations)) { 
				System.out.print('/');													
				
				for (int i = 0; i < numInstances; i++) {
					cummulativeCurrentInstanceIndex = totalRepetition * instances.length + i;
					trainWithInstance(population, i, evolve);
				}

				repetition++;
				totalRepetition++;
				trainsBeforeHook++;

				// check for duplicates on every repetition
				if (!thoroughlyCheckWIthPopulation) {
					assimilateDuplicateClassifiers(rulePopulation, evolve);
				}
			}

			if (hookCallbackRate < iterations) {
				System.out.print("] ");
				System.out.print("(" + repetition + "/" + iterations + ")");
				System.out.println();
			}
			executeCallbacks(population, repetition, evolve); 
			trainsBeforeHook = 0;
		}
	}

	/**
	 * Train with instance main template. Trains the classifier set with a
	 * single instance.
	 * 
	 * @param population
	 *            the classifier's population. olos o plh9usmos dld, [P]
	 * @param dataInstanceIndex
	 *            the index of the training data instance
	 * @param evolve
	 *            whether to evolve the set or just train by updating it
	 */
	public final void trainWithInstance(final ClassifierSet population, final int dataInstanceIndex, final boolean evolve) {
			
//		int index = totalRepetition * instances.length + dataInstanceIndex;
		
		final ClassifierSet matchSet = population.generateMatchSetNew(dataInstanceIndex);
		
		if (UPDATE_MODE == UPDATE_MODE_IMMEDIATE) 
			getUpdateStrategy().updateSet(population, matchSet, dataInstanceIndex, evolve);
		else if (UPDATE_MODE == UPDATE_MODE_HOLD) 
			getUpdateStrategy().updateSetNew(population, matchSet, dataInstanceIndex, evolve);
		
		
//		recordInTimeMeasurements(population, index);
	}

	
//	private void recordInTimeMeasurements(ClassifierSet population, int index) {
//		
////		MeanCoverageStatistic meanCov = new MeanCoverageStatistic();
////		double meanPopulationCoverage = meanCov.getMetric(this);
////		
////		
////		int numberOfMacroclassifiersCovered = 0;
////		int numberOfClassifiersCovered = 0;
////		
////		int numberOfMacroclassifiersGaed = 0;
////		int numberOfClassifiersGaed = 0;
////		
////		int numberOfMacroclassifiersInited = 0;
////		int numberOfClassifiersInited = 0;
////		
////		int numberOfSubsumptions = 0;
////		
////		double meanNs = 0;
////		
////		double meanAcc = 0;
////		double meanCoveredAcc = 0;
////		double meanGaedAcc = 0;
////		
////		double meanExplorationFitness = 0;
////		double meanCoveredExplorationFitness = 0;
////		double meanGaedExplorationFitness = 0;
////		
////		double meanPureFitness = 0;
////		double meanCoveredPureFitness = 0;
////		double meanGaedPureFitness = 0;
////		
////		for (int i = 0; i < population.getNumberOfMacroclassifiers(); i++) {
////			
////			Macroclassifier macro = population.getMacroclassifiersVector().get(i);
////			numberOfSubsumptions +=  macro.numberOfSubsumptions;
////			
////			if (macro.myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER) {
////				numberOfMacroclassifiersCovered++;
////				numberOfClassifiersCovered += macro.numerosity;
////			}
////			else if (macro.myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
////				numberOfMacroclassifiersGaed++;
////				numberOfClassifiersGaed += macro.numerosity;
////			}
////			else if (macro.myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT) {
////				numberOfMacroclassifiersInited++;
////				numberOfClassifiersInited += macro.numerosity;
////			}
////			
////			meanAcc += 					macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
////			meanExplorationFitness += 	macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
////			meanPureFitness += 			macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
////			meanNs += population.getClassifier(i).getNs();
////			
////			if (macro.myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER || macro.myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT) {
////				
////				meanCoveredAcc += 					macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
////				meanCoveredExplorationFitness += 	macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
////				meanCoveredPureFitness += 			macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
////			}
////			else if (macro.myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
////				
////				meanGaedAcc += 					macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
////				meanGaedExplorationFitness += 	macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
////				meanGaedPureFitness += 			macro.numerosity * macro.myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
////
////			}
////		}
////		
////		meanAcc /= population.getTotalNumerosity();
////		meanNs /= population.getNumberOfMacroclassifiers();
////		meanCoveredAcc /= (numberOfClassifiersCovered + numberOfClassifiersInited);
////		meanGaedAcc /= numberOfClassifiersGaed;
////		
////		meanExplorationFitness /= population.getTotalNumerosity();
////		meanCoveredExplorationFitness/= (numberOfClassifiersCovered + numberOfClassifiersInited);
////		meanGaedExplorationFitness /= numberOfClassifiersGaed;
////
////		meanPureFitness /= population.getTotalNumerosity();
////		meanCoveredPureFitness /= (numberOfClassifiersCovered + numberOfClassifiersInited);
////		meanGaedPureFitness /= numberOfClassifiersGaed;
//		
//		timeMeasurements[index][2] = (int) population.firstDeletionFormula;
//		timeMeasurements[index][3] = (int) population.secondDeletionFormula;
//
//	}
	
	/**
	 * Unregister an evaluator.
	 * 
	 * @param evaluator
	 *            the evaluator to register
	 * @return true if the evaluator has been unregisterd successfully
	 */
	public final boolean unregisterEvaluator(final ILCSMetric evaluator) {
		return hooks.remove(evaluator);
	}

	/**
	 * Update population with all train instances but do not perform evolution.
	 * 
	 * @param iterations
	 *            the number of full iterations (one iteration the LCS is
	 *            trained with all instances) to update the LCS
	 * @param population
	 *            the population of the classifiers to update.
	 */
	public final void updatePopulation(final int iterations,
									   final ClassifierSet population) {
		
		trainSet(iterations, population, false); // evolve = false
	}
	
	

	

	

}