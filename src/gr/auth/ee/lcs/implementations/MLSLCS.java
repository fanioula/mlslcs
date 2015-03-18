/*
 *	Copyright (C) 2011 by F. Tzima, M. Allamanis and A. Filotheou
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
package gr.auth.ee.lcs.implementations;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.calibration.InternalValidation;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.populationcontrol.FixedSizeSetWorstFitnessDeletion;
import gr.auth.ee.lcs.classifiers.statistics.MeanCoverageStatistic;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.data.representations.complex.GenericMultiLabelRepresentation;
import gr.auth.ee.lcs.data.representations.complex.GenericMultiLabelRepresentation.VotingClassificationStrategy;
import gr.auth.ee.lcs.data.updateAlgorithms.MLSLCSUpdateAlgorithm;
import gr.auth.ee.lcs.evaluators.AccuracyRecallEvaluator;
import gr.auth.ee.lcs.evaluators.ExactMatchEvalutor;
import gr.auth.ee.lcs.evaluators.HammingLossEvaluator;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.geneticalgorithm.algorithms.SteadyStateGeneticAlgorithm;
import gr.auth.ee.lcs.geneticalgorithm.operators.MultiPointCrossover;
import gr.auth.ee.lcs.geneticalgorithm.operators.SinglePointCrossover;
import gr.auth.ee.lcs.geneticalgorithm.operators.UniformBitMutation;
import gr.auth.ee.lcs.geneticalgorithm.selectors.RouletteWheelSelector;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import weka.core.Instances;

/**
 * An implementation of the MLS-LCS algorithm
 * 
 * @author F. Tzima and M. Allamanis and A. Filotheou
 * 
 */
public class MLSLCS extends AbstractLearningClassifierSystem {

	/**
	 * The input file used (.arff).
	 */
	private final String inputFile;

	/**
	 * The number of full iterations to train the UCS.
	 */
	private final int iterations;

	/**
	 * The size of the population to use.
	 */
	private final int populationSize;

	/**
	 * The GA crossover rate.
	 */
	private final float CROSSOVER_RATE = (float) SettingsLoader.getNumericSetting("crossoverRate", .8);

	/**
	 * The GA mutation rate.
	 */
	private final double MUTATION_RATE = (float) SettingsLoader.getNumericSetting("mutationRate", .04);

	/**
	 * The GA activation rate.
	 */
	private final int THETA_GA = (int) SettingsLoader.getNumericSetting("thetaGA", 100);

	/**
	 * The number of bits to use for representing continuous variables.
	 */
	private final int PRECISION_BITS = (int) SettingsLoader.getNumericSetting("precisionBits", 5);

	/**
	 * The UCS n power parameter.
	 */
	private final int N = (int) SettingsLoader.getNumericSetting("N", 10);

	/**
	 * The accuracy threshold parameter.
	 */
	private final double ACC0 = SettingsLoader.getNumericSetting("Acc0", .99);

	/**
	 * The UCS experience threshold.
	 */
	private final int EXPERIENCE_THRESHOLD = (int) SettingsLoader.getNumericSetting("THETA_EXP", 10);
	
	/**
	 * The attribute generalization rate.
	 */
	private final double ATTRIBUTE_GENERALIZATION_RATE = SettingsLoader.getNumericSetting("AttributeGeneralizationRate", 0.33);
	
	/**
	 * The attribute generalization rate when clustering
	 */
	private final double CLUSTERING_ATTRIBUTE_GENERALIZATION_RATE = SettingsLoader.getNumericSetting("ClusteringAttributeGeneralizationRate", 0);

	/**
	 * Percentage of only updates (and no exploration).
	 */
	private final double UPDATE_ONLY_ITERATION_PERCENTAGE = SettingsLoader.getNumericSetting("UpdateOnlyPercentage", .1);

	/**
	 * The label generalization rate.
	 */
	private final double LABEL_GENERALIZATION_RATE = SettingsLoader.getNumericSetting("LabelGeneralizationRate", 0.33);
	
	/**
	 * The label generalization rate when clustering.
	 */
	private final double CLUSTERING_LABEL_GENERALIZATION_RATE = SettingsLoader.getNumericSetting("ClusteringLabelGeneralizationRate", 0);
	
	private final int GENETIC_ALGORITHM_SELECTION = (int) SettingsLoader.getNumericSetting("gaSelection", 0);
	
	private final int CROSSOVER_OPERATOR = (int) SettingsLoader.getNumericSetting("crossoverOperator", 0);	


	/**
	 * The number of labels used at the dmlUCS.
	 */
	private final int numberOfLabels;

	/**
	 * The problem representation.
	 */
	private final GenericMultiLabelRepresentation rep;

	private String deletionsFile;
	
	private String zeroCoverageFile;
	
	long accEvalTime1;
	long recEvalTime1;
	long hamEvalTime1;
	long testEvalTime1;
	
	long accEvalTime2;
	long recEvalTime2;
	long hamEvalTime2;
	long testEvalTime2;
	
	long accEvalTime3;
	long recEvalTime3;
	long hamEvalTime3;
	long testEvalTime3;
	
	long bestClassificationModeTime;
	long proportionalCutCalibrationTime;
	long internalValidationCalibrationTime;
	
	long getConfidenceArrayTime;
	long calibrateTime;
	

	
	
	/**
	 * Constructor.
	 * 
	 * @throws IOException
	 */
	public MLSLCS() throws IOException {
		
		inputFile = SettingsLoader.getStringSetting("filename", "");
		numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		iterations = (int) SettingsLoader.getNumericSetting("trainIterations",1000);
		populationSize = (int) SettingsLoader.getNumericSetting("populationSize", 1500);
		
		boolean decideAtOnceForCrossOver = true;
		if (GENETIC_ALGORITHM_SELECTION ==0)
			decideAtOnceForCrossOver = false;
		
		final IGeneticAlgorithmStrategy ga = new SteadyStateGeneticAlgorithm(
			new RouletteWheelSelector(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION, true), 
			CROSSOVER_OPERATOR == 0 ? new SinglePointCrossover(this) : new MultiPointCrossover(this), 
			CROSSOVER_RATE,
			new UniformBitMutation(MUTATION_RATE), 
			THETA_GA, 
			this, decideAtOnceForCrossOver);
		
		rep = new GenericMultiLabelRepresentation(inputFile, 
												  PRECISION_BITS,
												  numberOfLabels, 
												  GenericMultiLabelRepresentation.EXACT_MATCH,
												  LABEL_GENERALIZATION_RATE, 
												  ATTRIBUTE_GENERALIZATION_RATE,
												  CLUSTERING_LABEL_GENERALIZATION_RATE,
												  CLUSTERING_ATTRIBUTE_GENERALIZATION_RATE,
												  this);
		
		rep.setClassificationStrategy(rep.new BestFitnessClassificationStrategy());

		MLSLCSUpdateAlgorithm strategy = new MLSLCSUpdateAlgorithm(N, 
																		 ACC0,
																	     EXPERIENCE_THRESHOLD, 
																		 ga,
																		 numberOfLabels,
																		 this);
		this.setElements(rep, strategy);

		rulePopulation = new ClassifierSet(
											new FixedSizeSetWorstFitnessDeletion(this,
																				 populationSize,
																				 new RouletteWheelSelector(AbstractUpdateStrategy.COMPARISON_MODE_DELETION, true)));
	}

	@Override
	public int[] classifyInstance(double[] instance) {
		
		return getClassifierTransformBridge().classify(this.getRulePopulation(), instance);
	}

	@Override
	public AbstractLearningClassifierSystem createNew() {
		try {
			return new MLSLCS();
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	@Override
	public String[] getEvaluationNames() {
		final String[] names = { "Accuracy(pcut)", 
								 "Recall(pcut)",
								 "HammingLoss(pcut)", 
								 "ExactMatch(pcut)", 
								 "Accuracy(ival)",
								 "Recall(ival)", 
								 "HammingLoss(ival)", 
								 "ExactMatch(ival)",
								 "Accuracy(best)", 
								 "Recall(best)", 
								 "HammingLoss(best)",
								 "ExactMatch(best)",
								 "Mean Coverage"
								 };
		return names;
	}

	@Override
	public double[] getEvaluations(Instances testSet) {
		

		final double[] results = new double[13];
		Arrays.fill(results, 0);

		proportionalCutCalibrationTime = -System.currentTimeMillis();
		final VotingClassificationStrategy pcut = proportionalCutCalibration();
		proportionalCutCalibrationTime += System.currentTimeMillis();
		
		System.out.println("Threshold (pcut) set to " + pcut.getThreshold());
		
		
		
		accEvalTime1 = -System.currentTimeMillis();
		final AccuracyRecallEvaluator accEval = new AccuracyRecallEvaluator(testSet, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY);
		results[0] = accEval.getMetric(this);
		accEvalTime1 += System.currentTimeMillis();


		recEvalTime1 = -System.currentTimeMillis();
		final AccuracyRecallEvaluator recEval = new AccuracyRecallEvaluator(testSet, false, this, AccuracyRecallEvaluator.TYPE_RECALL);
		results[1] = recEval.getMetric(this);
		recEvalTime1 += System.currentTimeMillis();


		hamEvalTime1 = -System.currentTimeMillis();
		final HammingLossEvaluator hamEval = new HammingLossEvaluator(testSet, false, numberOfLabels, this);
		results[2] = hamEval.getMetric(this);
		hamEvalTime1 += System.currentTimeMillis();

		testEvalTime1 = -System.currentTimeMillis();
		final ExactMatchEvalutor testEval = new ExactMatchEvalutor(testSet, false, this);
		results[3] = testEval.getMetric(this);
		testEvalTime1 += System.currentTimeMillis();
		
		
		final AccuracyRecallEvaluator selfAcc = new AccuracyRecallEvaluator(instances, false, this, AccuracyRecallEvaluator.TYPE_ACCURACY);
		
		internalValidationCalibrationTime = -System.currentTimeMillis();
		internalValidationCalibration(selfAcc);
		internalValidationCalibrationTime += System.currentTimeMillis();

		accEvalTime2 = -System.currentTimeMillis();
		results[4] = accEval.getMetric(this);
		accEvalTime2 += System.currentTimeMillis();

		recEvalTime2 = -System.currentTimeMillis();
		results[5] = recEval.getMetric(this);
		recEvalTime2 += System.currentTimeMillis();
		
		hamEvalTime2 = -System.currentTimeMillis();
		results[6] = hamEval.getMetric(this);
		hamEvalTime2 += System.currentTimeMillis();
		
		testEvalTime2 = -System.currentTimeMillis();		
		results[7] = testEval.getMetric(this);
		testEvalTime2 += System.currentTimeMillis();


		bestClassificationModeTime = -System.currentTimeMillis();
		useBestClassificationMode();
		bestClassificationModeTime += System.currentTimeMillis();
		

		accEvalTime3 = -System.currentTimeMillis();
		results[8] = accEval.getMetric(this);
		accEvalTime3 += System.currentTimeMillis();
		
		recEvalTime3 = -System.currentTimeMillis();
		results[9] = recEval.getMetric(this);
		recEvalTime3 += System.currentTimeMillis();
		
		hamEvalTime3 = -System.currentTimeMillis();
		results[10] = hamEval.getMetric(this);
		hamEvalTime3 += System.currentTimeMillis();
		
		testEvalTime3 = -System.currentTimeMillis();
		results[11] = testEval.getMetric(this);
		testEvalTime3 += System.currentTimeMillis();
		
		
		MeanCoverageStatistic meanCoverage = new MeanCoverageStatistic();
		results[12] = meanCoverage.getMetric(this);
		
		String testTimes = this.hookedMetricsFileDirectory + "/testTimes.txt";
		
		
		try {
			final FileWriter fstream = new FileWriter(testTimes, false);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			buffer.write("");
			buffer.flush();
			buffer.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		
		
		
		try {
			final FileWriter fstream = new FileWriter(testTimes, true);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			
			buffer.write(String.valueOf(proportionalCutCalibrationTime));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(getConfidenceArrayTime));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(calibrateTime));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(accEvalTime1));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(recEvalTime1));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(hamEvalTime1));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(testEvalTime1));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(System.getProperty("line.separator"));
			
			buffer.write(String.valueOf(internalValidationCalibrationTime));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(accEvalTime2));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(recEvalTime2));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(hamEvalTime2));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(testEvalTime2));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(System.getProperty("line.separator"));
			
			buffer.write(String.valueOf(bestClassificationModeTime));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(accEvalTime3));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(recEvalTime3));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(hamEvalTime3));
			buffer.write(System.getProperty("line.separator"));
			buffer.write(String.valueOf(testEvalTime3));
			buffer.write(System.getProperty("line.separator"));
			
			buffer.flush();
			buffer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		
		
		/*results[] =
		 * 
		 * |__________________Pcut________________|__________________Ival________________|__________________Best________________|
		 * |accuracy|recall|hammingDist|exactMatch|accuracy|recall|hammingDist|exactMatch|accuracy|recall|hammingDist|exactMatch|
		 * 
		 * */
		return results;
	}
	
	
	public void internalValidationCalibration(ILCSMetric selfAcc) {
		
		final VotingClassificationStrategy str = rep.new VotingClassificationStrategy();
		
		
		rep.setClassificationStrategy(str);
		
		final InternalValidation ival = new InternalValidation(this, str, selfAcc);
		ival.calibrate(10);
	}

	public VotingClassificationStrategy proportionalCutCalibration() {
		
		final VotingClassificationStrategy str = rep.new VotingClassificationStrategy();
		
		rep.setClassificationStrategy(str);

		str.proportionalCutCalibration(this.instances, rulePopulation);
		
		getConfidenceArrayTime = str.getConfidenceArrayTime;
		calibrateTime = str.calibrateTime;
		
		return str;
	}

	/**
	 * Runs the Direct-ML-UCS.
	 * 
	 */
	@Override
	public void train() {
		
//		timeMeasurements =  new double[(iterations + (int)(iterations * UPDATE_ONLY_ITERATION_PERCENTAGE)) * instances.length][5];
		
		trainSet(iterations, rulePopulation);
		
		updatePopulation((int) (iterations * UPDATE_ONLY_ITERATION_PERCENTAGE), rulePopulation);
		
		deletionsFile = this.hookedMetricsFileDirectory + "/deletions.txt";
		zeroCoverageFile = this.hookedMetricsFileDirectory + "/zeroCoverage.txt";	
		
		try {
			final FileWriter fstream3 = new FileWriter(deletionsFile, false);
			final FileWriter fstream4 = new FileWriter(zeroCoverageFile, false);

			final BufferedWriter buffer3 = new BufferedWriter(fstream3);
			final BufferedWriter buffer4 = new BufferedWriter(fstream4);
			
			buffer3.write("");
			buffer3.flush();
			buffer3.close();
			
			buffer4.write("");
			buffer4.flush();
			buffer4.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		

		
		try {
			final FileWriter fstream = new FileWriter(hookedMetricsFileDirectory + "/deletions.txt", true);
			final BufferedWriter buffer = new BufferedWriter(fstream);


			for (int i = 0 ; i < qualityIndexOfDeleted.size(); i++ ){
				buffer.write(
							   qualityIndexOfDeleted.elementAt(i) 
							 + "	" 
							 + accuracyOfDeleted.elementAt(i) 
							 + "	"
							 + iteration.elementAt(i)
							 + "	"
							 + originOfDeleted.elementAt(i)
							 + "	"
							 + accuracyOfCoveredDeletion.elementAt(i)
							 + "	"
							 + accuracyOfGaedDeletion.elementAt(i)
							 + "	"
							 + qualityIndexOfClassifiersCoveredDeleted.elementAt(i)
							 + "	"
							 + qualityIndexOfClassifiersGaedDeleted.elementAt(i)
							 + System.getProperty("line.separator"));
			}
			buffer.flush();
			buffer.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		
		
		
		try {
			
			final FileWriter fstream = new FileWriter(hookedMetricsFileDirectory + "/zeroCoverage.txt", true);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			for (int i = 0 ; i < rulePopulation.zeroCoverageVector.size(); i++ ){
				buffer.write(
							rulePopulation.zeroCoverageVector.elementAt(i)	
							+ "		"
							+ rulePopulation.zeroCoverageIterations.elementAt(i)
						   	+ System.getProperty("line.separator"));
			}
			buffer.flush();
			buffer.close();
		} 
		
		catch (Exception e) {
			e.printStackTrace();
		}
		

	}
	
	

	public void useBestClassificationMode() {
		rep.setClassificationStrategy(rep.new BestFitnessClassificationStrategy());
	}
	


}