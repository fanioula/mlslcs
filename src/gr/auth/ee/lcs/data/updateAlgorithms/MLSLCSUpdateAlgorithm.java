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
package gr.auth.ee.lcs.data.updateAlgorithms;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Vector;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.classifiers.statistics.MeanFitnessStatistic;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.utilities.SettingsLoader;

public class MLSLCSUpdateAlgorithm extends AbstractUpdateStrategy  {
	/**
	 * The experience threshold for subsumption.
	 */
	protected int subsumptionExperienceThreshold;
	
	/**
	 * The fitness threshold for subsumption.
	 */
	protected double subsumptionFitnessThreshold;
	
	/**
	 * A data object for the MLSLCS update algorithms.
	 * 
	 */
	final static class MLSLCSClassifierData implements Serializable {

		/**
		 * 
		 */
		private static final long serialVersionUID = 2584696442026755144L;

		/**
		 * d refers to the paper's d parameter in deletion possibility
		 */
		
		public double d = 0;
		
		/**
		 * The classifier's fitness
		 */
		public double fitness = 1;//Double.MIN_NORMAL; //.5;

		/**
		 * niche set size estimation.
		 */
		public double ns = 1; //20;

		/**
		 * Match Set Appearances.
		 */
		public double msa = 0;

		/**
		 * true positives.
		 */
		public double tp = 0;
		
		/**
		 * totalFitness = numerosity * fitness
		 */
		
		public double totalFitness = 1;
		
		
		// k for fitness sharing
		public double k = 0;
		
		public int minCurrentNs = 0;
		
		
		@Override
		public String toString(){
			return 	 "d = " + d 
					+ " fitness = " + fitness
					+ " ns = " + ns
					+ " msa= " + msa
					+ "tp = " + tp
					+ " minCurrentNs = " + minCurrentNs;
		} 
						
	}
	
	
	/**
	 * The way to choose the fitness calculation formula.
	 * 
	 * 0: Simple = (acc)^n
	 * 1: Complex = F + beta * (num * (acc)^n - F)
	 * 2: Sharing (see updateSet() and udpateSetNew() methods)
	 * 
	 * */
	public static final int FITNESS_MODE_SIMPLE 	= 0;
	public static final int FITNESS_MODE_COMPLEX 	= 1;
	public static final int FITNESS_MODE_SHARING 	= 2;
	public final int FITNESS_MODE = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);
	
	/**
	 * The way to choose the formula for computing core deletion probabilities.
	 * 
	 * 0: (cl.exp > THETA_DEL) && (cl.fitness < DELTA * meanPopulationFitness) ? cl.cs * (meanFitness / cl.fitness) : cl.cs
	 * 1: (cl.exp > THETA_DEL) && (cl.fitness < DELTA * meanPopulationFitness) ? cl.cs * (meanFitness / Math.pow(cl.fitness,N)) : cl.cs
	 * 2: (cl.exp < THETA_DEL) ? 1 / (cl.fitness * DELTA) : 1 / (cl.fitness * Math.exp(-cl.cs + 1))
	 * 3: (cl.exp < THETA_DEL) ?  Math.exp(1 / cl.fitness) : Math.exp(cl.cs - 1) / cl.fitness
	 * 
	 **/
	public static final int DELETION_MODE_DEFAULT = 0;
	public static final int DELETION_MODE_POWER = 1;
	public static final int DELETION_MODE_ICANNGA = 2;
	public static final int DELETION_MODE_JOURNAL = 3;
	public final int DELETION_MODE = (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0);
	
	/**
	 * The delta parameter used in determining the formula of possibility of deletion
	 */
	public static double DELTA = SettingsLoader.getNumericSetting("DELTA", .1);

	public static double ACC_0 = SettingsLoader.getNumericSetting("Acc0", .99);
	
	public static double a = SettingsLoader.getNumericSetting("Alpha", .1);
			
	/**
	 * do classifiers that don't decide clearly for the label, participate in the correct sets?
	 * */
	public final boolean wildCardsParticipateInCorrectSets = SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "false").equals("true");
	
	
	/** 
	if wildCardsParticipateInCorrectSets is true, and balanceCorrectSets is also true, control the population of the correct sets 
	by examining the numerosity of a correct set comprising only with wildcards against that of a correct set without them.
	if [C#only] <= wildCardParticipationRatio * [C!#], the correct set consists of wildcards AND non-wildcard rules 
	*/

	public final boolean balanceCorrectSets = SettingsLoader.getStringSetting("balanceCorrectSets", "false").equals("true");
	
	public final double wildCardParticipationRatio = SettingsLoader.getNumericSetting("wildCardParticipationRatio", 1);
	
	/**
	 * The learning rate.
	 */
	private final double LEARNING_RATE = SettingsLoader.getNumericSetting("beta", 0.2);
	
	
	/**
	 * The theta_del parameter.
	 */
	public static int THETA_DEL = (int) SettingsLoader.getNumericSetting("THETA_DEL", 20);
	
	
	/**
	 * The MLUCS omega parameter.
	 */	
	private final double OMEGA = SettingsLoader.getNumericSetting("OMEGA", 0.9);
	
	/**
	 * The MLUCS phi parameter.
	 */	
	private final double PHI =  SettingsLoader.getNumericSetting("PHI", 1);


	/**
	 * The LCS instance being used.
	 */
	protected final AbstractLearningClassifierSystem myLcs;

	/**
	 * Genetic Algorithm.
	 */
	public final IGeneticAlgorithmStrategy ga;

	/**
	 * Number of labels used.
	 */
	protected final int numberOfLabels;

	/**
	 * The n dumping factor for acc.
	 */
	protected final double n;
	
	/**
	 *  holds the classifiers' indices in the match set with the lowest coverage. used when deleting from [M]
	 * */
	private ArrayList <Integer> lowestCoverageIndices;
	
	private boolean commencedDeletions = false;
		
	
//	private int numberOfEvolutionsConducted;
//	
//	private int numberOfDeletionsConducted;
//	
//	private int numberOfSubsumptionsConducted;
//	
//	private int numberOfNewClassifiers;
//	
//	private long evolutionTime;
//	
//	private long subsumptionTime;
//	
//	private long deletionTime;
//	
//	private long generateCorrectSetTime;
//	
//	private long updateParametersTime;
//	
//	private long selectionTime;
	
	private String algorithmName;
	
	
	public MLSLCSUpdateAlgorithm(final double nParameter,
			final double fitnessThreshold, final int experienceThreshold,
			IGeneticAlgorithmStrategy geneticAlgorithm, int labels,
			AbstractLearningClassifierSystem lcs) {

		this.subsumptionFitnessThreshold = fitnessThreshold;
		this.subsumptionExperienceThreshold = experienceThreshold;
		this.algorithmName = getClass().getName();
		myLcs = lcs;
		numberOfLabels = labels;
		n = nParameter;
		ga = geneticAlgorithm;
		
		lowestCoverageIndices = new ArrayList <Integer>();

		System.out.println("Update algorithm states: ");
		System.out.println("fitness mode: 	" + FITNESS_MODE);
		System.out.println("deletion mode: 	" + DELETION_MODE);
		System.out.println("update algorithm: " + algorithmName);
		System.out.print("# => [C] " + wildCardsParticipateInCorrectSets);
		if (wildCardsParticipateInCorrectSets)
			System.out.println(", balance [C]: " + balanceCorrectSets + "\n");
		else
			System.out.println("\n");

	}
	
	/**
	 * This method provides a centralized point for computing each classifier's deletion probability
	 * because it is being called by two separate methods, computeDeletionProbabilities and computeDeletionProbabilitiesSmp
	 * 
	 * */
	protected void computeCoreDeletionProbabilities (final Macroclassifier cl, 
													final MLSLCSClassifierData data,
													final double meanFitness) {
		
		commencedDeletions = true;
		if (DELETION_MODE == DELETION_MODE_DEFAULT) {
			data.d = data.ns * ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanFitness) ? 
					meanFitness / data.fitness : 1);	
		}
		else if (DELETION_MODE == DELETION_MODE_POWER) {
			data.d = data.ns * ((cl.myClassifier.experience > THETA_DEL) && (data.fitness < DELTA * meanFitness) ? 
					meanFitness / Math.pow(data.fitness,n) : 1);	
		}
		else if (DELETION_MODE == DELETION_MODE_ICANNGA) {
			data.d = 1 / (data.fitness * ((cl.myClassifier.experience < THETA_DEL) ? DELTA : Math.exp(-data.ns  + 1)) );
		}
		else if (DELETION_MODE == DELETION_MODE_JOURNAL) {
			if (cl.myClassifier.experience < THETA_DEL) 
				data.d = Math.exp(1 / data.fitness) ;
			else 
				data.d = Math.exp(data.ns - 1) / data.fitness;
		}
	}
	
	

	/**
	 * 
	 * For every classifier, compute its deletion probability.
	 * 
	 * @param aSet
	 * 			the classifierset of which the classifiers' deletion probabilities we will compute
	 * */
	@Override
	public void computeDeletionProbabilities (ClassifierSet aSet) {

		
		final int numOfMacroclassifiers = aSet.getNumberOfMacroclassifiers();
				
		MeanFitnessStatistic meanFit = new MeanFitnessStatistic(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		double meanPopulationFitness = meanFit.getMetric(myLcs);
		

		/* update the d parameter, employed in the deletion mechanism, for each classifier in the match set, {currently population-wise} due to the change in 
		 * the classifiers's numerosities, niches' sizes, fitnesses and the mean fitness of the population
		 */
		for (int i = 0; i < numOfMacroclassifiers; i++) {
			
			final Macroclassifier cl = aSet.getMacroclassifier(i);
			final MLSLCSClassifierData data = (MLSLCSClassifierData) cl.myClassifier.getUpdateDataObject();
			
			computeCoreDeletionProbabilities(cl, data, meanPopulationFitness);
			
		}	
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#cover(gr.auth.ee.lcs.classifiers
	 * .ClassifierSet, int)
	 */
	@Override
	public void cover(ClassifierSet population, 
					    int instanceIndex) {
		
		final Classifier coveringClassifier = myLcs.getClassifierTransformBridge()
											  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);
		
		coveringClassifier.created = myLcs.totalRepetition;
		
		coveringClassifier.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();
		
		coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_COVER);
		myLcs.numberOfCoversOccured ++ ;
		population.addClassifier(new Macroclassifier(coveringClassifier, 1), false);
	}
	
	
	private Macroclassifier coverNew( int instanceIndex ) {
		
		final Classifier coveringClassifier = myLcs.getClassifierTransformBridge()
		  									  .createRandomCoveringClassifier(myLcs.instances[instanceIndex]);

		coveringClassifier.created = myLcs.totalRepetition;//ga.getTimestamp();
		
		coveringClassifier.cummulativeInstanceCreated = myLcs.getCummulativeCurrentInstanceIndex();
		
		coveringClassifier.setClassifierOrigin(Classifier.CLASSIFIER_ORIGIN_COVER);
		myLcs.numberOfCoversOccured ++ ;
		return new Macroclassifier(coveringClassifier, 1);
	}
	

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObject()
	 * */
	@Override				

	public Serializable createStateClassifierObject() {
		return new MLSLCSClassifierData();
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#createStateClassifierObjectArray()
	 * */
	@Override	
	public Serializable[] createClassifierObjectArray() {
		
		MLSLCSClassifierData classifierObjectArray[] = new MLSLCSClassifierData[(int) SettingsLoader.getNumericSetting("numberOfLabels", 1)];
		for (int i = 0; i < numberOfLabels; i++) {
			classifierObjectArray[i] = new MLSLCSClassifierData();
		}
		return classifierObjectArray;
	}
	
	
	
	/**
	 * Generates the correct set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @param labelIndex
	 *            the label index
	 * @return the correct set
	 */
	private ClassifierSet generateLabelCorrectSet(final ClassifierSet matchSet,
												   final int instanceIndex, 
												   final int labelIndex) {
		
		final ClassifierSet correctSet = new ClassifierSet(null);
		final ClassifierSet correctSetOnlyWildcards = new ClassifierSet(null);
		final ClassifierSet correctSetWithoutWildcards = new ClassifierSet(null);

		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		
		for (int i = 0; i < matchSetSize; i++) {
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i);
			
			if (wildCardsParticipateInCorrectSets) {
				
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) >= 0) // change: (=) means # => [C]
					correctSet.addClassifier(cl, false);
				
				if (balanceCorrectSets) {
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) == 0) 
						correctSetOnlyWildcards.addClassifier(cl, false);
					
					if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
						correctSetWithoutWildcards.addClassifier(cl, false);
				}
			}
			else 
				if (cl.myClassifier.classifyLabelCorrectly(instanceIndex, labelIndex) > 0)
					correctSet.addClassifier(cl, false);

		}
		
		if (wildCardsParticipateInCorrectSets && balanceCorrectSets) {
			int correctSetWithoutWildcardsNumerosity = correctSetWithoutWildcards.getNumberOfMacroclassifiers();
			int correctSetOnlyWildcardsNumerosity = correctSetOnlyWildcards.getNumberOfMacroclassifiers();
	
			if (correctSetOnlyWildcardsNumerosity <= wildCardParticipationRatio * correctSetWithoutWildcardsNumerosity)
				return correctSet;
			else	
				return correctSetWithoutWildcards;
		}
		
		else return correctSet;
	}
	

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int)
	 */
	@Override
	public double getComparisonValue(Classifier aClassifier, int mode) {
				
		final MLSLCSClassifierData data = (MLSLCSClassifierData) aClassifier.getUpdateDataObject();
		
		switch (mode) {
		case COMPARISON_MODE_EXPLORATION:
			return aClassifier.experience < THETA_DEL ? 0 : data.fitness;
		case COMPARISON_MODE_DELETION:
			return data.d;
		
		case COMPARISON_MODE_EXPLOITATION:
			return Double.isNaN(data.tp / data.msa) ? 0 : data.tp / data.msa;
			
		case COMPARISON_MODE_PURE_FITNESS:
			return data.fitness;
			
		case COMPARISON_MODE_PURE_ACCURACY:
			return Double.isNaN(data.tp / data.msa) ? 0 : data.tp / data.msa;
		
		case COMPARISON_MODE_ACCURACY:
			return (aClassifier.objectiveCoverage < 0) ? 2.0 : data.tp / data.msa;

		default:
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#getData(gr.auth.ee.lcs.classifiers
	 * .Classifier)
	 */
	@Override
	public String getData(Classifier aClassifier) {
		
		final MLSLCSClassifierData data = ((MLSLCSClassifierData) aClassifier.getUpdateDataObject());
		
        DecimalFormat df = new DecimalFormat("#.####");

		return  "tp:|" + df.format(data.tp)  + "|"
				+ "msa:|" + df.format(data.msa)  + "|"
				+ "ns:|" + df.format(data.ns) + "|";
//				+ "d:|" + df.format(data.d) + "|";
	}

	
	@Override
	public double getNs (Classifier aClassifier) {
		final MLSLCSClassifierData data = (MLSLCSClassifierData) aClassifier.getUpdateDataObject();
		return data.ns;
	}
	
	@Override
	public double getAccuracy (Classifier aClassifier) {
		final MLSLCSClassifierData data = (MLSLCSClassifierData) aClassifier.getUpdateDataObject();
		return (Double.isNaN(data.tp / data.msa) ? 0.0 : data.tp / data.msa);
	}
	
	
	
	@Override
	public void inheritParentParameters(Classifier parentA, 
										 Classifier parentB,
										 Classifier child) {
		
		final MLSLCSClassifierData childData = ((MLSLCSClassifierData) child.getUpdateDataObject());
		
		childData.ns = 1;
		child.setComparisonValue(COMPARISON_MODE_EXPLORATION, 1);
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#performUpdate(gr.auth.ee.lcs
	 * .classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public void performUpdate(ClassifierSet matchSet, ClassifierSet correctSet) {
		// Nothing here!
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.AbstractUpdateStrategy#setComparisonValue(gr.auth
	 * .ee.lcs.classifiers.Classifier, int, double)
	 */
	@Override
	public void setComparisonValue(Classifier aClassifier, 
									int mode,
									double comparisonValue) {
		
		final MLSLCSClassifierData data = ((MLSLCSClassifierData) aClassifier.getUpdateDataObject());
		data.fitness = comparisonValue;
	}
	
	
	
	
	/**
	 * Share a the fitness among a set.
	 * 
	 * @param matchSet
	 * 			the match set
	 * 
	 * @param labelCorrectSet
	 *           a correct set in which we share fitness
	 *            
	 * @param l
	 * 			 the index of the label for which the labelCorrectSet is formed
	 * 
	 * @param instanceIndex
	 * 			the index of the instance           
	 * 
	 * @author A. Filotheou
	 * 
	 */
	private void shareFitness(final ClassifierSet matchSet, 
								final ClassifierSet labelCorrectSet,
								final int l,
								int instanceIndex) {
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

		double relativeAccuracy = 0;
		
		for (int i = 0; i < matchSetSize; i++) { 
			
			final Macroclassifier cl = matchSet.getMacroclassifier(i); 
			final MLSLCSClassifierData dataArray[] = (MLSLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
			final MLSLCSClassifierData data = (MLSLCSClassifierData) cl.myClassifier.getUpdateDataObject();

			// Get classification ability for label l. 
			final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
			final int labelNs = labelCorrectSet.getTotalNumerosity();
			
			// update true positives, msa and niche set size
			if (classificationAbility == 0) {

				dataArray[l].tp += OMEGA;
				dataArray[l].msa += PHI;
				
				data.tp += OMEGA;
				data.msa += PHI;
				
				if (wildCardsParticipateInCorrectSets) {
					
					dataArray[l].minCurrentNs = Integer.MAX_VALUE;

					if (dataArray[l].minCurrentNs > labelNs) 
						dataArray[l].minCurrentNs = labelNs;

					if ((dataArray[l].tp / dataArray[l].msa) > ACC_0) {
						dataArray[l].k = 1;
					}
					else {
						dataArray[l].k = a * Math.pow(((dataArray[l].tp / dataArray[l].msa) / ACC_0), n);
						}
				}
				else
					dataArray[l].k = 0;
					
				
			}
			else if (classificationAbility > 0) {
				dataArray[l].minCurrentNs = Integer.MAX_VALUE;

				dataArray[l].tp += 1;
				data.tp += 1;
				
				if (dataArray[l].minCurrentNs > labelNs) 
					dataArray[l].minCurrentNs = labelNs;
				
				if ((dataArray[l].tp / dataArray[l].msa) > ACC_0) {
					dataArray[l].k = 1;
				}
				else {
					dataArray[l].k = a * Math.pow(((dataArray[l].tp / dataArray[l].msa) / ACC_0), n);
				}	
			}
			else dataArray[l].k = 0;
			
			
			// update msa for positive or negative decision (not updated above)
			if (classificationAbility != 0) {
				dataArray[l].msa += 1;
				data.msa += 1;
			}
			
			 relativeAccuracy += cl.numerosity * dataArray[l].k;
		} 
		
		if (relativeAccuracy == 0) relativeAccuracy = 1;

		for (int i = 0; i < matchSetSize; i++) {
			final Macroclassifier cl = matchSet.getMacroclassifier(i); 
			final MLSLCSClassifierData dataArray[] = (MLSLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
			dataArray[l].fitness += LEARNING_RATE * (cl.numerosity * dataArray[l].k / relativeAccuracy - dataArray[l].fitness);
		}
	}
	
	
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.AbstractUpdateStrategy#updateSet(gr.auth.ee.lcs.
	 * classifiers.ClassifierSet, gr.auth.ee.lcs.classifiers.ClassifierSet, int,
	 * boolean)
	 */
	@Override
	public void updateSet(ClassifierSet population, 
						   ClassifierSet matchSet,
						   int instanceIndex, 
						   boolean evolve) {
		
		if(commencedDeletions && SettingsLoader.getStringSetting("matchSetPopulationControl", "false").equals("true"))
			controlPopulationInMatchSet(population, matchSet);

		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];
			
//		generateCorrectSetTime = -System.currentTimeMillis(); 
		
		for (int i = 0; i < numberOfLabels; i++) { 
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); 		
		}		
		
//		generateCorrectSetTime += System.currentTimeMillis();

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

//		updateParametersTime = -System.currentTimeMillis();
		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			// For each classifier in the matchset
			for (int i = 0; i < matchSetSize; i++) { 

				final Macroclassifier cl = matchSet.getMacroclassifier(i); 
				

				int minCurrentNs = Integer.MAX_VALUE;
				final MLSLCSClassifierData data = (MLSLCSClassifierData) cl.myClassifier.getUpdateDataObject();
	
				for (int l = 0; l < numberOfLabels; l++) {
					// Get classification ability for label l.
					final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();

					if (classificationAbility == 0) {
						data.tp += OMEGA;
						data.msa += PHI;
						
						if (wildCardsParticipateInCorrectSets) {
							if (minCurrentNs > labelNs) { 
								minCurrentNs = labelNs;
							}
						}
					}
					else if (classificationAbility > 0) { 
						data.tp += 1;
						
						if (minCurrentNs > labelNs) { 
							minCurrentNs = labelNs;
						}
					}
					if (classificationAbility != 0) 
						data.msa += 1;
				} 
	
				cl.myClassifier.experience++;

				if (minCurrentNs != Integer.MAX_VALUE) {
					data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				}
				
				switch (FITNESS_MODE) {
				
				case FITNESS_MODE_SIMPLE:
					data.fitness = Math.pow ((data.tp) / (data.msa), n);
					break;

				case FITNESS_MODE_COMPLEX:
					data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);				 
					break;
				}
				updateSubsumption(cl.myClassifier);
			} 
		}
		
		
		
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			} 
			
			for (int i = 0; i < matchSetSize; i++) { 
				final Macroclassifier cl = matchSet.getMacroclassifier(i);	
				cl.myClassifier.experience++; 
				final MLSLCSClassifierData data = (MLSLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				final MLSLCSClassifierData dataArray[] = (MLSLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
				
				double fitnessSum = 0;
				double ns = 0;
				
				for (int l = 0; l < numberOfLabels; l++) {
					fitnessSum += dataArray[l].fitness;	
					ns += dataArray[l].minCurrentNs;
				}
				ns /= numberOfLabels;
				data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

				if (ns != Integer.MAX_VALUE) {
					data.ns += LEARNING_RATE * (ns - data.ns);
				}
					
				if (Math.pow(data.tp / data.msa, n) > ACC_0) {
					if (cl.myClassifier.experience >= this.subsumptionExperienceThreshold && cl.myClassifier.timestamp > 0)
						cl.myClassifier.setSubsumptionAbility(true);
				}
				else {
					cl.myClassifier.setSubsumptionAbility(false);
				}
				
			} 
		}
		
//		updateParametersTime += System.currentTimeMillis();
		
//		evolutionTime = 0;
		
//		numberOfEvolutionsConducted = 0;
//		numberOfSubsumptionsConducted = 0;
//		numberOfDeletionsConducted = 0;
//		numberOfNewClassifiers = 0;
//		subsumptionTime = 0;
//		deletionTime = 0;
//			
		if (evolve) {
//			evolutionTime = -System.currentTimeMillis();

			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					
					ga.evolveSet(labelCorrectSets[l], population, l);
					population.totalGAInvocations = ga.getTimestamp();
					
//					numberOfEvolutionsConducted += ga.evolutionConducted();
//					numberOfSubsumptionsConducted += ga.getNumberOfSubsumptionsConducted();
//					numberOfNewClassifiers += ga.getNumberOfNewClassifiers();
//					subsumptionTime += ga.getSubsumptionTime();
//					deletionTime += ga.getDeletionTime();
//					numberOfDeletionsConducted += ga.getNumberOfDeletionsConducted();
				} else {
					
					this.cover(population, instanceIndex);
//					numberOfNewClassifiers++;
//					IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
//					numberOfDeletionsConducted += theControlStrategy.getNumberOfDeletionsConducted(); 
//					deletionTime += theControlStrategy.getDeletionTime();
				}
			}
//			evolutionTime += System.currentTimeMillis();
		}
	}	
	
	
	@Override
	public void updateSetNew(ClassifierSet population, 
							   ClassifierSet matchSet,
							   int instanceIndex, 
							   boolean evolve) {
		
		/*
		 * If "&& evolve" is enabled in the condition below,
		 * rules will be deleted from match sets only during the
		 * training period (iterations), not during the update period that follows it.
		 * */
		
		if (commencedDeletions && SettingsLoader.getStringSetting("matchSetPopulationControl", "false").equals("true") /* && evolve */) {
				controlPopulationInMatchSet(population, matchSet);
		}
		
		// Create all label correct sets
		final ClassifierSet[] labelCorrectSets = new ClassifierSet[numberOfLabels];
		
//		generateCorrectSetTime = -System.currentTimeMillis();

		for (int i = 0; i < numberOfLabels; i++) { 
			labelCorrectSets[i] = generateLabelCorrectSet(matchSet, instanceIndex, i); 
		}
		
//		generateCorrectSetTime += System.currentTimeMillis();

		
		int CorrectSetsPopulation = 0;
		for (int i = 0; i < numberOfLabels; i++) {
			CorrectSetsPopulation += labelCorrectSets[i].getNumberOfMacroclassifiers() ;
		}
		myLcs.meanCorrectSetNumerosity = CorrectSetsPopulation / numberOfLabels;

		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();

//		updateParametersTime = -System.currentTimeMillis();
		
		if (FITNESS_MODE == FITNESS_MODE_SIMPLE || FITNESS_MODE == FITNESS_MODE_COMPLEX) {
			// For each classifier in the matchset
			for (int i = 0; i < matchSetSize; i++) { 
				
				final Macroclassifier cl = matchSet.getMacroclassifier(i); 
				
				int minCurrentNs = Integer.MAX_VALUE;
				final MLSLCSClassifierData data = (MLSLCSClassifierData) cl.myClassifier.getUpdateDataObject();
	
				for (int l = 0; l < numberOfLabels; l++) {
					// Get classification ability for label l. 
					final float classificationAbility = cl.myClassifier.classifyLabelCorrectly(instanceIndex, l);
					final int labelNs = labelCorrectSets[l].getTotalNumerosity();

					if (classificationAbility == 0) {
						data.tp += OMEGA;
						data.msa += PHI;
						
						if (wildCardsParticipateInCorrectSets) {
							if (minCurrentNs > labelNs) { 
								minCurrentNs = labelNs;
							}
						}
					}
					else if (classificationAbility > 0) {
						data.tp += 1;
						
						if (minCurrentNs > labelNs) { 
							minCurrentNs = labelNs;
						}
					}
					
					if (classificationAbility != 0) 
						data.msa += 1;
				} 
	
				cl.myClassifier.experience++;

				if (minCurrentNs != Integer.MAX_VALUE) {
					data.ns += LEARNING_RATE * (minCurrentNs - data.ns);
				}
				
				switch (FITNESS_MODE) {
				
				case FITNESS_MODE_SIMPLE:
					data.fitness = Math.pow((data.tp) / (data.msa), n);
					break;
				case FITNESS_MODE_COMPLEX:
					data.fitness += LEARNING_RATE * (Math.pow((data.tp) / (data.msa), n) - data.fitness);					 
					break;
				}
				updateSubsumption(cl.myClassifier);

			} 
		}
		
		
		
		else if (FITNESS_MODE == FITNESS_MODE_SHARING) {
			
			for (int l = 0; l < numberOfLabels; l++) {
				shareFitness(matchSet, labelCorrectSets[l], l, instanceIndex);
			} 
			
			for (int i = 0; i < matchSetSize; i++) { 
				final Macroclassifier cl = matchSet.getMacroclassifier(i);	
				cl.myClassifier.experience++; 
				final MLSLCSClassifierData data = (MLSLCSClassifierData) cl.myClassifier.getUpdateDataObject();
				final MLSLCSClassifierData dataArray[] = (MLSLCSClassifierData[]) cl.myClassifier.getUpdateDataArray();
				
				double fitnessSum = 0;
				double ns = 0;
				
				for (int l = 0; l < numberOfLabels; l++) {
					fitnessSum += dataArray[l].fitness;	
					ns += dataArray[l].minCurrentNs;
				}
				ns /= numberOfLabels;
				data.fitness = (fitnessSum / cl.numerosity) / numberOfLabels;

				if (ns != Integer.MAX_VALUE) {
					data.ns += LEARNING_RATE * (ns - data.ns);
				}
					
				if (Math.pow(data.tp / data.msa, n) > ACC_0) {
					if (cl.myClassifier.experience >= this.subsumptionExperienceThreshold && cl.myClassifier.timestamp > 0)
						cl.myClassifier.setSubsumptionAbility(true);
				}
				else {
					cl.myClassifier.setSubsumptionAbility(false);
				}
				
			} 
		}
		
//		updateParametersTime += System.currentTimeMillis(); 
//		
//		numberOfEvolutionsConducted = 0;
//		numberOfSubsumptionsConducted = 0;
//		numberOfDeletionsConducted = 0;
//		numberOfNewClassifiers = 0;
//		evolutionTime = 0;
//		subsumptionTime = 0;
//		deletionTime = 0;
		
		if (evolve) {
			
//			evolutionTime = -System.currentTimeMillis();
						
			Vector<Integer> labelsToEvolve = new Vector<Integer>();
			Vector<Integer> labelsToCover = new Vector<Integer>();
			
			for (int l = 0; l < numberOfLabels; l++) {
				if (labelCorrectSets[l].getNumberOfMacroclassifiers() > 0) {
					
					ga.increaseTimestamp();
					int meanAge = ga.getMeanAge(labelCorrectSets[l]);
					if ( !( ga.getTimestamp() - meanAge < ga.getActivationAge()) )
					{
						labelsToEvolve.add(l);
						for ( int i = 0; i < labelCorrectSets[l].getNumberOfMacroclassifiers(); i++ )
						{
							labelCorrectSets[l].getClassifier(i).timestamp = ga.getTimestamp();
						}
					}					
				} else {
					labelsToCover.add(l);
				}
			}
			
//			numberOfEvolutionsConducted = labelsToEvolve.size();
			
			Vector<Integer> indicesToSubsume = new Vector<Integer>();
			
			ClassifierSet newClassifiersSet = new ClassifierSet(null);
			
			for ( int i = 0; i < labelsToEvolve.size(); i++ )
			{
				ga.evolveSetNew(labelCorrectSets[labelsToEvolve.elementAt(i)], population, labelsToEvolve.get(i));
				indicesToSubsume.addAll(ga.getIndicesToSubsume());
				newClassifiersSet.merge(ga.getNewClassifiersSet());
				
//				subsumptionTime += ga.getSubsumptionTime();				
			}
			
			for ( int i = 0; i < labelsToCover.size(); i++ )
			{
				newClassifiersSet.addClassifier(this.coverNew(instanceIndex),false);
			}
			
			population.totalGAInvocations = ga.getTimestamp();

			
//			numberOfSubsumptionsConducted = indicesToSubsume.size();
//			numberOfNewClassifiers        = newClassifiersSet.getNumberOfMacroclassifiers();
			
			for ( int i = 0; i < indicesToSubsume.size() ; i++ )
			{
				population.getMacroclassifiersVector().get(indicesToSubsume.elementAt(i)).numerosity++; 
				population.getMacroclassifiersVector().get(indicesToSubsume.elementAt(i)).numberOfSubsumptions++; 
				population.totalNumerosity++;
			}
			
			population.mergeWithoutControl(newClassifiersSet);
			
//			deletionTime = -System.currentTimeMillis();
			final IPopulationControlStrategy theControlStrategy = population.getPopulationControlStrategy();
			theControlStrategy.controlPopulation(population);
//			deletionTime += System.currentTimeMillis();
			
//			numberOfDeletionsConducted = theControlStrategy.getNumberOfDeletionsConducted();
			
//			evolutionTime += System.currentTimeMillis();
			
		}
		
	}
	
	
	/**
	 * Implementation of the subsumption strength.
	 * 
	 * @param aClassifier
	 *            the classifier, whose subsumption ability is to be updated
	 */
	protected void updateSubsumption(final Classifier aClassifier) {
		aClassifier.setSubsumptionAbility(
				(aClassifier.getComparisonValue(COMPARISON_MODE_EXPLOITATION) > subsumptionFitnessThreshold)
						&& (aClassifier.experience > subsumptionExperienceThreshold) && (aClassifier.timestamp > 0));
	}
	
	
	/**
	 * Delete classifiers from every match set formed.
	 * 
	 */
	private void controlPopulationInMatchSet (final ClassifierSet population, final ClassifierSet matchSet) {
		double lowestCoverage = Double.MAX_VALUE;

		for (int i = 0; i < matchSet.getNumberOfMacroclassifiers(); i++) {
			
			final Classifier cl = matchSet.getClassifier(i);
			if (cl.objectiveCoverage > 0 && cl.objectiveCoverage <= lowestCoverage) {
				
				if (cl.objectiveCoverage < lowestCoverage) {
					lowestCoverageIndices.clear();
				}
				
				lowestCoverage = cl.objectiveCoverage;
				lowestCoverageIndices.add(i);
			}
		}
		
		if (lowestCoverageIndices.size() > 1) {
			
			double lowestFitness = Double.MAX_VALUE;
			int toBeDeleted = -1;

			for (int i = 0; i < lowestCoverageIndices.size(); i++) {
				
				final Macroclassifier macro = matchSet.getMacroclassifier(lowestCoverageIndices.get(i));
				final Classifier cl = macro.myClassifier;

				if (cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS) <= lowestFitness) {

					lowestFitness = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_FITNESS);
					toBeDeleted = lowestCoverageIndices.get(i);
				}
			}
			
			if (toBeDeleted >= 0) {
				myLcs.numberOfClassifiersDeletedInMatchSets++;
				population.deleteClassifier(matchSet.getMacroclassifier(toBeDeleted).myClassifier);
				matchSet.deleteClassifier(toBeDeleted);
			}
		}
		lowestCoverageIndices.clear();
	}
	
	

}
