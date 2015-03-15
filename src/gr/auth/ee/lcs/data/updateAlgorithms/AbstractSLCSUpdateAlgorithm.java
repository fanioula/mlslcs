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
package gr.auth.ee.lcs.data.updateAlgorithms;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;

import java.io.Serializable;

/**
 * An abstract *S-LCS update algorithm as described in Tzima-Mitkas paper.
 */
public abstract class AbstractSLCSUpdateAlgorithm extends
		AbstractUpdateStrategy {

	/**
	 * A data object for the *SLCS update algorithms.
	 * 
	 * @author F. Tzima and M. Allamanis
	 * 
	 */
	static class SLCSClassifierData implements Serializable {

		/**
		 * serial for versions.
		 */
		private static final long serialVersionUID = -20798032843413916L;

		/**
		 *
		 */
		public double fitness = .5;

		/**
		 * niche set size estimation.
		 */
		public double ns = 0;

		/**
		 * Match Set Appearances.
		 */
		public int msa = 0;

		/**
		 * true positives.
		 */
		public int tp = 0;

		/**
		 * false positives.
		 */
		public int fp = 0;

		/**
		 * Strength.
		 */
		public double str = 0;

	} 

	/**
	 * Genetic Algorithm.
	 */
	public IGeneticAlgorithmStrategy ga;

	/**
	 * A double indicating the probability that the GA will run on the matchSet (and not on the correct set).
	 */
	private final double matchSetRunProbability;

	/**
	 * The fitness threshold for subsumption.
	 */
	private final double subsumptionFitnessThreshold;

	/**
	 * The LCS instance being used.
	 */
	private final AbstractLearningClassifierSystem myLCS;

	/**
	 * The experience threshold for subsumption.
	 */
	private final int subsumptionExperienceThreshold;

	/**
	 * @param subsumptionFitness
	 *            the fitness threshold for subsumption
	 * @param subsumptionExperience
	 *            the experience threshold for subsumption
	 * @param gaMatchSetRunProbability
	 *            the probability of running the GA at the matchset
	 * @param geneticAlgorithm
	 *            the GA to use
	 * @param lcs
	 *            the LCS instance used
	 */
	protected AbstractSLCSUpdateAlgorithm(final double subsumptionFitness,
			final int subsumptionExperience,
			final double gaMatchSetRunProbability,
			final IGeneticAlgorithmStrategy geneticAlgorithm,
			final AbstractLearningClassifierSystem lcs) {
		this.subsumptionFitnessThreshold = subsumptionFitness;
		this.subsumptionExperienceThreshold = subsumptionExperience;
		this.matchSetRunProbability = gaMatchSetRunProbability;
		this.ga = geneticAlgorithm;
		myLCS = lcs;
	}

	/**
	 * Calls covering operator.
	 * 
	 * @param population
	 *            the rule population where the new rules will be added to
	 * @param instanceIndex
	 *            the index of the current sample
	 */
	@Override
	public final void cover(final ClassifierSet population, final int instanceIndex) {
		System.out.println("covering");		
		final Classifier coveringClassifier = myLCS
				.getClassifierTransformBridge().createRandomCoveringClassifier(myLCS.instances[instanceIndex]);
		
		population.addClassifier(new Macroclassifier(coveringClassifier, 1),
				false);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#
	 * createStateClassifierObject()
	 */
	@Override
	public final Serializable createStateClassifierObject() {
		return new SLCSClassifierData();
	}

	/**
	 * Generates the correct set.
	 * 
	 * @param matchSet
	 *            the match set
	 * @param instanceIndex
	 *            the global instance index
	 * @return the correct set
	 */
	private ClassifierSet generateCorrectSet(final ClassifierSet matchSet,
											 final int 			 instanceIndex) {
		
		final ClassifierSet correctSet = new ClassifierSet(null);
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < matchSetSize; i++) {
			Macroclassifier cl = matchSet.getMacroclassifier(i);
			if (cl.myClassifier.classifyCorrectly(instanceIndex) == 1)
				correctSet.addClassifier(cl, false);
		}
		return correctSet;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#getData(gr.auth
	 * .ee.lcs.classifiers.Classifier)
	 */
	@Override
	public final String getData(final Classifier aClassifier) {
		final SLCSClassifierData data = ((SLCSClassifierData) aClassifier
				.getUpdateDataObject());
		return "tp:" + data.tp + "msa:" + data.msa + "str: " + data.str + "ns:"
				+ data.ns;
	}
	

	@Override
	public final void performUpdate(final ClassifierSet matchSet,
									final ClassifierSet correctSet) {
		
		final int matchSetSize = matchSet.getNumberOfMacroclassifiers();  // the total number of macroclassifiers in the matchSet
		final int correctSetNumerosity = correctSet.getTotalNumerosity(); // the total number of microclassifiers in the correctSet
		
		for (int i = 0; i < matchSetSize; i++) {
			
			Classifier cl = matchSet.getClassifier(i); 
			SLCSClassifierData data = ((SLCSClassifierData) cl.getUpdateDataObject()); 
			
			if (correctSet.getClassifierNumerosity(cl) > 0) { 
				data.ns = ((data.msa * data.ns) + correctSetNumerosity) / (data.msa + 1); }
			
			data.msa++;

			updateFitness(cl, matchSet.getClassifierNumerosity(i), correctSet);
			this.updateSubsumption(cl); // set classifier's subsumption ability to true or false
			cl.experience++;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy#setComparisonValue
	 * (gr.auth.ee.lcs.classifiers.Classifier, int, double)
	 */
	@Override
	public final void setComparisonValue(final Classifier aClassifier,
			final int mode, final double comparisonValue) {
		final SLCSClassifierData data = ((SLCSClassifierData) aClassifier
				.getUpdateDataObject());
		data.fitness = comparisonValue; // TODO: More generic

	}

	/**
	 * The abstract function used to calculate the fitness of a classifier.
	 * 
	 * @param aClassifier
	 *            the classifier to calculate the fitness
	 * @param numerosity
	 *            the numerosity of the given classifier
	 * @param correctSet
	 *            the correct set, used at updating the fitness
	 */
	public abstract void updateFitness(Classifier aClassifier, int numerosity,
			ClassifierSet correctSet);

	@Override
	public final void updateSet(final ClassifierSet population,
								final ClassifierSet matchSet, 
								final int 			instanceIndex,
								final boolean 		evolve) {

		final ClassifierSet correctSet = generateCorrectSet(matchSet, instanceIndex);

		/*
		 * Cover if necessary
		 */
		if (correctSet.getNumberOfMacroclassifiers() == 0) {
			if (evolve)
				cover(population, instanceIndex);
			return;
		}

		performUpdate(matchSet, correctSet);

		if (!evolve)
			return;
		/*
		 * Run GA
		 */
		if (Math.random() < matchSetRunProbability)
			ga.evolveSet(matchSet, population, 0);
		else
			ga.evolveSet(correctSet, population, 0);

	}

	/**
	 * Implementation of the subsumption strength.
	 * 
	 * @param aClassifier
	 *            the classifier, whose subsumption ability is to be updated
	 */
	protected final void updateSubsumption(final Classifier aClassifier) {
		aClassifier
				.setSubsumptionAbility((aClassifier
						.getComparisonValue(COMPARISON_MODE_EXPLOITATION) > subsumptionFitnessThreshold)
						&& (aClassifier.experience > subsumptionExperienceThreshold));
	}

}