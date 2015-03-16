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
package gr.auth.ee.lcs.data.updateAlgorithms;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IGeneticAlgorithmStrategy;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.util.ArrayList;

/**
 * An alternative MlASLCS update algorithm.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public class MlASLCS4UpdateAlgorithm extends MLSLCSUpdateAlgorithm {

	
	
	/**
	 *  holds the classifiers' indices in the match set with the lowest coverage. used when deleting from [M]
	 * */
	private ArrayList <Integer> lowestCoverageIndices;
	
	private boolean commencedDeletions = false;
		
	public int numberOfAttributes;
	

	/**
	 * Constructor.
	 * 
	 * @param lcs
	 *            the LCS being used.
	 * @param labels
	 *            the number of labels
	 * @param geneticAlgorithm
	 *            the GA used
	 * @param nParameter
	 *            the ASLCS dubbing factor
	 * @param fitnessThreshold
	 *            the subsumption fitness threshold to be used.
	 * @param experienceThreshold
	 *            the subsumption experience threshold to be used
	 */
	public MlASLCS4UpdateAlgorithm(final double nParameter,
									final double fitnessThreshold, 
									final int experienceThreshold,
									IGeneticAlgorithmStrategy geneticAlgorithm, 
									int labels,
									AbstractLearningClassifierSystem lcs) {
		
		super(nParameter, fitnessThreshold, experienceThreshold, geneticAlgorithm, labels, lcs);
		lowestCoverageIndices = new ArrayList <Integer>();
		
	}

	protected void computeCoreDeletionProbabilities (final Macroclassifier cl, 
			final MlASLCSClassifierData data,
			final double meanFitness) {
		commencedDeletions = true;
		super.computeCoreDeletionProbabilities(cl, data, meanFitness);
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
	
	@Override
	public void updateSet(ClassifierSet population, 
						   ClassifierSet matchSet,
						   int instanceIndex, 
						   boolean evolve) {

		if(commencedDeletions && SettingsLoader.getStringSetting("matchSetPopulationControl", "false").equals("true"))
			controlPopulationInMatchSet(population, matchSet);
		
		super.updateSet(population, matchSet, instanceIndex, evolve);
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
		
		super.updateSetNew(population, matchSet, instanceIndex, evolve);
	}


	
}