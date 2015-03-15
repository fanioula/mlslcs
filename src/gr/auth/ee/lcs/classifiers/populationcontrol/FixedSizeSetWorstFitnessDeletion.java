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
package gr.auth.ee.lcs.classifiers.populationcontrol;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.IPopulationControlStrategy;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * A fixed size control strategy. Classifiers are deleted based on the selector
 * tournaments
 * 
 * @stereotype ConcreteStrategy
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public class FixedSizeSetWorstFitnessDeletion implements
		IPopulationControlStrategy {

	private AbstractLearningClassifierSystem myLcs;
	
	/**
	 * The Natural Selector used to select the the classifier to be deleted.
	 * @uml.property  name="mySelector"
	 * @uml.associationEnd  multiplicity="(1 1)"
	 */
	private final IRuleSelector mySelector;

	/**
	 * The fixed population size of the controlled set.
	 * @uml.property  name="populationSize"
	 */
	private final int populationSize;
	
	private int numberOfDeletions;
	
	private long deletionTime;
	
	private AbstractUpdateStrategy updateStrategy;

	/**
	 * Constructor of deletion strategy.
	 * 
	 * @param maxPopulationSize
	 *            the size that the population will have
	 * @param selector
	 *            the selector used for deleting
	 */
	public FixedSizeSetWorstFitnessDeletion(
											 final AbstractLearningClassifierSystem lcs,
											 final int maxPopulationSize, 
											 final IRuleSelector selector) {
		
		this.populationSize = maxPopulationSize;
		mySelector = selector; // roulette wheel for GMlASLCS3
		myLcs = lcs;
		updateStrategy = lcs.getUpdateStrategy();
	}

	/**
	 * @param aSet
	 *            the set to control
	 * @see gr.auth.ee.lcs.classifiers.IPopulationControlStrategy#controlPopulation(gr.auth.ee.lcs.classifiers.ClassifierSet)
	 */
	@Override
	public final void controlPopulation(final ClassifierSet aSet) {

		final ClassifierSet toBeDeleted = new ClassifierSet(null);

		numberOfDeletions = 0;
		deletionTime = 0;
		
		while (aSet.getTotalNumerosity() > populationSize) {
			long time1 = - System.currentTimeMillis();
			
			numberOfDeletions++;
			
			updateStrategy.computeDeletionProbabilities(aSet);
			
			mySelector.select(1, aSet, toBeDeleted); 
			Classifier cl = toBeDeleted.getClassifier(0);

			
			if (cl.formulaForD == 1)
				aSet.firstDeletionFormula++;
			else if (cl.formulaForD == 0) 
				aSet.secondDeletionFormula++;
			
			if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER || (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT))
				aSet.coveredDeleted++;
			else if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA)
				aSet.gaedDeleted++;
			
			// monitor deletions
			monitorDeletions(aSet, cl);
						
			aSet.deleteClassifier(cl);
			toBeDeleted.deleteClassifier(0);

			time1 += System.currentTimeMillis();
			
			deletionTime += time1;
		}
		
	}

	
	@Override
	public final int getNumberOfDeletionsConducted(){
		return numberOfDeletions;
	}
	
	@Override
	public final long getDeletionTime(){
		return deletionTime;
	}

	
	
	/**
	 * Record the progress of the deletion process.
	 * 
	 * @param aSet
	 *            the set to control
	 * @see gr.auth.ee.lcs.classifiers.IPopulationControlStrategy#controlPopulation(gr.auth.ee.lcs.classifiers.ClassifierSet)
	 * 
	 * @param cl
	 * 			  the classifier deleted
	 * */
	public void monitorDeletions(ClassifierSet aSet, Classifier cl) {

		double acc = cl.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION);
		
		double qualityIndex = -0.1;
		
		if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER|| cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT) {
			if (cl.objectiveCoverage > 0) 
				qualityIndex = cl.objectiveCoverage;

			myLcs.qualityIndexOfClassifiersCoveredDeleted.add((float) qualityIndex);
			myLcs.qualityIndexOfClassifiersGaedDeleted.add((float) -0.2);
			myLcs.originOfDeleted.add(0);
			myLcs.accuracyOfCoveredDeletion.add((float) acc);
			myLcs.accuracyOfGaedDeletion.add((float) -0.1);

		}
		
		else if (cl.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
			if (cl.objectiveCoverage > 0)
				qualityIndex = cl.objectiveCoverage;

			myLcs.qualityIndexOfClassifiersGaedDeleted.add((float) qualityIndex);
			myLcs.qualityIndexOfClassifiersCoveredDeleted.add((float) -0.2);
			myLcs.originOfDeleted.add(1);
			myLcs.accuracyOfGaedDeletion.add((float) acc);
			myLcs.accuracyOfCoveredDeletion.add((float) -0.1);

		}

		myLcs.qualityIndexOfDeleted.add((float) qualityIndex);
		myLcs.accuracyOfDeleted.add((float) acc);
		myLcs.iteration.add(myLcs.totalRepetition);

	}
}
