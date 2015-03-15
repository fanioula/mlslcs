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
package gr.auth.ee.lcs.geneticalgorithm.selectors;

import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

/**
 * 
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public final class BestClassifierSelector implements IRuleSelector {

	/**
	 * Boolean indicating if the selector selects the best or worst classifier.
	 */
	private final boolean max;

	/**
	 * The mode used for comparing classifiers.
	 */
	private final int mode;

	/**
	 * Default constructor.
	 * 
	 * @param maximum
	 *            if by best we mean the max fitness then true, else false
	 * @param comparisonMode
	 *            the mode of the values taken
	 */
	public BestClassifierSelector(final boolean maximum, final int comparisonMode) {
		this.max = maximum;
		this.mode = comparisonMode;
	}

	/**
	 * Select for population.
	 * 
	 * @param fromPopulation
	 *            the population to select from
	 * @return the index of the best macro-classifier in the set
	 */
	private int select(final ClassifierSet fromPopulation) {
		// Search for the best classifier (best macro-fitness and largest experience value OR worst macro-fitness and smallest experience value) 
		double bestFitness = max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		int bestExp = 0;
		int bestIndex = -1;
		final int popSize = fromPopulation.getNumberOfMacroclassifiers();
		for (int i = 0; i < popSize; i++) {
			final double temp = fromPopulation.getClassifier(i).getComparisonValue(mode) * fromPopulation.getClassifierNumerosity(i); 
			if ((max ? 1. : -1.) * (temp - bestFitness) > 0) {
				bestFitness = temp;
				bestIndex = i;
				bestExp = fromPopulation.getClassifier(i).experience;
			} 
			else if ((Double.compare(temp, bestFitness) == 0) && ((max ? 1. : -1.) * (fromPopulation.getClassifier(i).experience - bestExp) > 0)) {
				bestFitness = temp;
				bestIndex = i;
				bestExp = fromPopulation.getClassifier(i).experience;
			}
		}

		return bestIndex;
	}

	
	/**
	 * Selects the best/worst classifier (based on fitness) from the initial
	 * ClassifierSet and adds it to the target set.  
	 * The best classifier is added with howManyToSelect numerosity.
	 * 
	 * @param fromPopulation
	 *            the population to select from
	 * @param toPopulation
	 *            the population to add the selected classifier to
	 * @param howManyToSelect
	 *            the best classifier is added with howManyToSelect numerosity
	 */
	
	@Override
	public void select(final int howManyToSelect, final ClassifierSet fromPopulation, final ClassifierSet toPopulation) {
		// Add it toPopulation
		final int bestIndex = select(fromPopulation);
		if (bestIndex == -1)
			return;
		toPopulation.addClassifier(new Macroclassifier(fromPopulation.getClassifier(bestIndex), howManyToSelect), true);
	}

}