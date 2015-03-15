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
package gr.auth.ee.lcs.geneticalgorithm.selectors;

import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.Macroclassifier;
import gr.auth.ee.lcs.geneticalgorithm.IRuleSelector;

import java.util.Arrays;

/**
 * A tournament selecting the best fitness classifier.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public class TournamentSelector implements IRuleSelector {

	/**
	 * The size of the tournaments.
	 */
	private final int tournamentSize;

	/**
	 * The type of the tournaments.
	 */
	private final boolean max;

	/**
	 * The percentage of population size, used for tournament selection.
	 */
	private final double percentSize;

	/**
	 * The comparison mode for the tournaments.
	 */
	private final int mode;

	/**
	 * Constructor.
	 * 
	 * @param sizeOfTournaments
	 *            the size of the tournament as a percentage of the given set
	 *            size
	 * @param max
	 *            true if the tournament selects the max fitness
	 * @param comparisonMode
	 *            the comparison mode to be used
	 */
	public TournamentSelector(final double sizeOfTournaments,
			final boolean max, final int comparisonMode) {
		this.tournamentSize = 0;
		this.max = max;
		this.mode = comparisonMode;
		percentSize = sizeOfTournaments;
	}

	/**
	 * The default constructor of the selector.
	 * 
	 * @param sizeOfTournaments
	 *            the size of the tournament
	 * @param max
	 *            if true the we select the max fitness participant, else we
	 *            select the min.
	 * @param comparisonMode
	 *            comparison mode @see
	 *            gr.auth.ee.lcs.data.UpdateAlgorithmFactoryAndStrategy
	 */
	public TournamentSelector(final int sizeOfTournaments, final boolean max,
			final int comparisonMode) {
		this.tournamentSize = sizeOfTournaments;
		this.max = max;
		this.mode = comparisonMode;
		percentSize = 0;
	}

	/**
	 * Select  a rule form the population and return its index.
	 * The selection is made using a tournament with random 
	 * participants from the "fromPopulation".
	 * 
	 * @param fromPopulation
	 *            the population to select from
	 * @return the index of the selected classifier in the fromPopulation
	 */
	private int select(final ClassifierSet fromPopulation) {
		int size;
		if (tournamentSize == 0) {
			size = (int) Math.floor(fromPopulation.getTotalNumerosity() * percentSize);
		} 
		else {
			size = tournamentSize;
		}

		final int[] participants = new int[size];
		// Create random participants
		for (int j = 0; j < participants.length; j++) {
			participants[j] = (int) Math.floor((Math.random() * fromPopulation.getTotalNumerosity()));
		}
		return this.tournament(fromPopulation, participants);

	}

	/*
	 * Selects howManyToSelect classifiers (based on random tournaments) from the initial
	 * ClassifierSet and adds them to the target set as macro-classifiers with numerosity = 1.  
	 * 
	* @param fromPopulation
	 *            the population to select from
	 * @param toPopulation
	 *            the population to add the selected classifiers to
	 * @param howManyToSelect
	 *            the number of classifiers to select
	 */
	@Override
	public final void select(final int howManyToSelect,
			final ClassifierSet fromPopulation, final ClassifierSet toPopulation) {

		for (int i = 0; i < howManyToSelect; i++) {

			toPopulation.addClassifier(
					new Macroclassifier(fromPopulation.getClassifier(this
							.select(fromPopulation)), 1), false);

		}

	}

	/**
	 * Runs a tournament in the fromPopulation with the size defined in
	 * tournamentSize (during construction). The index of the tournament 
	 * winner is returned. 
	 * Best or worst fitness defines the winner, depending on the type of the 
	 * tournaments (max variable, set during construction).
	 * 
	 * @param fromPopulation
	 *            the source population to run the tournament in
	 * @param participants
	 *            the int[] of indexes of participants
	 * @return the index of the tournament winner
	 */
	public final int tournament(final ClassifierSet fromPopulation, final int[] participants) {

		// Sort the participants array
		Arrays.sort(participants);
		
		int i=1;
		int lastParticipant = participants[participants.length-i];
		while (lastParticipant > fromPopulation.getTotalNumerosity()-1)
		{
			participants[participants.length-i] = fromPopulation.getTotalNumerosity()-1;
			System.err.println("Participants array contains illegal index value (greater than the population's total numerosity) at " + (participants.length-i ) + 
					". Replacing with maximum allowed index value (" +  (fromPopulation.getTotalNumerosity()-1) + ").");
			i++;
			lastParticipant = participants[participants.length-i];
		}

		// best fitness found in tournament
		double bestFitness = max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

		// the "translated" (including numerosity) maximum index that corresponds to the current (macro-)classifier 
		// e.g. if we have two (macro-)classifiers, the first with numerosity 3 and the second with
		// numerosity 2, the "translated" maximum index that corresponds to the first (macro-)classifier 
		// is 2, while the same for the second (macro-)classifier is 4
		int maxIndexOfCurrentClassifier = -1;

		// the index of the best (macro-)classifier so far (in the ruleset)
		int bestClassifierIndex = -1;

		// the index of the current participant (in the participants array)
		int currentParticipantIndex = 0;

		// the index of the current (macro-)classifier (in the ruleset)
		int currentClassifierIndex = 0;
		
		// Run tournament
		do {
			maxIndexOfCurrentClassifier += fromPopulation.getClassifierNumerosity(currentClassifierIndex);
			while ((currentParticipantIndex < participants.length) && (participants[currentParticipantIndex] <= maxIndexOfCurrentClassifier)) {

				// currentParicipant is in the current (macro-)classifier
				final double fitness = fromPopulation.getClassifier(currentClassifierIndex).getComparisonValue(mode);

				if ((max ? 1. : -1.) * (fitness - bestFitness) > 0) {
					bestClassifierIndex = currentClassifierIndex;
					bestFitness = fitness;
				}
				currentParticipantIndex++; // next participant
			}
			currentClassifierIndex++; // next (macro-)classifier
		} while (currentParticipantIndex < participants.length);

		if (bestClassifierIndex >= 0)
			return bestClassifierIndex;
		else
			return 0;

	}


}
