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
package gr.auth.ee.lcs.classifiers;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.data.AbstractUpdateStrategy;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

/**
 * Implement set of Classifiers, counting numerosity for classifiers. This
 * object is serializable.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 * @has 1 - * Macroclassifier
 * @has 1 - 1 IPopulationControlStrategy
 */
public class ClassifierSet implements Serializable {
	
	
	/* d1 = the number of classifiers whose d is calculated by ns * <F> / f
	 * d2 = the number of classifiers whose d is calculated by ns
	 **/
	public int firstDeletionFormula = 0;
	public int secondDeletionFormula = 0;
	
	public int coveredDeleted = 0;
	public int gaedDeleted = 0;
	
	public int zeroCoverageDeletions = 0;
	public Vector<Integer> zeroCoverageVector = new Vector<Integer>();
	public Vector<Integer> zeroCoverageIterations = new Vector<Integer>();

	
	/**
	 * Serialization id for versioning.
	 */
	private static final long serialVersionUID = 2664983888922912954L;
	
	public int totalGAInvocations = 0;

	public int unmatched;
	
	
	/**
	 * Open a saved (and serialized) ClassifierSet.
	 * 
	 * @param path
	 *            the path of the ClassifierSet to be opened
	 * @param sizeControlStrategy
	 *            the ClassifierSet's
	 * @param lcs
	 *            the lcs which the new set will belong to
	 * @return the opened classifier set
	 */
	public static ClassifierSet openClassifierSet(final String path,
			final IPopulationControlStrategy sizeControlStrategy,
			final AbstractLearningClassifierSystem lcs) {
		FileInputStream fis = null;
		ObjectInputStream in = null;
		ClassifierSet opened = null;

		try {
			fis = new FileInputStream(path);
			in = new ObjectInputStream(fis);

			opened = (ClassifierSet) in.readObject();
			opened.myISizeControlStrategy = sizeControlStrategy;

			for (int i = 0; i < opened.getNumberOfMacroclassifiers(); i++) {
				final Classifier cl = opened.getClassifier(i);
				cl.setLCS(lcs);
			}

			in.close();
		} catch (IOException ex) {
			ex.printStackTrace();
		} catch (ClassNotFoundException ex) {
			ex.printStackTrace();
		}

		return opened;
	}

	/**
	 * A static function to save the classifier set.
	 * 
	 * @param toSave
	 *            the set to be saved
	 * @param filename
	 *            the path to save the set
	 */
	public static void saveClassifierSet(final ClassifierSet toSave,
			final String filename) {
		FileOutputStream fos = null;
		ObjectOutputStream out = null;

		try {
			fos = new FileOutputStream(filename);
			out = new ObjectOutputStream(fos);
			out.writeObject(toSave);
			out.close();

		} catch (IOException ex) {
			ex.printStackTrace();
		}

	}

	/**
	 * The total numerosity of all classifiers in set.
	 */
	public int totalNumerosity = 0;
	
	public int sumOfUnmatched;
	
	public Vector<Integer> deleteIndices;
	
	public boolean subsumed;
	

	/**
	 * Macroclassifier vector.
	 */
	private final ArrayList<Macroclassifier> myMacroclassifiers;

	/**
	 * An interface for a strategy on deleting classifiers from the set. This attribute is transient and therefore not serializable.
	 */
	private transient IPopulationControlStrategy myISizeControlStrategy;
	
	static int arrayList = 0;
	
	//padding variables
	long p0,p1,p2,p3,p4,p5,p6,p7;

	/**
	 * The default ClassifierSet constructor.
	 * 
	 * @param sizeControlStrategy
	 *            the size control strategy to use for controlling the set
	 */
	public ClassifierSet(final IPopulationControlStrategy sizeControlStrategy) {
		this.myISizeControlStrategy = sizeControlStrategy;
		this.myMacroclassifiers = new ArrayList<Macroclassifier>();

	}

	/**
	 * Adds a classifier with the a given numerosity to the set. It checks if
	 * the classifier already exists and increases its numerosity. It also
	 * checks for subsumption and updates the set's numerosity.
	 * 
	 * @param thoroughAdd
	 *            to thoroughly check addition
	 * @param macro
	 *            the macroclassifier to add to the set
	 */
	public final void addClassifier(final Macroclassifier macro,
									  final boolean thoroughAdd) {

		final int numerosity = macro.numerosity;
		// Add numerosity to the Set
		this.totalNumerosity += numerosity;
		
		
		subsumed = true;


		// Subsume if possible
		// if thoroughAdd = true, before adding the given macro to the population, 
		// check it against the whole population for subsumption
		if (thoroughAdd) { 
			Vector<Integer> indicesVector    = new Vector<Integer>();
			Vector<Float> 	fitnessVector    = new Vector<Float>();
			Vector<Integer> experienceVector = new Vector<Integer>();
			/* 0 for generality, 1 for equality */
			Vector<Integer> originVector = new Vector<Integer>();

			
			final Classifier aClassifier = macro.myClassifier;
			for (int i = 0; i < myMacroclassifiers.size(); i++) {
				
				final Classifier theClassifier = myMacroclassifiers.get(i).myClassifier;
				
				if (theClassifier.canSubsume()) {
					if (theClassifier.isMoreGeneral(aClassifier)) {
						
						indicesVector.add(i);
						originVector.add(0);
						fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
						experienceVector.add(theClassifier.experience);
					}
				} 
				else if (theClassifier.equals(aClassifier)) { 	// Or it can't
																// subsume but
																// it is equal
					indicesVector.add(i);
					originVector.add(1);
					fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
					experienceVector.add(theClassifier.experience);
				}

			} 
			
			int howManyGenerals = 0;
			int howManyEquals = 0;
			for (int i = 0; i < indicesVector.size(); i++) {
				if (originVector.elementAt(i) == 0)
						howManyGenerals++;
				else 
					howManyEquals++;
			}
			
			int indexOfSurvivor = 0;
			float maxFitness = 0;

			if (howManyGenerals !=  0) {
				
				for(int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 0) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
			}
			else if (howManyEquals != 0){
				
				for (int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 1) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
				
			}
			
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
				// Subsume and control size...
				myMacroclassifiers.get(indicesVector.elementAt(indexOfSurvivor)).numerosity += numerosity;
				myMacroclassifiers.get(indicesVector.elementAt(indexOfSurvivor)).numberOfSubsumptions++;
				
				indicesVector.clear();
				originVector.clear();
				fitnessVector.clear();
				experienceVector.clear();
				
				if (myISizeControlStrategy != null) {
					myISizeControlStrategy.controlPopulation(this);
				}
				return;
			}
			
		}
		
		subsumed = false;

		this.myMacroclassifiers.add(macro);
		if (myISizeControlStrategy != null) {
			myISizeControlStrategy.controlPopulation(this);
		}
	}

	/**
	 * Removes a micro-classifier from the set. It either completely deletes it
	 * (if the classsifier's numerosity is 0) or by decreasing the numerosity.
	 * 
	 * @param aClassifier
	 *            the classifier to delete
	 */
	public final void deleteClassifier(final Classifier aClassifier) {
		
		int index;
		final int macroSize = myMacroclassifiers.size();
		for (index = 0; index < macroSize; index++) {
			if (myMacroclassifiers.get(index).myClassifier.getSerial() ==  aClassifier.getSerial()) {
				break;
			}
		}

		if (index == macroSize)
			return;
		deleteClassifier(index);

	}


	/**
	 * Deletes a classifier with the given index. If the macroclassifier at the
	 * given index contains more than one classifier the numerosity is decreased
	 * by one.
	 * 
	 * @param index
	 *            the index of the classifier's macroclassifier to delete
	 */
	public final void deleteClassifier(final int index) {
		
		this.totalNumerosity--; // meiose to numerosity olou tou set
		
		if (this.myMacroclassifiers.get(index).numerosity > 1) {
			this.myMacroclassifiers.get(index).numerosity--; 
		} else {
			this.myMacroclassifiers.remove(index); 
		}
	}

	
	
	/**
	 * It completely removes the macroclassifier with the given index.
	 * Used by cleanUpZeroCoverage().
	 * 
	 * @param index
	 * 
	 * @author A. Filotheou
	 * 
	 * */
	
	
	public final void deleteMacroclassifier (final int index) {
		this.totalNumerosity -= this.myMacroclassifiers.get(index).numerosity;
		this.myMacroclassifiers.remove(index);
	
	}
	
	
	
	
	/**
	 * Generate a match set for a given instance.
	 * 
	 * @param dataInstance
	 *            the instance to be matched
	 * @return a ClassifierSet containing the match set
	 */
	public final ClassifierSet generateMatchSet(final double[] dataInstance) {
		final ClassifierSet matchSet = new ClassifierSet(null);
		final int populationSize = this.getNumberOfMacroclassifiers();
		// TODO: Parallelize for performance increase
		for (int i = 0; i < populationSize; i++) {
			if (this.getClassifier(i).isMatch(dataInstance)) {
				matchSet.addClassifier(this.getMacroclassifier(i), false);
			}
		}
		return matchSet;
	}

	/**
	 * Generate match set from data instance.
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	public final ClassifierSet generateMatchSet(final int dataInstanceIndex) {
		
		final ClassifierSet matchSet = new ClassifierSet(null); 
		deleteIndices = new Vector <Integer>(); // vector to hold the indices of macroclassifiers that are to be deleted due to zero coverage
		final int populationSize = this.getNumberOfMacroclassifiers();
		
		sumOfUnmatched = 0;

		
		// TODO: Parallelize for performance increase
		for (int i = 0; i < populationSize; i++) {
			
			// this = population (macroclassifiers)

			if (this.getClassifier(i).isMatch(dataInstanceIndex)) { 
				
				matchSet.addClassifier(this.getMacroclassifier(i), false); 
			}
			
			sumOfUnmatched += this.getClassifier(i).unmatched;

			
			boolean zeroCoverage = (this.getClassifier(i).getCheckedInstances() >= this.getClassifier(i).getLCS().instances.length) 
									 && (this.getClassifier(i).getCoverage() == 0);
			
			if (this.getClassifier(i).checked == this.getClassifier(i).getLCS().instances.length) 
				this.getClassifier(i).objectiveCoverage = this.getClassifier(i).getCoverage();
			
			if(zeroCoverage) {
				deleteIndices.add(i); // add the index of the macroclassifier with zero coverage 
				System.out.println("deleted due to 0-cov");
			}
		}
		
		for (int i = deleteIndices.size() - 1; i >= 0 ; i--) {
			this.deleteMacroclassifier(deleteIndices.elementAt(i));
		}		
		return matchSet;
	}

	public static double percentageOfBAMDiscovered(final ClassifierSet toSave, String[] BAMrules)
	{
		int numBAMRulesDiscovered = 0;
		Arrays.sort(BAMrules);
		final int populationSize = toSave.getNumberOfMacroclassifiers();
		for (int i=0; i<populationSize;i++)
		{
			if (Arrays.binarySearch(BAMrules, toSave.getClassifier(i).toString()) >= 0)
			{
				numBAMRulesDiscovered ++;
//				System.out.println("BAM rule discovered: " + toSave.getClassifier(i).toString());
				if (numBAMRulesDiscovered == BAMrules.length)
					break;
			}
		}
		double divBy = BAMrules.length;
		if (divBy == 0)
			divBy = 1;
			
		return numBAMRulesDiscovered/divBy;
	}
	
	public void deleteClassifiersBelowFitnessThreshold(double fitnessThreshold)
	{
		for (int i=0; i<getNumberOfMacroclassifiers();i++)
		{
			if (getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION) < fitnessThreshold)
			{
				this.deleteMacroclassifier(i);
				i--;
			}
		}
	}
	
	public void deleteClassifiersBelowNumerosityThreshold(int numerosityThreshold)
	{
		for (int i=0; i<getNumberOfMacroclassifiers();i++)
		{
			if (this.myMacroclassifiers.get(i).numerosity < numerosityThreshold && getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION) < 0.9)
			{
				this.deleteMacroclassifier(i);
				i--;
			}
		}
	}
	
	public int numberOfRulesForFullCoverageWithSpecificDecisions()
	{
		sortRulesAccordingToMacroFitness();
		
		final int populationSize = getNumberOfMacroclassifiers();
		byte[] lala = new byte[getClassifier(0).matchInstances.length];
		Arrays.fill(lala, (byte) 0); // fill it with zeros
		
		int numOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 2);
		byte[][] labels = new byte[getClassifier(0).matchInstances.length][numOfLabels];
		
		int i;
		for (i=0; i<populationSize;i++)
		{
			byte[] temp = getClassifier(i).matchInstances;

			for (int j= 0; j < lala.length; j++) 
			{
				if (temp[j] == -1 || temp[j] == 0)
					continue;
				else
					lala[j] = 1;
				
				for (int c=0; c<numOfLabels; c++)
				{
					if (getClassifier(i).classifyLabelCorrectly(j, c) == 0)
						continue;
					else
						labels[j][c] = 1;
				}
			}
			
			int sum = 0;
			int[] sumLabels = new int[numOfLabels];
			for (int j= 0; j < lala.length; j++) 
			{
				sum += lala[j];
				for (int l=0; l<numOfLabels; l++)
					sumLabels[l] += labels[j][l];
			}
			
			if (sum != lala.length)
				continue;
			boolean stop = true;
			for (int l=0; l<numOfLabels; l++)
			{
				if (sumLabels[l]  != lala.length)
				{
					stop = false;
					break;
				}
			}
			
			if (stop)
				break;
		}
		return i+1;
	}
	
	public void sortRulesAccordingToMacroFitness()
	{
		Collections.sort(myMacroclassifiers, new Comparator<Macroclassifier>() {
		    @Override
			public int compare(Macroclassifier o1, Macroclassifier o2) {
			    	int mode = AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION;
		    		if (o1.myClassifier.getComparisonValue(mode) * o1.numerosity == o2.myClassifier.getComparisonValue(mode) * o2.numerosity) { 
		             return 0;
		        } else { 
		           return o1.myClassifier.getComparisonValue(mode) * o1.numerosity < o2.myClassifier.getComparisonValue(mode) * o2.numerosity ? 1 : -1;
		        }
		    }
		});
	}
	
	
	
	
	/**
	 * Generate match set from data instance. New implementation separating the classifiers that match 
	 * the current data instance for the first time. 
	 * 
	 * @author Vag Skar
	 * 
	 * @param dataInstanceIndex
	 *            the index of the instance
	 * @return the match set
	 */
	
	public final ClassifierSet generateMatchSetNew(final int dataInstanceIndex){
		
		final ClassifierSet matchSet = new ClassifierSet(null);
		final ClassifierSet firstTimeSet = new ClassifierSet(null);

		deleteIndices = new Vector<Integer>();
		Vector<Integer> candidateDeleteIndices = new Vector<Integer>();
		
		final int populationSize = this.getNumberOfMacroclassifiers();
		
		for ( int i = 0; i < populationSize ; i++ )
		{
			Macroclassifier cl = this.getMacroclassifier(i);
			
			if ( cl.myClassifier.matchInstances == null )
			{
				cl.myClassifier.buildMatches();
			}
			
			
			if ( cl.myClassifier.matchInstances[dataInstanceIndex] == -1 )
			{
				firstTimeSet.addClassifier(cl, false);
				candidateDeleteIndices.add(i);

			}
			else if ( cl.myClassifier.matchInstances[dataInstanceIndex] == 1 )
			{
				matchSet.addClassifier(cl,false);
			}
			
		}
	
		for (int i = 0 ; i < firstTimeSet.getNumberOfMacroclassifiers()  ; i++)
		{
			Macroclassifier cl = firstTimeSet.getMacroclassifier(i);
			cl.myClassifier.matchInstances[dataInstanceIndex]
			= (byte)(cl.myClassifier.getLCS().getClassifierTransformBridge().isMatch
					(cl.myClassifier.getLCS().instances[dataInstanceIndex], cl.myClassifier)? 1 : 0);
			
			cl.myClassifier.checked++;
			cl.myClassifier.covered += cl.myClassifier.matchInstances[dataInstanceIndex];
			
			if(cl.myClassifier.matchInstances[dataInstanceIndex] == 1)
				matchSet.addClassifier(cl, false);
			
			boolean zeroCoverage = (cl.myClassifier.checked >= cl.myClassifier.getLCS().instances.length) 
			                       && (cl.myClassifier.covered == 0);
			
			if (cl.myClassifier.checked == cl.myClassifier.getLCS().instances.length) 
				cl.myClassifier.objectiveCoverage = cl.myClassifier.getCoverage();
			
			if (zeroCoverage)
				deleteIndices.add(candidateDeleteIndices.get(i));			
		}	
		
		
		for ( int i = deleteIndices.size() - 1 ; i >= 0 ; i-- )
		{
			zeroCoverageIterations.add(myMacroclassifiers.get(deleteIndices.elementAt(i)).myClassifier.getLCS().totalRepetition);

			this.deleteMacroclassifier(deleteIndices.elementAt(i));
			zeroCoverageDeletions++;
			zeroCoverageVector.add(zeroCoverageDeletions);
		}
		
		deleteIndices.clear();
		candidateDeleteIndices.clear();
		
		return matchSet;
	}
	
	public final ClassifierSet generateMatchSetCached (final int dataInstanceIndex) {
		
		final ClassifierSet matchSet = new ClassifierSet(null);
		final int populationSize = this.getNumberOfMacroclassifiers();
		
		for (int i= 0; i < populationSize; i++) {
			if (this.getClassifier(i).isMatchCached(dataInstanceIndex)) {
				matchSet.addClassifier(this.getMacroclassifier(i), false);
			}
		}
		return matchSet;
	}
	
	
	
	
	
	
	
	/**
	 * Return the classifier at a given index of the macroclassifier vector.
	 * 
	 * @param index
	 *            the index of the macroclassifier
	 * @return the classifier at the specified index
	 */
	public final Classifier getClassifier(final int index) {
		return this.myMacroclassifiers.get(index).myClassifier;
	}

	/**
	 * Returns a classifier's numerosity (the number of microclassifiers).
	 * 
	 * @param aClassifier
	 *            the classifier
	 * @return the given classifier's numerosity
	 */
	public final int getClassifierNumerosity(final Classifier aClassifier) {
		for (int i = 0; i < myMacroclassifiers.size(); i++) {
			if (myMacroclassifiers.get(i).myClassifier.getSerial() == aClassifier.getSerial()) 
				return this.myMacroclassifiers.get(i).numerosity;
		}
		return 0;
	}

	/**
	 * Overloaded function for getting a numerosity.
	 * 
	 * @param index
	 *            the index of the macroclassifier
	 * @return the index'th macroclassifier numerosity
	 */
	public final int getClassifierNumerosity(final int index) {
		return this.myMacroclassifiers.get(index).numerosity;
	}

	/**
	 * Returns (a copy of) the macroclassifier at the given index.
	 * 
	 * @param index
	 *            the index of the macroclassifier vector
	 * @return the macroclassifier at a given index
	 */
	public final Macroclassifier getMacroclassifier(final int index) {
		return new Macroclassifier(this.myMacroclassifiers.get(index));
	}
	
	
	
	/**
	 * returns the myMacroclassifiers vector
	 * 
	 * @author A. Filotheou
	 * 
	 */
	public ArrayList<Macroclassifier> getMacroclassifiersVector() {
		return myMacroclassifiers;
	}
	
	/**
	 * Returns the actual (not a copy as the aboce method) macroclassifier at the given index.
	 * 
	 * @param index
	 *            the index of the macroclassifier vector
	 * @return the macroclassifier at a given index
	 * 
	 * @author A. Filotheou
	 */
	
	public Macroclassifier getActualMacroclassifier(final int index) {
		return this.myMacroclassifiers.get(index);
	}
	
	/**
	 * Returns the actual (not a copy as the aboce method) macroclassifier that corresponds to the aClassifier classifier.
	 * 
	 * @param aClassifier
	 *            the classifier whose corresponding Macroclassifier we wish to obtain
	 * @return the macroclassifier
	 * 
	 * @author A. Filotheou
	 */
		
	
	public Macroclassifier getActualMacroclassifier(final Classifier aClassifier) {
		
		for (int i = 0; i < myMacroclassifiers.size(); i++) {
			if (myMacroclassifiers.get(i).myClassifier.getSerial() == aClassifier.getSerial()) 
				return this.myMacroclassifiers.get(i);
		}
		return null;	
	}


	/**
	 * Getter.
	 * 
	 * @return the number of macroclassifiers in the set
	 */
	public final int getNumberOfMacroclassifiers() {
		return this.myMacroclassifiers.size();
	}

	/**
	 * Get the set's population control strategy
	 * 
	 * @return the set's population control strategy
	 */
	public final IPopulationControlStrategy getPopulationControlStrategy() {
		return myISizeControlStrategy;
	}

	/**
	 * Returns the set's total numerosity (the total number of microclassifiers).
	 * @return  the sets total numerosity
	 */
	public final int getTotalNumerosity() {
		return this.totalNumerosity;
	}

	/**
	 * @return true if the set is empty
	 */
	public final boolean isEmpty() {
		return this.myMacroclassifiers.isEmpty();
	}
	
	
	
	
	public final int letPopulationSubsume(final Macroclassifier macro,
			  final boolean thoroughAdd) {
		
		// Subsume if possible
		// if thoroughAdd = true, before adding the given macro to the population, 
		// check it against the whole population for subsumption
		if (thoroughAdd) { 
			Vector<Integer> indicesVector    = new Vector<Integer>();
			Vector<Float> 	fitnessVector    = new Vector<Float>();
			Vector<Integer> experienceVector = new Vector<Integer>();
			/* 0 for generality, 1 for equality */
			Vector<Integer> originVector = new Vector<Integer>();

			
			final Classifier aClassifier = macro.myClassifier;
			for (int i = 0; i < myMacroclassifiers.size(); i++) {
				
				final Classifier theClassifier = myMacroclassifiers.get(i).myClassifier;
				
				if (theClassifier.canSubsume()) {
					if (theClassifier.isMoreGeneral(aClassifier)) {
						
						indicesVector.add(i);
						originVector.add(0);
						fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
						experienceVector.add(theClassifier.experience);
					}
				} else if (theClassifier.equals(aClassifier)) { // Or it can't
																// subsume but
																// it is equal
					indicesVector.add(i);
					originVector.add(1);
					fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float)theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
					experienceVector.add(theClassifier.experience);
				}

			} 
			
			int howManyGenerals = 0;
			int howManyEquals = 0;
			for (int i = 0; i < indicesVector.size(); i++) {
				if (originVector.elementAt(i) == 0)
						howManyGenerals++;
				else 
					howManyEquals++;
			}
			
			int indexOfSurvivor = 0;
			float maxFitness = 0;

			if (howManyGenerals !=  0) {
				
				for(int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 0) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
			}
			else if (howManyEquals != 0){
				
				for (int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 1) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						}
						else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}	
						}
					}
				}
				
			}
			
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
				
				int toBeReturned = indicesVector.elementAt(indexOfSurvivor);
				
				indicesVector.clear();
				originVector.clear();
				fitnessVector.clear();
				experienceVector.clear();
				
				return toBeReturned;
			}
			
		} 
		
		return -1;
		
	}
	
	public final void checkWholePopulationForPossibleSubsumptions() {
		
		Vector<Integer> indicesVector = new Vector<Integer>();
		Vector<Float> fitnessVector = new Vector<Float>();
		Vector<Integer> experienceVector = new Vector<Integer>();
		/* 0 for generality, 1 for equality */
		Vector<Integer> originVector = new Vector<Integer>();

		for (int c=0; c < myMacroclassifiers.size(); c++)
		{
			Classifier aClassifier = myMacroclassifiers.get(c).myClassifier;
			for (int i = c+1; i < myMacroclassifiers.size(); i++) {
	
				Classifier theClassifier = myMacroclassifiers.get(i).myClassifier;
	
				if (theClassifier.canSubsume()) {
					if (theClassifier.isMoreGeneral(aClassifier)) {
						indicesVector.add(i);
						originVector.add(0);
						fitnessVector.add(myMacroclassifiers.get(i).numerosity* (float) theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
						experienceVector.add(theClassifier.experience);
					}
				} 
				else if (theClassifier.equals(aClassifier)) { // Or it can't subsume but it is equal
					indicesVector.add(i);
					originVector.add(1);
					fitnessVector.add(myMacroclassifiers.get(i).numerosity * (float) theClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION));
					experienceVector.add(theClassifier.experience);
				}
			}
	
			int howManyGenerals = 0;
			int howManyEquals = 0;
			for (int i = 0; i < indicesVector.size(); i++) {
				if (originVector.elementAt(i) == 0)
					howManyGenerals++;
				else
					howManyEquals++;
			}
	
			int indexOfSurvivor = 0;
			float maxFitness = 0;
	
			if (howManyGenerals != 0) {
				for (int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 0) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						} else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}
						}
					}
				}
			} 
			else if (howManyEquals != 0) {
				for (int k = 0; k < indicesVector.size(); k++) {
					if (originVector.elementAt(k) == 1) {
						if (fitnessVector.elementAt(k) > maxFitness) {
							maxFitness = fitnessVector.elementAt(k);
							indexOfSurvivor = k;
						} else if (fitnessVector.elementAt(k) == maxFitness) {
							if (experienceVector.elementAt(k) >= experienceVector
									.elementAt(indexOfSurvivor)) {
								indexOfSurvivor = k;
							}
						}
					}
				}
	
			}
	
			// if subsumable:
			if (howManyGenerals != 0 || howManyEquals != 0) {
	
				int toBeReturned = indicesVector.elementAt(indexOfSurvivor);
				
				getMacroclassifiersVector().get(toBeReturned).numerosity += getMacroclassifiersVector().get(c).numerosity; 
				getMacroclassifiersVector().get(toBeReturned).numberOfSubsumptions++; 
				this.deleteMacroclassifier(c);
				
				indicesVector.clear();
				originVector.clear();
				fitnessVector.clear();
				experienceVector.clear();
	
				
			}
		}
		
	}


	/**
	 * Merge a set into this set.
	 * 
	 * @param aSet
	 *            the set to be merged.
	 */
	public final void merge(final ClassifierSet aSet) {
		final int setSize = aSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < setSize; i++) {
			final Macroclassifier ml = aSet.getMacroclassifier(i);
			this.addClassifier(ml, false);
		}
	}
	
	
	
	public final void mergeWithoutControl(final ClassifierSet aSet) {
		final int setSize = aSet.getNumberOfMacroclassifiers();
		for (int i = 0; i < setSize; i++) {
			final Macroclassifier ml = aSet.getMacroclassifier(i);
			final int numerosity = ml.numerosity;
			this.totalNumerosity += numerosity;
			this.myMacroclassifiers.add(ml);			
		}
	}

	

	/**
	 * Print all classifiers in the set.
	 */
	public final void print() {
		System.out.println(toString());
	}

	/**
	 * Remove all set's macroclassifiers.
	 */
	public final void removeAllMacroclassifiers() {
		this.myMacroclassifiers.clear();
		this.totalNumerosity = 0;
	}

	/**
	 * Self subsume. the fuck?
	 */
	public final void selfSubsume() {
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {
			final Macroclassifier cl = this.getMacroclassifier(0);
			final int numerosity = cl.numerosity;
			this.myMacroclassifiers.remove(0);
			this.totalNumerosity -= numerosity;
			this.addClassifier(cl, true);
		}
	}

	
	@Override
	public String toString() { // o buffer writes to population.txt. system.out -> console
		final StringBuffer response = new StringBuffer();
		
		double numOfCover = 0;
		double numOfGA = 0;
		double numOfInit = 0;
		int 	numOfSubsumptions = 0;
		int 	meanNs = 0;
		int 	coveredTotalNumerosity = 0;
		int 	gaedTotalNumerosity = 0;
		double meanAcc = 0;
		
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {

			double acc = this.getActualMacroclassifier(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
			if (Double.isNaN(acc)) 
				acc = 0;
			
			meanNs += this.getClassifier(i).getNs();
			meanAcc += acc * this.getMacroclassifier(i).numerosity;

		}
		
		if (this.getNumberOfMacroclassifiers() > 0) {
			meanNs /= this.getNumberOfMacroclassifiers();
			meanAcc /= this.getTotalNumerosity();
		}
		
        DecimalFormat df = new DecimalFormat("#.####");

        
        double accuracyOfCovered = 0;
        double accuracyOfGa = 0;
		for (int i = 0; i < this.getNumberOfMacroclassifiers(); i++) {

			
			//response.append(this.getClassifier(i).toString()
			response.append(
					myMacroclassifiers.get(i).myClassifier.toString() // antecedent => concequent
					+ "|"	
					//+ " total fitness: " + this.getClassifier(i).getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLOITATION) * this.getMacroclassifier(i).numerosity
					// myMacroclassifiers.elementAt(i).toString isos kalutera
					+ "macro fit:|" + df.format(myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION) 
							* myMacroclassifiers.get(i).numerosity) 
					+ "|"
					+ "fit:|" + df.format(myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_EXPLORATION))
					+ "|"
					+ "acc:|" + df.format(myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY))
					+ "|"
					+ "num:|" + myMacroclassifiers.get(i).numerosity 
					+ "|"
					+ "exp:|" + myMacroclassifiers.get(i).myClassifier.experience  
					+ "|"
					+ "cov:|" + (int) (myMacroclassifiers.get(i).myClassifier.objectiveCoverage * myMacroclassifiers.get(i).myClassifier.getLCS().instances.length)
					+ "|");
			
			response.append(myMacroclassifiers.get(i).myClassifier.getUpdateSpecificData());
						
			if (myMacroclassifiers.get(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_COVER) {
				numOfCover++;
				coveredTotalNumerosity += myMacroclassifiers.get(i).numerosity;
				accuracyOfCovered +=  myMacroclassifiers.get(i).numerosity * myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
				response.append("origin:|cover" + "|");
			}
			else if (myMacroclassifiers.get(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_GA) {
				numOfGA++;
				gaedTotalNumerosity += myMacroclassifiers.get(i).numerosity;
				accuracyOfGa +=  myMacroclassifiers.get(i).numerosity * myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
				response.append("origin:|ga" + "|");
			}
			else if (myMacroclassifiers.get(i).myClassifier.getClassifierOrigin() == Classifier.CLASSIFIER_ORIGIN_INIT) {
				numOfInit++;
				coveredTotalNumerosity += myMacroclassifiers.get(i).numerosity;
				accuracyOfCovered +=  myMacroclassifiers.get(i).numerosity * myMacroclassifiers.get(i).myClassifier.getComparisonValue(AbstractUpdateStrategy.COMPARISON_MODE_PURE_ACCURACY);
				response.append("origin:|init "+ "|");
			}	
			
			
			numOfSubsumptions += myMacroclassifiers.get(i).numberOfSubsumptions;
			response.append("created:|" + myMacroclassifiers.get(i).myClassifier.cummulativeInstanceCreated + "|");
			response.append("last in correctset:|" + myMacroclassifiers.get(i).myClassifier.timestamp + "|");
			response.append("subsumptions:|" + myMacroclassifiers.get(i).numberOfSubsumptions + "|");
			response.append("created:|" + (-Integer.MIN_VALUE + myMacroclassifiers.get(i).myClassifier.getSerial()) + "th" + "|");
			response.append(System.getProperty("line.separator"));
		}
		
		System.out.println("\nPopulation size (macro, micro): "  	+ "(" + this.getNumberOfMacroclassifiers() + "," + this.getTotalNumerosity() + ")");

		System.out.println("Classifiers in population covered: " 	+ (int) numOfCover);
		System.out.println("Classifiers in population ga-ed:   " 	+ (int) numOfGA);
		System.out.println("Classifiers in population init-ed: " 	+ (int) numOfInit);
		System.out.println();
		
		System.out.println("Accuracy of covered: " +  (Double.isNaN(accuracyOfCovered / numOfCover) ? 0 : accuracyOfCovered / coveredTotalNumerosity /*(numOfCover + numOfInit)*/));
		System.out.println("Accuracy of gaed:    " +  (Double.isNaN(accuracyOfGa/ numOfCover) ? 0 : accuracyOfGa / gaedTotalNumerosity/*numOfGA*/));
		System.out.println();

		
		System.out.println("Mean ns:   " + meanNs);
		System.out.println("Mean pure accuracy:   " + meanAcc);
		
		System.out.println("ga invocations: " 						+ this.totalGAInvocations);

		System.out.println("Subsumptions: " + numOfSubsumptions + "\n");

		return response.toString();
	}

}