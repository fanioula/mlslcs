package gr.auth.ee.lcs.geneticalgorithm.operators;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.geneticalgorithm.IBinaryGeneticOperator;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.SettingsLoader;

public class MultiPointCrossover implements IBinaryGeneticOperator {

	
	/**
	 * The LCS instance being used.
	 */
	final AbstractLearningClassifierSystem myLcs;
	
	
	public final int numberOfLabels; 

	
	/**
	 * Constructor.
	 */
	public MultiPointCrossover(AbstractLearningClassifierSystem lcs) {
		myLcs = lcs;
		numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);

	}
	
	
	@Override
	public Classifier operate(Classifier classifierA, 
							   Classifier classifierB, 
							   int label,
							   int mutationPoint) {		
				
		final Classifier child;
		child = myLcs.getNewClassifier(performCrossover(classifierA, classifierB, mutationPoint, label, numberOfLabels));

		return child;
	}
	
	
	/**
	 * A protected function that performs a multi point crossover.
	 * 
	 * @param chromosomeA
	 *            the first chromosome to crossover
	 * @param chromosomeB
	 *            the second chromosome to crossover
	 * @param position
	 *            the position (bit) to perform the crossover
	 * @return the new cross-overed (child) chromosome
	 */
	protected final ExtendedBitSet performCrossover(final ExtendedBitSet chromosomeA, 
													  final ExtendedBitSet chromosomeB,
													  final int position,
													  final int label, 
													  final int numberOfLabels) {
		// <numberOfLabels> should be a positive integer 
		if (numberOfLabels <= 0)
		{
				System.err.println("Multipoint crossover: argument <numberOfLabels> cannot have a negative or zero value.\nReturning a (non-mutated) clone of first parent.");
				return (ExtendedBitSet) chromosomeA.clone();
		}
		
		// chromosomeSize = number of representation bits for all attributes + 2 bits for one (the current) label 
		int chromosomeSize = chromosomeA.size() - 2 * (numberOfLabels - 1); 
		// <position> should be a positive integer less than or equal to (chromosomeSize - 1) 
		if (position > (chromosomeSize - 1) || position < 0)
		{
				System.err.println("Multipoint crossover: argument <position> cannot have a negative value or a value greater than chromosomeSize-1 (= " + (chromosomeSize - 1) + ").\nReturning a (non-mutated) clone of first parent.");
				return (ExtendedBitSet) chromosomeA.clone();
		}
		// <label> should be a positive integer less than or equal to (numberOfLabels - 1)
		if (label > (numberOfLabels - 1) || label < 0)
		{
				System.err.println("Multipoint crossover: argument <label> cannot have a negative value or a value greater than numberOfLabels (="+ numberOfLabels + ").\nReturning a (non-mutated) clone of first parent.");
				return (ExtendedBitSet) chromosomeA.clone();
		}

		final int antecedentBoundMax = chromosomeA.size() - 2 * numberOfLabels - 1;
		final int labelBoundMin = antecedentBoundMax + 2 * label + 1;
		
		final ExtendedBitSet child = (ExtendedBitSet) chromosomeA.clone();
		
//		if (position <= antecedentBoundMax) {
//			child.setSubSet(position, chromosomeB.getSubSet(position, antecedentBoundMax - position + 1)); 
//			child.setSubSet(labelBoundMin, chromosomeB.getSubSet(labelBoundMin, 2));
//		}
//		else {
//			int chromosomeSize = chromosomeA.size() - 2 * (numberOfLabels - 1); 
//			child.setSubSet(position + 2 * label, chromosomeB.getSubSet(position + 2 * label, chromosomeSize - position));
//		}
		
	
		// if <mutationPoint> corresponds to a bit within the attributes' representation bits, all attribute values in positions greater than or equal to <mutationPoint> are swapped
		if (position <= antecedentBoundMax) 
			child.setSubSet(position, chromosomeB.getSubSet(position, antecedentBoundMax - position + 1)); 
		
		// in any case, the current label (<label> parameter) is swapped
		child.setSubSet(labelBoundMin, chromosomeB.getSubSet(labelBoundMin, 2));

		return child;
	}
	

}
