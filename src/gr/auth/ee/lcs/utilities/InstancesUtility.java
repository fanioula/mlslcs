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
package gr.auth.ee.lcs.utilities;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;

/**
 * A utility class for converting a Weka Instance to a double array
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */

public final class InstancesUtility {
	
	public static Vector<Instances[]> testInstances = new Vector<Instances[]>();
	public static Vector<Instances[]> trainInstances = new Vector<Instances[]>();

	/**
	 * Perform the conversion.
	 * 
	 * @param set
	 *            the set containing the instances
	 * @return a double[][] containing the instances and their respective
	 *         attributes
	 */
	public static double[][] convertIntancesToDouble(final Instances set) {
		if (set == null)
			return null;

		final double[][] result = new double[set.numInstances()][set
				.numAttributes()];
		for (int i = 0; i < set.numInstances(); i++) {

			for (int j = 0; j < set.numAttributes(); j++) {
				result[i][j] = set.instance(i).value(j);
			}
		}

		return result;

	}

	/**
	 * Opens an file and creates an instance
	 * 
	 * @param filename
	 * @return the Weka Instances opened by the file
	 * @throws IOException
	 */
	public static Instances openInstance(final String filename)
			throws IOException {
		final FileReader reader = new FileReader(filename);
		return new Instances(reader);
	};

	/**
	 * Private Constructor to avoid instantiation.
	 */
	private InstancesUtility() {
	}
	

	
	/**
	 * Returns the label cardinality of the specified set.
	 * 
	 */
	public static double getLabelCardinality (final Instances set) { 
		if (set == null) return -1;
		
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		double sumOfLabels = 0;

		for (int i = 0; i < set.numInstances(); i++) {
			for (int j = set.numAttributes() - numberOfLabels; j < set.numAttributes(); j++) {
				sumOfLabels += set.instance(i).value(j);
			}
		}
		
		if (set.numInstances()!= 0) {

			return sumOfLabels / set.numInstances(); 
		}
		return 0;
	}
	
	/**
	 * The number of instances are multiple of the number of folds.
	 * From a se t of instances, it returns a chunk whose length is instances.numInstances / numberOfFolds
	 * with index = index. Index starts at zero.
	 * 
	 * In essencem this is used when splitting a partition of instances to a train and test set.
	 * 
	 * One chunk is the test set and the rest is the train set.
	 * We provide the index for the test set and the rest will automatically become the train set

	 * see splitPartitionIntoFolds
	 * 
	 * _____
	 * |_6_| index = 0
	 * |_6_|		 1
	 * |_6_|		 2 
	 * |_6_|		 3
	 * |_6_|		 4	
	 * |_6_|		 5
	 * |_6_|		 6
	 * |_6_|		 7		
	 * |_6_|		 8
	 * |_6_|		 9
	 * 
	 * */
	public static Instances getPartitionSegment(Instances instances, int index, int numberOfFolds) {
		
		if (instances.numInstances() % numberOfFolds != 0) {
			System.out.println("Number of instances not a multiple of " + numberOfFolds);
			return null;
		}
		
		int numberOfInstancesToGet = instances.numInstances() / numberOfFolds;
		Instances segment = new Instances(instances, numberOfInstancesToGet);
		
		for (int i = index * numberOfInstancesToGet; i < (index + 1) * numberOfInstancesToGet; i++) {
			segment.add(instances.instance(i));
		}
		return segment;
	}
	
	
	
	
	
	/**
	 * Splits the .arff input dataset to |number-of-distinct-label-combinations| Instances which are stored in the partitions[] array. 
	 * Called by initializePopulation() as a preparatory step to clustering.
	 * @throws Exception 
	 * 
	 * */
	
	public static Instances[] partitionInstances (final AbstractLearningClassifierSystem lcs, 
													final String filename) 
																			throws Exception {

		// Open .arff
		final Instances set = InstancesUtility.openInstance(filename);
		if (set.classIndex() < 0) {
			set.setClassIndex(set.numAttributes() - 1);
		}
		//set.randomize(new Random());
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		
		// the partitions vector holds the indices		
		String stringsArray[] = new String [lcs.instances.length];
		int indicesArray[] = new int [lcs.instances.length];

		// convert each instance's labelset into a string and store it in the stringsArray array
		for (int i = 0; i < set.numInstances(); i++) {
			stringsArray[i] = "";
			indicesArray[i] = i; 

			for (int j = set.numAttributes() - numberOfLabels; j < set.numAttributes(); j++) {
				stringsArray[i] += (int) set.instance(i).value(j);
			}
		}

		// contains the indicesVector(s)
		Vector<Vector> mothershipVector = new Vector<Vector>();
		
		String baseString = "";
		for (int i = 0; i < set.numInstances(); i++) {
			
			baseString = stringsArray[i];
			if (baseString.equals("")) continue;
			Vector<Integer> indicesVector = new Vector<Integer>();
			
			for (int j = 0; j < set.numInstances(); j++) {
				if (baseString.equals(stringsArray[j])) {
					stringsArray[j] = "";
					indicesVector.add(j);
				}
			}
			mothershipVector.add(indicesVector);
		}
		
		Instances[] partitions = new Instances[mothershipVector.size()];
		
		for (int i = 0; i < mothershipVector.size(); i++) {
			partitions[i] =  new Instances(set, mothershipVector.elementAt(i).size());
			for (int j = 0; j < mothershipVector.elementAt(i).size(); j++) {
				Instance instanceToAdd = set.instance((Integer) mothershipVector.elementAt(i).elementAt(j));
				partitions[i].add(instanceToAdd);
			}
		}	
		/*
		 * up to here, the partitions array has been formed. it contains the split dataset by label combinations
		 * it holds both the attributes and the labels, but for clustering the input should only be the attributes,
		 * so we need to delete the labels. this is taken care of by initializePopulation()
		 */
		return partitions;
	}
	
	
	
	public static Instances[] partitionInstances (final AbstractLearningClassifierSystem lcs, 
													final Instances trainSet) 
									throws Exception {

		// Open .arff
		final Instances set = trainSet;
		if (set.classIndex() < 0) {
			set.setClassIndex(set.numAttributes() - 1);
		}
		//set.randomize(new Random());
		int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1);
		
		// the partitions vector holds the indices		
		String stringsArray[] = new String[trainSet.numInstances()];
		int indicesArray[] = new int [trainSet.numInstances()];
		
		// convert each instance's labelset into a string and store it in the stringsArray array
		for (int i = 0; i < set.numInstances(); i++) {
			stringsArray[i] = "";
			indicesArray[i] = i; 
		
			for (int j = set.numAttributes() - numberOfLabels; j < set.numAttributes(); j++) {
				stringsArray[i] += (int) set.instance(i).value(j);
			}
		}
		
		// contains the indicesVector(s)
		Vector<Vector> mothershipVector = new Vector<Vector>();
		
		String baseString = "";
		for (int i = 0; i < set.numInstances(); i++) {
		
			baseString = stringsArray[i];
			if (baseString.equals("")) continue;
			Vector<Integer> indicesVector = new Vector<Integer>();
			
			for (int j = 0; j < set.numInstances(); j++) {
				if (baseString.equals(stringsArray[j])) {
					stringsArray[j] = "";
					indicesVector.add(j);
				}
			}
			mothershipVector.add(indicesVector);
		}
		
		Instances[] partitions = new Instances[mothershipVector.size()];
		
		for (int i = 0; i < mothershipVector.size(); i++) {
			partitions[i] =  new Instances(set, mothershipVector.elementAt(i).size());
			for (int j = 0; j < mothershipVector.elementAt(i).size(); j++) {
				Instance instanceToAdd = set.instance((Integer) mothershipVector.elementAt(i).elementAt(j));
				partitions[i].add(instanceToAdd);
			}
		}	
		/*
		 * up to here, the partitions array has been formed. it contains the split dataset by label combinations
		 * it holds both the attributes and the labels, but for clustering the input should only be the attributes,
		 * so we need to delete the labels. this is taken care of by initializePopulation()
		 */
		return partitions;
		}
	
	

	
	public static void splitDatasetIntoFolds (final AbstractLearningClassifierSystem lcs, 
												final Instances dataset,
												final int numberOfFolds) throws Exception {

		Instances[] partitions = InstancesUtility.partitionInstances(lcs, dataset);
		
		testInstances.setSize(partitions.length);
		trainInstances.setSize(partitions.length);
		
		
		int upperBound = (int) Math.ceil((double) dataset.numInstances() / (double) numberOfFolds);
		
		int [] numberOfTestInstancesPerFold = new int[numberOfFolds];
		
		
		/*
		 * let X partitions have partitions[i].numInstances() > numberOfFolds. 
		 * Then, vectors testInstances and trainInstances, after the call of splitPartitionIntoFolds(), will hold X arrays 
 		 *	meaning X elements.  
		 * */
		Vector<Integer> vectorOfPartitionIndices = new Vector<Integer>();
		


		for (int i = 0; i < partitions.length; i++) {
			
			if (partitions[i].numInstances() > numberOfFolds) {
				InstancesUtility.splitPartitionIntoFolds(partitions[i], numberOfFolds, i);
				vectorOfPartitionIndices.add(i);
			}	
			else {
				
				
				Instances[] emptyArrayTest = new Instances[numberOfFolds];
				Instances[] emptyArrayTrain = new Instances[numberOfFolds];

				for (int j = 0; j < numberOfFolds; j++) {
					emptyArrayTest[j] = new Instances (partitions[0], partitions[i].numInstances());
					emptyArrayTrain[j] = new Instances (partitions[0], partitions[i].numInstances());

				}
				//placeholders
				InstancesUtility.testInstances.add(i, emptyArrayTest);
				InstancesUtility.trainInstances.add(i, emptyArrayTrain);
			}	
		}
		
		/*
		 * At this point all partitions with numInstances > numFolds have been successfully been split.
		 * What is left is splitting the leftovers. 1st from the above partitions and 2nd from the ones that originally had numInstances < numFolds
		 * */

		
		
		for (int i = 0; i < numberOfFolds; i++) {
			int instancesSum = 0;
			for (int j = 0; j < vectorOfPartitionIndices.size(); j++) {
				instancesSum += InstancesUtility.testInstances.elementAt(vectorOfPartitionIndices.elementAt(j))[i].numInstances();	
			}
			
			// initial number of instances in test set per fold
			numberOfTestInstancesPerFold[i] = instancesSum;
		}
		
		/*
		 * 
		 *  i = 0 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 1 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 2 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 3 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 4 |_0|_0|_0|_0|_0|_0|_0|_0|_0|_0|
			i = 5 |_1|_1|_1|_1|_1|_1|_1|_1|_1|_1|
			i = 6 |_3|_3|_3|_3|_3|_3|_3|_3|_3|_3|
			i = 7 |_6|_6|_6|_6|_6|_6|_6|_6|_6|_6|
		 * 
		 * 
		 * */

		
		
		for (int i = 0; i < partitions.length; i++) {

			int numberOfLeftoverInstances = partitions[i].numInstances() % numberOfFolds; // eg 64 % 10 = 4
			Instances leftoverInstances = new Instances (partitions[i], numberOfLeftoverInstances);

			if (numberOfLeftoverInstances > 0) {
				/*
				 * Starting from the end. Anyhow they are the last {numberOfLeftoverInstances} instances in each partition
				 * that splitPartitionIntoFolds() has been called on.
				 * */
				for (int k = partitions[i].numInstances() - 1; k >= partitions[i].numInstances() - numberOfLeftoverInstances; k--) {
					leftoverInstances.add(partitions[i].instance(k));
				}

				
				/*
				 * For each partition, randomize the folds. Leftover instances will be placed in the first {numberOfLeftoverInstances} folds,
				 * that are already randomly distributed. If the first folds were not randomly distributed, there would be an uneven distribution,
				 * meaning that in the first ones there would be instances of the first partition and so on.
				 * 
				 * */
				
			    ArrayList<Integer> folds = new ArrayList<Integer>();
				
			    for (int k = 0; k < numberOfFolds; k++) {
			    	folds.add(k);
			    }
			 
			    Collections.shuffle(folds);  
			    
			    
			    
				int j = 0;
				while (leftoverInstances.numInstances() > 0) {
				    int foldIndex = folds.get(j);

					if (numberOfTestInstancesPerFold[foldIndex] < upperBound) {
	
						Instance toBeAdded = leftoverInstances.instance(0);
						
						// place the leftover first instance in a test set
						testInstances.elementAt(i)[foldIndex].add(toBeAdded);
						
						numberOfTestInstancesPerFold[foldIndex]++;
						
						// the instance placed in a test set for the current fold, needs to be put in the train set for all the other folds,
						// except for the current one of course
						for (int k = 0; k < numberOfFolds; k++) {
							if (k != foldIndex) {
								trainInstances.elementAt(i)[k].add(toBeAdded);
							}
						}
						
						// remove the instance placed in the test set
						leftoverInstances.delete(0);
						
					}
					j++;
					// if j hits the roof reset it. 
					// there may exist folds that have not reached their upper limit and abandon them
					if (j == numberOfFolds)
						j = 0;
				}
			}
		}
	}
	
	
	
	
	/**
	 * Splits a partition (collection of instances that belong to the same label combination) into train and test sets, leaving leftover instances.
	 * It presupposes that partition.numInstances > numberOfFolds.
	 * 
	 * Leftover instances should be distributed in a way that each test set holds
	 * 
	 * floor(totalNumInstances / numberOfFolds) <= testSetNumInstances <= ceil(totalNumInstances / numberOfFolds)
	 */
	public static void splitPartitionIntoFolds (Instances partition, int numberOfFolds, int partitionIndex) {
		
		int numberOfTestInstancesPerFold = partition.numInstances() / numberOfFolds; // eg 64 / 10 = 6
		int numberOfLeftoverInstances = partition.numInstances() % numberOfFolds; // eg 64 % 10 = 4
		
		Instances[] testArrayPerPartition = new Instances[numberOfFolds];
		Instances[] trainArrayPerPartition = new Instances[numberOfFolds];
		
		Instances bulk = new Instances(partition, partition.numInstances() - numberOfLeftoverInstances);
		
		/*
		 * E.g. I will split 64 total instances into 6 for testing, 54 for training and the rest (4) will be leftovers.
		 * 6 + 54 = 60 ~ 10
		 * The first 60 instances will be temporarily placed in the roundArray array
		 * */

		for (int i = 0; i < partition.numInstances() - numberOfLeftoverInstances; i++) {
			bulk.add(partition.instance(i));
		}
		
		
		for (int i = 0; i < numberOfFolds; i++) {
			testArrayPerPartition[i] = InstancesUtility.getPartitionSegment(bulk, i, numberOfFolds);
			trainArrayPerPartition[i] = new Instances(bulk, numberOfFolds);

			for (int j = 0; j < numberOfFolds; j++) {
				if (j != i) {
					for(int k = 0; k < numberOfTestInstancesPerFold; k++) {
						Instance kthInstance = InstancesUtility.getPartitionSegment(bulk, j, numberOfFolds).instance(k);
						trainArrayPerPartition[i].add(kthInstance);
					}
				}
			}	
		}
		
		/*
		 * In total, there will be partitions.length additions.
		 * Place each array in its respective place, depending on the partition index.
		 * */

		InstancesUtility.testInstances.add(partitionIndex, testArrayPerPartition);
		InstancesUtility.trainInstances.add(partitionIndex, trainArrayPerPartition);
	}	
}
		

