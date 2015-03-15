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

//Test GIT with comment. Fani.
//Test GIT with second comment. Fani.
//Test GIT with third comment. Fani.
//This is the last comment before merging back to master. 

/**
 * 
 */
package gr.auth.ee.lcs;

import gr.auth.ee.lcs.utilities.InstancesUtility;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Properties;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;

import weka.core.Instances;

/**
 * n-fold evaluator.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public class FoldEvaluator {

	/**
	 * The number of labels of the dataset 
	 */
	final int numberOfLabels = (int) SettingsLoader.getNumericSetting("numberOfLabels", 1); 
	
	final boolean initializePopulation = SettingsLoader.getStringSetting("initializePopulation", "false").equals("true");
	final String file = SettingsLoader.getStringSetting("filename", ""); // to trainSet.arff


	
	/**
	 * An inner class representing a fold of training. To be run as a thread
	 * 
	 * @author F. Tzima and M. Allamanis
	 * 
	 */
	private final class FoldRunnable implements Runnable {
		private final int i;
		private final int numOfFoldRepetitions;

		/**
		 * The train set.
		 */
		private Instances trainSet;

		/**
		 * The test set.
		 */
		private Instances testSet;

		/**
		 * Constructor
		 * 
		 * @param nFold
		 *            the fold number
		 * @param numOfFoldRepetitions
		 *            the number of repetitions to perform
		 */
		private FoldRunnable(int nFold,
				int numOfFoldRepetitions) {
			this.i = nFold;
			this.numOfFoldRepetitions = numOfFoldRepetitions;
			createStaticFolds();
		}

		@Override
		public void run() {
			
			
			
			// mark commencement time in console
			final Calendar cal = Calendar.getInstance();
			final SimpleDateFormat sdf = new SimpleDateFormat("kk:mm:ss, dd/MM/yyyy");
			String timestampStart = sdf.format(cal.getTime());
			System.out.println("Execution started @ " + timestampStart + "\n");
			
			
			double[][]  results 	= new double[numOfFoldRepetitions][];
			double[] 	pcutResults = new double[13];
			double[] 	ivalResults = new double[13];
			double[] 	bestResults = new double[13];
						
			for (int repetition = 0; repetition < numOfFoldRepetitions; repetition++) {
				
				AbstractLearningClassifierSystem foldLCS = prototype.createNew(); // foldLCS = new AbstractLearningClassifierSystem
				
				System.out.println("Training Fold " + i);
				
				try {
					loadStaticMlStratifiedFold(i, foldLCS);
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				
				foldLCS.registerMultilabelHooks(InstancesUtility.convertIntancesToDouble(testSet), numberOfLabels, i);
				
				if (initializePopulation) {
					try {
						foldLCS.setRulePopulation(foldLCS.initializePopulation(trainSet));
						System.out.println("Population initialized.");
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
				
				foldLCS.train();
				
				System.out.println("Label cardinality: " + foldLCS.labelCardinality);

				// Gather results...
				results[repetition] = foldLCS.getEvaluations(testSet);
				// print the results for the current repetition. added 21.09.2012
				final String[] names = prototype.getEvaluationNames();
				printEvaluations(results[repetition], names);
				System.out.println("\n");
				
				// log evals to files
				for (int i = 0; i < results[repetition].length; i++) {
					try {
						final FileWriter fstream = new FileWriter(foldLCS.hookedMetricsFileDirectory + "/evals/" + names[i] + ".txt", true);
						final BufferedWriter buffer = new BufferedWriter(fstream);
						buffer.write(String.valueOf(results[repetition][i]));	
						buffer.flush();
						buffer.close();
					} 
					catch (Exception e) {
						e.printStackTrace();
					}	
				} 
 
			}
			
			
			// Determine better repetition among numOfFoldRepetitions (not number of folds, mind you) per evaluation method
			int best = 0;
			for (int j = 1; j < numOfFoldRepetitions; j++) {
				if (results[j][0] > results[best][0]) // accuracy (pcut)
					best = j;
			}

			pcutResults = results[best];
			
			best = 0;
			for (int j = 1; j < numOfFoldRepetitions; j++) {
				if (results[j][4] > results[best][4]) // accuracy (ival)
					best = j;
			}

			ivalResults = results[best];
			
			best = 0;
			for (int j = 1; j < numOfFoldRepetitions; j++) {
				if (results[j][8] > results[best][8]) // accuracy (best)
					best = j;
			}

			bestResults = results[best];
					
			gatherResults(pcutResults, ivalResults, bestResults, i);
			

			// mark end of execution
			final Calendar cal_2 = Calendar.getInstance();
			final SimpleDateFormat sdf_2 = new SimpleDateFormat("kk:mm:ss, dd/MM/yyyy");
			String timestampStop = sdf_2.format(cal_2.getTime());
			System.out.println("\nExecution stopped @ " + timestampStop + "\n");
			
			

			
		}

		/**
		 * Load a fold into the evaluator.
		 * 
		 * @param foldNumber
		 *            the fold's index
		 * @param lcs
		 *            the LCS that will be used to evaluate this fold
		 */
		@SuppressWarnings("unused")
		private void loadFold(int foldNumber,
				AbstractLearningClassifierSystem lcs) {

			trainSet = instances.trainCV(numOfFolds, foldNumber);
			lcs.instances = InstancesUtility.convertIntancesToDouble(trainSet);
			testSet = instances.testCV(numOfFolds, foldNumber);
			lcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet);
		}
		
		/**
		 * Load a multilabel stratified fold into the evaluator.
		 * 
		 * @param foldNumber
		 *            the fold's index
		 * @param lcs
		 *            the LCS that will be used to evaluate this fold
		 */
		@SuppressWarnings("unused")
		private void loadMlStratifiedFold(int foldNumber,
											AbstractLearningClassifierSystem lcs) {
			
			Instances trainInstances = new Instances (instances, 0);
			Instances testInstances = new Instances (instances, 0);
			
			int numberOfPartitions = InstancesUtility.testInstances.size() / 2;
			
			for (int i = 0; i < numberOfPartitions; i++) {
				for (int j = 0; j < InstancesUtility.testInstances.elementAt(i)[foldNumber].numInstances(); j++) {
					testInstances.add(InstancesUtility.testInstances.elementAt(i)[foldNumber].instance(j));
					
				}
				for (int j = 0; j < InstancesUtility.trainInstances.elementAt(i)[foldNumber].numInstances(); j++) {
					trainInstances.add(InstancesUtility.trainInstances.elementAt(i)[foldNumber].instance(j));
					
				}
				
			}
			
			trainSet = trainInstances;
			trainSet.randomize(new Random());
			
			testSet = testInstances;
			testSet.randomize(new Random());

			lcs.instances = InstancesUtility.convertIntancesToDouble(trainSet);
			lcs.testInstances = InstancesUtility.convertIntancesToDouble(testSet);
			

			lcs.trainSet = trainSet;
			lcs.testSet = testSet;

			lcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet);
			

		}
		
		
		/**
		 * Creates the folds when using n-cross validation and stores them in a directory from where they will be used in the future.
		 * Each run tests whether this directory exists or not. If yes, it exits.
		 * 
		 * This method is used to provide concrete folds that do not change over time,
		 * in order to attain a firm base for analysing the behaviour of the LCS.
		 * 
		 * */
		private void createStaticFolds() {
			
			// create directory dataset.arff without the .arff
			File dir = new File(file.substring(0, (file.length() - 5))); 
			if (!dir.exists()) {
			  dir.mkdirs();
			
			
				for (int foldNumber = 0; foldNumber < numOfFolds; foldNumber++) {
					Instances trainInstances = new Instances (instances, 0);
					Instances testInstances = new Instances (instances, 0);

					int numberOfPartitions = InstancesUtility.testInstances.size() / 2;
										
					for (int i = 0; i < numberOfPartitions; i++) {
						for (int j = 0; j < InstancesUtility.testInstances.elementAt(i)[foldNumber].numInstances(); j++) {
							testInstances.add(InstancesUtility.testInstances.elementAt(i)[foldNumber].instance(j));
							
						}
						for (int j = 0; j < InstancesUtility.trainInstances.elementAt(i)[foldNumber].numInstances(); j++) {
							trainInstances.add(InstancesUtility.trainInstances.elementAt(i)[foldNumber].instance(j));
							
						}
						
					}
					
					trainInstances.randomize(new Random());
					testInstances.randomize(new Random());
					
					try{	
						final FileWriter fstream_train = new FileWriter(dir + "/train_" + foldNumber + ".arff", false);
						final BufferedWriter buffer_train = new BufferedWriter(fstream_train);
						buffer_train.write(trainInstances.toString());				

						buffer_train.write("");
						buffer_train.flush();
						buffer_train.close();
					} 
					catch (Exception e) {
						e.printStackTrace();
					}
					
					try {
						final FileWriter fstream_test = new FileWriter(dir + "/test_" + foldNumber + ".arff", true);
						final BufferedWriter buffer_test = new BufferedWriter(fstream_test);
						buffer_test.write(testInstances.toString());				
						buffer_test.flush();
						buffer_test.close();
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}
		}
		
		
		/**
		 * Load the stored sub-datasets.
		 */
		private void loadStaticMlStratifiedFold(int foldNumber, AbstractLearningClassifierSystem lcs) throws IOException {
			
			final String staticFoldsDirectory = file.substring(0, (file.length() - 5)); // 5 = . a r f f
			
			trainSet = InstancesUtility.openInstance(staticFoldsDirectory + "/train_" + foldNumber + ".arff");;
			testSet = InstancesUtility.openInstance(staticFoldsDirectory + "/test_" + foldNumber + ".arff");;


			lcs.instances = InstancesUtility.convertIntancesToDouble(trainSet);
			lcs.testInstances = InstancesUtility.convertIntancesToDouble(testSet);	
			
			lcs.trainSet = trainSet;
			lcs.testSet = testSet;

			lcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet);
		}
		
	}
	
	

	/**
	 * The number of folds to separate the dataset.
	 */
	private final int numOfFolds;

	/**
	 * The LCS prototype to be evaluated.
	 */
	private final AbstractLearningClassifierSystem prototype;

	/**
	 * The instances that the LCS will be evaluated on.
	 */
	private final Instances instances;

	/**
	 * The evaluations.
	 */
	private double[][] evals;
	
	private double [][] pcutEvals;
	
	private double [][] ivalEvals;
	
	private double [][] bestEvals;

	/**
	 * The runs to run.
	 */
	final int runs;

	/**
	 * An executor service containing a thread pool to run folds
	 */
	final ExecutorService threadPool;

	/**
	 * Constructor.
	 * 
	 * @param folds
	 *            the number of folds
	 * @param myLcs
	 *            the LCS instance to be evaluated
	 * @param filename
	 *            the filename of the .arff containing the instances
	 * @throws IOException
	 *             if the file is not found.
	 */
	public FoldEvaluator(int folds, AbstractLearningClassifierSystem myLcs,
			final String filename) throws IOException {
		numOfFolds = folds;
		prototype = myLcs; 

		instances = InstancesUtility.openInstance(filename);
		runs = (int) SettingsLoader.getNumericSetting("foldsToRun", numOfFolds);
		instances.randomize(new Random());
		int numOfThreads = (int) SettingsLoader.getNumericSetting("numOfThreads", 1);
		threadPool = Executors.newFixedThreadPool(numOfThreads);
		
		try {
			InstancesUtility.splitDatasetIntoFolds(myLcs, instances, numOfFolds);
		} catch (Exception e1) {
			e1.printStackTrace();
		}

	}

	/**
	 * Constructor.
	 * 
	 * @param folds
	 *            the number of folds used at evaluation
	 * @param numberOfRuns
	 *            the number of runs
	 * @param myLcs
	 *            the LCS under evaluation
	 * @param inputInstances
	 *            the instances to evaluate the LCS on
	 */
	public FoldEvaluator(int folds, int numberOfRuns,
			AbstractLearningClassifierSystem myLcs, Instances inputInstances) {
		numOfFolds = folds;
		prototype = myLcs;
		instances = inputInstances;
		runs = numberOfRuns;

		int numOfThreads = (int) SettingsLoader.getNumericSetting(
				"numOfThreads", 1);
		threadPool = Executors.newFixedThreadPool(numOfThreads);
	}

	/**
	 * Calculate the mean of all fold metrics.
	 * 
	 * @param results
	 *            the results double array
	 * @return the mean for each row
	 */
	
	public static double[] calcMean(double[][] results) {
		final double[] means = new double[results[0].length];
		for (int i = 0; i < means.length; i++) {
			double sum = 0;
			for (int j = 0; j < results.length; j++) {
				sum += results[j][i];
			}
			means[i] = (sum) / (results.length);
		}
		return means;
	}
	
	public static double[] calcMean(double[][] pcutResults, 
							 double[][] ivalResults, 
							 double[][] bestResults) {
		
		final double[] means = new double[pcutResults[0].length];
		
		for (int i = 0; i < 4; i++) {
			double sum = 0;
			
			for (int j = 0; j < pcutResults.length; j++) {
				sum += pcutResults[j][i];
			}
			
			means[i] = (sum) / (pcutResults.length);
		}
		
		for (int i = 4; i < 8; i++) {
			double sum = 0;
			
			for (int j = 0; j < ivalResults.length; j++) {
				sum += ivalResults[j][i];
			}
			
			means[i] = (sum) / (ivalResults.length);
		}
		
		for (int i = 8; i < 12; i++) {
			double sum = 0;
			
			for (int j = 0; j < bestResults.length; j++) {
				sum += bestResults[j][i];
			}
			
			means[i] = (sum) / (bestResults.length);
		}
		
		// mean coverage statistic based on ival's best folds
		double sum = 0;
		for (int j = 0; j < ivalResults.length; j++) {
			sum += ivalResults[j][12];
		}
		
		means[12] = (sum) / (ivalResults.length);
		
		return means;
	}
	
	/**
	 * Perform evaluation.
	 */
	public void evaluate() {
		
		final int numOfFoldRepetitions = (int) SettingsLoader.getNumericSetting("numOfFoldRepetitions", 1); // repeat process per fold

		// calls run() {runs} times
		for (int currentRun = 0; currentRun < runs; currentRun++) { // fold execution resumption
			//if (currentRun > 0) {
				Runnable foldEval = new FoldRunnable(currentRun, numOfFoldRepetitions);
				this.threadPool.execute(foldEval);
			//}
		}

		this.threadPool.shutdown();
		try {
			this.threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			System.out.println("Thread Pool Interrupted");
			e.printStackTrace();
		}
		//final double[] means = calcMean(this.evals);
		final double[] means = calcMean(this.pcutEvals, this.ivalEvals, this.bestEvals);
		// print results
		final String[] names = prototype.getEvaluationNames();
		printEvaluations(means, names);
		storeFinalEvaluations(means, names, "hookedMetrics/finals.txt");
		

	}
	
	public synchronized double[][] gatherResults(double[] results, int fold) {
		if (evals == null) {
			evals = new double[runs][results.length];
		}

		evals[fold] = results;

		return evals;

	}
	
	/**
	 * Gather the results from a specific fold.
	 * 
	 * @param results
	 *            the results array
	 * @param fold
	 *            the fold the function is currently gathering
	 * 
	 * @return the double containing all evaluations (up to the point being
	 *         added)
	 */
	public synchronized void gatherResults(double[] pcutResults, 
												 double[] ivalResults, 
												 double[] bestResults, 
												 int fold) {
		
		if (pcutEvals == null) {
			pcutEvals = new double[runs][pcutResults.length];
			ivalEvals = new double[runs][ivalResults.length];
			bestEvals = new double[runs][bestResults.length];
		}

		pcutEvals[fold] = pcutResults;
		ivalEvals[fold] = ivalResults;
		bestEvals[fold] = bestResults;
		
	}
	

	/**
	 * Print the evaluations.
	 * 
	 * @param means
	 *            the array containing the evaluation means
	 */
	public static void printEvaluations(double[] means, String[] names) {
		for (int i = 0; i < means.length; i++) {
			System.out.println(names[i] + ":\t" + means[i]);
			if ((i + 1) % 4 == 0) 
				System.out.println();
		}  
	}
	
	public static void printEvaluations(double[][] means, String[] names) {
		for (int i = 0; i < 4; i++) {
			System.out.print(names[i] + ": \t");
			for (int num=0; num<3; num++)
				System.out.print(means[num][i+4*num] + "\t");
			System.out.println();
		}  
	}
	
	//write final results in hookedMetrics/finals.txt
	public static void storeFinalEvaluations (double[] means, String[] names, String path) {

		for (int i = 0; i < means.length; i++) {
			try {
				final FileWriter fstream = new FileWriter(path, true);
				final BufferedWriter buffer = new BufferedWriter(fstream);
				buffer.write(String.valueOf(names[i] + ":\t" + means[i]) + "\n");	
				if ((i + 1) % 4 == 0) 
					buffer.write(System.getProperty("line.separator"));
				buffer.flush();
				buffer.close();
			} 
			catch (Exception e) {
				e.printStackTrace();
			}	
		}  
		
	}
	
	public static void main(String[] args) throws IOException {
//		categorizeExpResultDirectoriesBasedOnExpProperty();
		calculateBestResultsPerEvalMethod();
//		calculateFoldMeansFromDirectory();
//		categorizeExperimentResultDirectories();
//		calculateMEKAFoldMeansFromDirectory();
	}
		
	public static void calculateBestResultsPerEvalMethod() throws IOException{
		
		int numOfFolds = 10;
		Vector<Vector<double[]>> resultVecs = new Vector<Vector<double[]>>();
		Vector<Vector<String>> pathVecs = new Vector<Vector<String>>();
		for (int i=0; i<numOfFolds; i++)
		{
			resultVecs.add(new Vector<double[]>());
			pathVecs.add(new Vector<String>());
		}
		
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
		
		String dirPath = "/Users/fanitzima/git/results/genbase2";
		File dir = new File(dirPath);
		if (dir.isDirectory()) {
			File[] files = dir.listFiles();
			for (File temp : files) {
				if (temp.isDirectory() && temp.getName().startsWith("2014") && !temp.getName().contains("useless"))
				{
					String tempDirName = temp.getAbsolutePath();
					int foldNum = Integer.parseInt(tempDirName.split("_fold")[1]);
					double[] evals = new double[names.length];
					for (int j=0; j< names.length; j++)
					{
						String toRead = tempDirName + "/evals/" + names[j] + ".txt";
						BufferedReader br = null;		 
						try {
							String sCurrentLine;
							br = new BufferedReader(new FileReader(toRead));
							while ((sCurrentLine = br.readLine()) != null) {
								double lala = Double.parseDouble(sCurrentLine);
								evals[j] = lala;
							}
						} 
						catch (IOException e) {
							e.printStackTrace();
						} 
						catch (NumberFormatException e) {
							e.printStackTrace();
						} 
						finally {
							try {
								if (br != null)br.close();
							} 
							catch (IOException ex) {
								ex.printStackTrace();
							}
						}
					}
					resultVecs.get(foldNum).add(evals);
					pathVecs.get(foldNum).add(tempDirName);
				}
			}
		}
		else
			System.exit(0);
		
		int[] metricsToOptimize = {0, 3, 4, 7, 8, 11};
//		int[] metricsToOptimize = {0, 4, 8};
		double[][] evals = new double[numOfFolds][names.length];
		String[] foldersToCopy = new String[numOfFolds];
		
		double[][] meansAcc = new double[3][names.length];
		double[][] meansEm = new double[3][names.length];
		
		for (int m=0; m<metricsToOptimize.length; m++)
		{
			int metricToOptimize = metricsToOptimize[m];
			
			FileUtils.deleteDirectory(new File(dirPath + "/" + names[metricToOptimize]));
			new File(dirPath + "/" + names[metricToOptimize]).mkdir();
			
			for (int i=0; i<numOfFolds; i++)
			{
				for (int j=0; j<resultVecs.get(i).size(); j++)
				{
					double lala = resultVecs.get(i).get(j)[metricToOptimize];
//					System.out.println("fold: " + i + " " + names[metricToOptimize] + ": " + lala);
					if (lala > evals[i][metricToOptimize])
					{
						evals[i] = resultVecs.get(i).get(j);
						foldersToCopy[i] = pathVecs.get(i).get(j);
					}
				}
//				System.out.println("fold: " + i + " " + names[metricToOptimize] + " MAX: " + evals[i][metricToOptimize]);
//				System.out.println();
			}
			
			final double[] means = calcMean(evals, evals, evals);
			if (metricToOptimize == 3 || metricToOptimize == 7 || metricToOptimize == 11)
				meansEm[metricToOptimize%3] = means;
			if (metricToOptimize == 0 || metricToOptimize == 4 || metricToOptimize == 8)
				meansAcc[metricToOptimize%3] = means;
			
//			printEvaluations(means, names);
			storeFinalEvaluations(means, names, dirPath + "/" + names[metricToOptimize] + "/results.txt");
			
			for (int i=0; i<numOfFolds; i++)
				FileUtils.copyDirectory(new File(foldersToCopy[i]), new File(dirPath + "/" + names[metricToOptimize] + "/" + new File(foldersToCopy[i]).getName()));
				
		}
		System.out.println("\nResults optimized for accuracy");
		System.out.println("--------------------------");
		printEvaluations(meansAcc, names);
		
		System.out.println("\nResults optimized for exact match");
		System.out.println("------------------------------");
		printEvaluations(meansEm, names);
	}	

	
	public static void calculateFoldMeansFromDirectory() throws IOException {
		
		int numOfFolds = 10;
		double[][] evals = new double[numOfFolds][13];
		
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
		
		String dirS = "/Users/fanitzima/git/results/genbase2/";
//		String[] dirPaths = {dirS+names[0], dirS+names[3], dirS+names[4], dirS+names[7], dirS+names[8], dirS+names[11]};
		String[] dirPaths = {dirS+names[4]};
	
		for (int d=0; d<dirPaths.length; d++)
		{
			String dirPath = dirPaths[d];
			File dir = new File(dirPath);
			if (dir.isDirectory()) {
				File[] files = dir.listFiles();
				for (File temp : files) {
					if (temp.isDirectory() && !temp.getName().contains("useless"))
					{
						String tempDirName = temp.getAbsolutePath();
						int foldNum = Integer.parseInt(tempDirName.split("_fold")[1]);
						
						for (int j=0; j< names.length; j++)
						{
							String toRead = tempDirName + "/evals/" + names[j] + ".txt";
							BufferedReader br = null;		 
							try {
								String sCurrentLine;
								br = new BufferedReader(new FileReader(toRead));
								while ((sCurrentLine = br.readLine()) != null) {
									double lala = Double.parseDouble(sCurrentLine);
									evals[foldNum][j] = lala;
								}
							} 
							catch (IOException e) {
								e.printStackTrace();
							} 
							finally {
								try {
									if (br != null)br.close();
								} 
								catch (IOException ex) {
									ex.printStackTrace();
								}
							}
						}
					}
				}
			}
			else
				System.exit(0);
			
			for (int i=0; i<10; i++)
				System.out.println("fold: " + i + " accuracy (ival): " + evals[i][4]);
			
			System.out.println("\n\n");
			
			
			final double[] means = calcMean(evals, evals, evals);
			printEvaluations(means, names);
			storeFinalEvaluations(means, names, dirPath + "/results.txt");
		}
		
//		dirPath = "/Users/fanitzima/git/results/useless/enron";
//		dir = new File(dirPath);
//		if (dir.isDirectory()) {
//			File[] files = dir.listFiles();
//			for (File temp : files) {
//				if (temp.isDirectory() && !temp.getName().startsWith("fold"))
//				{
//					String tempDirName = temp.getAbsolutePath();
//					int foldNum = Integer.parseInt(tempDirName.split("_fold")[1]);
//					String toRead = tempDirName + "/evals/" + names[4] + ".txt";
//					BufferedReader br = null;		 
//					try {
//						String sCurrentLine;
//						br = new BufferedReader(new FileReader(toRead));
//						while ((sCurrentLine = br.readLine()) != null) {
//							double lala = Double.parseDouble(sCurrentLine);
//							if (lala >= evals[foldNum][4])
//								System.out.println("fold " + foldNum + ": " + lala + " \t" + tempDirName);
//							else
//								FileUtils.deleteDirectory(new File(tempDirName));
//						}
//					} 
//					catch (IOException e) {
//						e.printStackTrace();
//					} 
//					catch (NumberFormatException e) {
//						e.printStackTrace();
//					} 
//					finally {
//						try {
//							if (br != null)br.close();
//						} 
//						catch (IOException ex) {
//							ex.printStackTrace();
//						}
//					}
//				}
//			}
//		}
//		else
//			System.exit(0);
	}
	
	public static void calculateMEKAFoldMeansFromDirectory() throws IOException {
		
		int numOfFolds = 10;
		double[][] evals = new double[numOfFolds][3];
		
		final String[] names = { "Accuracy", 
				 "Exact match",
				 "Hamming loss"
				 };
		
		String dirS = "/Volumes/Data/CODE_PROJECTS/meka-1.6.2/CC_results/CC_enron_train_";
		for (int i=0; i<10; i++) 
		{	
			File temp = new File(dirS+i + ".out");
			for (int j=0; j<names.length;j++)
			{
				
				BufferedReader br = null;		 
				try {
					String sCurrentLine;
					br = new BufferedReader(new FileReader(temp.getAbsolutePath()));
					while ((sCurrentLine = br.readLine()) != null) {
						if (sCurrentLine.trim().startsWith(names[j]))
						{
							double lala = Double.parseDouble(sCurrentLine.split(":")[1]);
							evals[i][j] = lala;
							break;
						}
					}
				} 
				catch (IOException e) {
					e.printStackTrace();
				} 
				finally {
					try {
						if (br != null)br.close();
					} 
					catch (IOException ex) {
						ex.printStackTrace();
					}
				}
			}
		}
		
		double means[] = calcMean(evals);
		printEvaluations(means, names);
	}

	public static void categorizeExperimentResultDirectories() throws IOException {
		
		String dirPath = "/Users/fanitzima/Desktop/TempExps/";
		File dir = new File(dirPath);
		if (dir.isDirectory()) {
			File[] files = dir.listFiles();
			for (File temp : files) {
				if (temp.isDirectory() && temp.getName().startsWith("2014") && !temp.getName().contains("useless"))
				{
					System.out.println(temp.getAbsolutePath());
					String toRead = "defaultLcs.properties";
					try 
					{
						BufferedReader br = new BufferedReader(new FileReader(dirPath + temp.getName() + "/" + toRead));
						Properties prop = new Properties();
						prop.clear();
						prop.load(br);
						String trainFileName = prop.getProperty("filename");
						String tempfn = trainFileName.split("datasets/")[1];
						String dataset = "";
						int fold = -1;
						if (tempfn.contains("/"))
						{
							dataset = tempfn.split("/")[0].trim();
							fold = Integer.parseInt(tempfn.split("/")[1].replace("train_", "").replace(".arff", ""));
							if (!temp.getName().contains("fold"))
								temp.renameTo(new File(temp.getAbsolutePath() + "_fold" + fold));
						}
						else
						{
							dataset = tempfn.replace(".arff", "").trim();
							fold = Integer.parseInt(temp.getName().split("fold")[1].trim());
						}
						
						File datasetDir = new File(dirPath+dataset);
						if (!datasetDir.exists())
							datasetDir.mkdir();
						
						System.out.println(tempfn);
						System.out.println(dataset + ": fold " + fold);
						System.out.println();
						
						temp.renameTo(new File(dirPath + dataset + "/" + temp.getName()));
						
						br.close();
					
					} catch (IOException ex) {
						ex.printStackTrace();
					}
					catch (Exception ex) {
						System.err.println(ex.getMessage());
						System.exit(0);
					}
				}
			}
		}
		else
			System.exit(0);
		
	}
	
	public static void categorizeExpResultDirectoriesBasedOnExpProperty() throws IOException {
		
		String dirPath = "/Users/fanitzima/git/results/CAL500_final2/";
		File dir = new File(dirPath);
		if (dir.isDirectory()) {
			File[] files = dir.listFiles();
			for (File temp : files) {
				if (temp.isDirectory() && temp.getName().startsWith("2014") && !temp.getName().contains("useless"))
				{
					System.out.println(temp.getAbsolutePath());
					String toRead = "defaultLcs.properties";
					try 
					{
						BufferedReader br = new BufferedReader(new FileReader(dirPath + temp.getName() + "/" + toRead));
						Properties prop = new Properties();
						prop.clear();
						prop.load(br);
						String lala = prop.getProperty("foldsToRun");
						
						File datasetDir = new File(dirPath+lala);
						if (!datasetDir.exists())
							datasetDir.mkdir();
						
						temp.renameTo(new File(dirPath + lala + "/" + temp.getName()));
						
						br.close();
					
					} catch (IOException ex) {
						ex.printStackTrace();
					}
					catch (Exception ex) {
						System.err.println(ex.getMessage());
						System.exit(0);
					}
				}
			}
		}
		else
			System.exit(0);
		
	}
	
	


}
