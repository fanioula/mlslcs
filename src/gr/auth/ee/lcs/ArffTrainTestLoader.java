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
package gr.auth.ee.lcs;

import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.classifiers.statistics.bundles.SetStatisticsBundle;
import gr.auth.ee.lcs.evaluators.TestFileClassification;
import gr.auth.ee.lcs.implementations.global.RepetitiveExec;
import gr.auth.ee.lcs.utilities.InstancesUtility;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Random;

import weka.core.Instances;

/**
 * A simple loader using an .arff file.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public class ArffTrainTestLoader {

	/**
	 * A test set.
	 */
	public Instances testSet;

	/**
	 * The current trainSet.
	 */
	public Instances trainSet;

	/**
	 * The LCS instance.
	 */
	private final AbstractLearningClassifierSystem myLcs;

	/**
	 * Constructor. Creates a loader for the given LCS.
	 * 
	 * @param lcs
	 *            the lcs where instances will be loaded on
	 */
	public ArffTrainTestLoader(final AbstractLearningClassifierSystem lcs) {
		myLcs = lcs;
	}

	/**
	 * Perform evaluation.
	 */
	public void evaluate() {
		
		// mark commencement time in console
		final Calendar cal = Calendar.getInstance();
		final SimpleDateFormat sdf = new SimpleDateFormat("kk:mm:ss, dd/MM/yyyy");
		String timestampStart = sdf.format(cal.getTime());
		System.out.println("Execution started @ " + timestampStart + "\n");
		
		
		final String rulesLoadFile = SettingsLoader.getStringSetting("loadRulesFile", "");
		
		if (!rulesLoadFile.isEmpty())
			myLcs.rulePopulation = ClassifierSet.openClassifierSet(
																	rulesLoadFile,
																	myLcs.rulePopulation.getPopulationControlStrategy(), 
																	myLcs);

		myLcs.train();
		
		/*
		 * The following two arrays have the same number of positions (12).
		 * Each evaluation name (names[i]) corresponds, in sequence, to an evaluation, evals[i] 
		 * */
		final double[] evals = myLcs.getEvaluations(testSet);
		final String[] names = myLcs.getEvaluationNames();
		
//		System.out.println(myLcs.rulePopulation);


		final String classificationFile = SettingsLoader.getStringSetting("testClassificationFile", "");
		
		if (!classificationFile.equals("")) {
			try {
				final TestFileClassification classificationModule = new TestFileClassification(
						InstancesUtility.convertIntancesToDouble(testSet),
						classificationFile, 
						myLcs,
						(int) SettingsLoader.getNumericSetting("numberOfLabels", 1));
				
				classificationModule.produceClassification();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

//		final String outputDir = SettingsLoader.getStringSetting("outputDir", "output");
		String rulesSaveFile = SettingsLoader.getStringSetting("saveRulesFile", "");
//		if (!outputDir.equals("") && !rulesSaveFile.equals(""))
//			rulesSaveFile  = outputDir + "/" + rulesSaveFile;

		if (!rulesSaveFile.isEmpty())
			ClassifierSet.saveClassifierSet(myLcs.rulePopulation, myLcs.hookedMetricsFileDirectory + "/" + rulesSaveFile);
		
		
		System.out.println("\n");
		String[] BAMrules1 = {
				"00#### => 11##",
				"01#### => 10##",
				"1##### => 01##",
				"####00 => ##00",
				"####01 => ##01",
				"####10 => ##10",
				"####11 => ##11"
		};
		double temp1 = ClassifierSet.percentageOfBAMDiscovered(myLcs.rulePopulation, BAMrules1);
		System.out.println(temp1*100 + "% of BAM discovered");
		
		String[] BAMrules2 = {
				"00#### => 11##",
				"01#### => 10##",
				"1##### => 01##",
				"#####0 => ###0",
				"#####1 => ###1",
				"####0# => ##0#",
				"####1# => ##1#",
		};
		double temp2 = ClassifierSet.percentageOfBAMDiscovered(myLcs.rulePopulation, BAMrules2);
		System.out.println(temp2*100 + "% of BAM discovered");
		System.out.println("\n");
		
		for (int i = 0; i < evals.length; i++) {
			System.out.println(names[i] + ": " + evals[i]);
			if ((i + 1) % 4 == 0) System.out.println();
		}
		
		RepetitiveExec.evals.add(evals);
		
		for (int i = 0; i < evals.length; i++) {
			try {
				final FileWriter fstream = new FileWriter(myLcs.hookedMetricsFileDirectory + "/evals/" + names[i] + ".txt", true);
				final BufferedWriter buffer = new BufferedWriter(fstream);
				buffer.write(String.valueOf(evals[i]));	
				buffer.flush();
				buffer.close();
			} 
			catch (Exception e) {
				e.printStackTrace();
			}	
		}                      
		                        
		SetStatisticsBundle bundle = new SetStatisticsBundle(myLcs, (int) SettingsLoader.getNumericSetting("numberOfLabels", 1));
		System.out.println(bundle);
	

		// mark end of execution
		final Calendar cal_2 = Calendar.getInstance();
		final SimpleDateFormat sdf_2 = new SimpleDateFormat("kk:mm:ss, dd/MM/yyyy");
		String timestampStop = sdf_2.format(cal_2.getTime());
		System.out.println("\nExecution stopped @ " + timestampStop + "\n");
		
	}

	/**
	 * Load instances into the global train store and create test set.
	 * 
	 * @param filename
	 *            the .arff filename to be used
	 * @param generateTestSet
	 *            true if a test set is going to be generated
	 * @throws IOException
	 *             if the input file is not found
	 */
	public final void loadInstances(final String filename,
			final boolean generateTestSet) throws IOException {
		// Open .arff
		final Instances set = InstancesUtility.openInstance(filename);
		if (set.classIndex() < 0) {
			set.setClassIndex(set.numAttributes() - 1);
		}
		set.randomize(new Random());

		if (generateTestSet) {
			final int numOfFolds = (int) SettingsLoader.getNumericSetting("NumberOfFolds", 10);
			final int fold = (int) Math.floor(Math.random() * numOfFolds);
			trainSet = set.trainCV(numOfFolds, fold);
			testSet = set.testCV(numOfFolds, fold);
		} else {
			trainSet = set;
		}

		myLcs.instances = InstancesUtility.convertIntancesToDouble(trainSet);
		myLcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet);

	}

	/**
	 * Load instances into the global train store and create test set.
	 * 
	 * @param filename
	 *            the .arff filename to be used
	 * @param testFile
	 *            the test file to be loaded
	 * @throws IOException
	 *             if the input file is not found
	 */
	public final void loadInstancesWithTest(final String filename,
											  final String testFile) throws IOException {
		
		// Open .arff
		final Instances set = InstancesUtility.openInstance(filename);

		if (set.classIndex() < 0)
			set.setClassIndex(set.numAttributes() - 1);
		set.randomize(new Random());
		trainSet = set;

		myLcs.instances = InstancesUtility.convertIntancesToDouble(trainSet); 
		myLcs.labelCardinality = InstancesUtility.getLabelCardinality(trainSet);
		testSet = InstancesUtility.openInstance(testFile);


		myLcs.trainSet = trainSet;
		myLcs.testSet = testSet;
		
		myLcs.testInstances = InstancesUtility.convertIntancesToDouble(testSet);
		
		System.out.println("Label cardinality: " + myLcs.labelCardinality);

	}
	

}
