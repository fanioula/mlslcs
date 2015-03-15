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
package gr.auth.ee.lcs.evaluators;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.data.ILCSMetric;
import gr.auth.ee.lcs.utilities.SettingsLoader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;


/**
 * An evaluator logging output to a file.
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public class FileLogger implements ILCSMetric {

	/**
	 * The filename where output is logged.
	 */
	private final String file;
	
	
	/**
	 * The directory that the metric files will be stored.
	 * 
	 * @author alexandros filotheou
	 * 
	 * */
	
	private static String storeDirectory;

	/**
	 * The evaluator from which we log the output.
	 */
	private final ILCSMetric actualEvaluator;
	
	
	
	/**
	 * A FileLogger constructor to set the directory in which the metrics are stored for all FileLoggers
	 * and copy a backup of the src directory for debugging purposes
	 * 
	 * 
	 * @author alexandros filotheou
	 * 
	 * */
	public FileLogger(final AbstractLearningClassifierSystem lcs, int foldNumber) {
		
		file = null;
		actualEvaluator = null;
		
		final Calendar cal = Calendar.getInstance();
		final SimpleDateFormat sdf = new SimpleDateFormat("yyyy.MM.dd 'at' kk.mm.ss.SSS");
		
		String timestamp = sdf.format(cal.getTime());
		
		// make directory hookedMetrics/{simpleDateFormat}
		String dirName = "hookedMetrics/" + timestamp;
		if (foldNumber>=0)		
			dirName += "_fold" + foldNumber;
		File dir = new File(dirName); 
		System.out.println(dir.toString());
		if (!dir.exists()) {
		  dir.mkdirs();
		}
		
		storeDirectory = dirName;	
		
		dir = new File(storeDirectory + "/evals"); 
		if (!dir.exists()) {
		  dir.mkdirs();
		}
		
		// set the name of the directory in which the metrics will be stored
		if (lcs.hookedMetricsFileDirectory == null) 
			lcs.setHookedMetricsFileDirectory(storeDirectory); 
		
		try {
			// keep a copy of thedefaultLcs.properties configuration 
			File getConfigurationFile = new File(storeDirectory, "defaultLcs.properties");
			
			if (!getConfigurationFile.exists()) {
				FileInputStream in = new FileInputStream("defaultLcs.properties");
				FileOutputStream out = new FileOutputStream(storeDirectory + "/defaultLcs.properties");
				byte[] buf = new byte[1024];
				int len;
				while ((len = in.read(buf)) > 0) {
				   out.write(buf, 0, len);
				}
				in.close();
				out.close();
			}

		} 
		catch (Exception e) {
			e.printStackTrace();
		}	
		
		
//		// copy the /src directory into storeDirectory
//		String sourceDir = "src";
//		File srcDir = new File(sourceDir);
//		String destinationDir = storeDirectory + "/src";
//		File destDir = new File(destinationDir);
//		if (!destDir.exists()) destDir.mkdir();
//		
//		try {
//			FileUtils.copyDirectory(srcDir, destDir);
//		} 
//		catch (Exception e) {
//			e.printStackTrace();
//		}
		
		try {
			// record fitness mode, deletion mode and whether # participate in the correct sets in the file essentialSettings.txt
			final FileWriter fstream = new FileWriter(storeDirectory + "/essentialSettings.txt", true);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			
			int fitness_mode = (int) SettingsLoader.getNumericSetting("FITNESS_MODE", 0);
			int deletion_mode = (int) SettingsLoader.getNumericSetting("DELETION_MODE", 0);
			boolean wildCardsParticipateInCorrectSets = String.valueOf(SettingsLoader.getStringSetting("wildCardsParticipateInCorrectSets", "true")).equals("true");
			boolean initializePopulation = String.valueOf(SettingsLoader.getStringSetting("initializePopulation", "true")).equals("true");

			buffer.write(					
					  "fitness mode: " + fitness_mode
					+ System.getProperty("line.separator")
					+ "deletion mode:  " + deletion_mode
					+ System.getProperty("line.separator")
					+ "# in correct sets :" + wildCardsParticipateInCorrectSets 
					+ System.getProperty("line.separator")
					+ (wildCardsParticipateInCorrectSets ? 
					 "balance correct sets: " + String.valueOf(SettingsLoader.getStringSetting("balanceCorrectSets", "true").equals("true"))
					+ (String.valueOf(SettingsLoader.getStringSetting("balanceCorrectSets", "true").equals("true") 
							? ", with ratio: " +  SettingsLoader.getNumericSetting("wildCardParticipationRatio", 0)
							: "")) : ""
					+ System.getProperty("line.separator")
					+ (initializePopulation ? "population initialized via clustering: " + initializePopulation : "")
					+ System.getProperty("line.separator"))
			);
			buffer.flush();
			buffer.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}	
		
		
	}
	

	/**
	 * FileLogger constructor.
	 * 
	 * @param filename
	 *            the filename of the file where log will be output.
	 * @param evaluator
	 *            the evaluator which we are going to output.
	 */
	public FileLogger(
					   final String filename, 
					   final ILCSMetric evaluator) {
		
		
		file = storeDirectory + "/" + filename + ".txt"; 
		actualEvaluator = evaluator;
		
		try {
			final FileWriter fstream = new FileWriter(file, false);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			buffer.write("");
			buffer.flush();
			buffer.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}

	}

	@Override
	public final double getMetric(final AbstractLearningClassifierSystem lcs) {
		
		final double evalResult = actualEvaluator.getMetric(lcs);
		
		try {
			
			final FileWriter fstream = new FileWriter(file, true);
			final BufferedWriter buffer = new BufferedWriter(fstream);
			buffer.write(
/*						String.valueOf(lcs.repetition) 
						+ ":" 
						+ */String.valueOf(evalResult)
						+ System.getProperty("line.separator"));
			buffer.flush();
			buffer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	@Override
	public String getMetricName() {
		return actualEvaluator.getMetricName();
	}
	
	



}
