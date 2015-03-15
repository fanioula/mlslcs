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
package gr.auth.ee.lcs.data.representations.complex;

import gr.auth.ee.lcs.AbstractLearningClassifierSystem;
import gr.auth.ee.lcs.classifiers.Classifier;
import gr.auth.ee.lcs.classifiers.ClassifierSet;
import gr.auth.ee.lcs.data.ClassifierTransformBridge;
import gr.auth.ee.lcs.data.IClassificationStrategy;
import gr.auth.ee.lcs.utilities.ExtendedBitSet;
import gr.auth.ee.lcs.utilities.InstancesUtility;

import java.io.IOException;
import java.util.Enumeration;

import weka.core.Instances;

/**
 * A Complex representation for the chromosome.
 * 
 * @has 1 contains * AbstractAttribute
 * 
 * @author F. Tzima and M. Allamanis
 * 
 */
public abstract class ComplexRepresentation extends ClassifierTransformBridge {

	/**
	 * A inner abstract class that represents a single attribute.
	 * @author  F. Tzima and M. Allamanis
	 */
	public static abstract class AbstractAttribute {
		/**
		 * The length in bits of the attribute in the chromosome.
		 */
		protected int lengthInBits;

		/**
		 * The attribute's position in the chromosome.
		 */
		protected int positionInChromosome;

		/**
		 * The human-readable name of the attribute.
		 */
		protected String nameOfAttribute;

		/**
		 * The generalization rate used when covering.
		 */
		protected double generalizationRate;
		
		
		/**
		 * The generalization rate used when covering when clustering.
		 */
		protected double clusteringGeneralizationRate;
		
		

		/**
		 * The default constructor.
		 * 
		 * @param startPosition
		 *            the position in the chromosome to start the attribute
		 *            representation
		 * @param attributeName
		 *            the name of the attribute
		 * @param generalizationProbability
		 *            the generalization rate used
		 */
		protected AbstractAttribute(final int startPosition,
									final String attributeName,
									final double generalizationProbability) {
			
			nameOfAttribute = attributeName;
			positionInChromosome = startPosition;
			this.generalizationRate = generalizationProbability;
		}

		/**
		 * Fix an attribute representation.
		 * 
		 * @param generatedClassifier
		 *            the chromosome to fix
		 */
		public abstract void fixAttributeRepresentation(
				ExtendedBitSet generatedClassifier);

		/**
		 * @return  the length in bits of the chromosome.
		 */
		public final int getLengthInBits() {
			return lengthInBits;
		}

		/**
		 * Tests equality between genes.
		 * 
		 * @param baseChromosome
		 *            the base chromosome
		 * @param testChromosome
		 *            the test chromosome
		 * @return true if the genes are equivalent
		 */
		public abstract boolean isEqual(ExtendedBitSet baseChromosome,
				ExtendedBitSet testChromosome);

		/**
		 * Check if the attribute vision is a match to the chromosome.
		 * 
		 * @param attributeVision
		 *            the attribute vision value
		 * @param testedChromosome
		 *            the chromosome to test
		 * @return true if it is a match
		 */
		public abstract boolean isMatch(float attributeVision,
				ExtendedBitSet testedChromosome);

		/**
		 * Tests is baseChromsome's gene is more general than the test
		 * chromosome.
		 * 
		 * @param baseChromosome
		 *            the base chromosome
		 * @param testChromosome
		 *            the test chromosome
		 * @return true if base is more general than test
		 */
		public abstract boolean isMoreGeneral(ExtendedBitSet baseChromosome,
				ExtendedBitSet testChromosome);

		/**
		 * Returns true if the specific attribute is specific (not generic).
		 * 
		 * @return true when attribute is specific
		 */
		public abstract boolean isSpecific(final ExtendedBitSet testedChromosome);
		
		
		/**
		 * Create a random gene for the vision attribute.
		 * 
		 * @param attributeValue
		 *            the attribute value to cover
		 * @param generatedClassifier
		 *            the classifier where the chromosome will be generated
		 */
		public abstract void randomClusteringValue(float attributeValue,
				Classifier generatedClassifier);

		/**
		 * Create a random gene for the vision attribute.
		 * 
		 * @param attributeValue
		 *            the attribute value to cover
		 * @param generatedClassifier
		 *            the classifier where the chromosome will be generated
		 */
		public abstract void randomCoveringValue(float attributeValue,
				Classifier generatedClassifier);

		/**
		 * Convert Attribute to human-readable String.
		 * 
		 * @param convertingClassifier
		 *            the chromosome of the classifier to convert
		 * @return the string representation of the attribute
		 */
		public abstract String toString(ExtendedBitSet convertingClassifier);
	}

	/**
	 * A boolean attribute representation.
	 * 
	 * @author F. Tzima and M. Allamanis
	 * 
	 */
	public class BooleanAttribute extends AbstractAttribute {

		/**
		 * The constructor.
		 * 
		 * @param startPosition
		 *            the position attribute gene starts in chromosome
		 * @param attributeName
		 *            the name of the attribute
		 * @param generalizationRate
		 *            the generalization rate, used at covering
		 */
		public BooleanAttribute(final int startPosition,
								 final String attributeName, 
								 final double generalizationRate) {
			
			super(startPosition, attributeName, generalizationRate);
			lengthInBits = 2;
			chromosomeSize += lengthInBits;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #fixAttributeRepresentation
		 * (gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final void fixAttributeRepresentation(
				final ExtendedBitSet generatedClassifier) {
			return;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isEqual(gr.auth.ee.lcs.classifiers.ExtendedBitSet,
		 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isEqual(final ExtendedBitSet baseChromosome,
				final ExtendedBitSet testChromosome) {

			// if the base classifier is specific and the test is # return false
			if (baseChromosome.get(positionInChromosome) != testChromosome
					.get(positionInChromosome)) {
				return false;
			} else if ((baseChromosome.get(positionInChromosome + 1) != testChromosome
					.get(positionInChromosome + 1))
					&& baseChromosome.get(positionInChromosome)) {
				return false;
			}

			return true;
			
			
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isMatch(float, gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isMatch(final float attributeVision,
				final ExtendedBitSet testedChromosome) {

			if (testedChromosome.get(this.positionInChromosome)) {
				final boolean matches = (((attributeVision == 0)) ? false : true) == testedChromosome.get(this.positionInChromosome + 1); 
				return matches;

			} else {
				return true;	
			}
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isMoreGeneral(gr.auth.ee.lcs.classifiers.ExtendedBitSet,
		 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isMoreGeneral(final ExtendedBitSet baseChromosome,
				final ExtendedBitSet testChromosome) {
			
			// if the base classifier is specific and the test is # return false
			if (baseChromosome.get(positionInChromosome)
					&& !testChromosome.get(positionInChromosome))
				return false;
			if ((baseChromosome.get(positionInChromosome + 1) != testChromosome
					.get(positionInChromosome + 1))
					&& baseChromosome.get(positionInChromosome))
				return false;

			return true;
		}

		@Override
		public boolean isSpecific(final ExtendedBitSet testedChromosome) {
			return testedChromosome.get(this.positionInChromosome);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #randomCoveringValue(float, gr.auth.ee.lcs.classifiers.Classifier)
		 */
		@Override
		public final void randomCoveringValue(final float attributeValue,
				final Classifier generatedClassifier) {
			if (attributeValue == 0)
				generatedClassifier.clear(positionInChromosome + 1);
			else
				generatedClassifier.set(positionInChromosome + 1);

			if (Math.random() < generalizationRate)
				generatedClassifier.clear(positionInChromosome);
			else
				generatedClassifier.set(positionInChromosome);

		}
		
		@Override
		public final void randomClusteringValue(final float attributeValue,
				final Classifier generatedClassifier) {
			if (attributeValue == 0)
				generatedClassifier.clear(positionInChromosome + 1);
			else
				generatedClassifier.set(positionInChromosome + 1);

			if (Math.random() < clusteringAttributeGeneralizationRate)
				generatedClassifier.clear(positionInChromosome);
			else
				generatedClassifier.set(positionInChromosome);
			
		}

		
		
		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #toString(gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final String toString(final ExtendedBitSet convertingClassifier) {
			if (convertingClassifier.get(this.positionInChromosome)) {
				return convertingClassifier.get(this.positionInChromosome + 1) ? "1"
						: "0";
			} else {
				return "#";
			}

		}

	}

	/**
	 * A private inner abstract class that represents a numeric interval.
	 * 
	 * @author F. Tzima and M. Allamanis
	 */
	public class IntervalAttribute extends AbstractAttribute {

		/**
		 * The minimum and the maximum value that the attribute can receive.
		 */
		private final float minValue;

		/**
		 * The minimum and the maximum value that the attribute can receive.
		 */
		private final float maxValue;

		/**
		 * The number of bits to use for representing the interval limits.
		 */
		private final int precisionBits;

		/**
		 * The number of parts we have split the interval from min to max.
		 */
		private int totalParts = 0;

		/**
		 * The Interval Attribute Constructor.
		 * 
		 * @param startPosition
		 *            the position in the chromosome where the gene starts
		 * @param attributeName
		 *            the name of the attribute
		 * @param minValue
		 *            the minimum value of the attribute
		 * @param maxValue
		 *            the maximum value of the attribute
		 * @param precisionBits
		 *            the number of bits to use per number
		 * @param generalizationRate
		 *            the rate at which to create general attributes when
		 *            covering
		 */
		public IntervalAttribute(final int startPosition,
								  final String attributeName, 
								  final float minValue,
								  final float maxValue, 
								  final int precisionBits,
								  final double generalizationRate) {
			
			
			super(startPosition, attributeName, generalizationRate);
			this.minValue = minValue;
			this.maxValue = maxValue;
			this.precisionBits = precisionBits;
			// 2 values of bits + 1 activation bit
			lengthInBits = (2 * precisionBits) + 1;
			for (int i = 0; i < precisionBits; i++)
				totalParts |= 1 << i;
			chromosomeSize += lengthInBits;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #fixAttributeRepresentation
		 * (gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final void fixAttributeRepresentation(
				final ExtendedBitSet chromosome) {
			if (getLowBoundValue(chromosome) > getHighBoundValue(chromosome)) {
				// Swap
				final ExtendedBitSet low = chromosome.getSubSet(
						positionInChromosome + 1, precisionBits);
				final ExtendedBitSet high = chromosome
						.getSubSet(positionInChromosome + 1 + precisionBits,
								precisionBits);
				chromosome.setSubSet(positionInChromosome + 1, high);
				chromosome.setSubSet(positionInChromosome + 1 + precisionBits,
						low);
			}

		}

		/**
		 * Return the numeric high bound of the interval.
		 * 
		 * @param chromosome
		 *            the chromosome containing the interval
		 * @return the numeric value of the high bound
		 */
		private float getHighBoundValue(final ExtendedBitSet chromosome) {
			final int part = chromosome.getIntAt(positionInChromosome + 1
					+ precisionBits, precisionBits);

			return ((((float) part) / ((float) totalParts)) * (maxValue - minValue))
					+ minValue;
		}

		/**
		 * Return the numeric lower bound of the interval.
		 * 
		 * @param chromosome
		 *            the chromosome containing the interval
		 * @return the numeric value of the lower bound
		 */
		private float getLowBoundValue(final ExtendedBitSet chromosome) {
			final int part = chromosome.getIntAt(positionInChromosome + 1,
					precisionBits);

			return ((((float) part) / ((float) totalParts)) * (maxValue - minValue))
					+ minValue;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isEqual(gr.auth.ee.lcs.classifiers.ExtendedBitSet,
		 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isEqual(final ExtendedBitSet baseChromosome,
				final ExtendedBitSet testChromosome) {

			if (baseChromosome.get(positionInChromosome) != testChromosome
					.get(positionInChromosome))
				return false;

			if (!baseChromosome.get(positionInChromosome))
				return true;
			if (!baseChromosome.getSubSet(positionInChromosome + 1,
					precisionBits).equals(
					testChromosome.getSubSet(positionInChromosome + 1,
							precisionBits)))
				return false;

			return true;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isMatch(float, gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isMatch(final float attributeVision,
				final ExtendedBitSet testedChromosome) {

			if (!testedChromosome.get(positionInChromosome))
				return true;								
		

			return ((attributeVision >= getLowBoundValue(testedChromosome)) && (attributeVision <= getHighBoundValue(testedChromosome)));

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isMoreGeneral(gr.auth.ee.lcs.classifiers.ExtendedBitSet,
		 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isMoreGeneral(final ExtendedBitSet baseChromosome,
				final ExtendedBitSet testChromosome) {
			if (!baseChromosome.get(positionInChromosome))
				return true;
			if (!testChromosome.get(positionInChromosome))
				return false;
			if ((getHighBoundValue(baseChromosome) >= getHighBoundValue(testChromosome))
					&& (getLowBoundValue(baseChromosome) <= getLowBoundValue(testChromosome)))
				return true;
			return false;
		}

		@Override
		public boolean isSpecific(final ExtendedBitSet testedChromosome) {
			return testedChromosome.get(positionInChromosome);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #randomCoveringValue(float, gr.auth.ee.lcs.classifiers.Classifier)
		 */
		@Override
		public final void randomCoveringValue(final float attributeValue,
				final Classifier generatedClassifier) {
			// First find a random value that is smaller than the attribute
			// value & convert it to fraction
			final int newLowBound = (int) Math
					.floor((((attributeValue - minValue) * Math.random()) / (maxValue - minValue))
							* totalParts);
			final int newMaxBound = (int) Math
					.ceil((((maxValue - minValue - ((maxValue - attributeValue) * Math
							.random())) / (maxValue - minValue)) * totalParts));

			// Then set at chromosome
			if (Math.random() < (1 - generalizationRate))
				generatedClassifier.set(positionInChromosome);
			else
				generatedClassifier.clear(positionInChromosome);

			generatedClassifier.setIntAt(positionInChromosome + 1,
					precisionBits, newLowBound);
			generatedClassifier.setIntAt(positionInChromosome + 1
					+ precisionBits, precisionBits, newMaxBound);
		}

		@Override
		public final void randomClusteringValue(final float attributeValue,
				final Classifier generatedClassifier) {
			// First find a random value that is smaller than the attribute
			// value & convert it to fraction
			final int newLowBound = (int) Math
					.floor((((attributeValue - minValue) * Math.random()) / (maxValue - minValue))
							* totalParts);
			final int newMaxBound = (int) Math
					.ceil((((maxValue - minValue - ((maxValue - attributeValue) * Math
							.random())) / (maxValue - minValue)) * totalParts));

			// Then set at chromosome
			if (Math.random() < (1 - clusteringAttributeGeneralizationRate))
				generatedClassifier.set(positionInChromosome);
			else
				generatedClassifier.clear(positionInChromosome);

			generatedClassifier.setIntAt(positionInChromosome + 1,
					precisionBits, newLowBound);
			generatedClassifier.setIntAt(positionInChromosome + 1
					+ precisionBits, precisionBits, newMaxBound);
		}
		
		
		
		
		
		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #toString(gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final String toString(final ExtendedBitSet convertingChromosome) {
			// Check if condition active
			if (!convertingChromosome.get(positionInChromosome))
				return nameOfAttribute + ":#";

			final String value = nameOfAttribute
					+ " in ["
					+ String.format("%.3f",
							getLowBoundValue(convertingChromosome))
					+ ","
					+ String.format("%.3f",
							getHighBoundValue(convertingChromosome)) + "]";
			return value;
		}

	}

	/**
	 * A nominal attribute. It is supposed that the vision vector "sees" a
	 * number from 0 to n-1. Where n is the number of all possible values.
	 * 
	 * @author F. Tzima and M. Allamanis
	 */
	public class NominalAttribute extends AbstractAttribute {

		/**
		 * The names of the nominal values.
		 */
		private final String[] nominalValuesNames;

		/**
		 * Constructor.
		 * 
		 * @param startPosition
		 *            the start position of the gene
		 * @param attributeName
		 *            the name of the attribute
		 * @param nominalValuesNames
		 *            a string[] containing the names of the nominal values
		 * @param generalizationRate
		 *            the generalization rate used for covering
		 */
		public NominalAttribute(final int startPosition,
								 final String attributeName, 
								 final String[] nominalValuesNames,
								 final double generalizationRate) {
			
			super(startPosition, attributeName, generalizationRate);
			this.nominalValuesNames = nominalValuesNames;
			// We are going to use one bit per possible value plus one
			// activation bit
			lengthInBits = nominalValuesNames.length + 1;
			chromosomeSize += lengthInBits;
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #fixAttributeRepresentation
		 * (gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final void fixAttributeRepresentation(
				final ExtendedBitSet chromosome) {
			int ones = 0;
			if (chromosome.get(positionInChromosome)) // Specific
				for (int i = 1; i < this.lengthInBits; i++)
					if (chromosome.get(positionInChromosome + i))
						ones++;

			// Fix (we have none or all set)
			if ((ones == 0) || (ones == (lengthInBits - 1)))
				chromosome.clear(positionInChromosome);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isEqual(gr.auth.ee.lcs.classifiers.ExtendedBitSet,
		 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isEqual(final ExtendedBitSet baseChromosome,
				final ExtendedBitSet testChromosome) {
			if (baseChromosome.get(positionInChromosome) != testChromosome
					.get(positionInChromosome))
				return false;
			if (!baseChromosome.get(positionInChromosome))
				return true;
			if (!baseChromosome.getSubSet(positionInChromosome + 1,
					nominalValuesNames.length).equals(
					testChromosome.getSubSet(positionInChromosome + 1,
							nominalValuesNames.length)))
				return false;

			return true;

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isMatch(float, gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isMatch(final float attributeVision,
									   final ExtendedBitSet testedChromosome) {
			
			if (!testedChromosome.get(positionInChromosome)) 
				return true;								
			final int genePosition = (int) attributeVision + 1 + positionInChromosome;
			return testedChromosome.get(genePosition);

		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #isMoreGeneral(gr.auth.ee.lcs.classifiers.ExtendedBitSet,
		 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final boolean isMoreGeneral(final ExtendedBitSet baseChromosome,
				final ExtendedBitSet testChromosome) {
			if (!baseChromosome.get(positionInChromosome))
				return true;
			if (!testChromosome.get(positionInChromosome))
				return false;
			if (!baseChromosome.getSubSet(positionInChromosome + 1,
					nominalValuesNames.length).equals(
					testChromosome.getSubSet(positionInChromosome + 1,
							nominalValuesNames.length)))
				return false;

			return true;
		}

		@Override
		public boolean isSpecific(final ExtendedBitSet testedChromosome) {
			return testedChromosome.get(positionInChromosome);
		}

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #randomCoveringValue(float, gr.auth.ee.lcs.classifiers.Classifier)
		 */
		@Override
		public final void randomCoveringValue(final float attributeValue, // visionVector[i]
				final Classifier myChromosome) {
			// Clear everything
			myChromosome.clear(positionInChromosome, this.lengthInBits);
			if (Math.random() < (1 - generalizationRate))
				myChromosome.set(positionInChromosome);
			else
				myChromosome.clear(positionInChromosome);

			// Randomize all bits of gene
			for (int i = 1; i < lengthInBits; i++) {
				if (Math.random() < (.5))
					myChromosome.set(positionInChromosome + i);
				else
					myChromosome.clear(positionInChromosome + i);
			}

			// and set as "1" the nominal values that we are trying to match
			myChromosome.set(positionInChromosome + 1 + (int) attributeValue);

		}
		
		@Override
		public final void randomClusteringValue(final float attributeValue, // visionVector[i]
				final Classifier myChromosome) {
			// Clear everything
			myChromosome.clear(positionInChromosome, this.lengthInBits);
			if (Math.random() < (1 - clusteringAttributeGeneralizationRate))
				myChromosome.set(positionInChromosome);
			else
				myChromosome.clear(positionInChromosome);

			// Randomize all bits of gene
			for (int i = 1; i < lengthInBits; i++) {
				if (Math.random() < (.5))
					myChromosome.set(positionInChromosome + i);
				else
					myChromosome.clear(positionInChromosome + i);
			}

			// and set as "1" the nominal values that we are trying to match
			myChromosome.set(positionInChromosome + 1 + (int) attributeValue);

		}
		
		
		

		/*
		 * (non-Javadoc)
		 * 
		 * @see
		 * gr.auth.ee.lcs.data.representations.ComplexRepresentation.Attribute
		 * #toString(gr.auth.ee.lcs.classifiers.ExtendedBitSet)
		 */
		@Override
		public final String toString(final ExtendedBitSet convertingChromosome) {
			// Check if attribute is active
			if (!convertingChromosome.get(positionInChromosome))
				return nameOfAttribute + ":#";
			String attr = nameOfAttribute + " in [";
			for (int i = 0; i < nominalValuesNames.length; i++)
				if (convertingChromosome.get(positionInChromosome + 1 + i))
					attr += nominalValuesNames[i] + ", ";
			return attr + "]";
		}

	}

	/**
	 * The list of all attributes.
	 */
	protected AbstractAttribute[] attributeList;

	/**
	 * The attribute generalization rate.
	 */
	private final double attributeGeneralizationRate;
	
	/**
	 * The attribute generalization rate when clustering.
	 */
	private final double clusteringAttributeGeneralizationRate;

	/**
	 * The size of the chromosomes of the representation.
	 */
	protected int chromosomeSize = 0;

	/**
	 * The number of labels used.
	 */
	protected int numberOfLabels;

	/**
	 * The number of bits to be used for numerical values.
	 */
	protected int precision;

	/**
	 * The rule consequents for all classes.
	 */
	protected String[] ruleConsequents;

	/**
	 * The default classification strategy.
	 */
	private IClassificationStrategy defaultClassificationStrategy = null;

	/**
	 * The LCS instance used by the representation.
	 */
	protected final AbstractLearningClassifierSystem myLcs;

	
	/**
	 * Constructor.
	 * 
	 * @param attributes
	 *            the attribute objects of the representation
	 * @param ruleConsequentsNames
	 *            the rule consequents
	 * @param labels
	 *            the number of labels
	 * @param generalizationRate
	 *            the attribute generalization rate
	 * @param lcs
	 *            the LCS instance
	 */
	protected ComplexRepresentation(final AbstractAttribute[] attributes,
			final String[] ruleConsequentsNames, final int labels,
			final double generalizationRate, final double clusteringAttributeGeneralizationRate,
			final AbstractLearningClassifierSystem lcs) {
		super();
		this.attributeList = attributes;
		this.numberOfLabels = labels;
		this.ruleConsequents = ruleConsequentsNames;
		this.attributeGeneralizationRate = generalizationRate;
		this.clusteringAttributeGeneralizationRate = clusteringAttributeGeneralizationRate;
		this.myLcs = lcs;
	}
	
	/**
	 * Arff Loader.
	 * 
	 * @param inputArff
	 *            the input .arff
	 * @param precisionBits
	 *            bits used for precision
	 * @param labels
	 *            the number of labels (classes) in the set
	 * @param generalizationRate
	 *            the attribute generalization rate (P#)
	 * @param lcs
	 *            the LCS instance the representation belongs to
	 * @throws IOException
	 *             when .arff not found
	 */
	protected ComplexRepresentation(final String inputArff,
								    final int precisionBits, 
								    final int labels,
								    final double generalizationRate,
								    final double clusteringAttributeGeneralizationRate,
								    final AbstractLearningClassifierSystem lcs) throws IOException {

		this.myLcs = lcs;
		final Instances instances = InstancesUtility.openInstance(inputArff);

		this.numberOfLabels = labels;
		attributeList = new AbstractAttribute[instances.numAttributes()];
		this.attributeGeneralizationRate = generalizationRate;
		this.clusteringAttributeGeneralizationRate = clusteringAttributeGeneralizationRate;
		this.precision = precisionBits;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#areEqual(gr.auth.ee.lcs
	 * .classifiers.Classifier, gr.auth.ee.lcs.classifiers.Classifier)
	 */
	@Override
	public final boolean areEqual(final Classifier cl1, final Classifier cl2) {
		final ExtendedBitSet baseChromosome = cl1;
		final ExtendedBitSet testChromosome = cl2;

		// Check for equality starting with class
		for (int i = attributeList.length - 1; i >= 0; i--) {
			if (!attributeList[i].isEqual(baseChromosome, testChromosome))
				return false;
		}

		return true;
	}

	/**
	 * Build the representation for some instances.
	 * 
	 * @param instances
	 *            the instances
	 */
	
	protected void buildRepresentationFromInstance(final Instances instances) { 
		
		for (int i = 0; i < (instances.numAttributes() - numberOfLabels); i++) { 
			

			final String attributeName = instances.attribute(i).name();

			if (instances.attribute(i).isNominal()) {
				
				String[] attributeNames = new String[instances.attribute(i).numValues()]; 
				
				final Enumeration<?> values = instances.attribute(i).enumerateValues();
				
				for (int j = 0; j < attributeNames.length; j++) {
					attributeNames[j] = (String) values.nextElement();
				}
				
				// Create boolean or generic nominal
				if (attributeNames.length > 2)
					attributeList[i] = new ComplexRepresentation.NominalAttribute(
							this.chromosomeSize, 
							attributeName, 
							attributeNames, 
							attributeGeneralizationRate);
				else 
					attributeList[i] = new ComplexRepresentation.BooleanAttribute(
							chromosomeSize, 
							attributeName,
							attributeGeneralizationRate);

			} else if (instances.attribute(i).isNumeric()) {
				float minValue, maxValue;
				minValue = (float) instances.instance(0).toDoubleArray()[i];
				maxValue = minValue;
				for (int sample = 0; sample < instances.numInstances(); sample++) {
					final float currentVal = (float) instances.instance(sample)
							.toDoubleArray()[i];
					if (currentVal > maxValue)
						maxValue = currentVal;
					if (currentVal < minValue)
						minValue = currentVal;
				}
				attributeList[i] = new ComplexRepresentation.IntervalAttribute( 
																				this.chromosomeSize, 
																				attributeName, 
																				minValue, 
																				maxValue,
																				precision, 
																				attributeGeneralizationRate);
			}
		} 
		
		createClassRepresentation(instances);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#buildRepresentationModel()
	 */
	@Override
	public void buildRepresentationModel() {

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#classify(gr.auth.ee.lcs
	 * .classifiers.ClassifierSet, double[])
	 */
	@Override
	public final int[] classify(final ClassifierSet aSet,
							    final double[] visionVector) {
		
		return defaultClassificationStrategy.classify(aSet, visionVector);
	}

	/**
	 * Classify using another classification Strategy.
	 * 
	 * @param aSet
	 *            the set of classifiers used at the classification
	 * @param visionVector
	 *            the vision vector of the instance to be classified
	 * @param strategy
	 *            the strategy to use for classification
	 * @return the classification of the given instance
	 */
	public final int[] classify(final ClassifierSet aSet,
			final double[] visionVector, final IClassificationStrategy strategy) {
		return strategy.classify(aSet, visionVector);
	}

	/**
	 * Create the class representation depending on the problem.
	 * 
	 * @param instances
	 *            the Weka instances
	 */
	protected abstract void createClassRepresentation(Instances instances);
	
	
	
	
	@Override
	public final Classifier createRandomClusteringClassifier(final double[] visionVector) {
		
		final Classifier generatedClassifier = myLcs.getNewClassifier(); 
		
		for (int i = 0; i < attributeList.length; i++) {
			attributeList[i].randomClusteringValue((float) visionVector[i], generatedClassifier);
		}

		return generatedClassifier;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#createRandomCoveringClassifier
	 * (double[], int)
	 */
	@Override
	public final Classifier createRandomCoveringClassifier(final double[] visionVector) {
		
		final Classifier generatedClassifier = myLcs.getNewClassifier(); // kataskeuazo enan kainourio classifier
		
		for (int i = 0; i < attributeList.length; i++) {
			attributeList[i].randomCoveringValue((float) visionVector[i], generatedClassifier);
		}

		return generatedClassifier;
	}

	
	

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#fixChromosome(gr.auth.ee
	 * .lcs.classifiers.ExtendedBitSet)
	 */
	@Override
	public final void fixChromosome(final ExtendedBitSet aChromosome) {
		for (int i = 0; i < attributeList.length; i++) {
			attributeList[i].fixAttributeRepresentation(aChromosome);
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.ClassifierTransformBridge#getChromosomeSize()
	 */
	/**
	 * @return
	 */
	@Override
	public final int getChromosomeSize() {
		return chromosomeSize;
	}

	@Override
	public final String[] getLabelNames() {
		return ruleConsequents;
	}

	@Override
	public final int getNumberOfAttributes() {
		return attributeList.length - numberOfLabels;
	}

	@Override
	public boolean isAttributeSpecific(final Classifier aClassifier,
			final int attributeIndex) {
		return attributeList[attributeIndex].isSpecific(aClassifier);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.ClassifierTransformBridge#isMatch(double[],
	 * gr.auth.ee.lcs.classifiers.ExtendedBitSet)
	 */
	@Override
	public final boolean isMatch(final double[] visionVector, 
								   final ExtendedBitSet chromosome) {
		
		for (int i = 0; i < (attributeList.length - numberOfLabels); i++) {
			if (!attributeList[i].isMatch((float) visionVector[i], chromosome))
				return false;
		}
		return true;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#isMoreGeneral(gr.auth.ee
	 * .lcs.classifiers.Classifier, gr.auth.ee.lcs.classifiers.Classifier)
	 */
	@Override
	public final boolean isMoreGeneral(final Classifier baseClassifier,
			final Classifier testClassifier) {
		// Start from labels to the attributes
		for (int i = attributeList.length - 1; i >= 0; i--) {
			if (!attributeList[i].isMoreGeneral(baseClassifier, testClassifier))
				return false;
		}
		return true;
	}

	/**
	 * Setter of the classification strategy.
	 * 
	 * @param strategy
	 *            the strategy to use
	 */
	public final void setClassificationStrategy(final IClassificationStrategy strategy) {
		
		defaultClassificationStrategy = strategy;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see gr.auth.ee.lcs.data.ClassifierTransformBridge#
	 * setRepresentationSpecificClassifierData
	 * (gr.auth.ee.lcs.classifiers.Classifier)
	 */
	@Override
	public void setRepresentationSpecificClassifierData(
			final Classifier aClassifier) {

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#toBitSetString(gr.auth.
	 * ee.lcs.classifiers.Classifier)
	 */
	@Override
	public final String toBitSetString(final Classifier classifier) {

		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * gr.auth.ee.lcs.data.ClassifierTransformBridge#toNaturalLanguageString
	 * (gr.auth.ee.lcs.classifiers.Classifier)
	 */
	@Override
	public final String toNaturalLanguageString(final Classifier aClassifier) {
		final StringBuffer nlRule = new StringBuffer();
		for (int i = 0; i < (attributeList.length - numberOfLabels); i++) {
			
			final String attributeString = attributeList[i].toString(aClassifier);

			nlRule.append(attributeString); 
		}

		// Add consequence
		nlRule.append(" => ");
		for (int i = attributeList.length - numberOfLabels; i < attributeList.length; i++) {
			nlRule.append(attributeList[i].toString(aClassifier));
		}
		return nlRule.toString();
	}

}