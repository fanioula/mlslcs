/*
 *	Copyright (C) 2011 by AUTH
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
package gr.auth.ee.lcs.utilities;

import java.io.Serializable;
import java.util.Random;

/**
 * <p>
 * Title: ExtendedBitSet
 * </p>
 * <p>
 * Description: Bit string class that implements a more flexible set of
 * operators than
 * </p>
 * <p>
 * java.util.BitSet, though it implements all the routines of that class
 * </p>
 * <p>
 * so as to make this one plug-compatible. It also emulates BitSet's
 * </p>
 * <p>
 * functionality of transparently extending the set with zero bits if a bit
 * </p>
 * <p>
 * outside its current length is accessed.
 * </p>
 * <p>
 * Copyright: Copyright (c) 2003
 * </p>
 * <p>
 * Company: Aristotle University of Thessaloniki
 * </p>
 * 
 * @author Vangelis Valtos
 * @author Seraphim Seroglou
 * @version 1.0
 * @see java.util.BitSet
 */
@SuppressWarnings("serial")
public class ExtendedBitSet implements Cloneable, Serializable {
	/**
	 * Holds the bits of the bit set, in units of BITSINLONG=64 bits.
	 * @uml.property  name="bitUnits" multiplicity="(0 -1)" dimension="1"
	 */
	protected long[] bitUnits;

	/**
	 * Number of bits in the bit set.
	 * @uml.property  name="lenBits"
	 */
	protected int lenBits;

	protected static final int BITSINBYTE = Byte.SIZE;
	protected static final int BITSINCHAR = Character.SIZE;
	protected static final int BITSINSHORT = Short.SIZE;
	protected static final int BITSININT = Integer.SIZE;
	protected static final int BITSINLONG = Long.SIZE;

	private static Random random = new Random();

	private static final boolean compileTest = false;

	/**
	 * Simple unit test to check that this class is at least vaguely working.
	 * 
	 * @param args
	 *            Unused
	 **/
	public static void main(String[] args) {
		if (compileTest) {
			final ExtendedBitSet bitSet = new ExtendedBitSet("10010110");
			System.out.println("Initial set: " + bitSet);
			bitSet.rotateRight(0, 4, 2);
			System.out.println("rotateRight( 0, 4, 2 ): " + bitSet);
			bitSet.rotateLeft(4, 4, 2);
			System.out.println("rotateLeft( 4, 4, 2 ): " + bitSet);
			bitSet.set(8);
			System.out.println("set( 8 ): " + bitSet);
			bitSet.deleteSubSet(8, 1);
			System.out.println("deleteBitSet( 8, 1 ): " + bitSet);
			System.out.println("get( 7 ): " + bitSet.get(7));
			bitSet.setByteAt(1, (byte) 47);
			System.out.println("setByteAt( 1, 47 ): " + bitSet);
			System.out.println("getByteAt( 1 ): " + (int) bitSet.getByteAt(1));
			bitSet.setLongAt(7, -7L);
			System.out.println("setLongAt( 7, -7 ): " + bitSet);
			System.out.println("getLongAt( 7 ): " + bitSet.getLongAt(7));
			bitSet.shiftRightUnsigned(33, 30, 16);
			System.out.println("shiftRightUnsigned( 33, 33, 16 ): " + bitSet);
		}
	}

	/**
	 * Zero length set constructor.
	 */
	public ExtendedBitSet() {
		lenBits = 0;
	}

	/**
	 * Construct a copy of another extended bit set.
	 * 
	 * @param bitSet
	 *            the other bit set
	 */
	public ExtendedBitSet(final ExtendedBitSet bitSet) {
		lenBits = bitSet.lenBits;
		if ((bitUnits == null) || (bitUnits.length < bitSet.bitUnits.length)) {
			bitUnits = new long[bitSet.bitUnits.length];
		}
		System.arraycopy(bitSet.bitUnits, 0, bitUnits, 0,
				bitSet.bitUnits.length);
	}

	/**
	 * Construct with a given length initialised to zeros.
	 * 
	 * @param length
	 *            number of bits in the set
	 */
	public ExtendedBitSet(int length) {
		if (length < 1) {
			length = 1;
		}
		lenBits = length;
		bitUnits = new long[((lenBits - 1) / BITSINLONG) + 1];
	}

	/**
	 * Construct from an array of bytes. The contents of the bytes directly
	 * dictate the bits. Bit 0 of array element 0 is bit 0 of the bit set. bit 7
	 * of array element n is bit n*8+63 of the set, subject to the maximum
	 * length given by <code>length</code>.
	 * 
	 * @param length
	 *            number of bits in set
	 * @param bitPattern
	 *            initial bits values
	 */
	public ExtendedBitSet(int length, final byte[] bitPattern) {
		bitUnits = new long[((length - 1) / BITSINLONG) + 1];
		lenBits = length;
		for (int index = 0; length > 0; length -= BITSINLONG, ++index) {
			for (int j = 0; (j < 8) && (((index * 8) + j) < bitPattern.length); ++j) {
				bitUnits[index] |= (bitPattern[(index * 8) + j] & 0xFFL) << (8 * j);
			}
		}
	}

	/**
	 * Construct with a given length initialised to random values.
	 * 
	 * @param length
	 *            number of bits in the set
	 * @param randomSeed
	 *            seed for generator; if zero the generator is not re-seeded
	 */
	public ExtendedBitSet(final int length, final long randomSeed) {
		this(length);

		// Assign random values
		synchronized (random) {
			if (randomSeed != 0) {
				random.setSeed(randomSeed);
			}
			for (int i = 0; i < bitUnits.length; ++i) {
				bitUnits[i] = random.nextLong();
			}
		}

		// Set unused bits at the top of the last long to zero
		// The shift works because the JLS defines that only the bottommost
		// bits of the result are used to generate a value between 0-63.
		bitUnits[bitUnits.length - 1] &= -1L >>> (BITSINLONG - length); // Zero-fill
																		// right
																		// shift
	}

	/**
	 * Construct from an array of longs. The contents of the longs directly
	 * dictate the bits. Bit 0 of array element 0 is bit 0 of the bit set. bit
	 * 63 of array element n is bit n*64+63 of the set, subject to the maximum
	 * length given by <code>length</code>.
	 * 
	 * @param length
	 *            number of bits in set
	 * @param bitPattern
	 *            initial bits values
	 */
	public ExtendedBitSet(final int length, final long[] bitPattern) {
		initAll(length, bitPattern);
	}

	/**
	 * Construct from a string bit pattern. Each char should be 1 or 0 (though
	 * any non '0' is treated as a '1'). Using this constructor on the result of
	 * toString called on another bit set should result in a duplicate copy.
	 * 
	 * @param bitPattern
	 *            bit set as string of 1s and 0s.
	 */
	public ExtendedBitSet(final String bitPattern) {
		int index = 0;

		lenBits = bitPattern.length();
		bitUnits = new long[((lenBits - 1) / BITSINLONG) + 1];

		for (int i = lenBits; i > 0; i -= BITSINLONG) {
			long mask = 1;
			long val = 0;

			for (int j = 0; (j < BITSINLONG) && ((lenBits - j) > 0); ++j) {
				if (bitPattern.charAt(i - j - 1) != '0') {
					val |= mask;
				}
				// Shifts the bits mask to the left, by 1, with 0's shifted in
				// from the right.
				mask <<= 1;
			}

			bitUnits[index++] = val;
			// System.out.println(val);
		}
	}

	/**
	 * Logically ANDs the <strong>whole</strong> of this bit set with the
	 * specified set of bits. If <code>bitSet</code> is shorter than this set,
	 * the additional values are assumed to be zero; if it is longer, it is
	 * truncated.
	 * 
	 * @param bitSet
	 *            the bit set to be ANDed with
	 * @return this
	 */
	public final ExtendedBitSet and(ExtendedBitSet bitSet) {
		if (bitSet.lenBits < lenBits) {
			bitSet = (new ExtendedBitSet(lenBits)).setSubSet(0, bitSet);
		}
		return and(0, bitSet);
	}

	/**
	 * Logically ANDs a range of bits in this bit set with the specified set of
	 * bits. <code>offset</code> gives the start of the range, and the length of
	 * <code>bitSet</code> defines its length.
	 * 
	 * @param offset
	 *            the start of the range of bits to AND
	 * @param bitSet
	 *            the bit set to be ANDed with
	 * @return this
	 */
	public final ExtendedBitSet and(int offset, final ExtendedBitSet bitSet) {
		extendIfNeeded(offset + bitSet.lenBits);

		int otherOffset = 0;
		for (int length = bitSet.lenBits; length > 0; length -= BITSINLONG) {
			long newValue = getLongAt(offset, length)
					& bitSet.getLongAt(otherOffset, length);
			setLongAt(offset, length, newValue);
			offset += BITSINLONG;
			otherOffset += BITSINLONG;
			length -= BITSINLONG;
		}
		return this;
	}

	/**
	 * Returns the value of the bit set as a byte (truncating it if it is longer
	 * than eight bits).
	 * 
	 * @return the bit set represented as a byte
	 */
	public final byte byteValue() {
		return getByteAt(0, BITSINBYTE);
	}

	/**
	 * Returns the value of the bit set as a char (truncating it if it is longer
	 * than sixteen bits).
	 * 
	 * @return the bit set represented as a char
	 */
	public final char charValue() {
		return getCharAt(0, BITSINCHAR);
	}

	/**
	 * Set the value of a particular bit to false.
	 * 
	 * @param offset
	 *            the bit to clear
	 * @return this
	 */
	public final ExtendedBitSet clear(final int offset) {
		extendIfNeeded(offset + 1);

		final int thisLong = offset / BITSINLONG;
		final long thisBit = offset % BITSINLONG;

		bitUnits[thisLong] &= ~(1L << thisBit);

		return this;
	}

	/**
	 * Sets a range of bits to false. <code>length</code> bits starting at
	 * <code>offset</code> are set to false.
	 * 
	 * @param offset
	 *            start of range
	 * @param length
	 *            length of range
	 * @return this
	 */
	public final ExtendedBitSet clear(int offset, int length) {
		extendIfNeeded(offset + length);

		while (length > 0) {
			setLongAt(offset, length, 0);
			offset += BITSINLONG;
			length -= BITSINLONG;
		}

		return this;
	}

	/**
	 * Create and return a copy of this bit set.
	 * 
	 * @return a copy of this bit set
	 */
	@Override
	public Object clone() {
		return new ExtendedBitSet(this);
	}

	/**
	 * Copy the contents of another extended bit set.
	 * 
	 * @param bitSet
	 *            the other bit set
	 * @return the current bit set
	 */
	public final ExtendedBitSet copy(final ExtendedBitSet bitSet) {
		if (bitSet != this) {
			lenBits = bitSet.lenBits;
			if ((bitUnits == null)
					|| (bitUnits.length < bitSet.bitUnits.length)) {
				bitUnits = new long[bitSet.bitUnits.length];
			}
			System.arraycopy(bitSet.bitUnits, 0, bitUnits, 0,
					bitSet.bitUnits.length);
		}
		return this;
	}

	/**
	 * Move a subset of bits left. Shifts bits within a field of
	 * <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the left. The bottommost <code>shift</code>
	 * bits are left as they were. To give an example: <code>
	 *                      76543210
	 *      Initial bits:   01100101
	 * 
	 *      creepLeft( 2, 4, 1 )
	 * 
	 *      01|1001|01 -> 01|0011|01
	 * </code> Bits 0, 1, 6 and 7 are left untouched as they are outside the
	 * field defined by <code>offset</code> and <code>length</code>. Bit 5
	 * "drops of the end" of the field and is lost, bits 3 and 4 become bits 4
	 * and 5, and bit 2's value remains unchanged.
	 * 
	 * @param offset
	 *            start of field to creep
	 * @param length
	 *            length of field to creep
	 * @param shift
	 *            number of positions to creep left
	 * @return this
	 */
	public final ExtendedBitSet creepLeft(final int offset, int length,
			int shift) {
		extendIfNeeded(offset + length);

		// Emulate the behaviour of Java << operator, i.e. normalise shift
		// value to be somewhere between 0 and length - 1.

		shift %= length;
		if (shift < 0) {
			shift += length;
		}
		if (shift > 0) {
			// Shift the bits

			int destOffset = (offset + length) - BITSINLONG;
			int sourceOffset = destOffset - shift;
			if (sourceOffset < 0) {
				destOffset = offset + shift;
				sourceOffset = offset;
			}

			length -= shift;
			while (length > 0) {
				setLongAt(destOffset, length, getLongAt(sourceOffset, length));
				sourceOffset -= BITSINLONG;
				destOffset -= BITSINLONG;
				length -= BITSINLONG;
			}
		}

		return this;
	}

	/**
	 * Move a subset of bits right. Shifts bits within a field of
	 * <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the right. The topmost <code>shift</code>
	 * bits are left as they were. Example: <code>
	 *      01|1001|01 -> creepRight( 2, 4, 1 ) -> 01|1100|01
	 * </code> Read the explanation for creepLeft if this seems opaque.
	 * 
	 * @param offset
	 *            start of field to creep
	 * @param length
	 *            length of field to creep
	 * @param shift
	 *            number of positions to creep right
	 * @return this
	 * 
	 * @see creepLeft
	 */
	public final ExtendedBitSet creepRight(final int offset, int length,
			int shift) {
		extendIfNeeded(offset + length);

		// Emulate the behaviour of Java >>> operator, i.e. normalise shift
		// value to be somewhere between 0 and length - 1.

		shift %= length;
		if (shift < 0) {
			shift += length;
		}
		if (shift > 0) {
			int destOffset = offset;
			int sourceOffset = offset + shift;

			length -= shift;
			while (length > 0) {
				setLongAt(destOffset, length, getLongAt(sourceOffset, length));
				sourceOffset += BITSINLONG;
				destOffset += BITSINLONG;
				length -= BITSINLONG;
			}
		}

		return this;
	}

	/**
	 * Remove a chunk from a bit set and shuffle up the remainder. The overall
	 * length of the bit set is reduced by <code>length</code>.
	 * 
	 * @param offset
	 *            deletion point
	 * @param length
	 *            number of bits to remove
	 * @return this
	 */
	public final ExtendedBitSet deleteSubSet(final int offset, final int length) {
		shiftRightUnsigned(offset, lenBits - offset, length);
		lenBits -= length;

		return this;
	}

	/**
	 * Returns the value of the bit set as a double (truncating it if it is
	 * longer than sixtyfour bits).
	 * 
	 * @return the bit set represented as a double
	 */
	public final double doubleValue() {
		return getDoubleAt(0);
	}

	/**
	 * Compares this object against the specified object.
	 * 
	 * @param obj
	 *            the object to compare with
	 * @return true if the objects are the same; false otherwise
	 */
	@Override
	public final boolean equals(final Object obj) {
		boolean retValue = false;

		if (obj instanceof ExtendedBitSet) {
			final ExtendedBitSet other = (ExtendedBitSet) obj;

			if (this == other) {
				retValue = true;
			} else if (bitUnits == null) {
				retValue = (other.bitUnits == null);
			} else if (lenBits == other.lenBits) {
				// Ignore unused parts of the bit set (which might be there
				// if the set has shrunk in size)
				final int limit = (bitUnits.length < other.bitUnits.length) ? bitUnits.length
						: other.bitUnits.length;

				retValue = true;
				for (int i = 0; i < limit; ++i) {
					if (bitUnits[i] != other.bitUnits[i]) {
						retValue = false;
						break;
					}
				}
			}
		}

		return retValue;
	}

	/**
	 * Ensures that the bit set is large enough to hold <code>length</code> bits
	 * and extends it if it isn't.
	 * 
	 * @param length
	 *            number of bits set is required to hold
	 */
	protected final void extendIfNeeded(final int length) {
		if (length > lenBits) {
			final int chunks = ((length - 1) / BITSINLONG) + 1;

			if ((bitUnits == null) || (chunks > bitUnits.length)) {
				final long[] newBits = new long[chunks];

				if (bitUnits != null) {
					System.arraycopy(bitUnits, 0, newBits, 0, bitUnits.length);
				}
				bitUnits = newBits;
			}
			lenBits = length;
		}
	}

	/**
	 * Returns the value of the bit set as a float (truncating it if it is
	 * longer than thirtytwo bits).
	 * 
	 * @return the bit set represented as a float
	 */
	public final float floatValue() {
		return getFloatAt(0);
	}

	/**
	 * Get the value of a particular bit.
	 * 
	 * @param offset
	 *            the bit to read
	 * @return the bit's value
	 */
	public final boolean get(final int offset) {
		extendIfNeeded(offset + 1);

		final int thisLong = offset / BITSINLONG;
		final int thisBit = offset % BITSINLONG;

		return ((bitUnits[thisLong] & (1L << thisBit)) != 0);
	}

	/**
	 * Returns the value of the eight bits at a given offset in the set as a
	 * byte.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as a byte
	 */
	public final byte getByteAt(final int offset) {
		return getByteAt(offset, BITSINBYTE);
	}

	/**
	 * Returns the value of the <code>length</code> bits at a given offset in
	 * the set as a byte. If <code>length</code> is less than eight, the value
	 * is zero extended; if greater it is truncated.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @param length
	 *            the number of bits to read
	 * @return the bits represented as a byte
	 */
	public final byte getByteAt(final int offset, int length) {
		if (length > BITSINBYTE) {
			length = BITSINBYTE;
		}
		return (byte) getLongAt(offset, length);
	}

	/**
	 * Returns the value of the sixteen bits at a given offset in the set as a
	 * char.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as a char
	 */
	public final char getCharAt(final int offset) {
		return getCharAt(offset, BITSINCHAR);
	}

	/**
	 * Returns the value of the <code>length</code> bits at a given offset in
	 * the set as a char. If <code>length</code> is less than sixteen, the value
	 * is zero extended; if greater it is truncated.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @param length
	 *            the number of bits to read
	 * @return the bits represented as a char
	 */
	public final char getCharAt(final int offset, int length) {
		if (length > BITSINCHAR) {
			length = BITSINCHAR;
		}
		return (char) getLongAt(offset, length);
	}

	/**
	 * Returns the value of the sixtyfour bits at a given offset in the set as a
	 * double.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as a double
	 */
	public final double getDoubleAt(final int offset) {
		return Double.longBitsToDouble(getLongAt(offset, BITSINLONG));
	}

	/**
	 * Returns the value of the thirtytwo bits at a given offset in the set as a
	 * float.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as a float
	 */
	public final float getFloatAt(final int offset) {
		return Float.intBitsToFloat(getIntAt(offset, BITSININT));
	}

	/**
	 * Returns the value of the thirtytwo bits at a given offset in the set as
	 * an int.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as an int
	 */
	public final int getIntAt(final int offset) {
		return getIntAt(offset, BITSININT);
	}

	/**
	 * Returns the value of the <code>length</code> bits at a given offset in
	 * the set as an int. If <code>length</code> is less than thirtytwo, the
	 * value is zero extended; if greater it is truncated.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @param length
	 *            the number of bits to read
	 * @return the bits represented as an int
	 */
	public final int getIntAt(final int offset, int length) {
		if (length > BITSININT) {
			length = BITSININT;
		}
		return (int) getLongAt(offset, length);
	}

	/**
	 * Returns the value of the sixtyfour bits at a given offset in the set as a
	 * long.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as a long
	 */
	public final long getLongAt(final int offset) {
		return getLongAt(offset, BITSINLONG);
	}

	/**
	 * Returns the value of the <code>length</code> bits at a given offset in
	 * the set as a long. If <code>length</code> is less than sixtyfour, the
	 * value is zero extended; if greater it is truncated.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @param length
	 *            the number of bits to read
	 * @return the bits represented as a long
	 */
	public final long getLongAt(final int offset, int length) {
		if (length > BITSINLONG) {
			length = BITSINLONG;
		}
		final int block = offset / BITSINLONG;
		final int shift = offset % BITSINLONG;
		final long mask = -1L >>> (BITSINLONG - length);
		long retValue;

		extendIfNeeded(offset + length);

		if (shift == 0) {
			retValue = bitUnits[block];
		} else {
			retValue = bitUnits[block] >>> shift;
			if ((BITSINLONG - shift) < length) {
				retValue |= bitUnits[block + 1] << (BITSINLONG - shift);
			}
		}
		return retValue & mask;
	}

	/**
	 * Returns the value of the sixteen bits at a given offset in the set as a
	 * short.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @return the bits at that offset represented as a short
	 */
	public final short getShortAt(final int offset) {
		return getShortAt(offset, BITSINSHORT);
	}

	/**
	 * Returns the value of the <code>length</code> bits at a given offset in
	 * the set as a short. If <code>length</code> is less than sixteen, the
	 * value is zero extended; if greater it is truncated.
	 * 
	 * @param offset
	 *            the offset to read from
	 * @param length
	 *            the number of bits to read
	 * @return the bits represented as a short
	 */
	public final short getShortAt(final int offset, int length) {
		if (length > BITSINSHORT) {
			length = BITSINSHORT;
		}
		return (short) getLongAt(offset, length);
	}

	/**
	 * Obtains a subset of bits. Builds another bit set formed by the bits from
	 * offset to offset + length - 1.
	 * 
	 * @param offset
	 *            start of subset
	 * @param length
	 *            length of subset
	 * @return the indicated subset of bits
	 */
	public final ExtendedBitSet getSubSet(int offset, int length) {
		extendIfNeeded(offset + length);

		final int chunks = ((length - 1) / BITSINLONG) + 1;
		final long[] newBits = new long[chunks];
		final ExtendedBitSet retValue = new ExtendedBitSet();

		retValue.lenBits = length;

		for (int i = 0; i < chunks; ++i) {
			newBits[i] = getLongAt(offset, length);
			offset += BITSINLONG;
			length -= BITSINLONG;
		}
		retValue.bitUnits = newBits;
		return retValue;
	}

	/**
	 * Gets the hashCode.
	 * 
	 * @return the hashCode
	 */
	@Override
	public final int hashCode() {
		int retValue = lenBits;

		if (bitUnits != null) {
			for (int i = 0; i < bitUnits.length; ++i) {
				// is this different for non-equal objects???
				retValue ^= (int) (bitUnits[i] ^ (bitUnits[i] >>> 32));
			}
		}
		return retValue;
	}

	/**
	 * Copy bits from an array of longs. The contents of the longs directly
	 * dictate the bits. Bit 0 of array element 0 is bit 0 of the bit set bit 63
	 * of array element n is bit n*64+63 of the set, subject to the maximum
	 * length given by <code>length</code>.
	 * 
	 * @param length
	 *            number of bits in set
	 * @param inBits
	 *            initial bits values
	 * @return this
	 */
	public final ExtendedBitSet initAll(final int length, final long[] inBits) {
		int longsNeeded = ((length - 1) / BITSINLONG) + 1;

		if ((bitUnits == null) || (longsNeeded > bitUnits.length)) {
			bitUnits = new long[longsNeeded];
		}
		if (inBits.length < longsNeeded) {
			longsNeeded = inBits.length;
		}
		System.arraycopy(inBits, 0, bitUnits, 0, longsNeeded);
		lenBits = length;
		return this;
	}

	/**
	 * Insert a subset of bits into the current set before bit
	 * <code>offset</code>. Bits <code>offset</code> and greater are shuffled up
	 * to make room. If <code>offset</code> is negative, insert at the end.
	 * 
	 * @param offset
	 *            insertion point
	 * @param bitSet
	 *            bit set to insert
	 * @return this
	 */
	public final ExtendedBitSet insertSubSet(int offset,
			final ExtendedBitSet bitSet) {
		if (offset < 0) {
			offset = lenBits;
		} else {
			final int bitsToShift = (lenBits - offset) + bitSet.bitUnits.length;

			shiftLeft(offset, bitsToShift, bitSet.bitUnits.length);
		}
		return setSubSet(offset, bitSet);
	}

	/**
	 * Returns the value of the bit set as an int (truncating it if it is longer
	 * than thirtytwo bits).
	 * 
	 * @return the bit set represented as an int
	 */
	public final int intValue() {
		return getIntAt(0, BITSININT);
	}

	/**
	 * Invert the value of a particular bit.
	 * 
	 * @param offset
	 *            the bit to flip
	 * @return this
	 */
	public final ExtendedBitSet invert(final int offset) {
		extendIfNeeded(offset + 1);

		final int thisLong = offset / BITSINLONG;
		final long thisBit = offset % BITSINLONG;

		bitUnits[thisLong] ^= 1L << thisBit;

		return this;
	}

	/**
	 * Inverts the values of a range of bits. <code>length</code> bits at
	 * <code>offset</code> are flipped.
	 * 
	 * @param offset
	 *            start of range
	 * @param length
	 *            length of range
	 * @return this
	 */
	public final ExtendedBitSet invert(int offset, int length) {
		extendIfNeeded(offset + length);
		while (length > 0) {
			long newValue = getLongAt(offset, length) ^ -1L;
			setLongAt(offset, length, newValue);
			offset += BITSINLONG;
			length -= BITSINLONG;
		}
		return this;
	}

	/**
	 * Returns the value of the bit set as a long (truncating it if it is longer
	 * than sixtyfour bits).
	 * 
	 * @return the bit set represented as a long
	 */
	public final long longValue() {
		return getLongAt(0, BITSINLONG);
	}

	/**
	 * Logically ORs the <strong>whole</strong> of this bit set with the
	 * specified set of bits. If <code>bitSet</code> is shorter than this set,
	 * the additional values are assumed to be zero; if it is longer, it is
	 * truncated.
	 * 
	 * @param bitSet
	 *            the bit set to be ORed with
	 * @return this
	 */
	public final ExtendedBitSet or(ExtendedBitSet bitSet) {
		if (bitSet.lenBits < lenBits) {
			bitSet = (new ExtendedBitSet(lenBits)).setSubSet(0, bitSet);
		}
		return or(0, bitSet);
	}

	/**
	 * Logically ORs a range of bits in this bit set with the specified set of
	 * bits. <code>offset</code> gives the start of the range, and the length of
	 * <code>bitSet</code> defines its length.
	 * 
	 * @param offset
	 *            the start of the range of bits to OR
	 * @param bitSet
	 *            the bit set to be ORed with
	 * @return this
	 */
	public final ExtendedBitSet or(int offset, final ExtendedBitSet bitSet) {
		extendIfNeeded(offset + bitSet.lenBits);

		int otherOffset = 0;
		for (int length = bitSet.lenBits; length > 0; length -= BITSINLONG) {
			long newValue = getLongAt(offset, length)
					| bitSet.getLongAt(otherOffset, length);
			setLongAt(offset, length, newValue);
			offset += BITSINLONG;
			otherOffset += BITSINLONG;
			length -= BITSINLONG;
		}
		return this;
	}

	/**
	 * Rotate a subset of bits leftward. Rotates bits within a field of
	 * <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the left. Bits lost off of the top of the
	 * field are re-inserted at the bottom. Example: <code>
	 *      01|1001|01 -> rotateLeft( 2, 4, 1 ) -> 01|0110|01
	 * </code> Read the explanation for creepLeft if this seems opaque.
	 * 
	 * @param offset
	 *            start of field to rotate
	 * @param length
	 *            length of field to rotate
	 * @param shift
	 *            number of positions to rotate left
	 * @return this
	 * 
	 * @see creepLeft
	 */
	public final ExtendedBitSet rotateLeft(final int offset, final int length,
			final int shift) {
		if ((shift % length) != 0) {
			final ExtendedBitSet save = getSubSet((offset + length) - shift,
					shift);

			creepLeft(offset, length, shift);
			setSubSet(offset, save);
		}

		return this;
	}

	/**
	 * Rotate a subset of bits rightward. Rotates bits within a field of
	 * <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the right. Bits lost off of the bottom of
	 * the field are re-inserted at the top. Example: <code>
	 *      01|1001|01 -> rotateRight( 2, 4, 1 ) -> 01|0110|01
	 * </code> Read the explanation for creepLeft if this seems opaque.
	 * 
	 * @param offset
	 *            start of field to rotate
	 * @param length
	 *            length of field to rotate
	 * @param shift
	 *            number of positions to rotate right
	 * @return this
	 * 
	 * @see creepLeft
	 */
	public final ExtendedBitSet rotateRight(final int offset, final int length,
			final int shift) {
		if ((shift % length) != 0) {
			final ExtendedBitSet save = getSubSet(offset, shift);

			creepRight(offset, length, shift);
			setSubSet((offset + length) - shift, save);
		}

		return this;
	}

	/**
	 * Set the value of a particular bit to true.
	 * 
	 * @param offset
	 *            the bit to set
	 * @return this
	 */
	public final ExtendedBitSet set(final int offset) {
		extendIfNeeded(offset + 1);

		final int thisLong = offset / BITSINLONG;
		final long thisBit = offset % BITSINLONG;

		bitUnits[thisLong] |= (1L << thisBit);

		return this;
	}

	/**
	 * Sets a range of bits to true. <code>length</code> bits at
	 * <code>offset</code> are set to true.
	 * 
	 * @param offset
	 *            start of range
	 * @param length
	 *            length of range
	 * @return this
	 */
	public final ExtendedBitSet set(int offset, int length) {
		extendIfNeeded(offset + length);

		while (length > 0) {
			setLongAt(offset, length, -1L);
			offset += BITSINLONG;
			length -= BITSINLONG;
		}

		return this;
	}

	/**
	 * Set the value of a particular bit to the boolean given.
	 * 
	 * @param offset
	 *            the bit to set
	 * @param value
	 *            the value to set it to
	 * @return this
	 */
	public final ExtendedBitSet setBooleanAt(final int offset,
			final boolean value) {
		if (value) {
			return set(offset);
		} else {
			return clear(offset);
		}
	}

	/**
	 * Sets the value of the eight bits at a given offset in the set to
	 * represent a byte.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setByteAt(final int offset, final byte value) {
		return setByteAt(offset, BITSINBYTE, value);
	}

	/**
	 * Sets the value of <code>length</code> bits at a given offset in the set
	 * to represent a byte. If <code>length</code> is less than eight, the value
	 * is truncated; lengths greater than eight are treated as eight.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param length
	 *            the number of bits of the value to write
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setByteAt(final int offset, int length,
			final byte value) {
		if (length > BITSINBYTE) {
			length = BITSINBYTE;
		}
		return setLongAt(offset, length, value);
	}

	/**
	 * Sets the value of the sixteen bits at a given offset in the set to
	 * represent a char.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setCharAt(final int offset, final char value) {
		return setCharAt(offset, BITSINCHAR, value);
	}

	/**
	 * Sets the value of <code>length</code> bits at a given offset in the set
	 * to represent a char. If <code>length</code> is less than sixteen, the
	 * value is truncated; lengths greater than sixteen are treated as sixteen.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param length
	 *            the number of bits of the value to write
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setCharAt(final int offset, int length,
			final char value) {
		if (length > BITSINCHAR) {
			length = BITSINCHAR;
		}
		return setLongAt(offset, length, value);
	}

	/**
	 * Sets the value of sixtyfour bits at a given offset in the set to
	 * represent a double.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setDoubleAt(final int offset, final double value) {
		return setLongAt(offset, BITSINLONG, Double.doubleToLongBits(value));
	}

	/**
	 * Sets the value of thirtytwo bits at a given offset in the set to
	 * represent a float.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setFloatAt(final int offset, final float value) {
		return setIntAt(offset, BITSININT, Float.floatToIntBits(value));
	}

	/**
	 * Sets the value of the thirtytwo bits at a given offset in the set to
	 * represent an int.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setIntAt(final int offset, final int value) {
		return setIntAt(offset, BITSININT, value);
	}

	/**
	 * Sets the value of <code>length</code> bits at a given offset in the set
	 * to represent an int. If <code>length</code> is less than thirtytwo, the
	 * value is truncated; lengths greater than thirtytwo are treated as
	 * thirtytwo.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param length
	 *            the number of bits of the value to write
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setIntAt(final int offset, int length,
			final int value) {
		if (length > BITSININT) {
			length = BITSININT;
		}
		return setLongAt(offset, length, value);
	}

	/**
	 * Sets the value of <code>length</code> bits at a given offset in the set
	 * to represent a long. If <code>length</code> is less than sixtyfour, the
	 * value is truncated; lengths greater than sixtyfour are treated as
	 * sixtyfour.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param length
	 *            the number of bits of the value to write
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setLongAt(final int offset, int length,
			long value) {
		if (length > BITSINLONG) {
			length = BITSINLONG;

		}
		final int block = offset / BITSINLONG;
		final int shift = offset % BITSINLONG;
		final long mask = -1L >>> (BITSINLONG - length);

		extendIfNeeded(offset + length);

		value &= mask;

		if (shift == 0) {
			bitUnits[block] = (bitUnits[block] & ~mask) | value;
		} else {
			bitUnits[block] = (bitUnits[block] & ~(mask << shift))
					| (value << shift);
			if ((BITSINLONG - shift) < length) {
				bitUnits[block + 1] = (bitUnits[block + 1] & ~(mask >>> (BITSINLONG - shift)))
						| (value >>> (BITSINLONG - shift));
			}
		}

		return this;
	}

	/**
	 * Sets the value of the sixtyfour bits at a given offset in the set to
	 * represent a long.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setLongAt(final int offset, final long value) {
		return setLongAt(offset, BITSINLONG, value);
	}

	/**
	 * Sets the value of <code>length</code> bits at a given offset in the set
	 * to represent a short. If <code>length</code> is less than sixteen, the
	 * value is truncated; lengths greater than sixteen are treated as sixteen.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param length
	 *            the number of bits of the value to write
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setShortAt(final int offset, int length,
			final short value) {
		if (length > BITSINSHORT) {
			length = BITSINSHORT;
		}
		return setLongAt(offset, length, value);
	}

	/**
	 * Sets the value of the sixteen bits at a given offset in the set to
	 * represent a short.
	 * 
	 * @param offset
	 *            the offset to assign to
	 * @param value
	 *            the value to write into the set
	 * @return this
	 */
	public final ExtendedBitSet setShortAt(final int offset, final short value) {
		return setShortAt(offset, BITSINSHORT, value);
	}

	/**
	 * Extend or reduce the size of the bit set.
	 * 
	 * @param length
	 *            new size of the bit set.
	 * @return this
	 */
	public final ExtendedBitSet setSize(final int length) {
		if (length < lenBits) {
			// Clear the bits to ensure that all unused bits are always
			// zero. It makes life easier for comparing and extending.
			clear(length, lenBits - length);
			lenBits = length;
		} else {
			extendIfNeeded(length);
		}
		return this;
	}

	/**
	 * Sets the value of a subset of bits. Bits at offset <code>offset</code>
	 * are set to bit values obtained from bit set <code>bitSet</code>. The
	 * number of bits affected is dictated by the size of bitSet.
	 * 
	 * @param offset
	 *            start of subset to alter
	 * @param bitSet
	 *            bit set to obtain the new values from
	 * @return this
	 */
	public final ExtendedBitSet setSubSet(int offset,
			final ExtendedBitSet bitSet) {
		extendIfNeeded(offset + bitSet.lenBits);
		int otherOffset = 0;
		for (int length = bitSet.lenBits; length > 0; length -= BITSINLONG) {
			setLongAt(offset, length, bitSet.getLongAt(otherOffset, length));
			offset += BITSINLONG;
			otherOffset += BITSINLONG;
		}
		return this;
	}

	/**
	 * Move a subset of bits left. Shifts bits within a field of
	 * <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the left. The bottommost <code>shift</code>
	 * bits are zeroed. Example: <code>
	 *      01|1001|01 -> shiftLeft( 2, 4, 1 ) -> 01|0010|01
	 * </code> Read the explanation for creepLeft if this seems opaque.
	 * 
	 * @param offset
	 *            start of field to shift
	 * @param length
	 *            length of field to shift
	 * @param shift
	 *            number of positions to shift
	 * @return this
	 * 
	 * @see creepLeft
	 */
	public final ExtendedBitSet shiftLeft(final int offset, final int length,
			final int shift) {
		if ((shift % length) != 0) {
			// Shift the bits and zero what's left over
			creepLeft(offset, length, shift);
			clear(offset, shift);
		}

		return this;
	}

	/**
	 * Shift a subset of bits right with sign extension. Shifts bits within a
	 * field of <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the right. The topmost <code>shift</code>
	 * bits are all set to the value of the topmost bit. Example: <code>
	 *      01|1001|01 -> shiftRightSigned( 2, 4, 1 ) -> 01|1100|01
	 * </code> Read the explanation for creepLeft if this seems opaque.
	 * 
	 * @param offset
	 *            start of field to shift
	 * @param length
	 *            length of field to shift
	 * @param shift
	 *            number of positions to shift right
	 * @return this
	 * 
	 * @see creepLeft
	 */
	public final ExtendedBitSet shiftRightSigned(final int offset,
			final int length, final int shift) {
		if ((shift % length) != 0) {
			// Shift the bits and sign extend what's left over

			creepRight(offset, length, shift);
			if (get((offset + length) - 1)) {
				set((offset + length) - shift, shift - 1);
			} else {
				clear((offset + length) - shift, shift - 1);
			}
		}

		return this;
	}

	/**
	 * Shifts a subset of bits right. Shifts bits within a field of
	 * <code>length</code> bits starting at <code>offset</code>,
	 * <code>shift</code> places to the right. The topmost <code>shift</code>
	 * bits are zeroed. Example: <code>
	 *      01|1001|01 -> shiftRightUnsigned( 2, 4, 1 ) -> 01|0100|01
	 * </code> Read the explanation for creepLeft if this seems opaque.
	 * 
	 * @param offset
	 *            start of field to shift
	 * @param length
	 *            length of field to shift
	 * @param shift
	 *            number of positions to shift right
	 * @return this
	 * 
	 * @see creepLeft
	 */
	public final ExtendedBitSet shiftRightUnsigned(final int offset,
			final int length, final int shift) {
		if ((shift % length) != 0) {
			// Shift the bits and zero what's left over
			creepRight(offset, length, shift);
			clear((offset + length) - shift, shift);
		}

		return this;
	}

	/**
	 * Returns the value of the bit set as a short (truncating it if it is
	 * longer than sixteen bits).
	 * 
	 * @return the bit set represented as a short
	 */
	public final short shortValue() {
		return getShortAt(0, BITSINSHORT);
	}

	/**
	 * Return the number of bits in the set.
	 * 
	 * @return the number of bits in the set
	 */
	public final int size() {
		return lenBits;
	}

	/**
	 * Swaps the value of a subset of bits with another bit set.
	 * <code>length</code> bits at offset <code>offset</code> in each bit set
	 * are swapped with each other.
	 * 
	 * @param offset
	 *            start of subset to alter
	 * @param length
	 *            number of bits to swap
	 * @param bitSet
	 *            bit set to swap values with
	 * @return this
	 */
	public final ExtendedBitSet swapSubSet(int offset, final int length,
			final ExtendedBitSet bitSet) {
		extendIfNeeded(offset + length);
		for (int len = length; len > 0; len -= BITSINLONG) {
			long saved = getLongAt(offset, len);
			setLongAt(offset, len, bitSet.getLongAt(offset, len));
			bitSet.setLongAt(offset, len, saved);
			offset += BITSINLONG;
		}
		return this;
	}

	/**
	 * Return the bit set represented as a binary string.
	 * 
	 * @return the string representation of the set
	 */
	@Override
	public String toString() {
		final StringBuffer buf = new StringBuffer(lenBits);

		if (bitUnits != null) {
			long mask = 1L << (lenBits - 1);

			for (int i = bitUnits.length - 1; i >= 0; --i, mask = 1L << 63) {
				for (; mask != 0; mask >>>= 1) {
					if ((bitUnits[i] & mask) != 0) {
						buf.append('1');
					} else {
						buf.append('0');
					}
				}
			}
		}
		return buf.toString();
	}

	/**
	 * Logically XORs the <strong>whole</strong> of this bit set with the
	 * specified set of bits. If <code>bitSet</code> is shorter than this set,
	 * the additional values are assumed to be zero; if it is longer, it is
	 * truncated.
	 * 
	 * @param bitSet
	 *            the bit set to be XORed with
	 * @return this
	 */
	public final ExtendedBitSet xor(ExtendedBitSet bitSet) {
		if (bitSet.lenBits < lenBits) {
			bitSet = (new ExtendedBitSet(lenBits)).setSubSet(0, bitSet);
		}
		return xor(0, bitSet);
	}

	/**
	 * Logically XORs a range of bits in this bit set with the specified set of
	 * bits. <code>offset</code> gives the start of the range, and the length of
	 * <code>bitSet</code> defines its length.
	 * 
	 * @param offset
	 *            the start of the range of bits to XOR
	 * @param bitSet
	 *            the bit set to be XORed with
	 * @return this
	 */
	public final ExtendedBitSet xor(int offset, final ExtendedBitSet bitSet) {
		extendIfNeeded(offset + bitSet.lenBits);

		int otherOffset = 0;
		for (int length = bitSet.lenBits; length > 0; length -= BITSINLONG) {
			long newValue = getLongAt(offset, length)
					^ bitSet.getLongAt(otherOffset, length);
			setLongAt(offset, length, newValue);
			offset += BITSINLONG;
			otherOffset += BITSINLONG;
			length -= BITSINLONG;
		}
		return this;
	}
}
