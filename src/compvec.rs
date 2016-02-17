//! To CompRawVec as Vec is to RawVec.


use std;

use ::WORD_SIZE;
use comprawvec::CompRawVec;



/// First value to use in CompVec::next(masked__valid, ...)
pub const VALID_MAX: usize = std::usize::MAX;


// A bitwise left-shift that returns zero if the shift would overflow
#[inline]
fn shl_or_zero(i: usize, shift: u32) -> usize {
    if shift >= WORD_SIZE as u32 {
        0
    } else {
        i << shift
    }
}


/// A simple sparse vector.  The `valid` word is a bitmap of which indeces
/// have values.  The maximum size of this vector is equal to the number of
/// bits in a word (32 or 64).
pub struct CompVec<T> {
    data: CompRawVec<T>,
}


/// A type that implements Iterator for CompVec
pub struct Iter<'a, T: 'a> {
    vec: &'a CompVec<T>,
    masked_valid: usize,
    compressed: usize,
}


impl<T> CompVec<T> {
    // Take an index in the range 0..usize::BITS and compress it down
    // omitting empty entries. Will panic if index is outside of valid range.
    fn compress_index(valid: usize, index: usize) -> (usize, usize) {
        let bit = 1usize << index;
        let marker = shl_or_zero(bit, 1);
        let mask = marker.wrapping_sub(1);
        let compressed = (((valid | bit) & mask).count_ones() - 1) as usize;
        (bit, compressed)
    }

    /// Move a value into the node at the given index. Returns a reference
    /// to the location where the value is stored.
    pub fn set(&mut self, index: usize, value: T) -> &mut T {
        let valid = self.data.valid();
        let (bit, compressed) = Self::compress_index(valid, index);

        if valid & bit == 0 {
            unsafe { self.data.insert(bit, compressed, value) }
        } else {
            unsafe { self.data.replace(compressed, value) }
        }
    }

    /// Return the mutable value at the given index if it exists, otherwise
    /// return None.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let valid = self.data.valid();
        let (bit, compressed) = Self::compress_index(valid, index);

        if valid & bit != 0 {
            unsafe { Some(self.data.get_mut(compressed)) }
        } else {
            None
        }
    }

    /// Return the value at the given index if it exists, otherwise return
    /// None.
    pub fn get(&self, index: usize) -> Option<&T> {
        let valid = self.data.valid();
        let (bit, compressed) = Self::compress_index(valid, index);

        if valid & bit != 0 {
            unsafe { Some(self.data.get(compressed)) }
        } else {
            None
        }
    }

    /// Return the value at the given index if it exists, otherwise call the
    /// provided function to get the default value to insert and return.
    pub fn get_default_mut<F>(&mut self, index: usize, default: F) -> &mut T
        where F: Fn() -> T
    {
        let valid = self.data.valid();
        let (bit, compressed) = Self::compress_index(valid, index);

        if valid & bit == 0 {
            unsafe { self.data.insert(bit, compressed, default()) }
        } else {
            unsafe { self.data.get_mut(compressed) }
        }
    }

    /// Remove an entry, returning the entry if it was present at the given
    /// index.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let valid = self.data.valid();
        let (bit, compressed) = Self::compress_index(valid, index);

        if valid & bit != 0 {
            unsafe { Some(self.data.remove(bit, compressed)) }
        } else {
            None
        }
    }

    /// Number of objects stored.
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Number of objects that can be stored without reallocation.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Return true if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.data.valid() == 0
    }

    /// Return the next Some(((masked_valid, compressed), (index, &value)))
    /// or None
    ///
    /// `masked_valid` is the last valid bitmap with already-visited indeces
    /// masked out, starts with std::usize::MAX for the first call.
    /// `compressed` is the last compressed vector index, always starting
    /// at zero for the first call.
    #[inline]
    pub fn next(&self,
                masked_valid: usize,
                compressed: usize)
                -> Option<((usize, usize), (usize, &T))> {
        let valid = self.data.valid();
        let index = (valid & masked_valid).trailing_zeros();

        if index < (WORD_SIZE as u32) {
            let mask = shl_or_zero(std::usize::MAX, index + 1);

            Some(((mask, compressed + 1),
                  (index as usize, unsafe { self.data.get(compressed) })))
        } else {
            None
        }
    }

    /// Return the next Some(((masked_valid, compressed), (index, &mut value)))
    /// or None
    ///
    /// `masked_valid` is the last valid bitmap with already-visited indeces
    /// masked out, starts with std::usize::MAX for the first call.
    /// `compressed` is the last compressed vector index, always starting
    /// at zero for the first call.
    #[inline]
    pub fn next_mut(&mut self,
                    masked_valid: usize,
                    compressed: usize)
                    -> Option<((usize, usize), (usize, &mut T))> {
        let valid = self.data.valid();
        let index = (valid & masked_valid).trailing_zeros();

        if index < (WORD_SIZE as u32) {
            let mask = shl_or_zero(std::usize::MAX, index + 1);

            Some(((mask, compressed + 1),
                  (index as usize, unsafe { self.data.get_mut(compressed) })))
        } else {
            None
        }
    }

    /// Create an iterator over the contents
    pub fn iter(&self) -> Iter<T> {
        Iter {
            vec: self,
            masked_valid: VALID_MAX,
            compressed: 0,
        }
    }

    pub fn new() -> CompVec<T> {
        CompVec { data: CompRawVec::new() }
    }
}


impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {

        if let Some(((masked_valid, compressed),
                     (index, value))) = self.vec.next(self.masked_valid, self.compressed) {

            self.masked_valid = masked_valid;
            self.compressed = compressed;
            Some((index, value))
        } else {
            None
        }
    }
}
