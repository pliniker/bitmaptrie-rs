//! This module borrows patterns from RawVec to build a raw compressed array.


use std::cmp::max;
use std::intrinsics::{needs_drop, drop_in_place, abort};
use std::mem::{size_of, align_of};
use std::ptr::{copy, read, write, Unique};

extern crate alloc;
use self::alloc::heap::{allocate, reallocate, deallocate};


/// The initial sparse-vector capacity.
const MIN_CAPACITY: usize = 4;


/// A compressed vector of objects where the first element of the array is a
/// bitmap of which indeces are valid. This is an unsafe data structure as it
/// is not bounds-checked. Maximum capacity is the word size
pub struct CompRawVec<T> {
    // bitmap of which indeces have values
    valid: usize,

    // a whole word for the tiny capacity: is there a more efficient encoding?
    capacity: usize,

    // pointer to the array of length `capacity`
    ptrs: Unique<T>,
}


impl<T> CompRawVec<T> {
    fn oom() -> ! {
        unsafe { abort() }
    }

    /// Allocate a block of objects
    fn allocate(capacity: usize) -> Unique<T> {
        unsafe {
            let size = capacity * size_of::<T>();

            let array = allocate(size, align_of::<T>());
            if array.is_null() {
                Self::oom();
            }

            Unique::new(array as *mut T)
        }
    }

    /// Resize an allocated block of objects
    #[inline]
    fn reallocate(data: &mut Unique<T>, old_capacity: usize, new_capacity: usize) {
        let old_size = old_capacity * size_of::<T>();
        let new_size = new_capacity * size_of::<T>();

        unsafe {
            let array = reallocate(data.as_ptr() as *mut u8,
                                   old_size,
                                   new_size,
                                   align_of::<T>());
            if array.is_null() {
                Self::oom();
            }

            *data = Unique::new(array as *mut T);
        }
    }

    /// Free a block of objects
    fn deallocate(data: &mut Unique<T>, capacity: usize) {
        unsafe {
            let size = capacity * size_of::<T>();
            deallocate(data.as_ptr() as *mut u8, size, align_of::<T>());
        }
    }

    /// Access an object mutably in the array.
    /// Does not bounds-check index.
    pub unsafe fn get_mut(&mut self, index: usize) -> &mut T {
        let p: *mut T = self.ptrs.as_ptr().offset(index as isize);
        &mut *p
    }

    /// Access an object in the array.
    /// Does not bounds-check index.
    pub unsafe fn get(&self, index: usize) -> &T {
        let p: *const T = self.ptrs.as_ptr().offset(index as isize);
        &*p
    }

    /// Insert a new value at the given position, shifting subsequent
    /// values to the right.
    /// Does not bounds-check index.
    #[inline]
    pub unsafe fn insert(&mut self, valid_bit: usize, index: usize, value: T) -> &mut T {

        let size = self.size();
        let capacity = self.capacity;

        // see if insertion will overflow the current capacity
        if size == capacity {
            let new_capacity = capacity * 2;

            Self::reallocate(&mut self.ptrs, capacity, new_capacity);

            self.capacity = new_capacity;
        }

        // shift later objects and insert before them
        let ptr: *mut T = self.ptrs.as_ptr().offset(index as isize);

        copy(ptr, ptr.offset(1), size - index);
        write(ptr, value);

        self.valid |= valid_bit;

        &mut *ptr
    }

    /// Replace the value at an index with a new value.
    /// Does not bounds-check index.
    pub unsafe fn replace(&mut self, index: usize, value: T) -> &mut T {
        let mut ptr = self.get_mut(index);
        *ptr = value;
        ptr
    }

    /// Remove an object from the array, assuming it is present.
    /// Does not bounds-check index.
    pub unsafe fn remove(&mut self, valid_bit: usize, index: usize) -> T {

        let size = self.size();
        let capacity = self.capacity;

        let value;

        let ptr: *mut T = self.ptrs.as_ptr().offset(index as isize);

        value = read(ptr);

        copy(ptr.offset(1), ptr, size - index - 1);

        // clear the valid bitmap bit
        self.valid ^= valid_bit;

        // if removal results in a size half the capacity, reallocate
        if ((size - 1) * 2 == capacity) && (capacity > MIN_CAPACITY) {
            let new_capacity = max(capacity >> 1, MIN_CAPACITY);

            Self::reallocate(&mut self.ptrs, capacity, new_capacity);

            self.capacity = new_capacity;
        }

        value
    }

    /// Return the valid bitmap.
    pub fn valid(&self) -> usize {
        self.valid
    }

    /// Return the count of objects in the array.
    pub fn size(&self) -> usize {
        self.valid.count_ones() as usize
    }

    /// Return the current array capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Instantiate a new CompRawVec with NODE_CAPACITY_INC capacity.
    pub fn new() -> CompRawVec<T> {
        CompRawVec {
            valid: 0,
            capacity: MIN_CAPACITY,
            ptrs: Self::allocate(MIN_CAPACITY),
        }
    }
}


impl<T> Drop for CompRawVec<T> {
    fn drop(&mut self) {
        unsafe {
            if needs_drop::<T>() {
                let size = self.size();

                for index in 0..size {
                    let object = self.get_mut(index);
                    drop_in_place(object);
                }
            }
        }

        Self::deallocate(&mut self.ptrs, self.capacity);
    }
}
