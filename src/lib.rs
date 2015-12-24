
//! A bitmapped vector trie with node compression and a path cache.
//!
//! The trie does not prescribe a length or capacity beside the range of values
//! of it's index: usize.
//!
//! Branching factor is the word-size: 32 or 64. This makes the depth 6 for
//! 32bit systems and 11 for 64bit systems.
//!
//! Possible performance improvements:
//!
//!  * enable popcnt instruction when supported
//!
//! # Usage
//!
//! ```
//! use bitmaptrie::Trie;
//!
//! let mut t: Trie<String> = Trie::new();
//! t.set(123, "testing 123".to_owned());
//! ```


// so much features for doing simple things!
#![feature(drop_in_place)]
#![feature(associated_consts)]
#![feature(core_intrinsics)]
#![feature(num_bits_bytes)]
#![feature(unique)]
#![feature(heap_api)]
#![feature(alloc)]


use std::ptr::null_mut;
use std::cell::Cell;
use std::mem::transmute;

mod comprawvec;
use comprawvec::CompRawVec;


// 32 or 64
const BRANCHING_FACTOR_BITS: usize = (0b100 | (std::usize::BYTES >> 2));
// 0x3 or or 0x7
const BRANCHING_INDEX_MASK: usize = (1 << BRANCHING_FACTOR_BITS) - 1;
// 6 or 11
const BRANCHING_DEPTH: usize = std::usize::BITS / BRANCHING_FACTOR_BITS as usize + 1;

/// First value to use in CompVec::next(masked__valid, ...)
pub const VALID_MAX: usize = std::usize::MAX;


// A bitwise left-shift that returns zero if the shift would overflow
#[inline]
fn default_shl(i: usize, shift: u32) -> usize {
    if shift >= std::usize::BITS as u32 {
        0
    } else {
        i << shift
    }
}


/// A simple sparse vector.  The `valid` word is a bitmap of which indeces
/// have values.  The maximum size of this vector is equal to the number of
/// bits in a word (32 or 64).
pub struct CompVec<T> {
    data: CompRawVec<T>
}


impl<T> CompVec<T> {

    // Take an index in the range 0..usize::BITS and compress it down
    // omitting empty entries. Will panic if index is outside of valid range.
    fn compress_index(valid: usize, index: usize) -> (usize, usize) {
        let bit = 1usize << index;
        let marker = default_shl(bit, 1);
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
    pub fn get_default_mut<F>(&mut self,
                              index: usize,
                              default: F) -> &mut T where F: Fn() -> T {
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
                compressed: usize) -> Option<((usize, usize), (usize, &T))> {
        let valid = self.data.valid();
        let index = (valid & masked_valid).trailing_zeros();

        if index < (std::usize::BITS as u32) {
            let mask = default_shl(std::usize::MAX, index + 1);

            Some(((mask, compressed + 1),
                  (index as usize, unsafe { self.data.get(compressed) })))
        } else {
            None
        }
    }

    pub fn new() -> CompVec<T> {
        CompVec{data: CompRawVec::new()}
    }
}


/// The identity function.
///
/// Also forces the argument to move.
fn moving<T>(x: T) -> T { x }


/// An interior (branch) or exterior (leaf) trie node, defined recursively.
pub enum TrieNode<T> {
    Interior(CompVec<TrieNode<T>>),
    Exterior(CompVec<T>)
}


impl<T> TrieNode<T> {

    fn new_branch() -> TrieNode<T> {
        TrieNode::Interior(CompVec::new())
    }

    fn new_leaf() -> TrieNode<T> {
        TrieNode::Exterior(CompVec::new())
    }

    fn set(&mut self,
           index: usize,
           value: T,
           depth: usize,
           cache: &mut PathCache<T>) -> &mut T {

        let mut depth = depth;
        let mut shift = depth * BRANCHING_FACTOR_BITS;
        let mut current = self;

        loop {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;

            match moving(current) {
                &mut TrieNode::Interior(ref mut branch) => {
                    let child = branch.get_default_mut(
                        local_index,
                        || {
                            if depth > 1 {
                                Self::new_branch()
                            } else {
                                Self::new_leaf()
                            }
                        });

                    depth -= 1;
                    shift -= BRANCHING_FACTOR_BITS;

                    cache.set(depth, local_index, child);

                    current = child;
                },

                &mut TrieNode::Exterior(ref mut leaf) => {
                    return leaf.set(local_index, value);
                }
            }
        }
    }

    fn get_mut(&mut self,
               index: usize,
               depth: usize,
               cache: &mut PathCache<T>) -> Option<&mut T> {

        let mut depth = depth;
        let mut shift = depth * BRANCHING_FACTOR_BITS;
        let mut current = self;

        loop {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;

            match moving(current) {
                &mut TrieNode::Interior(ref mut branch) => {

                    if let Some(child) = branch.get_mut(local_index as usize) {
                        depth -= 1;
                        shift -= BRANCHING_FACTOR_BITS;

                        cache.set(depth, local_index, child);;

                        current = child;
                    } else {
                        return None
                    }
                },

                &mut TrieNode::Exterior(ref mut leaf) => {
                    return leaf.get_mut(local_index as usize)
                }
            }
        }
    }

    fn get(&self,
           index: usize,
           depth: usize,
           cache: &mut PathCache<T>) -> Option<&T> {

        let mut depth = depth;
        let mut shift = depth * BRANCHING_FACTOR_BITS;
        let mut current = self;

        loop {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;

            match moving(current) {
                &TrieNode::Interior(ref branch) => {

                    if let Some(child) = branch.get(local_index as usize) {
                        depth -= 1;
                        shift -= BRANCHING_FACTOR_BITS;

                        cache.set(depth, local_index, child);;

                        current = child;
                    } else {
                        return None
                    }
                },

                &TrieNode::Exterior(ref leaf) => {
                    return leaf.get(local_index as usize)
                }
            }
        }
    }

    /// Remove an entry if it exists, returning it if it existed.
    fn remove(&mut self,
              index: usize,
              depth: usize,
              cache: &mut PathCache<T>) -> (Option<T>, bool) {
        // must be recursive in order to delete empty leaves and branches

        match self {
            &mut TrieNode::Interior(ref mut branch) => {
                let shift = depth * BRANCHING_FACTOR_BITS;
                let local_index = (index >> shift) & BRANCHING_INDEX_MASK;

                let (value, empty_child) = {
                    let mut maybe_child = branch.get_mut(local_index as usize);

                    if let Some(ref mut child) = maybe_child {
                        child.remove(index, depth - 1, cache)
                    } else {
                        (None, false)
                    }
                };

                if empty_child {
                    branch.remove(local_index as usize);
                    cache.invalidate(depth);
                }

                (value, branch.is_empty())
            },

            &mut TrieNode::Exterior(ref mut leaf) => {
                let local_index = index & BRANCHING_INDEX_MASK;
                let value = leaf.remove(local_index as usize);
                cache.invalidate(depth);
                (value, leaf.is_empty())
            }
        }
    }
}


/// An index path cache line, with the index and the node it refers to.
struct TrieNodePtr<T> {
    index: usize,
    /// A null pointer here indicates an invalid cache line.
    node: *mut TrieNode<T>
}


impl<T> Clone for TrieNodePtr<T> {
    fn clone(&self) -> TrieNodePtr<T> {
        TrieNodePtr{index: self.index,
                    node: self.node}
    }
}


impl<T> Copy for TrieNodePtr<T> {}


impl<T> Default for TrieNodePtr<T> {
    fn default() -> TrieNodePtr<T> {
        TrieNodePtr{index: 0, node: null_mut()}
    }
}


impl<T> TrieNodePtr<T> {

    fn set(&mut self, index: usize, node: &TrieNode<T>) {
        self.index = index;
        self.node = unsafe { transmute(node) };
    }

    fn is_hit(&self, index: usize) -> bool {
        (self.index == index) && (!self.node.is_null())
    }

    fn get(&self) -> *const TrieNode<T> {
        self.node
    }

    fn get_mut(&self) -> *mut TrieNode<T> {
        self.node
    }

    fn invalidate(&mut self) {
        self.node = null_mut();
    }
}


/// A cached path into a trie
pub struct PathCache<T> {
    generation: usize,
    index_cache: Option<usize>,
    path_cache: [TrieNodePtr<T>; BRANCHING_DEPTH],
}


impl<T> PathCache<T> {

    fn new() -> PathCache<T> {
        PathCache{generation: 0,
                  index_cache: None,
                  path_cache: [TrieNodePtr::<T>::default(); BRANCHING_DEPTH]}
    }

    /// Calculate where to start looking in the cache
    fn get_start(&mut self, index: usize) -> (u8, u8) {
        let mut depth = BRANCHING_DEPTH;

        // calulate the bit difference between the last access and the current
        // one and use the count of leading zeros (indicating no difference)
        // to predict where the least significant cache-hit might be
        if let Some(last_index) = self.index_cache {
            // the msb bits that are covered by self.root, no cache lookup
            let msb_ignore = std::usize::BITS % BRANCHING_FACTOR_BITS;

            let diff = (index ^ last_index) << msb_ignore;
            let similarity = diff.leading_zeros();

            depth -= similarity as usize / BRANCHING_FACTOR_BITS + 1;
        }

        self.index_cache = Some(index);

        let shift = (depth + 1) * BRANCHING_FACTOR_BITS;

        (depth as u8, shift as u8)
    }

    /// Update the cache line at the given depth with a new index and node.
    fn set(&mut self, depth: usize, index: usize, node: &TrieNode<T>) {
        self.path_cache[depth].set(index, node);
    }

    /// Invalidate the cache line at the given depth.
    fn invalidate(&mut self, depth: usize) {
        self.path_cache[depth].invalidate();
    }

    /// Get the deepest node that can match the index against the cache.
    // TODO: integrate lookup into path_cache in here, rolling back to nearest
    // entry if the first lookup is to an invalid entry
    fn get_node(&mut self, index: usize) -> Option<(*const TrieNode<T>, u8)> {
        let (cached_depth, shift) = self.get_start(index);

        if cached_depth < BRANCHING_DEPTH as u8 {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;
            let cache = &self.path_cache[cached_depth as usize];

            if cache.is_hit(local_index) {
                return Some((cache.get(), cached_depth));
            }
        }

        None
    }

    /// Get the deepest node that can match the index against the cache.
    fn get_node_mut(&mut self, index: usize) -> Option<(*mut TrieNode<T>, u8)> {
        // look in the cache first, to see if we already have references
        // to the appropriate interior nodes
        let (cached_depth, shift) = self.get_start(index);

        if cached_depth < BRANCHING_DEPTH as u8 {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;
            let cache = &self.path_cache[cached_depth as usize];

            if cache.is_hit(local_index) {
                return Some((cache.get_mut(), cached_depth));
            }
        }

        None
    }

    /// Increment the generation number, wrapping back to zero on overflow
    fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Are these caches synchronized on the same generation?
    fn in_sync_to(&self, other: &PathCache<T>) -> bool {
        self.generation == other.generation
    }

}


impl<T> Clone for PathCache<T> {
    fn clone(&self) -> PathCache<T> {
        PathCache{index_cache: self.index_cache,
                  generation: self.generation,
                  path_cache: self.path_cache.clone()}
    }
}


impl<T> Copy for PathCache<T> {}


/// Path-cached bitmap trie.
///
/// Caveats for *_with_cache() functions:
///  - no way to prevent a PathCache being used with the wrong MultiCacheTrie
///    instance: safety fail
///  - structure-modifying writes are more expensive due to cache invalidation
pub struct Trie<T> {
    root: TrieNode<T>,
    cache: Cell<*mut PathCache<T>>,
}


impl<T> Trie<T> {

    /// Instantiate a new Trie, indexed by a usize integer and storing values
    /// of type T.
    pub fn new() -> Trie<T> {
        let root = TrieNode::new_branch();

        let cache = Box::into_raw(Box::new(PathCache::new()));

        Trie{root: root, cache: Cell::new(cache)}
    }

    /// Set an entry to a value.
    pub fn set(&mut self, index: usize, value: T) -> &mut T {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.set(index, value, depth as usize, cache)
        } else {
            self.root.set(index, value, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Retrieve a value.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.get_mut(index, depth as usize, cache)
        } else {
            self.root.get_mut(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Retrieve a value.
    pub fn get(&self, index: usize) -> Option<&T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node(index) {
            unsafe { &*start_node }.get(index, depth as usize, cache)
        } else {
            self.root.get(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Remove an entry, returning the associated value.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let (Some(value), _) =
            if let Some((start_node, depth)) = cache.get_node_mut(index) {
                unsafe { &mut *start_node }.remove(index, depth as usize, cache)
            } else {
                self.root.remove(index, BRANCHING_DEPTH - 1, cache)
            } {
                Some(value)
            } else {
                None
            }
    }

    // TODO:
    // Per-node generation counter for cache entries to match against: this
    // would mean checking the full path for node-generation changes on every
    // access, but would allow concurrent non-overlapping structural changes
    // without cache invalidation.
    // Another benefit is that doing this might pave the way for a truly
    // concurrent thread-safe trie.

    /// Set an entry to a value, accelerating access with the given cache,
    /// updating the cache with the new path. This function causes a new
    /// generation since it can possibly cause memory to move around.
    pub fn set_with_cache(&mut self,
                          cache: &mut PathCache<T>,
                          index: usize,
                          value: T) -> &mut T {
        let gen_cache = unsafe { &mut *self.cache.get() };

        if !cache.in_sync_to(gen_cache) {
            *cache = *gen_cache;
        }

        let rv = if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.set(index, value, depth as usize, cache)
        } else {
            self.root.set(index, value, BRANCHING_DEPTH - 1, cache)
        };

        cache.new_generation();
        *gen_cache = *cache;

        return rv;
    }

    /// Retrieve a value, accelerating access with the given cache, updating
    /// the cache with the new path.
    pub fn get_mut_with_cache(&mut self,
                              cache: &mut PathCache<T>,
                              index: usize) -> Option<&mut T> {
        let gen_cache = unsafe { &mut *self.cache.get() };

        if !cache.in_sync_to(gen_cache) {
            *cache = *gen_cache;
        }

        if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.get_mut(index, depth as usize, cache)
        } else {
            self.root.get_mut(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Retrieve a value, accelerating access with the given cache, updating
    /// the cache with the new path.
    pub fn get_with_cache(&self,
                          cache: &mut PathCache<T>,
                          index: usize) -> Option<&T> {

        let gen_cache = unsafe { &mut *self.cache.get() };

        if !cache.in_sync_to(gen_cache) {
            *cache = *gen_cache;
        }

        if let Some((start_node, depth)) = cache.get_node(index) {
            unsafe { &*start_node }.get(index, depth as usize, cache)
        } else {
            self.root.get(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Remove an entry, accelerating access with the given cache, updating
    /// the cache with the new path. This function causes a new generation
    /// since it can cause memory to move around.
    pub fn remove_with_cache(&mut self,
                             cache: &mut PathCache<T>,
                             index: usize) -> Option<T> {
        let gen_cache = unsafe { &mut *self.cache.get() };

        if !cache.in_sync_to(gen_cache) {
            *cache = *gen_cache;
        }

        let rv = if let (Some(value), _) =
            if let Some((start_node, depth)) = cache.get_node_mut(index) {
                unsafe { &mut *start_node }.remove(index, depth as usize, cache)
            } else {
                self.root.remove(index, BRANCHING_DEPTH - 1, cache)
            } {
                Some(value)
            } else {
                None
            };

        cache.new_generation();
        *gen_cache = *cache;

        return rv;
    }

    /// Create a new cache for this instance.
    pub fn new_cache(&self) -> PathCache<T> {
        PathCache::new()
    }

    /// Create an iterator over immutable data
    pub fn iter(&self) -> Iter<T> {
        Iter::new(&self.root)
    }
}


impl<T> Drop for Trie<T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(self.cache.get()) };
    }
}


/// Iterator over Trie
pub struct Iter<'a, T: 'a> {
    nodes: [*const TrieNode<T>; BRANCHING_DEPTH],
    points: [(usize, usize); BRANCHING_DEPTH],
    depth: usize,
    current: &'a TrieNode<T>,
    index: usize,
}


impl<'a, T> Iter<'a, T> {
    pub fn new(root: &TrieNode<T>) -> Iter<T> {
        Iter{nodes: [null_mut(); BRANCHING_DEPTH],
             points: [(VALID_MAX, 0); BRANCHING_DEPTH],
             depth: BRANCHING_DEPTH - 1,
             current: root,
             index: 0}
    }
}


impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<(usize, &'a T)> {
        loop {
            match self.current {

                &TrieNode::Interior(ref node) => {
                    let point = self.points[self.depth];

                    if let Some(((mask, comp), (index_part, child)))
                        = node.next(point.0, point.1) {

                            self.points[self.depth] = (mask, comp);

                            let shift = self.depth * BRANCHING_FACTOR_BITS;
                            let index_mask = !(BRANCHING_INDEX_MASK << shift);

                            self.index &= index_mask;
                            self.index |= index_part << shift;

                            self.nodes[self.depth] = self.current;
                            self.current = child;
                            self.depth -= 1;

                        } else {
                            self.points[self.depth] = (VALID_MAX, 0usize);

                            self.depth += 1;
                            if self.depth == BRANCHING_DEPTH {
                                return None;
                            }

                            self.current = unsafe { &*self.nodes[self.depth] };
                        }
                },

                &TrieNode::Exterior(ref node) => {
                    let point = self.points[0];

                    if let Some(((mask, comp), (index_part, value)))
                        = node.next(point.0, point.1) {

                            self.points[0] = (mask, comp);

                            let index_mask = !BRANCHING_INDEX_MASK;
                            self.index = self.index & index_mask | index_part;

                            return Some((self.index, value));

                        } else {
                            self.points[0] = (VALID_MAX, 0usize);
                            self.depth = 1;
                            self.current = unsafe { &*self.nodes[1] };
                        }
                }
            }
        }
    }
}
