//! A bitmapped vector trie with node compression and a path cache. Values are always sorted by
//! their index; thus iterating is always in index order.
//!
//! The trie does not prescribe a length or capacity beside the range of values of it's index:
//! usize. It could be used to compose a data structure that does behave like Vec.
//!
//! The branching factor is the word-size: 32 or 64. This makes the depth 6 for 32bit systems and
//! 11 for 64bit systems. Because of the path cache, spatially dense indexes will not cause full
//! depth traversal.
//!
//! Performance improvements:
//!
//!  * enable popcnt instruction when supported
//!
//! Possible code improvements:
//!
//!  * with better use of generics, some code duplication might be avoided
//!
//! # Usage
//!
//! ```
//! use bitmaptrie::Trie;
//!
//! let mut t: Trie<String> = Trie::new();
//! t.set(123, "testing 123".to_owned());
//!
//! if let Some(ref value) = t.get(123) {
//!     println!("value = {}", *value);
//! }
//! ```


#![feature(alloc)]
#![feature(associated_consts)]
#![feature(core_intrinsics)]
#![feature(drop_in_place)]
#![feature(heap_api)]
#![feature(unique)]


use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::{Index, IndexMut};
use std::ptr::null_mut;

mod comprawvec;
mod compvec;

pub use compvec::{CompVec, VALID_MAX};


// need these to be consts so they can be plugged into array sizes
#[cfg(target_pointer_width = "32")]
pub const USIZE_BYTES: usize = 4;

#[cfg(target_pointer_width = "64")]
pub const USIZE_BYTES: usize = 8;

pub const WORD_SIZE: usize = USIZE_BYTES * 8;

// number of bits represented by each trie node: 5 or 6
const BRANCHING_FACTOR_BITS: usize = (0b100 | (USIZE_BYTES >> 2));
// bit mask for previous: 31 or or 63
const BRANCHING_INDEX_MASK: usize = (1 << BRANCHING_FACTOR_BITS) - 1;
// 6 or 11
const BRANCHING_DEPTH: usize = (USIZE_BYTES * 8) / BRANCHING_FACTOR_BITS as usize + 1;


/// The identity function.
///
/// Also forces the argument to move.
fn moving<T>(x: T) -> T {
    x
}


/// An interior (branch) or exterior (leaf) trie node, defined recursively.
pub enum TrieNode<T> {
    // node that contains further nodes
    Interior(CompVec<TrieNode<T>>),

    // node that contains the T-typed values
    Exterior(CompVec<T>),
}


/// An index path cache line, with the index and the node it refers to.
struct TrieNodePtr<T> {
    // the node-local index for this node pointer
    index: usize,

    // A null pointer here indicates an invalid cache line.
    node: *mut TrieNode<T>,
}


/// A cached path into a trie
pub struct PathCache<T> {
    // used to compare caches against each other, the newest is always valid
    generation: usize,

    // last index accessed using this cache
    index_cache: Option<usize>,

    // the path through the trie to the exterior node for the last index
    path_cache: [TrieNodePtr<T>; BRANCHING_DEPTH],
}


/// Path-cached bitmap trie.
///
/// Caveats for *_with_cache() functions:
///  - no way to prevent a PathCache being used with the wrong MultiCacheTrie
///    instance: safety fail
///  - structure-modifying writes are more expensive due to cache invalidation
pub struct Trie<T> {
    root: TrieNode<T>,

    // this pattern of putting a raw pointer in a Cell probably isn't the right abstraction
    cache: Cell<*mut PathCache<T>>,
}


/// Iterator over Trie
pub struct Iter<'a, T: 'a> {
    // current path down to the exterior node
    nodes: [*const TrieNode<T>; BRANCHING_DEPTH],
    // position in each node of the child node (masked_valid, compressed_index)
    points: [(usize, usize); BRANCHING_DEPTH],

    // current position in the current path
    depth: usize,
    current: &'a TrieNode<T>,

    // current full index pieced together from all current nodes
    index: usize,
}


/// Iterator over Trie
pub struct IterMut<'a, T: 'a> {
    // current path down to the exterior node
    nodes: [*mut TrieNode<T>; BRANCHING_DEPTH],
    // position in each node of the child node (masked_valid, compressed_index)
    points: [(usize, usize); BRANCHING_DEPTH],

    // current position in the current path
    depth: usize,
    current: *mut TrieNode<T>,

    // current full index pieced together from all current nodes
    index: usize,

    // because we aren't borrowing a &mut in this struct
    _lifetime: PhantomData<&'a mut T>,
}


/// Borrows a Trie mutably, exposing a Sync-safe subset of it's methods. No trie structure
/// modifying methods are included.
/// The lifetime of this type is the lifetime of the mutable borrow.
pub struct ParallelBorrow<'a, T: 'a> {
    trie: *mut Trie<T>,
    _marker: PhantomData<&'a T>
}


/// A type that references an interior Trie node. For splitting a Trie into sub-nodes, each of
/// which can be passed to a different thread for mutable structural changes.
pub struct SubTrie<'a, T: 'a> {
    node: *mut TrieNode<T>,
    _marker: PhantomData<&'a T>
}


/// Iterator type that returns interior nodes in the SubTrie type, on which mutable operations
/// can be performed.
pub struct SubTrieIter<'a, T: 'a> {
    _marker: PhantomData<&'a T>
}


unsafe impl<T: Send> Send for TrieNodePtr<T> {}
unsafe impl<T: Send> Send for PathCache<T> {}

unsafe impl<T: Send> Send for TrieNode<T> {}
unsafe impl<T: Send> Send for Trie<T> {}

unsafe impl<'a, T: Send> Send for Iter<'a, T> {}
unsafe impl<'a, T: Send> Send for IterMut<'a, T> {}

unsafe impl<'a, T: Send> Send for ParallelBorrow<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for ParallelBorrow<'a, T> {}


impl<T> TrieNode<T> {
    fn new_branch() -> TrieNode<T> {
        TrieNode::Interior(CompVec::new())
    }

    fn new_leaf() -> TrieNode<T> {
        TrieNode::Exterior(CompVec::new())
    }

    fn set(&mut self, index: usize, value: T, depth: usize, cache: &mut PathCache<T>) -> &mut T {

        let mut depth = depth;
        let mut shift = depth * BRANCHING_FACTOR_BITS;
        let mut current = self;

        loop {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;

            match moving(current) {
                &mut TrieNode::Interior(ref mut branch) => {
                    let child = branch.get_default_mut(local_index, || {
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
                }

                &mut TrieNode::Exterior(ref mut leaf) => {
                    return leaf.set(local_index, value);
                }
            }
        }
    }

    fn get_default_mut<F>(&mut self,
                          index: usize,
                          depth: usize,
                          default: F,
                          cache: &mut PathCache<T>)
                          -> &mut T
        where F: Fn() -> T
    {
        let mut depth = depth;
        let mut shift = depth * BRANCHING_FACTOR_BITS;
        let mut current = self;

        loop {
            let local_index = (index >> shift) & BRANCHING_INDEX_MASK;

            match moving(current) {
                &mut TrieNode::Interior(ref mut branch) => {
                    let child = branch.get_default_mut(local_index, || {
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
                }

                &mut TrieNode::Exterior(ref mut leaf) => {
                    return leaf.get_default_mut(local_index, default);
                }
            }
        }
    }

    fn get_mut(&mut self, index: usize, depth: usize, cache: &mut PathCache<T>) -> Option<&mut T> {

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
                        cache.invalidate_down(depth);
                        return None;
                    }
                }

                &mut TrieNode::Exterior(ref mut leaf) => return leaf.get_mut(local_index as usize),
            }
        }
    }

    fn get(&self, index: usize, depth: usize, cache: &mut PathCache<T>) -> Option<&T> {

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
                        cache.invalidate_down(depth);
                        return None;
                    }
                }

                &TrieNode::Exterior(ref leaf) => return leaf.get(local_index as usize),
            }
        }
    }

    /// Remove an entry if it exists, returning it if it existed.
    fn remove(&mut self,
              index: usize,
              depth: usize,
              cache: &mut PathCache<T>)
              -> (Option<T>, bool) {
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
            }

            &mut TrieNode::Exterior(ref mut leaf) => {
                let local_index = index & BRANCHING_INDEX_MASK;
                let value = leaf.remove(local_index as usize);
                cache.invalidate(depth);
                (value, leaf.is_empty())
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain_if<F>(&mut self, index: usize, depth: usize, f: &mut F) -> bool
        where F: FnMut(usize, &mut T) -> bool
    {
        // must be recursive in order to delete empty leaves and branches

        match self {
            &mut TrieNode::Interior(ref mut int_vec) => {
                // interior CompVec
                let shift = depth * BRANCHING_FACTOR_BITS;
                let mut masked_valid = VALID_MAX;
                let mut compressed = 0;

                // for each child node...
                loop {
                    let (do_remove, local_index, next_masked_valid, next_compressed) = {
                        // look up child mutably inside this scope
                        if let Some(((v, c), (i, ref mut child))) = int_vec.next_mut(masked_valid,
                                                                                     compressed) {

                            let index = index | (i << shift);

                            // recurse into child
                            (child.retain_if(index, depth - 1, f), i, v, c)
                        } else {
                            break;
                        }
                    };

                    if do_remove {
                        // remove is a mutable operation and can't be called in the same scope as
                        // int_vec.next_mut()
                        int_vec.remove(local_index as usize);
                    } else {
                        compressed = next_compressed;  // only advance compressed if not removed
                    }

                    masked_valid = next_masked_valid;
                }

                int_vec.is_empty()
            }

            &mut TrieNode::Exterior(ref mut ext_vec) => {
                // exterior CompVec
                let mut masked_valid = VALID_MAX;
                let mut compressed = 0;

                loop {
                    let (do_remove, local_index, next_masked_valid, next_compressed) = {

                        if let Some(((v, c), (i, ref mut value))) = ext_vec.next_mut(masked_valid,
                                                                                     compressed) {

                            let index = index | i;

                            (!f(index, value), i, v, c)
                        } else {
                            break;
                        }
                    };

                    if do_remove {
                        ext_vec.remove(local_index as usize);
                    } else {
                        compressed = next_compressed;
                    }

                    masked_valid = next_masked_valid;
                }

                ext_vec.is_empty()
            }
        }
    }
}


impl<T> Clone for TrieNodePtr<T> {
    fn clone(&self) -> TrieNodePtr<T> {
        TrieNodePtr {
            index: self.index,
            node: self.node,
        }
    }
}


impl<T> Copy for TrieNodePtr<T> {}


impl<T> Default for TrieNodePtr<T> {
    fn default() -> TrieNodePtr<T> {
        TrieNodePtr {
            index: 0,
            node: null_mut(),
        }
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


impl<T> PathCache<T> {
    fn new() -> PathCache<T> {
        PathCache {
            generation: 0,
            index_cache: None,
            path_cache: [TrieNodePtr::<T>::default(); BRANCHING_DEPTH],
        }
    }

    /// Calculate where to start looking in the cache
    fn get_start(&mut self, index: usize) -> u8 {
        // calulate the bit difference between the last access and the current
        // one and use the count of leading zeros (indicating no difference)
        // to predict where the least significant cache-hit might be
        let depth = if let Some(last_index) = self.index_cache {
            let diff = WORD_SIZE - (index ^ last_index).leading_zeros() as usize;

            (diff / BRANCHING_FACTOR_BITS) + if diff % BRANCHING_FACTOR_BITS > 0 { 1 } else { 0 }
        } else {
            BRANCHING_DEPTH
        };

        self.index_cache = Some(index);

        depth as u8
    }

    /// Update the cache line at the given depth with a new index and node.
    fn set(&mut self, depth: usize, index: usize, node: &TrieNode<T>) {
        self.path_cache[depth].set(index, node);
    }

    /// Invalidate the cache line at the given depth.
    fn invalidate(&mut self, depth: usize) {
        self.path_cache[depth].invalidate();
    }

    /// Invalidated the whole cache
    fn invalidate_all(&mut self) {
        for d in 0..BRANCHING_DEPTH {
            self.path_cache[d].invalidate();
        }
    }

    /// Invalidate cache from the given depth down.
    fn invalidate_down(&mut self, depth: usize) {
        for d in 0..(depth + 1) {
            self.path_cache[d].invalidate();
        }
    }

    /// Get the deepest node that can match the index against the cache.
    fn get_node(&mut self, index: usize) -> Option<(*const TrieNode<T>, u8)> {
        // down to this depth at least matches the last access
        let mut cached_depth = self.get_start(index);

        while cached_depth < (BRANCHING_DEPTH as u8 - 1) {
            let parent_shift = (cached_depth + 1) * BRANCHING_FACTOR_BITS as u8;
            let parent_index = (index >> parent_shift) & BRANCHING_INDEX_MASK;

            let cache = &self.path_cache[cached_depth as usize];

            if cache.is_hit(parent_index) {
                return Some((cache.get(), cached_depth));
            }

            cached_depth += 1;
        }

        None
    }

    /// Get the deepest node that can match the index against the cache.
    fn get_node_mut(&mut self, index: usize) -> Option<(*mut TrieNode<T>, u8)> {
        // down to this depth at least matches the last access
        let mut cached_depth = self.get_start(index);

        while cached_depth < (BRANCHING_DEPTH as u8 - 1) {
            let parent_shift = (cached_depth + 1) * BRANCHING_FACTOR_BITS as u8;
            let parent_index = (index >> parent_shift) & BRANCHING_INDEX_MASK;

            let cache = &self.path_cache[cached_depth as usize];

            if cache.is_hit(parent_index) {
                return Some((cache.get_mut(), cached_depth));
            }

            cached_depth += 1;
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
        PathCache {
            index_cache: self.index_cache,
            generation: self.generation,
            path_cache: self.path_cache.clone(),
        }
    }
}


impl<T> Copy for PathCache<T> {}


impl<T> Trie<T> {
    /// Instantiate a new Trie, indexed by a usize integer and storing values
    /// of type T.
    pub fn new() -> Trie<T> {
        let root = TrieNode::new_branch();

        let cache = Box::into_raw(Box::new(PathCache::new()));

        Trie {
            root: root,
            cache: Cell::new(cache),
        }
    }

    /// Set an entry to a value, moving the new value into the trie. Updates the internal path
    /// cache to point at this index.
    pub fn set(&mut self, index: usize, value: T) -> &mut T {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.set(index, value, depth as usize, cache)
        } else {
            self.root.set(index, value, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Retrieve a mutable reference to the value at the given index. If the index does not have
    /// an associated value, call the default function to generate a value for that index and
    /// return the reference to it. Updates the internal path cache to point at this index.
    pub fn get_default_mut<F>(&mut self, index: usize, default: F) -> &mut T
        where F: Fn() -> T
    {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.get_default_mut(index, depth as usize, default, cache)
        } else {
            self.root.get_default_mut(index, BRANCHING_DEPTH - 1, default, cache)
        }
    }

    /// Retrieve a mutable reference to the value at the given index. Updates the internal path
    /// cache to point at this index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.get_mut(index, depth as usize, cache)
        } else {
            self.root.get_mut(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Retrieve a reference to the value at the given index. Updates the internal path cache to
    /// point at this index.
    pub fn get(&self, index: usize) -> Option<&T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node(index) {
            unsafe { &*start_node }.get(index, depth as usize, cache)
        } else {
            self.root.get(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Remove an entry, returning the associated value. Invalidates the internal path cache from
    /// the depth of tree modification out to the leaf if anything was removed.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let (Some(value), _) = if let Some((start_node, depth)) = cache.get_node_mut(index) {
            unsafe { &mut *start_node }.remove(index, depth as usize, cache)
        } else {
            self.root.remove(index, BRANCHING_DEPTH - 1, cache)
        } {
            Some(value)
        } else {
            None
        }
    }

    /// Retains only the elements specified by the predicate. Invalidates the cache entirely.
    pub fn retain_if<F>(&mut self, mut f: F)
        where F: FnMut(usize, &mut T) -> bool
    {
        self.root.retain_if(0usize, BRANCHING_DEPTH - 1, &mut f);
        unsafe { &mut *self.cache.get() }.invalidate_all();
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
    pub fn set_with_cache(&mut self, cache: &mut PathCache<T>, index: usize, value: T) -> &mut T {
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
    pub fn get_mut_with_cache(&mut self, cache: &mut PathCache<T>, index: usize) -> Option<&mut T> {
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
    pub fn get_with_cache(&self, cache: &mut PathCache<T>, index: usize) -> Option<&T> {

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
    pub fn remove_with_cache(&mut self, cache: &mut PathCache<T>, index: usize) -> Option<T> {
        let gen_cache = unsafe { &mut *self.cache.get() };

        if !cache.in_sync_to(gen_cache) {
            *cache = *gen_cache;
        }

        let rv = if let (Some(value), _) = if let Some((start_node, depth)) =
                                                  cache.get_node_mut(index) {
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

    /// Create an iterator over mutable data
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(&mut self.root)
    }

    /// Create a mutable borrow that gives a subset of functions that can be accessed across
    /// threads.
    pub fn parallel_borrow(&mut self) -> ParallelBorrow<T> {
        ParallelBorrow::from_trie(self)
    }
}


impl<T> Drop for Trie<T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(self.cache.get()) };
    }
}


impl<T> Index<usize> for Trie<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index).unwrap()
    }
}


impl<T> IndexMut<usize> for Trie<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).unwrap()
    }
}


impl<'a, T> Iter<'a, T> {
    pub fn new(root: &TrieNode<T>) -> Iter<T> {
        Iter {
            nodes: [null_mut(); BRANCHING_DEPTH],
            points: [(VALID_MAX, 0); BRANCHING_DEPTH],
            depth: BRANCHING_DEPTH - 1,
            current: root,
            index: 0,
        }
    }
}


impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    // I'm not too happy with this state machine design. It surely is possible to use generics
    // a bit more to modularize this code.
    fn next(&mut self) -> Option<(usize, &'a T)> {
        loop {
            match self.current {

                &TrieNode::Interior(ref node) => {
                    let point = self.points[self.depth];

                    if let Some(((mask, comp), (index_part, child))) = node.next(point.0, point.1) {

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
                }

                &TrieNode::Exterior(ref node) => {
                    let point = self.points[0];

                    if let Some(((mask, comp), (index_part, value))) = node.next(point.0, point.1) {

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


impl<'a, T> IterMut<'a, T> {
    pub fn new(root: &mut TrieNode<T>) -> IterMut<T> {
        IterMut {
            nodes: [null_mut(); BRANCHING_DEPTH],
            points: [(VALID_MAX, 0); BRANCHING_DEPTH],
            depth: BRANCHING_DEPTH - 1,
            current: root,
            index: 0,
            _lifetime: PhantomData,
        }
    }
}


impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (usize, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match unsafe { &mut *self.current } {

                &mut TrieNode::Interior(ref mut node) => {
                    let point = self.points[self.depth];

                    if let Some(((mask, comp), (index_part, child))) = node.next_mut(point.0,
                                                                                     point.1) {

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

                        self.current = unsafe { &mut *self.nodes[self.depth] };
                    }
                }

                &mut TrieNode::Exterior(ref mut node) => {
                    let point = self.points[0];

                    if let Some(((mask, comp), (index_part, value))) = node.next_mut(point.0,
                                                                                     point.1) {

                        self.points[0] = (mask, comp);

                        let index_mask = !BRANCHING_INDEX_MASK;
                        self.index = self.index & index_mask | index_part;

                        return Some((self.index, value));

                    } else {
                        self.points[0] = (VALID_MAX, 0usize);
                        self.depth = 1;
                        self.current = unsafe { &mut *self.nodes[1] };
                    }
                }
            }
        }
    }
}


impl<'a, T> ParallelBorrow<'a, T> {
    fn from_trie(trie: &'a mut Trie<T>) -> ParallelBorrow<'a, T> {
        ParallelBorrow {
            trie: trie,
            _marker: PhantomData
        }
    }

    pub fn clone(&self) -> ParallelBorrow<'a, T> {
        ParallelBorrow {
            trie: self.trie,
            _marker: PhantomData
        }
    }

    fn trie(&self) -> &'a Trie<T> {
        unsafe { &*self.trie }
    }

    fn trie_mut(&mut self) -> &'a mut Trie<T> {
        unsafe { &mut *self.trie }
    }

    pub fn get(&self, index: usize) -> Option<&'a T> {
        self.trie().get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&'a mut T> {
        self.trie_mut().get_mut(index)
    }
}
