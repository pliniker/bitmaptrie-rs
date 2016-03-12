//! # Bitmappped Vector Trie
//!
//! A bitmapped vector trie with node compression and a path cache. Values are always sorted by
//! their index; thus iterating is always in index order.
//!
//! The trie does not prescribe a length or capacity beside the range of values of it's index:
//! usize. It could be used to compose a data structure that does behave more like Vec.
//!
//! The branching factor is the word-size: 32 or 64. This makes the depth 6 for 32bit systems and
//! 11 for 64bit systems. Because of the path cache, spatially dense indexes will not cause full
//! depth traversal.
//!
//! There is support for sharding the Trie such that each shard might be passed to a different
//! thread for processing.
//!
//! Performance improvements:
//!
//!  * enable popcnt instruction when supported
//!
//! Possible code improvements:
//!
//!  * with better use of generics, some code duplication might be avoided
//!
//! # Basic Usage
//!
//! ```
//! use bitmaptrie::Trie;
//!
//! let mut t: Trie<String> = Trie::new();
//!
//! // set a key/value. Will overwrite any previous value at the index
//! t.set(123, String::from("testing 123"));
//!
//! // look up a key returning a reference to the value if it exists
//! if let Some(ref value) = t.get(123) {
//!     println!("value = {}", *value);
//! }
//!
//! // will remove the only entry
//! t.retain_if(|key, _value| key != 123);
//! ```
//!
//! # Thread Safety
//!
//! The trie can be borrowed in two ways:
//!  * For `T: Send`: mutable sharding, where each shard can safely be accessed mutably in it's
//!    own thread, allowing destructive updates. This is analagous to `Vec::chunks()`.
//!  * For `T: Send + Sync`: a Sync-safe borrow that can itself be sharded, but prevents
//!    destructive updates. This is analagous to a borrow of `Vec<T: Send + Sync>`.
//!
//! Since the trie is borrowed and not shared using something like `Arc`, it follows that these
//! methods will only work with scoped threading.
//!
//! Sharding works by doing a breadth-first search into the trie to find the depth at which there
//! are at least the number of interior nodes as requested. The returned number of nodes may be
//! much greater than the requested number, or may be less if the trie is small.
//!
//! As there is no knowledge of the balanced-ness of the trie, the more shards that are returned by
//! `borrow_sharded()`, the more evenly the number of leaves on each shard will likely be
//! distributed.
//!
//! ## Mutable Sharding
//!
//! ```
//! use bitmaptrie::Trie;
//!
//! let mut t: Trie<String> = Trie::new();
//! t.set(123, String::from("testing xyzzy"));
//!
//! let mut shards = t.borrow_sharded(4); // shard at least 4 ways if possible
//! for mut shard in shards.drain() {
//!     // launch a scoped thread or submit a task to a queue here and move shard into it for
//!     // processing
//!
//!     // destructive update to the trie, only touching this one shard
//!     shard.retain_if(|_, value| *value == String::from("testing 123"));
//! }
//! ```
//!
//! ## Sync-safe Borrow and Sharding
//!
//! `T` in `Trie<T>` must be Sync in order to make value changes.
//!
//! ```
//! use bitmaptrie::Trie;
//!
//! let mut t: Trie<usize> = Trie::new();
//! t.set(123, 246);
//!
//! let num_threads = 4;
//!
//! let shared = t.borrow_sync();
//! let shards = shared.borrow_sharded(num_threads);
//!
//! for shard in shards.iter() {
//!     let shared = shared.clone();
//!     // launch a scoped thread here or submit a task to a queue and move shard and borrow into
//!     // it for processing
//!
//!     // iterate over the shard's key/values
//!     for (_, value) in shard.iter() {
//!         if let Some(other) = shared.get(*value) {
//!             println!("found a cross reference");
//!         }
//!     }
//! }
//! ```


#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(heap_api)]
#![feature(unique)]


use std::cell::Cell;
use std::collections::VecDeque;
use std::collections::vec_deque::Iter as VecDequeIter;
use std::collections::vec_deque::Drain as VecDequeDrain;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::{Index, IndexMut};
use std::ptr::null_mut;

mod comprawvec;
mod compvec;

pub use compvec::{CompVec,
                  Iter as CompVecIter,
                  IterMut as CompVecIterMut,
                  VALID_MAX};


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
struct PathCache<T> {
    // last index accessed using this cache
    index_cache: Option<usize>,

    // the path through the trie to the exterior node for the last index
    path_cache: [TrieNodePtr<T>; BRANCHING_DEPTH],
}


/// Path-cached bitmap trie type. The key is always a `usize`.
pub struct Trie<T> {
    root: TrieNode<T>,

    // this pattern of putting a raw pointer in a Cell probably isn't the right abstraction
    cache: Cell<*mut PathCache<T>>,
}


/// Iterator over `(key, &T)`s of `Trie<T>`
pub struct Iter<'a, T: 'a> {
    // current path down to the exterior node
    nodes: [*const TrieNode<T>; BRANCHING_DEPTH],
    // position in each node of the child node (masked_valid, compressed_index)
    points: [(usize, usize); BRANCHING_DEPTH],

    // current position in the current path
    depth: usize,
    current: &'a TrieNode<T>,

    // depth at which the iterator is finished
    escape_depth: usize,

    // current full index pieced together from all current nodes
    index: usize,
}


/// Iterator over mutable `(key, &mut T)`s of `Trie<T>`
pub struct IterMut<'a, T: 'a> {
    // current path down to the exterior node
    nodes: [*mut TrieNode<T>; BRANCHING_DEPTH],
    // position in each node of the child node (masked_valid, compressed_index)
    points: [(usize, usize); BRANCHING_DEPTH],

    // current position in the current path
    depth: usize,
    current: *mut TrieNode<T>,

    // depth at which the iterator is finished
    escape_depth: usize,

    // current full index pieced together from all current nodes
    index: usize,

    // because we aren't borrowing a &mut in this struct
    _lifetime: PhantomData<&'a mut T>,
}


/// Borrows a Trie exposing a Sync-type-safe subset of it's methods. No structure
/// modifying methods are included. Each borrow gets it's own path cache.
/// The lifetime of this type is the lifetime of the mutable borrow.
/// This can be used to parallelize structure-immutant changes.
pub struct BorrowSync<'a, T: 'a + Send + Sync> {
    root: *const TrieNode<T>,
    cache: Cell<*mut PathCache<T>>,
    _marker: PhantomData<&'a T>,
}


/// Borrows a BorrowSync, splitting it into interior nodes each of which can be iterated over
/// separately while still giving access to the full Trie.
pub struct BorrowShardImmut<'a, T: 'a + Send + Sync> {
    buffer: VecDeque<ShardImmut<'a, T>>,
}


/// Immutable borrow of an interior Trie node.
pub struct ShardImmut<'a, T: 'a + Send + Sync> {
    index: usize,
    depth: usize,
    node: *const TrieNode<T>,
    _marker: PhantomData<&'a T>,
}


/// Type that borrows a Trie mutably, giving an iterable type that returns interior nodes on
/// which structure-mutating operations can be performed. This can be used to parallelize
/// destructive operations.
pub struct BorrowShardMut<'a, T: 'a + Send> {
    buffer: VecDeque<ShardMut<'a, T>>,
}


/// A type that references an interior Trie node. For splitting a Trie into sub-nodes, each of
/// which can be passed to a different thread for mutable structural changes.
pub struct ShardMut<'a, T: 'a + Send> {
    // higher bits that form the roots of this node
    index: usize,
    depth: usize,
    node: *mut TrieNode<T>,
    _marker: PhantomData<&'a T>,
}


unsafe impl<T: Send> Send for TrieNodePtr<T> {}
unsafe impl<T: Send> Send for PathCache<T> {}

unsafe impl<T: Send> Send for TrieNode<T> {}
unsafe impl<T: Send> Send for Trie<T> {}

unsafe impl<'a, T: 'a + Send> Send for Iter<'a, T> {}
unsafe impl<'a, T: 'a + Send> Send for IterMut<'a, T> {}

unsafe impl<'a, T: 'a + Send + Sync> Send for BorrowSync<'a, T> {}
unsafe impl<'a, T: 'a + Send + Sync> Sync for BorrowSync<'a, T> {}

unsafe impl<'a, T: 'a + Send + Sync> Send for ShardImmut<'a, T> {}
unsafe impl<'a, T: 'a + Send + Sync> Sync for ShardImmut<'a, T> {}

unsafe impl<'a, T: 'a + Send> Send for BorrowShardMut<'a, T> {}
unsafe impl<'a, T: 'a + Send> Send for ShardMut<'a, T> {}


/// TrieNode is a recursive tree data structure where each node has up to 32 or 64 branches
/// depending on the system word size. The tree has a depth attribute which is counted inversely,
/// that is, zero is the leaves of the tree rather than the root.
///
/// Each accessor method takes a PathCache<T> parameter. On access, the path cache is updated
/// with the path taken through the tree. In some cases, the cache is invalidated if a path
/// is or might have been destroyed or wasn't fully followed.
impl<T> TrieNode<T> {
    fn new_branch() -> TrieNode<T> {
        TrieNode::Interior(CompVec::new())
    }

    fn new_leaf() -> TrieNode<T> {
        TrieNode::Exterior(CompVec::new())
    }

    // Inserts a new value at the given index or replaces an existing value at that index.
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

    // Returns a mutable reference to the given index. If no value is at that index, inserts a
    // value using the default function and returns the reference to that.
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

    // Return a mutable reference to a value at the given index or None if there is no value there.
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

    // Return a reference to a value at the given index or None if there is no value there.
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

    // Remove an entry if it exists, returning it if it existed.
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

    // Retains only the elements specified by the predicate.
    fn retain_if<F>(&mut self, index: usize, depth: usize, f: &mut F) -> bool
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


impl<T> PathCache<T> {
    fn new() -> PathCache<T> {
        PathCache {
            index_cache: None,
            path_cache: [TrieNodePtr::<T>::default(); BRANCHING_DEPTH],
        }
    }

    // Calculate where to start looking in the cache
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

    // Update the cache line at the given depth with a new index and node.
    fn set(&mut self, depth: usize, index: usize, node: &TrieNode<T>) {
        self.path_cache[depth].set(index, node);
    }

    // Invalidate the cache line at the given depth.
    fn invalidate(&mut self, depth: usize) {
        self.path_cache[depth].invalidate();
    }

    // Invalidated the whole cache
    fn invalidate_all(&mut self) {
        for d in 0..BRANCHING_DEPTH {
            self.path_cache[d].invalidate();
        }
    }

    // Invalidate cache from the given depth down.
    fn invalidate_down(&mut self, depth: usize) {
        for d in 0..(depth + 1) {
            self.path_cache[d].invalidate();
        }
    }

    // Get the deepest node that can match the index against the cache.
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

    // Get the deepest node that can match the index against the cache.
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
}


impl<T> Clone for PathCache<T> {
    fn clone(&self) -> PathCache<T> {
        PathCache {
            index_cache: self.index_cache,
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
        unsafe { &mut *self.cache.get() }.invalidate_all();
        self.root.retain_if(0usize, BRANCHING_DEPTH - 1, &mut f);
    }

    /// Create an iterator over immutable data
    pub fn iter(&self) -> Iter<T> {
        Iter::new(&self.root, BRANCHING_DEPTH, 0)
    }

    /// Create an iterator over mutable data
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(&mut self.root, BRANCHING_DEPTH, 0)
    }
}


impl<T: Send> Trie<T> {
    /// Shard the trie into at minimum `n` nodes (by doing a breadth-first search for the depth
    /// with at least that many interior nodes) and return an guard type that provides an iterator
    /// to iterate over the nodes. There is no upper bound on the number of nodes returned and less
    /// than n may be returned. The guard type, `BorrowShard`, guards the lifetime of the mutable
    /// borrow, making this suitable for use in a scoped threading context. Invalidates the
    /// cache entirely.
    pub fn borrow_sharded(&mut self, n: usize) -> BorrowShardMut<T> {
        unsafe { &mut *self.cache.get() }.invalidate_all();
        BorrowShardMut::new(&mut self.root, n)
    }

    /// Cleans up any empty interior nodes that may be left dangling. This is only useful in
    /// conjunction with borrow_split() where sub-tries may be left empty but not deleted
    /// themselves and is entirely optional.
    pub fn prune(&mut self) {
        self.retain_if(|_, _| true);
    }
}


impl<T: Send + Sync> Trie<T> {
    /// Create a mutable borrow that gives a subset of functions that can be accessed across
    /// threads if `T` is Sync. Suitable only for a scoped thread as the lifetime of the
    /// `BorrowSync` instance is not `'static` but the same duration as the borrow. Each borrow
    /// contains it's own path cache.
    pub fn borrow_sync(&self) -> BorrowSync<T> {
        BorrowSync::new(&self.root)
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
    pub fn new(root: &TrieNode<T>, escape_depth: usize, base_index: usize) -> Iter<T> {
        Iter {
            nodes: [null_mut(); BRANCHING_DEPTH],
            points: [(VALID_MAX, 0); BRANCHING_DEPTH],
            depth: escape_depth - 1,
            escape_depth: escape_depth,
            current: root,
            index: base_index,
        }
    }
}


impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (usize, &'a T);

    // I'm not too happy with this state machine design. It surely is possible to use generics
    // or macros a bit more to modularize this code.
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
                        if self.depth == self.escape_depth {
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
    pub fn new(root: &mut TrieNode<T>, escape_depth: usize, base_index: usize) -> IterMut<T> {
        IterMut {
            nodes: [null_mut(); BRANCHING_DEPTH],
            points: [(VALID_MAX, 0); BRANCHING_DEPTH],
            depth: escape_depth - 1,
            escape_depth: escape_depth,
            current: root,
            index: base_index,
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
                        if self.depth == self.escape_depth {
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


impl<'a, T: Send + Sync> BorrowSync<'a, T> {
    fn new(root: &'a TrieNode<T>) -> BorrowSync<'a, T> {
        let cache = Box::into_raw(Box::new(PathCache::new()));

        BorrowSync {
            root: root,
            cache: Cell::new(cache),
            _marker: PhantomData
        }
    }

    /// Retrieve a reference to the value at the given index. Updates the local path
    /// cache to point at this index.
    pub fn get(&self, index: usize) -> Option<&'a T> {
        let cache = unsafe { &mut *self.cache.get() };

        if let Some((start_node, depth)) = cache.get_node(index) {
            unsafe { &*start_node }.get(index, depth as usize, cache)
        } else {
            unsafe { &*self.root }.get(index, BRANCHING_DEPTH - 1, cache)
        }
    }

    /// Return a type that can be iterated over to produce interior nodes that themselves can
    /// be iterated over. This type is immutable.
    pub fn borrow_sharded(&self, n: usize) -> BorrowShardImmut<'a, T> {
        BorrowShardImmut::new(unsafe { &*self.root }, n)
    }
}


impl<'a, T: Send + Sync> Clone for BorrowSync<'a, T> {
    /// Clone this instance: the new instance can be sent to another thread. It has it's own
    /// path cache.
    fn clone(&self) -> BorrowSync<'a, T> {
        let cache = Box::into_raw(Box::new(PathCache::new()));

        BorrowSync {
            root: self.root,
            cache: Cell::new(cache),
            _marker: PhantomData
        }
    }
}


// clean up the path cache
impl<'a, T: Send + Sync> Drop for BorrowSync<'a, T> {
    fn drop(&mut self) {
        unsafe { Box::from_raw(self.cache.get()) };
    }
}


impl<'a, T: 'a + Send + Sync> BorrowShardImmut<'a, T> {
    fn new(root: &'a TrieNode<T>, n: usize) -> BorrowShardImmut<'a, T> {
        // breadth-first search into trie to find at least n nodes
        let mut depth = BRANCHING_DEPTH - 1;

        let mut buf = VecDeque::with_capacity(WORD_SIZE);
        buf.push_back(ShardImmut::new(0, depth, root));

        loop {
            if let Some(subtrie) = buf.pop_front() {
                // If we've just switched to popping the next depth and there are sufficient
                // nodes in the buffer, we're done.
                // If we've hit depth 2, we're done because the unit of work per split isn't worth
                // being smaller.
                if (subtrie.depth < depth && buf.len() >= n) ||
                    subtrie.depth == 2 {

                    buf.push_front(subtrie);
                    break;
                }

                depth = subtrie.depth;

                // otherwise keep looking deeper
                if let &TrieNode::Interior(ref int_vec) = unsafe { &*subtrie.node } {
                    for child in int_vec.iter() {
                        let index = subtrie.index | child.0 << (depth * BRANCHING_FACTOR_BITS);
                        buf.push_back(ShardImmut::new(index, subtrie.depth - 1, child.1));
                    }
                } else {
                    unreachable!();
                }
            } else {
                break;
            }
        }

        BorrowShardImmut {
            buffer: buf
        }
    }

    /// Return an Iterator that provides `ShardMut` instances that can be independently mutated.
    pub fn iter(&'a self) -> VecDequeIter<'a, ShardImmut<'a, T>> {
        self.buffer.iter()
    }
}


impl<'a, T: 'a + Send + Sync> ShardImmut<'a, T> {
    fn new(index: usize, depth: usize, node: *const TrieNode<T>) -> ShardImmut<'a, T> {
        ShardImmut {
            index: index,
            depth: depth,
            node: node,
            _marker: PhantomData
        }
    }

    /// Return an iterator across this sub tree
    pub fn iter(&self) -> Iter<T> {
        Iter::new(unsafe { &*self.node }, self.depth + 1, self.index)
    }
}


impl<'a, T: 'a + Send> BorrowShardMut<'a, T> {
    fn new(root: &'a mut TrieNode<T>, n: usize) -> BorrowShardMut<'a, T> {

        // breadth-first search into trie to find at least n nodes
        let mut depth = BRANCHING_DEPTH - 1;

        let mut buf = VecDeque::with_capacity(WORD_SIZE);
        buf.push_back(ShardMut::new(0, depth, root));

        loop {
            if let Some(subtrie) = buf.pop_front() {
                // If we've just switched to popping the next depth and there are sufficient
                // nodes in the buffer, we're done.
                // If we've hit depth 2, we're done because the unit of work per split isn't worth
                // being smaller.
                if (subtrie.depth < depth && buf.len() >= n) ||
                    subtrie.depth == 2 {

                    buf.push_front(subtrie);
                    break;
                }

                depth = subtrie.depth;

                // otherwise keep looking deeper
                if let &mut TrieNode::Interior(ref mut int_vec) = unsafe { &mut *subtrie.node } {
                    for child in int_vec.iter_mut() {
                        let index = subtrie.index | child.0 << (depth * BRANCHING_FACTOR_BITS);
                        buf.push_back(ShardMut::new(index, subtrie.depth - 1, child.1));
                    }
                } else {
                    unreachable!();
                }
            } else {
                break;
            }
        }

        BorrowShardMut {
            buffer: buf
        }
    }

    /// Return a draining Iterator across the whole list of nodes.
    pub fn drain(&'a mut self) -> VecDequeDrain<'a, ShardMut<'a, T>> {
        self.buffer.drain(..)
    }
}


impl<'a, T: 'a + Send> ShardMut<'a, T> {
    fn new(index: usize, depth: usize, node: *mut TrieNode<T>) -> ShardMut<'a, T> {
        ShardMut {
            index: index,
            depth: depth,
            node: node,
            _marker: PhantomData
        }
    }

    /// Return an iterator across this sub tree
    pub fn iter(&self) -> Iter<T> {
        Iter::new(unsafe { &*self.node }, self.depth + 1, self.index)
    }

    /// Return an iterator across mutable values of this sub tree
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(unsafe { &mut *self.node }, self.depth + 1, self.index)
    }

    /// Retains only the elements specified by the predicate `f`.
    pub fn retain_if<F>(&mut self, mut f: F)
        where F: FnMut(usize, &mut T) -> bool
    {
        unsafe { &mut *self.node }.retain_if(self.index, self.depth, &mut f);
    }
}
