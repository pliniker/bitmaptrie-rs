# A bitmapped vector trie for Rust

[![Build Status](https://travis-ci.org/pliniker/bitmaptrie-rs.svg?branch=master)](https://travis-ci.org/pliniker/bitmaptrie-rs)

Requires Rust-nightly due to use of low-level unstable APIs.

This is a non-persistent bitmapped vector trie with word-size indexing: thus
on a 32-bit system the indexing is 32 bits; on 64-bit it is 64 bits.

It essentially behaves as an unbounded (except by the word-size index) sparse
vector.

Performance is good for spatially-close accesses but deteriorates for random
spatially-sparse accesses. Performance improves significantly if compiled 
with popcnt and lzcnt instructions.

The last access path is cached to accelerate the next nearby access.

Multi-path-cache methods are available for accelerating read-only accesses
at multiple positions but the current design causes write performance to 
degrade.

### Usage

```
extern crate bitmaptrie;
use bitmaptrie::Trie;
  
let mut trie: Trie<String> = Trie::new();

trie.set(123usize, "testing 123".to_owned());
```

### TODO

- implement Index and IndexMut traits
