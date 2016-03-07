# A bitmapped vector trie for Rust

[![Build Status](https://travis-ci.org/pliniker/bitmaptrie-rs.svg?branch=master)](https://travis-ci.org/pliniker/bitmaptrie-rs)
[![Latest Version](https://img.shields.io/crates/v/bitmaptrie.svg)](https://crates.io/crates/bitmaptrie)

## [Documentation](https://crates.fyi/crates/bitmaptrie/)

Requires Rust-nightly due to use of low-level unstable APIs.

This is a non-persistent bitmapped vector trie with word-size indexing: thus
on a 32-bit system the indexing is 32 bits; on 64-bit it is 64 bits.

It essentially behaves as an unbounded (except by the word-size index) sparse
vector.

Performance is good for spatially-close accesses but deteriorates for random
spatially-sparse accesses. Performance improves significantly if compiled
with popcnt and lzcnt instructions.
See [wiki](https://github.com/pliniker/bitmaptrie-rs/wiki/Benchmark-information)
for more.

The last access path is cached to accelerate the next nearby access.

Multi-path-cache methods are available for accelerating read-only accesses
at multiple positions but the current design causes write performance to
degrade.

### Usage

```rust
extern crate bitmaptrie;
use bitmaptrie::Trie;

fn main() {
    let mut trie: Trie<String> = Trie::new();

    trie.set(123usize, "testing 123".to_owned());

    if let Some(value) = trie.get_mut(123) {
        *value = "test pass".to_owned();
    }
}
```

### Author

Peter Liniker

### License

Dual MIT/Apache 2.0
