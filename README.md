# A bitmapped vector trie for Rust

This is a non-persistent bitmapped vector trie with word-size indexing: thus
on a 32-bit system the indexing is 32 bits; on 64-bit it is 64 bits.

It essentially behaves as an unbounded (except by the word-size index) sparse
vector.

Performance is good for spatially-close accesses but deteriorates for random
spatially-sparse accesses. Performance improves significantly if compiled 
with popcnt and lzcnt instructions.

### Usage

```
extern crate bitmaptrie;
use bitmaptrie::Trie;
  
let mut trie: Trie<String> = Trie::new();

trie.set(123usize, "testing 123".to_owned());
```

### TODO

- implement Index and IndexMut traits
