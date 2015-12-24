

extern crate bitmaptrie;
use bitmaptrie::Trie;


fn main() {
    let mut t: Trie<usize> = Trie::new();

    for i in 0..(1usize << 17) {
        t.set(i, i);
    }
}
