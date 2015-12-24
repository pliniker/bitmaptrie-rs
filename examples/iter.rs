

extern crate bitmaptrie;
use bitmaptrie::Trie;


fn main() {
    let mut t: Trie<usize> = Trie::new();

    for i in 0..(1usize << 7) {
        t.set(i, i);
    }

    let mut accum = 0;
    for (_index, value) in t.iter() {
        accum += *value;
    }

    println!("{:?}", accum);
}
