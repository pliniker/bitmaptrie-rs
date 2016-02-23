

extern crate bitmaptrie;
use bitmaptrie::Trie;


const TEST_SIZE: usize = 1 << 24;


fn main() {
    let mut t: Trie<usize> = Trie::new();

    for i in 0..TEST_SIZE {
        t.set(i, i);
    }

    for x in 3..5 {
        t.retain_if(|index, _value| (index & 0xff) == x);
    }

    for (index, value) in t.iter() {
        println!("i = {}", index);
        assert!(index == *value);
    }
}
