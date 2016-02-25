

extern crate bitmaptrie;
use bitmaptrie::Trie;


const TEST_SIZE: usize = 32;


fn main() {
    let mut t: Trie<String> = Trie::new();

    for i in 0..TEST_SIZE {
        t.set(i, String::from("testing"));
    }

    t.retain_if(|index, _value| {
        // keep odd numbered indeces
        index & 0x1 == 0x1
    });

    for (index, value) in t.iter() {
        println!("index = {}, value = {}", index, *value);
    }
}
