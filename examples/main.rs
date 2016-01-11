

extern crate bitmaptrie;
use bitmaptrie::Trie;


fn main() {
    let mut t: Trie<usize> = Trie::new();

    t.set(0, 42);
    if let Some(value) = t.get(0) {
        println!("value at 0th index is {}", value);
    }

    t[0] = 17;
    println!("0th = {}", t[0]);

    t.remove(0);
    if let None = t.get(0) {
        println!("0th was removed");
    }

    t[0] = 1;  // panics because index doesn't exist any more
}
