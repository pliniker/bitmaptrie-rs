

extern crate bitmaptrie;
use bitmaptrie::Trie;


const TEST_SIZE: usize = 1 << 20;


fn main() {
    let mut t: Trie<usize> = Trie::new();

    for _ in 0..TEST_SIZE {
        for i in 0..TEST_SIZE {
            let b = Box::new(String::from("testing"));
            let r = Box::into_raw(b);

            t.set(r as usize, i);
        }

        println!("sweeping");

        t.retain_if(|ptr, _| {
            let b = unsafe { Box::from_raw(ptr as *mut String) };
            drop(b);
            false
        });

        for (index, value) in t.iter() {
            println!("i = {}", index);
        }
    }
}
