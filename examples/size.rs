
use std::mem::size_of;

extern crate bitmaptrie;
use bitmaptrie::TrieNode;


fn main() {
    println!("The size of TrieNode<_> is {}", size_of::<TrieNode<String>>());
    println!("It would be nice if it could be {}", size_of::<usize>());
}
