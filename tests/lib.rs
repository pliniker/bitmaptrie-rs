#![feature(test)]

extern crate test;
extern crate bitmaptrie;


mod tests {

    use std;
    use bitmaptrie::{CompVec, Trie, VALID_MAX};


    fn shl_or_zero(value: usize, shift: u32) -> usize {
        if shift >= (std::mem::size_of::<usize>() * 8) as u32 {
            0
        } else {
            value << shift
        }
    }

    #[test]
    fn test_node_0() {
        let mut n: CompVec<usize> = CompVec::new();

        n.set(3, 42);

        if let Some(x) = n.get(3) {
            assert!(*x == 42);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_node_n() {
        let mut n: CompVec<usize> = CompVec::new();

        for i in 0..(std::mem::size_of::<usize>() * 8) {
            n.set(i, i);

            if let Some(x) = n.get(i) {
                assert!(*x == i);
            } else {
                assert!(false);
            }
        }

        for i in 0..(std::mem::size_of::<usize>() * 8) {
            if let Some(x) = n.get(i) {
                assert!(*x == i);
            } else {
                assert!(false);
            }
        }
    }

    #[test]
    fn test_node_mut() {
        let mut n: CompVec<usize> = CompVec::new();
        n.set(0, 2);

        {
            if let Some(ref mut k) = n.get_mut(0) {
                **k = 3;
            } else {
                assert!(false);
            }
        }

        if let Some(i) = n.get(0) {
            assert!(*i == 3);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_node_default() {
        let n: CompVec<usize> = CompVec::new();

        if let None = n.get(0) {
            assert!(true);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_node_remove() {
        let mut t: CompVec<String> = CompVec::new();

        t.set(0, "thing1".to_owned());
        t.set(1, "thing2".to_owned());
        t.set(2, "cat".to_owned());
        t.set(3, "zoo".to_owned());
        t.set(4, "animals".to_owned());
        t.set(5, "species".to_owned());
        t.set(6, "genus".to_owned());

        assert_eq!(t.size(), 7);

        t.set(7, "flora".to_owned());

        assert_eq!(t.size(), 8);

        t.remove(3);

        if let Some(s) = t.get(4) {
            assert_eq!(s, "animals");
        } else {
            assert!(false);
        }

        assert_eq!(t.size(), 7);
    }

    #[test]
    fn test_node_next() {
        let vm = VALID_MAX;

        let mut t: CompVec<usize> = CompVec::new();

        assert_eq!(t.next(vm, 0), None);

        t.set(3, 42);

        if let Some(((mask, comp), (index, value))) = t.next(vm, 0) {
            assert_eq!(mask, shl_or_zero(vm, 4));
            assert_eq!(comp, 1);
            assert_eq!(index, 3);
            assert_eq!(*value, 42);
        } else {
            assert!(false);
        }

        assert_eq!(t.next(shl_or_zero(vm, 4), 1), None);
    }

    #[test]
    fn test_node_recur() {
        let mut t: CompVec<CompVec<usize>> = CompVec::new();

        t.set(3, CompVec::new());
    }

    #[test]
    fn test_trie_0() {
        let mut t: Trie<usize> = Trie::new();

        t.set(0, 42);
        t.set(1, 99);
        if let Some(x) = t.get(0) {
            assert!(*x == 42);
        } else {
            assert!(false);
        }
    }

    // check an entire space for panics
    #[test]
    fn test_trie_n() {
        let bits = 14;

        let mut t: Trie<usize> = Trie::new();

        for i in 0..(1 << bits) {
            t.set(i, i as usize);
        }

        for i in 0..(1 << bits) {
            let value = t.get(i);

            println!("expect={:?} got={:?}", i, value);

            if let Some(&k) = value {
                assert!(i as usize == k);
            } else {
                assert!(false);
            }
        }

        // assert!(false);
    }

    #[test]
    fn test_trie_easy_remove() {
        let mut t: Trie<String> = Trie::new();

        t.set(0xABCD, "abcd".to_owned());

        assert!(t.remove(0) == None);

        if let Some(ref k) = t.get(0xABCD) {
            assert_eq!(*k, "abcd");
        } else {
            assert!(false);
        }

        if let Some(k) = t.remove(0xABCD) {
            assert_eq!(k, "abcd");
        } else {
            assert!(false);
        }

        assert!(t.remove(0xABCD) == None);
    }

    #[test]
    fn test_trie_iter() {
        let mut t: Trie<String> = Trie::new();

        t.set(1234567, "abcd".to_owned());
        t.set(9999999, "foo".to_owned());

        let mut i = t.iter();

        if let Some((index, value)) = i.next() {
            assert_eq!(index, 1234567);
            assert_eq!(*value, "abcd".to_owned());
        } else {
            assert!(false);
        }

        if let Some((index, value)) = i.next() {
            assert_eq!(index, 9999999);
            assert_eq!(*value, "foo".to_owned());
        } else {
            assert!(false);
        }

        assert_eq!(i.next(), None);
    }

    #[test]
    fn test_trie_iter_mut() {
        let test_data = [(1, 0), (1000, 999), (100000, 99999), (1000000, 999999)];
        let mut t: Trie<usize> = Trie::new();

        for &(k, v) in &test_data {
            t.set(k, v);
        }

        for (_, v) in t.iter_mut() {
            *v += 1;
        }

        for (k, v) in t.iter() {
            assert_eq!(k, *v);
        }
    }

    #[test]
    fn test_trie_retain_if() {
        let count = 1000000;
        let mut t: Trie<usize> = Trie::new();

        for i in 0..count {
            t.set(i, i + 1);
        }

        // keep values whose index's 3 lsb are 0; equalize value with index
        t.retain_if(|k, v| {
            if k & 0x7 == 0 {
                *v -= 1;
                true
            } else {
                false
            }
        });

        for (k, v) in t.iter() {
            println!("{} => {}", k, *v);
        }

        for i in 0..count {
            if let Some(v) = t.get(i) {
                println!("i = {}, v = {}", i, *v);
                assert!(i & 0x7 == 0);
                assert!(*v == i);
            } else {
                println!("i = {}", i);
                assert!(i & 0x7 != 0);
            }
        }
    }

    #[test]
    fn test_trie_get_default() {
        let mut t: Trie<String> = Trie::new();

        if let Some(_) = t.get(123) {
            assert!(false);
        }

        {
            let value = t.get_default_mut(123, || { String::from("testing") });

            assert!(*value == "testing");

            *value = String::from("tested");
        }

        if let Some(new_val) = t.get(123) {
            assert!(*new_val == "tested");
        }
    }

    #[test]
    fn test_trie_borrow_sync() {
        let mut t: Trie<usize> = Trie::new();
        t.set(123, 0);

        {
            let mut sync_1 = t.borrow_sync();

            if let Some(value) = sync_1.get_mut(123) {
                *value += 1;
            }

            let sync_2 = sync_1.clone();

            if let Some(_) = sync_2.get(124) {
                assert!(false);
            }
        }

        if let Some(value) = t.get(123) {
            assert!(*value == 1);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn test_trie_borrow_split() {
        let magic_number = 4;

        let word_size = std::mem::size_of::<usize>() * 8;
        let factor = word_size * word_size * word_size * word_size;

        let mut t: Trie<usize> = Trie::new();

        for i in 0..magic_number {
            let index = i * factor;
            t.set(index, 0);
            println!("set i={}", index);
        }

        // check number of chunks the trie got split into
        {
            let mut guard = t.borrow_split(magic_number);
            assert!(guard.iter_mut().count() == magic_number);
        }

        // remove all values concurrently
        {
            let mut guard = t.borrow_split(magic_number);

            for node in guard.iter_mut() {
                node.retain_if(|index, _| {

                    // index must be one of the original set
                    let mut valid = false;
                    for i in 0..magic_number {
                        if index == i * factor {
                            valid = true;
                        }
                    }

                    println!("get i={}", index);
                    assert!(valid);

                    false
                });
            }
        }

        // trie should be empty
        assert!(t.iter().count() == 0);
    }
}
