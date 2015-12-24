
//! Benchmarks
//!
//! Performance is measured against Rust's std HashMap with usize key and
//! value.
//!

#![feature(test)]
#![feature(num_bits_bytes)]

extern crate test;
extern crate rand;
extern crate bitmaptrie;


mod bench {

    use test::Bencher;

    use std;
    use std::collections::HashMap;

    use rand::{Rng, SeedableRng};
    use rand::chacha::ChaChaRng;

    use bitmaptrie::Trie;

    // arbitrary benchmark parameters
    const BENCH_LENGTH: usize = 1 << 14;

    const IS_64BIT: usize = std::usize::BYTES >> 3;

    // As the original purpose of this is to map memory addresses, the least
    // significant bits can be ignored due to alignment (2 bits on 32bit, 3
    // bits on 64bit).
    const SPARSE_RANGE: usize = 1 << (32 - 2 - IS_64BIT); // 4GB range
    const DENSE_RANGE: usize = 1 << (26 - 2 - IS_64BIT); // 64MB range


    // generate a repeatable pseudo random sequence of numbers, repeatable
    // so that the same sequence is used in every test
    fn generate_pseudo_randoms(length: usize, max: usize) -> Vec<usize> {
        let seed: &[_] = &[1, 2, 3, 4];
        let mut rng = ChaChaRng::from_seed(seed);

        let mut sequence: Vec<usize> = Vec::with_capacity(length);

        for _i in 0..length {
            sequence.push(rng.gen_range(0, max));
        }

        sequence
    }


    #[bench]
    fn bench_trie_insert_randoms_dense(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);

        b.iter(|| {
            let mut t: Trie<usize> = Trie::new();
            for i in &seq {
                t.set(*i, *i as usize);
            }
        });
    }


    #[bench]
    fn bench_hashmap_insert_randoms_dense(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);

        b.iter(|| {
            let mut h: HashMap<usize, usize> = HashMap::new();
            for i in &seq {
                h.insert(*i, *i as usize);
            }
        });
    }


    #[bench]
    fn bench_trie_insert_randoms_sparse(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, SPARSE_RANGE);

        b.iter(|| {
            let mut t: Trie<usize> = Trie::new();
            for i in &seq {
                t.set(*i, *i as usize);
            }
        });
    }


    #[bench]
    fn bench_hashmap_insert_randoms_sparse(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, SPARSE_RANGE);

        b.iter(|| {
            let mut h: HashMap<usize, usize> = HashMap::new();
            for i in &seq {
                h.insert(*i, *i as usize);
            }
        });
    }


    #[bench]
    fn bench_trie_insert_sequence(b: &mut Bencher) {
        b.iter(|| {
            let mut t: Trie<usize> = Trie::new();
            for i in 0..BENCH_LENGTH {
                t.set(i, i as usize);
            }
        });
    }


    #[bench]
    fn bench_hashmap_insert_sequence(b: &mut Bencher) {
        b.iter(|| {
            let mut h: HashMap<usize, usize> = HashMap::new();
            for i in 0..BENCH_LENGTH {
                h.insert(i, i as usize);
            }
        });
    }


    #[bench]
    fn bench_trie_insert_repeat(b: &mut Bencher) {
        b.iter(|| {
            let mut t: Trie<usize> = Trie::new();
            for i in 0..BENCH_LENGTH {
                t.set(0xABCDEF0, i as usize);
            }
        });
    }


    #[bench]
    fn bench_hashmap_insert_repeat(b: &mut Bencher) {
        b.iter(|| {
            let mut h: HashMap<usize, usize> = HashMap::new();
            for i in 0..BENCH_LENGTH {
                h.insert(0xABCDEF0, i as usize);
            }
        });
    }


    #[bench]
    fn bench_trie_crud(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);

        b.iter(|| {
            let mut t: Trie<usize> = Trie::new();

            for i in &seq {
                t.set(*i, *i as usize);
                if let Some(ref mut v) = t.get_mut(*i) {
                    **v = 0xABCDEF0 as usize;
                }
                t.remove(*i);
            }
        });
    }


    #[bench]
    fn bench_trie_multicache_crud(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);

        b.iter(|| {
            let mut t: Trie<usize> = Trie::new();
            let mut k = t.new_cache();

            for i in &seq {
                t.set_with_cache(&mut k, *i, *i as usize);
                if let Some(ref mut v) = t.get_mut_with_cache(&mut k, *i) {
                    **v = 0xABCDEF0 as usize;
                }
                t.remove_with_cache(&mut k, *i);
            }
        });
    }


    #[bench]
    fn bench_hashmap_crud(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);

        b.iter(|| {
            let mut h: HashMap<usize, usize> = HashMap::new();

            for i in &seq {
                h.insert(*i, *i as usize);
                if let Some(ref mut v) = h.get_mut(i) {
                    **v = 0xABCDEF0 as usize;
                }
                h.remove(i);
            }
        });
    }

    #[bench]
    fn bench_trie_iter_dense(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);
        let mut t: Trie<usize> = Trie::new();

        for i in &seq {
            t.set(*i, *i as usize);
        }

        b.iter(|| {
            let mut _x = 0;
            for (_key, value) in t.iter() {
                _x ^= *value;
            }
        });
    }

    #[bench]
    fn bench_hashmap_iter_dense(b: &mut Bencher) {
        let seq = generate_pseudo_randoms(BENCH_LENGTH, DENSE_RANGE);
        let mut h: HashMap<usize, usize> = HashMap::new();

        for i in &seq {
            h.insert(*i, *i as usize);
        }

        b.iter(|| {
            let mut _x = 0;
            for (_key, value) in h.iter() {
                _x ^= *value;
            }
        });
    }

    #[bench]
    fn bench_trie_iter_seq(b: &mut Bencher) {
        let mut t: Trie<usize> = Trie::new();

        for i in 0..BENCH_LENGTH {
            t.set(i, i as usize);
        }

        b.iter(|| {
            let mut _x = 0;
            for (_key, value) in t.iter() {
                _x ^= *value;
            }
        });
    }

    #[bench]
    fn bench_hashmap_iter_seq(b: &mut Bencher) {
        let mut h: HashMap<usize, usize> = HashMap::new();

        for i in 0..BENCH_LENGTH {
            h.insert(i, i);
        }

        b.iter(|| {
            let mut _x = 0;
            for (_key, value) in h.iter() {
                _x ^= *value;
            }
        });
    }
}
