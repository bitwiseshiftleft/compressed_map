use criterion::{criterion_group, criterion_main, Criterion};
use compressed_map::{CompressedRandomMap,BuildOptions};
use rand::{Rng,SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;

criterion_group!{
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark
}
fn criterion_benchmark(crit: &mut Criterion) {
    let sizes = vec![1000usize,10000,100000,1000000];
    let nbits = 8;
    let mask = if nbits == 64 { !0 } else { (1u64 << nbits) - 1 };
    for size in sizes {
        /* It's fine to use the same keys every time, due to the random hash key */
        let mut seed = [0u8;32];
        seed[0..4].copy_from_slice(&(size as u32).to_le_bytes());
        let mut rng : StdRng = SeedableRng::from_seed(seed);
        let keys   : Vec<u64> = (0..size).map(|_| rng.gen::<u64>()).collect();
        let values : Vec<u64> = (0..size).map(|_| rng.gen::<u64>() & mask).collect();

        // /* Bench hashmap insert because the C includes it in build */
        let mut map = HashMap::new();
        crit.bench_function(&format!("uniform hashmap {}",size),
            |crit| crit.iter(|| {
            map = HashMap::new();
            for i in 0..size {
                map.insert(keys[i],values[i]);
            }
        }));

        /* Bench building */
        let mut umap = None;
        let mut total_tries = 0;
        let mut total_builds = 0;
        let mut options = BuildOptions::default();
        options.key_gen = Some(seed[..16].try_into().unwrap());
        crit.bench_function(&format!("uniform build {}",size),
            |crit| crit.iter(|| {
                options.try_num = 0;
                umap = CompressedRandomMap::<_,_>::build(&map, &mut options);
                assert!(umap.is_some()); /* Assert success */
                total_tries += options.try_num + 1;
                total_builds += 1;
            }));

        if let Some(umap_some) = umap {
            println!("Building for size {} success rate {}%", size,
                total_builds as f64 * 100. / total_tries as f64);
            /* Bench querying */
            let mut qi = 0;
            crit.bench_function(&format!("uniform query {}",size),
                |crit| crit.iter(|| {
                    assert_eq!(umap_some.query(&keys[qi]), values[qi]);
                    qi = (qi+1) % keys.len();
                }));
        }
    }
}

criterion_group!{
    name = benches_slow;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark_slow
}
fn criterion_benchmark_slow(crit: &mut Criterion) {
    let sizes = vec![10000000usize,100000000];
    let nbits = 8;
    let mask = if nbits == 64 { !0 } else { (1u64 << nbits) - 1 };
    for size in sizes {
        let mut seed = [0u8;32];
        seed[0..4].copy_from_slice(&(size as u32).to_le_bytes());
        let mut rng : StdRng = SeedableRng::from_seed(seed);

        /* It's fine to use the same keys every time, due to the random hash key */
        let keys   : Vec<u64> = (0..size).map(|_| rng.gen::<u64>()).collect();
        let values : Vec<u64> = (0..size).map(|_| rng.gen::<u64>() & mask).collect();

        // /* Bench hashmap insert because the C includes it in build */
        let mut map = HashMap::new();
        for i in 0..size {
            map.insert(keys[i],values[i]);
        }

        /* Bench building */
        let mut umap = None;
        let mut total_tries = 0;
        let mut total_builds = 0;
        let mut options = BuildOptions::default();
        options.key_gen = Some(seed[..16].try_into().unwrap());
        crit.bench_function(&format!("uniform build {}",size),
            |crit| crit.iter(|| {
                options.try_num = 0;
                umap = CompressedRandomMap::<_,_>::build(&map, &mut options);
                assert!(umap.is_some()); /* Assert success */
                total_tries += options.try_num + 1;
                total_builds += 1;
            }));

        if let Some(umap_some) = umap {
            println!("Building for size {} success rate {}%", size,
                total_builds as f64 * 100. / total_tries as f64);
            /* Bench querying */
            let mut qi = 0;
            crit.bench_function(&format!("uniform query {}",size),
                |crit| crit.iter(|| {
                    assert_eq!(umap_some.query(&keys[qi]), values[qi]);
                    qi = (qi+1) % keys.len();
                }));
        }
    }
}

criterion_main!(benches, benches_slow);
