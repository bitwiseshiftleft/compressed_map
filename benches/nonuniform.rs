use criterion::{criterion_group, criterion_main, Criterion};
use compressed_map::{CompressedMap,BuildOptions,STD_BINCODE_CONFIG};
use bincode::{encode_to_vec};
use rand::{Rng,SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}

fn criterion_benchmark(crit: &mut Criterion) {
    let no = 100000000;
    let yes  = 700000;

    let seed = [0u8;32];
    let mut rng : StdRng = SeedableRng::from_seed(seed);

    let mut map = HashMap::new();
    for _ in 0..yes { map.insert(rng.gen::<[u8;32]>(), true); }
    for _ in 0..no { map.insert(rng.gen::<[u8;32]>(), false); }
    let mut query = Vec::with_capacity(yes+no);
    for (k,v) in &map {
        query.push((*k,*v));
    }

    let mut options = BuildOptions::default();
    options.key_gen = Some(seed[..16].try_into().unwrap());
    let mut umap = None;
    crit.bench_function(&format!("nonu build {}+{}",yes,no),
    |crit| crit.iter(|| {
        umap = CompressedMap::<_,_>::build(&map, &mut options);
        assert!(umap.is_some()); /* Assert success */
    }));

    if let Some(umap_some) = umap {
        let p = yes as f64 / (yes + no) as f64;
        let shannon = -(p * p.log2() + (1.-p)*(1.-p).log2());
        let shannon = (shannon * (yes + no) as f64 / 8.) as usize;
        let size = encode_to_vec(&umap_some, STD_BINCODE_CONFIG).unwrap().len();
        println!("Shannon = {}, size = {}, ratio = {}",
            shannon,
            size,
            size as f64 / shannon as f64
        );

        let mut qi = 0;
        crit.bench_function(&format!("nonu query {}+{}",yes,no),
            |crit| crit.iter(|| {
                let (k,v) = query[qi];
                assert_eq!(*umap_some.query(&k), v);
                qi = (qi+1) % query.len();
            }));
    }
}

criterion_main!(benches);
