
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(bench)]
fn criterion_benchmark(crit: &mut Criterion) {
    /* Real matrix benchmark */
    use std::cmp::max;
    use compressed_map::tilematrix::matrix::Matrix;
    let sizes = vec![100usize,1000,10000];
    for size in sizes {
        let mut a = Matrix::new(size,size,0);
        let mut b = Matrix::new(size,size,0);
        a.randomize();
        b.randomize();
        let mut c = Matrix::new(a.rows,b.cols_main,max(a.cols_aug,b.cols_aug));
        
        crit.bench_function(&format!("matrix mulacc {}",size), |crit| crit.iter(|| c.accum_mul(&a,&b)));

        crit.bench_function(&format!("matrix randomize {}",size), |crit| crit.iter(|| {
            a.randomize();
        }));

        crit.bench_function(&format!("matrix randomize+rref {}",size), |crit| crit.iter(|| {
            a.randomize();
            a.rref();
        }));
    }
}

#[cfg(not(bench))]
fn criterion_benchmark(_crit: &mut Criterion) {
    /* Empty matrix benchmark, because matrix code is private */
    println!("Matrix benches are disabled unless you pass --cfg bench");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
