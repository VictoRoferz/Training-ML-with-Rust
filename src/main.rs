use ndarray::{Array1, Array2};
use numpy::ndarray;
use rand::Rng;
use std::time::Instant;

use rustdt::DecisionTreeClassifier;

fn main() {
    // Generate a synthetic dataset
    let n_samples = 20000;
    let n_features = 10;
    let n_classes = 3;

    let mut rng = rand::rng();
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let class = rng.random_range(0..n_classes);
        y_data.push(class);
        for j in 0..n_features {
            // Simple linear separability
            let val = class as f64 * 2.0 + rng.random::<f64>();
            x_data.push(val + j as f64 * 0.1);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from(y_data);

    let mut clf = DecisionTreeClassifier::new(Some(10), 2, 1);
    println!("Fitting tree...");
    clf.fit(x.view(), y.view());

    // Warm up prediction (optional)
    let preds = clf.predict(x.view());
    println!("Warmup done. Example prediction: {:?}", &preds[..5]);

    // Profile just the prediction part
    println!("Profiling predict()...");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = clf.predict(x.view());
    }
    let elapsed = start.elapsed();
    println!("Prediction took {:.2?}", elapsed);
}
