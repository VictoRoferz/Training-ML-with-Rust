use numpy::ndarray::{ArrayView1, ArrayView2, Axis};
use pyo3::prelude::*;
use rayon::{join, prelude::*};
use std::cmp::Ordering;

type Class = usize;

/// Decision Tree Classifier (CART-style)
///
/// Massimo’s version: clear, honest, and efficient — without over-optimization.
pub struct DecisionTreeClassifier {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    root: Option<TreeNode>,
    n_classes: usize,
}

#[derive(Debug)]
enum TreeNode {
    Leaf {
        class: Class,
    },
    Internal {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl DecisionTreeClassifier {
    pub fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        Self {
            max_depth,
            min_samples_split,
            min_samples_leaf,
            root: None,
            n_classes: 0,
        }
    }

    pub fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, usize>) {
        assert_eq!(
            x.nrows(),
            y.len(),
            "x and y must have the same number of samples"
        );

        self.n_classes = y.iter().max().copied().unwrap_or(0) + 1;
        self.root = Some(self.build_tree(x, y, 0));
    }

    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Vec<Class> {
        let root = self.root.as_ref().expect("Model not fitted yet.");
        let mut preds = Vec::with_capacity(x.nrows());
        for row in x.outer_iter() {
            let row_vec: Vec<f64> = row.to_vec();
            preds.push(self.predict_one(&row_vec, root));
        }
        preds
    }

    fn predict_one<'a>(&self, row: &[f64], mut node: &'a TreeNode) -> Class {
        loop {
            match node {
                TreeNode::Leaf { class } => return *class,
                TreeNode::Internal {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    let v = row[*feature_index];
                    node = if v.is_nan() || v > *threshold {
                        right
                    } else {
                        left
                    };
                }
            }
        }
    }

    fn build_tree(
        &self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        depth: usize,
    ) -> TreeNode {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return TreeNode::Leaf { class: 0 };
        }

        let counts = self.class_counts(y);
        let n_unique = counts.iter().filter(|&&c| c > 0).count();

        if n_unique == 1
            || self.max_depth.map_or(false, |m| depth >= m)
            || n_samples < self.min_samples_split
        {
            return TreeNode::Leaf {
                class: self.majority_class(&counts),
            };
        }

        let parent_gini = self.gini(&counts);
        let (feature, threshold, gain_found) = self.find_best_split(x, y, &counts, parent_gini);

        if !gain_found {
            return TreeNode::Leaf {
                class: self.majority_class(&counts),
            };
        }

        let (left_idx, right_idx) = self.partition(x, feature, threshold);
        if left_idx.is_empty() || right_idx.is_empty() {
            return TreeNode::Leaf {
                class: self.majority_class(&counts),
            };
        }

        let left_x = x.select(Axis(0), &left_idx);
        let left_y = y.select(Axis(0), &left_idx);
        let right_x = x.select(Axis(0), &right_idx);
        let right_y = y.select(Axis(0), &right_idx);

        // Limiting recursion parallelism, might get out of hand with larger trees
        if depth < 4 {
            let (left, right) = join(
                || self.build_tree(left_x.view(), left_y.view(), depth + 1),
                || self.build_tree(right_x.view(), right_y.view(), depth + 1),
            );
            TreeNode::Internal {
                feature_index: feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        } else {
            TreeNode::Internal {
                feature_index: feature,
                threshold,
                left: Box::new(self.build_tree(left_x.view(), left_y.view(), depth + 1)),
                right: Box::new(self.build_tree(right_x.view(), right_y.view(), depth + 1)),
            }
        }
    }

    fn find_best_split(
        &self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, usize>,
        counts: &[usize],
        parent_gini: f64,
    ) -> (usize, f64, bool) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Each feature is independent — parallelize across features
        let best = (0..n_features)
            .into_par_iter()
            .map(|f| {
                // Pair (feature value, index)
                let mut sorted: Vec<(f64, usize)> =
                    (0..n_samples).map(|i| (x[[i, f]], i)).collect();

                sorted.sort_unstable_by(|a, b| match (a.0.is_nan(), b.0.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal),
                });

                let mut left_counts = vec![0; self.n_classes];
                let mut right_counts = counts.to_vec();
                let mut n_left = 0;
                let mut n_right = n_samples;

                let mut best_gain = 0.0;
                let mut best_thresh = 0.0;
                let mut found = false;

                // Try each possible split point
                for i in 0..(n_samples - 1) {
                    let (_, idx) = sorted[i];
                    let cls = y[idx];
                    n_left += 1;
                    n_right -= 1;
                    left_counts[cls] += 1;
                    right_counts[cls] -= 1;

                    if n_left < self.min_samples_leaf || n_right < self.min_samples_leaf {
                        continue;
                    }

                    let val = sorted[i].0;
                    let next = sorted[i + 1].0;
                    if (val - next).abs() <= f64::EPSILON {
                        continue; // skip identical values
                    }

                    let thresh = 0.5 * (val + next);
                    let g_left = self.gini(&left_counts);
                    let g_right = self.gini(&right_counts);
                    let weighted = (n_left as f64 / n_samples as f64) * g_left
                        + (n_right as f64 / n_samples as f64) * g_right;
                    let gain = parent_gini - weighted;

                    if gain > best_gain {
                        best_gain = gain;
                        best_thresh = thresh;
                        found = true;
                    }
                }

                (f, best_thresh, best_gain, found)
            })
            .reduce(
                || (0, 0.0, 0.0, false),
                |a, b| if b.2 > a.2 { b } else { a },
            );

        let (feature, thresh, gain, found) = best;
        (feature, thresh, found && gain > 0.0)
    }

    fn class_counts(&self, y: ArrayView1<'_, usize>) -> Vec<usize> {
        y.iter().filter(|&&cls| cls < self.n_classes).fold(
            vec![0; self.n_classes],
            |mut counts, &cls| {
                counts[cls] += 1;
                counts
            },
        )
    }

    fn partition(&self, x: ArrayView2<'_, f64>, f: usize, t: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for i in 0..x.nrows() {
            let v = x[[i, f]];
            if v.is_nan() || v > t {
                right.push(i);
            } else {
                left.push(i);
            }
        }
        (left, right)
    }

    fn majority_class(&self, counts: &[usize]) -> usize {
        counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn gini(&self, counts: &[usize]) -> f64 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let n = total as f64;
        1.0 - counts
            .iter()
            .map(|&c| {
                let p = c as f64 / n;
                p * p
            })
            .sum::<f64>()
    }
}

#[pymodule]
mod a2 {
    use super::*;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2};

    #[pyclass]
    pub struct PyDecisionTreeClassifier {
        inner: DecisionTreeClassifier,
    }

    #[pymethods]
    impl PyDecisionTreeClassifier {
        #[new]
        #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1))]
        pub fn new(
            max_depth: Option<usize>,
            min_samples_split: usize,
            min_samples_leaf: usize,
        ) -> Self {
            Self {
                inner: DecisionTreeClassifier::new(max_depth, min_samples_split, min_samples_leaf),
            }
        }

        pub fn fit<'py>(
            &mut self,
            _py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
            y: PyReadonlyArray1<'py, i64>,
        ) -> PyResult<()> {
            let x_view = x.as_array();
            let y_vec: Vec<usize> = y
                .as_array()
                .iter()
                .map(|&v| Ok(v as usize))
                .collect::<PyResult<_>>()?;

            let y_arr = ArrayView1::from(&y_vec);
            self.inner.fit(x_view, y_arr);
            Ok(())
        }

        pub fn predict<'py>(
            &self,
            _py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
        ) -> PyResult<Vec<usize>> {
            Ok(self.inner.predict(x.as_array()))
        }
    }
}
