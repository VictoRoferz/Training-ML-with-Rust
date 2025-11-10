use numpy::ndarray::{ArrayView1, ArrayView2, Axis};
use pyo3::prelude::*;
use rayon::{join, prelude::*};
use std::cmp::Ordering;

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
        class: usize,
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

    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Vec<usize> {
        // processing predictions in batches to avoid excessive parallelization
        let chunk_size = 1024;
        let root_idx = self.root.as_ref().expect("Model not fitted yet.");
        let n_samples = x.nrows();

        (0..n_samples)
            .into_par_iter()
            .chunks(chunk_size)
            .flat_map_iter(|chunk| {
                chunk
                    .into_iter()
                    .map(|i| {
                        x.index_axis(Axis(0), i)
                            .as_slice()
                            .map(|row| self.predict_one(row, root_idx))
                            .expect("Row not contiguous")
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn predict_one<'a>(&self, row: &[f64], mut node: &'a TreeNode) -> usize {
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

        let stop_splitting = n_unique == 1
            || self.max_depth.map_or(false, |m| depth >= m)
            || n_samples < self.min_samples_split;

        if stop_splitting {
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

        if [left_idx.is_empty(), right_idx.is_empty()]
            .iter()
            .any(|&b| b)
        {
            return TreeNode::Leaf {
                class: self.majority_class(&counts),
            };
        }

        let left_x = x.select(Axis(0), &left_idx);
        let left_y = y.select(Axis(0), &left_idx);
        let right_x = x.select(Axis(0), &right_idx);
        let right_y = y.select(Axis(0), &right_idx);

        // Limiting parallelization of recursive calls, might get out of hand with larger trees
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

        let (f, t, g, found) = (0..n_features)
            .into_par_iter()
            .map(|f| {
                let mut sorted: Vec<(f64, usize)> =
                    (0..n_samples).map(|i| (x[[i, f]], i)).collect();

                sorted.sort_by(|a, b| match (a.0.is_nan(), b.0.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal),
                });

                let mut left_counts = vec![0; self.n_classes];
                let mut right_counts = counts.to_vec();
                let mut n_left = 0;
                let mut n_right = n_samples;

                (0..n_samples - 1)
                    .filter_map(|i| {
                        let (_, idx) = sorted[i];
                        let cls = y[idx];
                        n_left += 1;
                        n_right -= 1;
                        left_counts[cls] += 1;
                        right_counts[cls] -= 1;

                        if n_left < self.min_samples_leaf || n_right < self.min_samples_leaf {
                            return None;
                        }

                        let val = sorted[i].0;
                        let next = sorted[i + 1].0;
                        if (val - next).abs() <= f64::EPSILON {
                            return None;
                        }

                        let thresh = 0.5 * (val + next);
                        let g_left = self.gini(&left_counts);
                        let g_right = self.gini(&right_counts);
                        let weighted = (n_left as f64 / n_samples as f64) * g_left
                            + (n_right as f64 / n_samples as f64) * g_right;
                        let gain = parent_gini - weighted;

                        Some((thresh, gain))
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    .map_or((f, 0.0, 0.0, false), |(best_thresh, best_gain)| {
                        (f, best_thresh, best_gain, best_gain > 0.0)
                    })
            })
            .reduce(
                || (0, 0.0, 0.0, false),
                |a, b| if b.2 > a.2 { b } else { a },
            );

        (f, t, found && g > 0.0)
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
        (0..x.nrows()).fold((vec![], vec![]), |mut acc, i| {
            let v = x[[i, f]];
            if v.is_nan() || v > t {
                acc.1.push(i);
            } else {
                acc.0.push(i);
            }
            acc
        })
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
        let total = counts.iter().sum::<usize>() as f64;
        if total == 0.0 {
            return 0.0;
        }
        1.0 - counts
            .iter()
            .map(|&c| {
                let p = c as f64 / total;
                p * p
            })
            .sum::<f64>()
    }
}

#[pymodule]
mod rustdt {
    use super::*;
    use numpy::{PyReadonlyArray1, PyReadonlyArray2, ndarray::Array2};

    #[pyclass]
    pub struct RustDecisionTreeClassifier {
        inner: DecisionTreeClassifier,
    }

    #[pymethods]
    impl RustDecisionTreeClassifier {
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
            py: Python<'py>,
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

            py.detach(|| {
                self.inner.fit(x_view, y_arr);
            });

            Ok(())
        }

        pub fn predict<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray2<'py, f64>,
        ) -> PyResult<Vec<usize>> {
            let x_array: Array2<f64> = x.as_array().to_owned();
            let predictions = py.detach(|| self.inner.predict(x_array.view()));
            Ok(predictions)
        }
    }
}
