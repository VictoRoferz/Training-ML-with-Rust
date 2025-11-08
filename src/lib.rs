use itertools::Itertools;
use pyo3::prelude::*;
use rayon::prelude::*;

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

struct SplitResult {
    feature_index: usize,
    threshold: f64,
    impurity_decrease: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
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

    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<usize>) {
        assert!(!X.is_empty());
        let n_samples = X.len();
        let n_features = X[0].len();
        assert_eq!(y.len(), n_samples);

        self.n_classes = y.iter().max().unwrap() + 1;
        let indices: Vec<usize> = (0..n_samples).collect();
        let root = self.build_tree(&X, &y, indices, 0, n_features);
        self.root = Some(root);
    }

    pub fn predict(&self, X: &[Vec<f64>]) -> Vec<usize> {
        X.iter().map(|x| self.predict_one(x)).collect()
    }

    fn predict_one(&self, x: &[f64]) -> usize {
        let mut node = self.root.as_ref().expect("Tree not fitted yet");
        loop {
            match node {
                TreeNode::Leaf { class } => return *class,
                TreeNode::Internal {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    node = if x[*feature_index] <= *threshold {
                        left
                    } else {
                        right
                    };
                }
            }
        }
    }

    fn build_tree(
        &self,
        X: &[Vec<f64>],
        y: &[usize],
        indices: Vec<usize>,
        depth: usize,
        n_features: usize,
    ) -> TreeNode {
        let n_samples = indices.len();

        if indices.iter().map(|&i| y[i]).all_equal() {
            return TreeNode::Leaf {
                class: y[indices[0]],
            };
        }

        if self.max_depth.map_or(false, |max| depth >= max) || n_samples < self.min_samples_split {
            return TreeNode::Leaf {
                class: majority_class(y, &indices, self.n_classes),
            };
        }

        let best_split = (0..n_features)
            .into_iter()
            .filter_map(|j| {
                find_best_split_for_feature(
                    X,
                    y,
                    &indices,
                    j,
                    self.n_classes,
                    self.min_samples_leaf,
                )
            })
            .max_by(|a, b| {
                a.impurity_decrease
                    .partial_cmp(&b.impurity_decrease)
                    .unwrap()
            });

        match best_split {
            None => TreeNode::Leaf {
                class: majority_class(y, &indices, self.n_classes),
            },
            Some(split) => {
                let left = self.build_tree(X, y, split.left_indices, depth + 1, n_features);
                let right = self.build_tree(X, y, split.right_indices, depth + 1, n_features);
                TreeNode::Internal {
                    feature_index: split.feature_index,
                    threshold: split.threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
        }
    }
}

fn gini_from_labels(labels: &[usize], n_classes: usize) -> f64 {
    let n = labels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let counts = (0..n_classes)
        .map(|c| labels.iter().filter(|&&yi| yi == c).count() as f64)
        .collect_vec();
    1.0 - counts.iter().map(|&p| (p / n).powi(2)).sum::<f64>()
}

fn majority_class(y: &[usize], indices: &[usize], n_classes: usize) -> usize {
    let counts = (0..n_classes)
        .map(|c| indices.iter().filter(|&&i| y[i] == c).count())
        .collect_vec();
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| c)
        .map(|(cls, _)| cls)
        .unwrap()
}

fn find_best_split_for_feature(
    X: &[Vec<f64>],
    y: &[usize],
    indices: &[usize],
    feature_index: usize,
    n_classes: usize,
    min_samples_leaf: usize,
) -> Option<SplitResult> {
    let n_samples = indices.len();
    if n_samples <= 2 * min_samples_leaf {
        return None;
    }

    let sorted = indices
        .iter()
        .map(|&i| (X[i][feature_index], i))
        .sorted_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .collect_vec();

    let parent_impurity = gini_from_labels(&indices.iter().map(|&i| y[i]).collect_vec(), n_classes);

    let mut best: Option<SplitResult> = None;

    for i in 0..(n_samples - 1) {
        let (val, _) = sorted[i];
        let next_val = sorted[i + 1].0;

        if (val - next_val).abs() < f64::EPSILON {
            continue;
        }

        let left_indices = sorted[..=i].iter().map(|&(_, idx)| idx).collect_vec();
        let right_indices = sorted[i + 1..].iter().map(|&(_, idx)| idx).collect_vec();

        if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
            continue;
        }

        let g_left = gini_from_labels(&left_indices.iter().map(|&i| y[i]).collect_vec(), n_classes);
        let g_right = gini_from_labels(
            &right_indices.iter().map(|&i| y[i]).collect_vec(),
            n_classes,
        );

        let n_left = left_indices.len() as f64;
        let n_right = right_indices.len() as f64;
        let n_total = n_samples as f64;

        let weighted = (n_left / n_total) * g_left + (n_right / n_total) * g_right;
        let impurity_decrease = parent_impurity - weighted;

        if impurity_decrease <= 0.0 {
            continue;
        }

        if best
            .as_ref()
            .map_or(true, |b| impurity_decrease > b.impurity_decrease)
        {
            best = Some(SplitResult {
                feature_index,
                threshold: 0.5 * (val + next_val),
                impurity_decrease,
                left_indices,
                right_indices,
            });
        }
    }

    best
}

#[pymodule]
mod a2 {
    use super::*;

    #[pyclass]
    struct PyDecisionTreeClassifier {
        inner: DecisionTreeClassifier,
    }

    #[pymethods]
    impl PyDecisionTreeClassifier {
        #[new]
        #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1))]
        fn new(
            max_depth: Option<usize>,
            min_samples_split: usize,
            min_samples_leaf: usize,
        ) -> Self {
            let inner = DecisionTreeClassifier::new(max_depth, min_samples_split, min_samples_leaf);
            Self { inner }
        }

        fn fit(&mut self, X: Vec<Vec<f64>>, y: Vec<usize>) -> PyResult<()> {
            self.inner.fit(X, y);
            Ok(())
        }

        fn predict(&self, X: Vec<Vec<f64>>) -> PyResult<Vec<usize>> {
            Ok(self.inner.predict(&X))
        }
    }
}
