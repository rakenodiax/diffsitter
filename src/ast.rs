//! Utilities for processing the ASTs provided by `tree_sitter`

use crate::diff::Hunks;
use anyhow::Result;
use logging_timer::time;
use std::{cell::RefCell, collections::VecDeque, ops::Index, path::PathBuf};
use tree_sitter::Node as TSNode;
use tree_sitter::Tree as TSTree;

/// Get the minium of an arbitrary number of elements
macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $($z: expr),+) => (::std::cmp::min($x, min!($($z),*)));
}

/// The internal variant of an edit
///
/// This is the edit enum that's used for the minimum edit distance algorithm. It features a
/// variant, `Substitution`, that isn't exposed externally. When recreating the edit path,
/// [Substitution](Edit::Substitution) variant turns into an
/// [Addition](Edit::Addition) and [Deletion](Internal::Deletion).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Edit<'a> {
    /// A no-op
    ///
    /// There is no edit
    Noop,

    /// Some text was added
    ///
    /// An addition refers to the text from a node that was added from b
    Addition(Entry<'a>),

    /// Some text was deleted
    ///
    /// An addition refers to text from a node that was deleted from source a
    Deletion(Entry<'a>),

    /// Some text was replaced
    ///
    /// A substitution refers to text from a node that was replaced, holding a reference to the old
    /// AST node and the AST node that replaced it
    Substitution {
        /// The old text
        old: Entry<'a>,

        /// The new text that took its palce
        new: Entry<'a>,
    },
}

/// A mapping between a tree-sitter node and the text it corresponds to
#[derive(Debug, Clone, Copy)]
pub struct Entry<'a> {
    /// The node an entry in the diff vector refers to
    ///
    /// We keep a reference to the leaf node so that we can easily grab the text and other metadata
    /// surrounding the syntax
    pub reference: TSNode<'a>,

    /// A reference to the text the node refers to
    ///
    /// This is different from the `source_text` that the [AstVector](AstVector) refers to, as the
    /// entry only holds a reference to the specific range of text that the node covers.
    pub text: &'a str,
}

/// A vector that allows for linear traversal through the leafs of an AST.
///
/// This representation of the tree leaves is much more convenient for things like dynamic
/// programming, and provides useful for formatting.
#[derive(Debug)]
pub struct AstVector<'a> {
    /// The leaves of the AST, build with an in-order traversal
    pub leaves: Vec<Entry<'a>>,

    /// The full source text that the AST refers to
    pub source_text: &'a str,
}

impl<'a> Eq for Entry<'a> {}

/// A wrapper struct for AST vector data that owns the data that the AST vector references
///
/// Ideally we would just have the AST vector own the actual string and tree, but it makes things
/// extremely messy with the borrow checker, so we have this wrapper struct that holds the owned
/// data that the vector references. This gets tricky because the tree sitter library uses FFI so
/// the lifetime references get even more mangled.
#[derive(Debug)]
pub struct AstVectorData {
    /// The text in the file
    pub text: String,

    /// The tree that was parsed using the text
    pub tree: TSTree,

    /// The file path that the text corresponds to
    pub path: PathBuf,
}

impl<'a> AstVector<'a> {
    /// Create a `DiffVector` from a `tree_sitter` tree
    ///
    /// This method calls a helper function that does an in-order traversal of the tree and adds
    /// leaf nodes to a vector
    #[time("info", "ast::{}")]
    pub fn from_ts_tree(tree: &'a TSTree, text: &'a str) -> Self {
        let leaves = RefCell::new(Vec::new());
        build(&leaves, tree.root_node(), text);
        AstVector {
            leaves: leaves.into_inner(),
            source_text: text,
        }
    }

    /// Return the number of nodes in the diff vector
    pub fn len(&self) -> usize {
        self.leaves.len()
    }
}

impl<'a> Index<usize> for AstVector<'a> {
    type Output = Entry<'a>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.leaves[index]
    }
}

impl<'a> PartialEq for Entry<'a> {
    fn eq(&self, other: &Entry) -> bool {
        self.text == other.text
    }
}

impl<'a> PartialEq for AstVector<'a> {
    fn eq(&self, other: &AstVector) -> bool {
        if self.leaves.len() != other.leaves.len() {
            return false;
        }

        // Zip through each entry to determine whether the elements are equal. We start with a
        // `false` value for not equal and accumulate any inequalities along the way.
        let not_equal = self
            .leaves
            .iter()
            .zip(other.leaves.iter())
            .fold(false, |not_equal, (entry_a, entry_b)| {
                not_equal | (entry_a != entry_b)
            });
        !not_equal
    }
}

/// Recursively build a vector from a given node
///
/// This is a helper function that simply walks the tree and collects leaves in an in-order manner.
/// Every time it encounters a leaf node, it stores the metadata and reference to the node in an
/// `Entry` struct.
fn build<'a>(vector: &RefCell<Vec<Entry<'a>>>, node: tree_sitter::Node<'a>, text: &'a str) {
    // If the node is a leaf, we can stop traversing
    if node.child_count() == 0 {
        // We only push an entry if the referenced text range isn't empty, since there's no point
        // in having an empty text range. This also fixes a bug where the program would panic
        // because it would attempt to access the 0th index in an empty text range.
        if !node.byte_range().is_empty() {
            let node_text: &'a str = &text[node.byte_range()];
            vector.borrow_mut().push(Entry {
                reference: node,
                text: node_text,
            });
        }
        return;
    }

    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        build(vector, child, text);
    }
}

/// Recreate the paths for additions and deletions given a [PredecessorMap]
///
/// This will generate the hunks for both documents in one shot as we reconstruct the path.
#[time("info", "ast::{}")]
fn recreate_path(last_idx: (usize, usize), preds: PredecessorVec) -> Result<(Hunks, Hunks)> {
    // The hunks for the old document. Deletions correspond to this.
    let mut hunks_old = Hunks::new();
    // The hunks for the new document. Additions correspond to this.
    let mut hunks_new = Hunks::new();
    let mut curr_idx = last_idx;

    while let Some(entry) = preds[curr_idx.0][curr_idx.1] {
        match entry.edit {
            Edit::Noop => (),
            Edit::Addition(x) => hunks_new.push_front(x)?,
            Edit::Deletion(x) => hunks_old.push_front(x)?,
            Edit::Substitution { old, new } => {
                hunks_new.push_front(new)?;
                hunks_old.push_front(old)?;
            }
        }
        curr_idx = entry.previous_idx;
    }
    Ok((hunks_old, hunks_new))
}

/// An entry in the precedecessor table
///
/// This entry contains information about the type of edit, and which index to backtrack to
#[derive(Debug, Clone, Copy)]
struct PredEntry<'a> {
    /// The edit in question
    pub edit: Edit<'a>,

    /// The index the edit came from
    pub previous_idx: (usize, usize),
}

type PredecessorVec<'a> = Vec<Vec<Option<PredEntry<'a>>>>;

/// Helper function to use the minimum edit distance algorithm on two [AstVectors](AstVector)
#[time("info", "ast::{}")]
fn min_edit<'a>(a: &'a AstVector, b: &'a AstVector) -> PredecessorVec<'a> {
    // The optimal move that led to the edit distance at an index. We use this 2D vector to
    // backtrack and build the edit path once we find the optimal edit distance
    let mut predecessors: PredecessorVec<'a> = vec![vec![None; b.len() + 1]; a.len() + 1];

    // Initialize the dynamic programming array
    // dp[i][j] is the edit distance between a[:i] and b[:j]
    let mut dp: Vec<Vec<u32>> = vec![vec![0; b.len() + 1]; a.len() + 1];

    // Sanity check that the dimensions of the DP table are correct
    debug_assert!(dp.len() == a.len() + 1);
    debug_assert!(dp[0].len() == b.len() + 1);

    for i in 0..=a.len() {
        for j in 0..=b.len() {
            // If either string is empty, the minimum edit is just to add strings
            if i == 0 {
                dp[i][j] = j as u32;

                if j > 0 {
                    let pred_entry = PredEntry {
                        edit: Edit::Addition(b[j - 1]),
                        previous_idx: (i, j - 1),
                    };
                    predecessors[i][j] = Some(pred_entry);
                }
            } else if j == 0 {
                dp[i][j] = i as u32;

                if i > 0 {
                    let pred_entry = PredEntry {
                        edit: Edit::Deletion(a[i - 1]),
                        previous_idx: (i - 1, j),
                    };
                    predecessors[i][j] = Some(pred_entry);
                }
            }
            // If the current letter for each string matches, there is no change
            else if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
                let pred_entry = PredEntry {
                    edit: Edit::Noop,
                    previous_idx: (i - 1, j - 1),
                };
                predecessors[i][j] = Some(pred_entry);
            }
            // Otherwise, there is either a substitution, a deletion, or an addition
            else {
                let min = min!(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);

                // Store the current minimum edit in the precedecessor map based on which path has
                // the lowest edit distance
                let pred_entry = if min == dp[i - 1][j] {
                    PredEntry {
                        edit: Edit::Deletion(a[i - 1]),
                        previous_idx: (i - 1, j),
                    }
                } else if min == dp[i][j - 1] {
                    PredEntry {
                        edit: Edit::Addition(b[j - 1]),
                        previous_idx: (i, j - 1),
                    }
                } else {
                    PredEntry {
                        edit: Edit::Substitution {
                            old: a[i - 1],
                            new: b[j - 1],
                        },
                        previous_idx: (i - 1, j - 1),
                    }
                };
                // Store the precedecessor so we can backtrack and recreate the path that led to
                // the minimum edit path
                predecessors[i][j] = Some(pred_entry);

                // Store the current minimum edit distance for a[:i] <-> b[:j]. An addition,
                // deletion, and substitution all have an edit cost of 1, which is why we're adding
                // one to the minimum.
                dp[i][j] = 1 + min;
            }
        }
    }
    predecessors
}

/// Input parameters for a diff algorithm
struct DiffInputs<'a, T: Eq> {
    /// A slice of comparable objects.
    ///
    /// This is considered the "first" input, and an element that's present in `a` but not present
    /// in `b` is considered a [deletion](EditType::Deletion).
    pub a: &'a [T],

    /// A slice of comparable objects
    pub b: &'a [T],
}

type Diff<T> = VecDeque<EditType<T>>;

/// The different types of elements that can be in an edit script
#[derive(Debug, Eq, PartialEq)]
pub enum EditType<T> {
    /// An element that was added in the edit script
    Addition(T),

    /// An element that was deleted in the edit script
    Deletion(T),
}

/// Implementation of the Myers diff algorithm
pub mod myers {
    use std::fmt::Debug;

    use super::*;

    /// Bisect a diff AST to find the "middle snake"
    ///
    /// Split a diff into two subproblems along the middle, and recursively construct a diff for
    /// each half.
    fn bisect<'a, T>() {}

    /// Compute an edit script from Myers diff states
    ///
    /// * `states`: The states from the Myers diff algorithm for each `d`-length path
    /// * `last_k`: The `k` corresponding to the last found `d`-path
    fn backtrack<'a, T: PartialEq + Debug>(
        states: &[Vec<i32>],
        a: &'a [T],
        b: &'a [T],
    ) -> VecDeque<EditType<&'a T>> {
        println!("a length: {}", a.len());
        println!("b length: {}", b.len());

        // We know that there can only be `d` states, so we implicitly store the length of the
        // D-path for the edit graph in the length of the state vector.
        let max_d = (a.len() + b.len()) as i32;
        let mut diff = VecDeque::new();

        // We know that the edit graph must end on (m, n), which corresponds to the endpoints of a
        // and b
        let mut idx_a = (a.len() - 1) as i32;
        let mut idx_b = (b.len() - 1) as i32;

        // Iterate backwards through the d-paths to reconstruct the minimal edit path
        for d in (0..states.len()).rev() {
            let k = idx_a - idx_b;
            let k_idx = (k + max_d) as usize;
            let state = &states[d];

            println!("[1] idx_a: {}, idx_b: {}, k: {}, d: {}", idx_a, idx_b, k, d);

            // Find the previous k value by determining whether the horizontally or vertically
            // connected d - 1 path is maximal
            let prev_k_idx = if k == -(d as i32) || state[k_idx - 1] < state[k_idx + 1] {
                k_idx + 1
            } else {
                k_idx - 1
            };
            let prev_k = (prev_k_idx as i32) - max_d;
            let prev_a_idx = state[prev_k_idx];
            let prev_b_idx = prev_a_idx - (prev_k as i32);
            println!("prev a: {}, prev b: {}", prev_a_idx, prev_b_idx);

            while idx_a > prev_a_idx && idx_b > prev_b_idx {
                idx_a -= 1;
                idx_b -= 1;
            }

            debug_assert!(idx_b < b.len() as i32);
            debug_assert!(idx_a < a.len() as i32);

            if (idx_a as i32) == prev_a_idx {
                diff.push_front(EditType::Addition(&b[idx_b as usize]));
            } else if (idx_b as i32) == prev_b_idx {
                diff.push_front(EditType::Deletion(&a[idx_a as usize]));
            }
            println!("edit: {:#?}", diff.back());

            idx_a = prev_a_idx;
            idx_b = prev_b_idx;
        }
        diff
    }

    /// An implementation of the Myers diff algorithm
    ///
    /// Generate the minimum edit script between two data slices using an implementation of the Myers
    /// diff algorithm.
    // TODO(afnan) switch this back to returning the edit script
    pub fn diff<'a, T: Eq + Debug>(a: &'a [T], b: &'a [T]) -> Diff<&'a T> {
        println!("a: {:#?}", a);
        println!("b: {:#?}", b);
        // The maximum possible length of an edit script between a and b, if they have no elements in common
        let max = a.len() + b.len();

        // The states of v at each d
        let mut states = Vec::new();
        states.reserve(max + 1);

        // v[k] returns the x coordinate for the largest D path seen on diagonal k + max. We can't have negative indices
        // so there is an offset of -max applied to the index.
        // The diagonals range from [-d, d], which is [-max, max]. We offset the k with max when indexing, so we need all
        // integers in [0, 2 * max] to be indexable in the vector.
        let mut v: Vec<i32> = vec![0; (2 * max) + 1];
        let a_len = a.len() as i32;
        let b_len = b.len() as i32;

        for d in 0..=(max as i32) {
            for k in (-d..=d).step_by(2) {
                // We need to convert the unsigned integer to signed before adding to a negative number. We guarantee that
                // offsetting by max will shift the k index back to the range of [0, max] so we don't have to worry about
                // an overflow on the outer conversion.
                let k_idx = (k + (max as i32)) as usize;
                let go_down = k == -d || v[k_idx - 1] < v[k_idx + 1];
                let mut x = if go_down {
                    v[k_idx + 1]
                } else {
                    v[k_idx - 1] + 1
                };
                let mut y = x - (k as i32);

                // Extend the diagonal snake as far as possible
                // This isn't a mistake, though the lint seems to find this particular line very suspicious
                #[allow(clippy::suspicious_operation_groupings)]
                while x < a_len && y < b_len && a[x as usize] == b[y as usize] {
                    x += 1;
                    y += 1;
                }

                // Once the longest path coordinates have reached (a.len(), b.len()), we know we've found the shortest edit
                // script that covers the edits between the entirety of both inputs
                if x >= a_len && y >= b_len {
                    // backtrack
                    return backtrack(&states, a, b);
                }

                println!(
                    "d: {}, x: {}, y: {}, x len: {}, y len: {}, k: {}",
                    d, x, y, a_len, b_len, k
                );
                //debug_assert!(x <= a_len);
                //debug_assert!(y <= b_len);
                v[k_idx as usize] = x;
            }
            // Record the state at each `d` so we can backtrack later once we reach the end of the path
            states.push(v.clone());
        }
        backtrack(&states, a, b)
    }
}

/// Compute the hunks corresponding to the minimum edit path between two documents
///
/// This method computes the minimum edit distance between two `DiffVector`s, which are the leaf
/// nodes of an AST, using the standard DP approach to the longest common subsequence problem, the
/// only twist is that here, instead of operating on raw text, we're operating on the leaves of an
/// AST.
///
/// This has O(mn) space complexity and uses O(mn) space to compute the minimum edit path, and then
/// has O(mn) space complexity and uses O(mn) space to backtrack and recreate the path.
///
/// This will return two groups of [hunks](diff::Hunks) in a tuple of the form
/// `(old_hunks, new_hunks)`.
pub fn edit_hunks<'a>(a: &'a AstVector, b: &'a AstVector) -> Result<(Hunks<'a>, Hunks<'a>)> {
    let predecessors = min_edit(a, b);
    recreate_path((a.len(), b.len()), predecessors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    /*
    #[test_case(vec![0], vec![0] => 0 ; "When both strings are the same")]
    #[test_case(vec![0, 0, 0, 0], vec![0, 0, 0, 0] => 0 ; "When both strings are the same (longer)")]
    #[test_case(vec![0], vec![1] => 2 ; "test substitution")]
    #[test_case(vec![0], vec![0, 0] => 1 ; "test addition")]
    #[test_case(vec![3, 0], vec![0, 2] => 2 ; "One edit in each")]
    fn test_myers(a: Vec<i32>, b: Vec<i32>) -> u32 {
        let edit_script = myers::diff(&a, &b);
        println!("{:#?}", edit_script);
        edit_script.len() as u32
    }

    #[test]
    fn test_myers_addition() {
        let a = vec![0];
        let b = vec![0, 0];
        let edit_script: VecDeque<EditType<&i32>> = myers::diff(&a, &b);
        let expected: VecDeque<EditType<&i32>> = VecDeque::new();
        assert!(edit_script == expected);
    }
    */
}
