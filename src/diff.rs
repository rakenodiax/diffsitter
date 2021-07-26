//! Structs and other convenience methods for handling logical concepts pertaining to diffs, such
//! as hunks.

use crate::ast::{EditType, Entry};
use crate::neg_idx_vec::NegIdxVec;
use anyhow::Result;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;
use thiserror::Error;

/// Find the length of the common prefix between the ranges specified for `a` and `b`.
fn common_prefix_len<T: PartialEq>(
    a: &[T],
    a_range: Range<usize>,
    b: &[T],
    b_range: Range<usize>,
) -> usize {
    let mut l = 0;

    while a_range.start + l < a_range.end
        && b_range.start + l < b_range.end
        && a[a_range.start + l] == b[b_range.start + l]
    {
        l += 1;
    }
    l
}

/// Find the length of the common suffix between the ranges specified for `a` and `b`.
/// The ranges are assumed to be [inclusive, exclusive).
fn common_suffix_len<T: PartialEq>(
    a: &[T],
    a_range: Range<usize>,
    b: &[T],
    b_range: Range<usize>,
) -> usize {
    let mut l = 1;

    while (a_range.end as isize) - (l as isize) >= a_range.start as isize
        && (b_range.end as isize) - (l as isize) >= b_range.start as isize
        && a[a_range.end - l] == b[b_range.end - l]
    {
        l += 1;
    }
    l - 1
}

/// The edit information representing a line
#[derive(Debug, Clone, PartialEq)]
pub struct Line<'a> {
    /// The index of the line in the original document
    pub line_index: usize,
    /// The entries corresponding to the line
    pub entries: VecDeque<Entry<'a>>,
}

impl<'a> Line<'a> {
    pub fn new(line_index: usize) -> Self {
        Line {
            line_index,
            entries: VecDeque::new(),
        }
    }
}

/// A grouping of consecutive edit lines for a document
///
/// Every line in a hunk must be consecutive and in ascending order.
#[derive(Debug, Clone, PartialEq)]
pub struct Hunk<'a>(pub VecDeque<Line<'a>>);

/// Types of errors that come up when inserting an entry to a hunk
#[derive(Debug, Error)]
pub enum HunkInsertionError {
    #[error(
        "Non-adjacent entry (line {incoming_line:?}) added to hunk (last line: {last_line:?})"
    )]
    NonAdjacentHunk {
        incoming_line: usize,
        last_line: usize,
    },
    #[error("Attempted to prepend an entry with a line index ({incoming_line:?}) greater than the first line's index ({first_line:?})")]
    LaterLine {
        incoming_line: usize,
        first_line: usize,
    },
    #[error("Attempted to prepend an entry with a column ({incoming_col:?}) greater than the first entry's column ({first_col:?})")]
    LaterColumn {
        incoming_col: usize,
        first_col: usize,
    },
}

impl<'a> Hunk<'a> {
    /// Create a new, empty hunk
    pub fn new() -> Self {
        Hunk(VecDeque::new())
    }

    /// Returns the first line number of the hunk
    ///
    /// This will return [None] if the internal vector is empty
    pub fn first_line(&self) -> Option<usize> {
        self.0.front().map(|x| x.line_index)
    }

    /// Returns the last line number of the hunk
    ///
    /// This will return [None] if the internal vector is empty
    pub fn last_line(&self) -> Option<usize> {
        self.0.back().map(|x| x.line_index)
    }

    /// Prepend an [entry](Entry) to a hunk
    ///
    /// Entries can only be prepended in descending order (from last to first)
    pub fn push_front(&mut self, entry: Entry<'a>) -> Result<(), HunkInsertionError> {
        let incoming_line_idx = entry.reference.start_position().row;

        // Add a new line vector if the entry has a greater line index, or if the vector is empty.
        // We ensure that the last line has the same line index as the incoming entry.
        if let Some(first_line) = self.0.front() {
            let first_line_idx = first_line.line_index;

            if incoming_line_idx > first_line_idx {
                return Err(HunkInsertionError::LaterLine {
                    incoming_line: incoming_line_idx,
                    first_line: first_line_idx,
                });
            }

            if first_line_idx - incoming_line_idx > 1 {
                return Err(HunkInsertionError::NonAdjacentHunk {
                    incoming_line: incoming_line_idx,
                    last_line: first_line_idx,
                });
            }

            // Only add a new line here if the the incoming line index is one after the last entry
            // If this isn't the case, the incoming line index must be the same as the last line
            // index, so we don't have to add a new line.
            if first_line_idx - incoming_line_idx == 1 {
                self.0.push_front(Line::new(incoming_line_idx));
            }
        } else {
            // line is empty
            self.0.push_front(Line::new(incoming_line_idx));
        }

        // Add the entry to the last line
        let first_line = self.0.front_mut().unwrap();

        // Entries must be added in order, so ensure the last entry in the line has an ending
        // column less than the incoming entry's starting column.
        if let Some(&first_entry) = first_line.entries.back() {
            let first_col = first_entry.reference.end_position().column;
            let incoming_col = entry.reference.end_position().column;

            if incoming_col > first_col {
                return Err(HunkInsertionError::LaterColumn {
                    incoming_col,
                    first_col,
                });
            }
        }
        first_line.entries.push_front(entry);
        Ok(())
    }
}

/// The hunks that correspond to a document
///
/// This type implements a helper builder function that can take
#[derive(Debug, Clone, PartialEq)]
pub struct Hunks<'a>(pub VecDeque<Hunk<'a>>);

impl<'a> Hunks<'a> {
    pub fn new() -> Self {
        Hunks(VecDeque::new())
    }

    /// Push an entry to the front of the hunks
    ///
    /// This will expand the list of hunks if necessary, though the entry must precede the foremost
    /// hunk in the document (by row/column). Failing to do so will result in an error.
    pub fn push_front(&mut self, entry: Entry<'a>) -> Result<()> {
        // If the hunk isn't empty, attempt to prepend an entry into the first hunk
        if let Some(hunk) = self.0.front_mut() {
            let res = hunk.push_front(entry);

            // If the hunk insertion fails because an entry isn't adjacent, then we can create a
            // new hunk. Otherwise we propagate the error since it is a logic error.
            if let Err(HunkInsertionError::NonAdjacentHunk {
                incoming_line: _,
                last_line: _,
            }) = res
            {
                self.0.push_front(Hunk::new());
                self.0.front_mut().unwrap().push_front(entry)?;
            } else {
                res.map_err(|x| anyhow::anyhow!(x))?;
            }
        } else {
            self.0.push_front(Hunk::new());
            self.0.front_mut().unwrap().push_front(entry)?;
        }
        Ok(())
    }
}

/// A difference engine provider
///
/// Any entity that implements this trait is responsible for providing a method
/// that computes the diff between two inputs.
pub trait DiffEngine<'elem, T>
where
    T: Eq + 'elem,
{
    /// The container type to returned from the `diff` function
    type Container;

    /// Compute the shortest edit sequence that will turn `a` into `b`
    fn diff(&self, a: &'elem [T], b: &'elem [T]) -> Self::Container;
}

#[derive(Eq, PartialEq, Copy, Clone, Debug, Default)]
pub struct Myers {}

impl<'elem, T> DiffEngine<'elem, T> for Myers
where
    T: Eq + 'elem + std::fmt::Debug,
{
    type Container = Vec<EditType<&'elem T>>;

    fn diff(&self, a: &'elem [T], b: &'elem [T]) -> Self::Container {
        let mut res = Vec::new();
        // We know the worst case is deleting everything from a and inserting everything from b
        res.reserve(a.len() + b.len());
        Myers::diff_impl(&mut res, a, 0..a.len(), b, 0..b.len());
        res
    }
}

/// Information relevant for a middle snake calculation
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MidSnakeInfo {
    /// The index in `a` that corresponds to the middle snake
    pub a_split: i32,

    /// The index in `b` that corresponds to the middle snake
    pub b_split: i32,

    /// the full length of the optimal path between the two inputs
    pub optimal_len: u32,
}

/// Split a range at the specified index
fn split_range(r: &Range<usize>, idx: usize) -> (Range<usize>, Range<usize>) {
    (r.start..idx, idx..r.end)
}

impl Myers {
    fn diff_impl<'elem, T: Eq + Debug + 'elem>(
        res: &mut Vec<EditType<&'elem T>>,
        old: &'elem [T],
        mut old_range: Range<usize>,
        new: &'elem [T],
        mut new_range: Range<usize>,
    ) {
        // Initial optimizations: we can skip the common prefix + suffix
        let common_pref_len = common_prefix_len(old, old_range.clone(), new, new_range.clone());
        old_range.start += common_pref_len;
        new_range.start += common_pref_len;

        let common_suf_len = common_suffix_len(old, old_range.clone(), new, new_range.clone());
        old_range.end -= common_suf_len;
        new_range.end -= common_suf_len;

        // We know if either or both of the inputs are empty, we don't have to bother finding the
        // middle snake
        if old_range.is_empty() && new_range.is_empty() {
            return;
        }

        if old_range.is_empty() {
            for i in new_range {
                res.push(EditType::Addition(&new[i]));
            }
            return;
        }

        if new_range.is_empty() {
            for i in old_range {
                res.push(EditType::Deletion(&old[i]));
            }
            return;
        }

        let (x_start, y_start) =
            Myers::middle_snake(old, old_range.clone(), new, new_range.clone());

        // divide and conquer along the middle snake
        let (old_first_half, old_second_half) = split_range(&old_range, x_start);
        let (new_first_half, new_second_half) = split_range(&new_range, y_start);

        Myers::diff_impl(res, old, old_first_half, new, new_first_half);
        Myers::diff_impl(res, old, old_second_half, new, new_second_half);
    }

    /// Find the middle snake that splits an optimal path in half, if one exists
    ///
    /// This implementation directly derives from "An O(ND) Difference Algorithm and Its Variations"
    /// by Myers. This will compute the location of the middle snake and the length of the optimal
    /// shortest edit script.
    pub fn middle_snake<T: Eq>(
        old: &[T],
        old_range: Range<usize>,
        new: &[T],
        new_range: Range<usize>,
    ) -> (usize, usize) {
        let n = old_range.len() as i32;
        let m = new_range.len() as i32;
        let delta = n - m;
        let is_odd = delta % 2 == 1;
        let midpoint = ((m + n) as f32 / 2.00).ceil() as i32 + 1;

        // The size of the frontier vector
        let vec_length = (midpoint * 2) as usize;

        // We initialize with sentinel index values instead of 0 and n - 1 because there's no guarantee that the first
        // element will match.
        // TODO(afnan): allocate the vectors outside this method because it's inefficient to keep
        // reallocating
        let mut fwd_front: NegIdxVec<i32> = vec![0; vec_length].into();
        let mut rev_front: NegIdxVec<i32> = vec![0; vec_length].into();

        for d in 0..=midpoint {
            // Find the end of the furthest reaching forward d-path
            for k in (-d..=d).rev().step_by(2) {
                // k == -d and k != d are just bounds checks to make sure we don't try to compare
                // values outside of the [-d, d] range. We check for the furthest reaching forward
                // frontier by seeing which diagonal has the highest x value.
                let mut x = if k == -d || (k != d && fwd_front[k + 1] >= fwd_front[k - 1]) {
                    // If the longest diagonal is from the vertically connected d - 1 path. The y
                    // component is implicitly added when we compute y below with a different k value.
                    fwd_front[k + 1]
                } else {
                    // If the longest diagonal is from the horizontally connected d - 1 path. We
                    // add one here for the horizontal connection (x, y) -> (x + 1, y).
                    fwd_front[k - 1] + 1
                };
                let y = x - k;

                // Coordinates of the first point in the snake
                let (x0, y0) = (x, y);

                // Extend the snake
                if x < n && y < m {
                    debug_assert!(x >= 0);
                    debug_assert!(y >= 0);
                    let common_pref_len = common_prefix_len(
                        old,
                        old_range.start + (x as usize)..old_range.end,
                        new,
                        new_range.start + (y as usize)..new_range.end,
                    );
                    x += common_pref_len as i32;
                }

                fwd_front[k] = x;

                // If delta is odd and k is in the defined range
                if is_odd && (k - delta).abs() < d {
                    // If the path overlaps the furthest reaching reverse d - 1 path in diagonal k
                    // then the length of an SES is 2D - 1, and the last snake of the forward path
                    // is the middle snake.
                    let reverse_x = rev_front[-(k - delta)];

                    // NOTE: that we can convert x and y to `usize` because they are both within
                    // the range of the length of the inputs, which are valid usize values. This property
                    // is also checked with assertions in debug releases.
                    if x + reverse_x >= n {
                        return (old_range.start + x0 as usize, new_range.start + y0 as usize);
                    }
                }
            }

            // Find the end of the furthest reaching reverse d-path
            for k in (-d..=d).rev().step_by(2) {
                // k == d and k != -d are just bounds checks to make sure we don't try to compare
                // anything out of range, as explained above. In the reverse path we check to see
                // which diagonal has the *smallest* x value because we're trying to go from the
                // bottom-right to the top-left of the matrix.
                let mut x = if k == -d || (k != d && rev_front[k + 1] >= rev_front[k - 1]) {
                    // If the longest diagonal is from the horizontally connected d - 1 path. We
                    // subtract 1 from x because we're doing a horizontal jump.
                    rev_front[k + 1]
                } else {
                    // If the longest diagonal is from the vertically connected d - 1 path. The y
                    // value is implicitly handled when we compute y with a different k value.
                    rev_front[k - 1] + 1
                };
                let mut y = x - k;

                if x < n && y < m {
                    debug_assert!(x >= 0, "x={} must be more than 0", x);
                    debug_assert!(y >= 0, "y={} must be more than 0", y);
                    debug_assert!(n - x >= 0);
                    debug_assert!(m - y >= 0);

                    let common_suf_len = common_suffix_len(
                        old,
                        old_range.start..old_range.start + (n as usize) - (x as usize),
                        new,
                        new_range.start..new_range.start + (m as usize) - (y as usize),
                    );
                    x += common_suf_len as i32;
                    y += common_suf_len as i32;
                }

                rev_front[k] = x;

                // If delta is even and k is in the defined range, check for an overlap
                if !is_odd && (k - delta).abs() <= d {
                    // If the path overlaps with the furthest reaching forward D-path in the
                    // diagonal k + delta, then the length of an SES is 2D and the last snake of
                    // the reverse path is the middle snake.

                    // Sanity checks
                    //check_bounds(x, y);
                    let forward_x = fwd_front[-(k - delta)];

                    // If forward_x >= reverse_x, we have an overlap, so return the furthest
                    // reaching reverse path as the middle snake
                    // NOTE: that we can convert x and y to `usize` because they are both within
                    // the range of the length of the inputs, which are valid usize values.
                    if forward_x + x >= n {
                        return (
                            (n as usize) - (x as usize) + old_range.start,
                            (m as usize) - (y as usize) + new_range.start,
                        );
                    }
                }
            }
        }
        panic!("Algorithm error: this branch should never be taken");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq as p_assert_eq;
    use test_case::test_case;

    #[test]
    fn test_mid_snake() {
        let input_a = &b"ABCABBA"[..];
        let input_b = &b"CBABAC"[..];
        let mid_snake = Myers::middle_snake(
            &input_a[..],
            0..input_a.len(),
            &input_b[..],
            0..input_b.len(),
        );
        let expected = (4, 1);
        p_assert_eq!(expected, mid_snake);
    }

    fn myers_diff<'a, T: Eq + 'a + Debug>(a: &'a Vec<T>, b: &'a Vec<T>) -> Vec<EditType<&'a T>> {
        let myers = Myers::default();
        myers.diff(&a[..], &b[..])
    }

    #[test]
    fn myers_diff_empty_inputs() {
        let input_a: Vec<i32> = vec![];
        let input_b: Vec<i32> = vec![];
        let edit_script = myers_diff(&input_a, &input_b);
        assert!(edit_script.is_empty());
    }

    #[test]
    fn myers_diff_no_diff() {
        let input_a: Vec<i32> = vec![0; 4];
        let input_b: Vec<i32> = vec![0; 4];
        let edit_script = myers_diff(&input_a, &input_b);
        assert!(edit_script.is_empty());
    }

    #[test]
    fn myers_diff_one_addition() {
        let input_a: Vec<i32> = Vec::new();
        let input_b: Vec<i32> = vec![0];
        let expected = vec![EditType::Addition(&input_b[0])];
        let edit_script = myers_diff(&input_a, &input_b);
        p_assert_eq!(expected, edit_script);
    }

    #[test]
    fn myers_diff_one_deletion() {
        let input_a: Vec<i32> = vec![0];
        let input_b: Vec<i32> = Vec::new();
        let expected = vec![EditType::Deletion(&input_a[0])];
        let edit_script = myers_diff(&input_a, &input_b);
        p_assert_eq!(expected, edit_script);
    }

    #[test]
    fn myers_diff_single_substitution() {
        let myers = Myers::default();
        let input_a = vec![1];
        let input_b = vec![2];
        let edit_script = myers.diff(&input_a[..], &input_b[..]);
        let expected = vec![
            EditType::Addition(&input_b[0]),
            EditType::Deletion(&input_a[0]),
        ];
        p_assert_eq!(expected, edit_script);
    }

    #[test]
    fn myers_diff_single_substitution_with_common_elements() {
        let myers = Myers::default();
        let input_a = vec![0, 0, 0];
        let input_b = vec![0, 1, 0];
        let edit_script = myers.diff(&input_a[..], &input_b[..]);
        let expected = vec![
            EditType::Addition(&input_b[1]),
            EditType::Deletion(&input_a[1]),
        ];
        p_assert_eq!(expected, edit_script);
    }

    #[test_case(b"BAAA", b"CAAA" => 0 ; "no common prefix")]
    #[test_case(b"AAABA", b"AAACA" => 3 ; "with common prefix")]
    fn common_prefix_tests(a: &[u8], b: &[u8]) -> usize {
        common_prefix_len(&a[..], 0..a.len(), &b[..], 0..b.len())
    }

    #[test_case(b"AAAB", b"AAAC" => 0 ; "no common suffix")]
    #[test_case(b"ABAAA", b"ACAAA" => 3 ; "with common suffix")]
    fn common_suffix_tests(a: &[u8], b: &[u8]) -> usize {
        common_suffix_len(&a[..], 0..a.len(), &b[..], 0..b.len())
    }
}
