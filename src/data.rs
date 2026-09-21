//! Dataset loading: the IDX format used by MNIST, plus batching.
//!
//! Only unsigned-byte IDX files are supported. Files must already be
//! decompressed (`gunzip`): decoding gzip would need an inflate implementation,
//! and we have no dependencies.

use crate::tensor::Tensor;
use crate::Real;

pub struct Idx {
    pub dims: Vec<usize>,
    pub data: Vec<u8>,
}

/// Parses an IDX file: bytes 0-1 are zero, byte 2 is the dtype (0x08 = u8),
/// byte 3 is the number of dimensions, followed by big-endian u32 sizes, then data.
pub fn parse_idx(bytes: &[u8]) -> Result<Idx, String> {
    if bytes.len() < 4 {
        return Err("file too short for an IDX header".into());
    }
    if bytes[0] != 0 || bytes[1] != 0 {
        return Err("bad IDX magic number (is the file still gzip-compressed?)".into());
    }
    if bytes[2] != 0x08 {
        return Err(format!("unsupported IDX dtype 0x{:02x} (only 0x08 = u8)", bytes[2]));
    }
    let nd = bytes[3] as usize;
    let header = 4 + 4 * nd;
    if bytes.len() < header {
        return Err("file too short for the declared dimensions".into());
    }
    let mut dims = Vec::with_capacity(nd);
    let mut count: usize = 1;
    for i in 0..nd {
        let o = 4 + 4 * i;
        let d = u32::from_be_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        count = count.checked_mul(d).ok_or("IDX dimensions overflow")?;
        dims.push(d);
    }
    if bytes.len() - header != count {
        return Err(format!("IDX size mismatch: header says {count} bytes, file has {}", bytes.len() - header));
    }
    Ok(Idx { dims, data: bytes[header..].to_vec() })
}

pub struct Dataset {
    /// `[n, d]`, pixel values scaled to [0, 1] by dividing by 255.
    pub images: Tensor,
    pub labels: Vec<usize>,
}

impl Dataset {
    pub fn from_idx(images: &Idx, labels: &Idx) -> Result<Self, String> {
        if images.dims.len() != 3 {
            return Err(format!("expected image dims [n, rows, cols], got {:?}", images.dims));
        }
        if labels.dims.len() != 1 {
            return Err(format!("expected label dims [n], got {:?}", labels.dims));
        }
        let n = images.dims[0];
        if labels.dims[0] != n {
            return Err(format!("{n} images but {} labels", labels.dims[0]));
        }
        let d = images.dims[1] * images.dims[2];
        let px: Vec<Real> = images.data.iter().map(|&b| b as Real / 255.0).collect();
        Ok(Dataset {
            images: Tensor::from_vec(&[n, d], px),
            labels: labels.data.iter().map(|&b| b as usize).collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Copies the rows in `idx` into a batch `([len(idx), d], labels)`.
    pub fn batch(&self, idx: &[usize]) -> (Tensor, Vec<usize>) {
        let (_, d) = self.images.dims2();
        let mut x = Vec::with_capacity(idx.len() * d);
        let mut y = Vec::with_capacity(idx.len());
        for &i in idx {
            x.extend_from_slice(&self.images.data()[i * d..(i + 1) * d]);
            y.push(self.labels[i]);
        }
        (Tensor::from_vec(&[idx.len(), d], x), y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx_bytes(dims: &[u32], data: &[u8]) -> Vec<u8> {
        let mut v = vec![0, 0, 0x08, dims.len() as u8];
        for d in dims {
            v.extend_from_slice(&d.to_be_bytes());
        }
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn parses_images_and_labels() {
        let im = parse_idx(&idx_bytes(&[2, 2, 2], &[0, 255, 51, 102, 255, 0, 0, 0])).unwrap();
        let lb = parse_idx(&idx_bytes(&[2], &[7, 3])).unwrap();
        assert_eq!(im.dims, vec![2, 2, 2]);
        let ds = Dataset::from_idx(&im, &lb).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.images.shape(), &[2, 4]);
        assert_eq!(ds.images.data()[1], 1.0);
        assert_eq!(ds.images.data()[2], 51.0 / 255.0);
        assert_eq!(ds.labels, vec![7, 3]);
    }

    #[test]
    fn batch_copies_selected_rows() {
        let im = parse_idx(&idx_bytes(&[3, 1, 2], &[1, 2, 3, 4, 5, 6])).unwrap();
        let lb = parse_idx(&idx_bytes(&[3], &[0, 1, 2])).unwrap();
        let ds = Dataset::from_idx(&im, &lb).unwrap();
        let (x, y) = ds.batch(&[2, 0]);
        assert_eq!(x.shape(), &[2, 2]);
        assert_eq!(x.data()[0], 5.0 / 255.0);
        assert_eq!(x.data()[3], 2.0 / 255.0);
        assert_eq!(y, vec![2, 0]);
    }

    #[test]
    fn rejects_malformed_files() {
        assert!(parse_idx(&[]).is_err());
        assert!(parse_idx(&[0x1f, 0x8b, 8, 0]).is_err()); // gzip magic
        assert!(parse_idx(&[0, 0, 0x0d, 1, 0, 0, 0, 1, 0]).is_err()); // wrong dtype
        assert!(parse_idx(&idx_bytes(&[2], &[1])).is_err()); // truncated data
        assert!(parse_idx(&idx_bytes(&[1], &[1, 2])).is_err()); // extra data
    }

    #[test]
    fn rejects_count_mismatch() {
        let im = parse_idx(&idx_bytes(&[2, 1, 1], &[1, 2])).unwrap();
        let lb = parse_idx(&idx_bytes(&[1], &[0])).unwrap();
        assert!(Dataset::from_idx(&im, &lb).is_err());
    }
}
