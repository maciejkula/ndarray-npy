//! A simple serialization of ndarray arrays of simple types (`f32`, `f64`) into
//! [NumPy](http://www.numpy.org/)'s [`.npy`
//! format](https://docs.scipy.org/doc/numpy/neps/npy-format.html).
//!
//! Files produced this way can be loaded with `numpy.load`.
//!
//! ```
//! extern crate ndarray;
//! extern crate ndarray_npy;
//!
//! use ndarray::prelude::*;
//! use std::io;
//!
//! fn main() {
//!     let arr: Array2<f64> = Array2::zeros((3, 4));
//!
//!     let stdout = io::stdout();
//!     let mut handle = stdout.lock();
//!
//!     ndarray_npy::write(&mut handle, &arr).unwrap();
//! }
//! ```
//!
extern crate byteorder;
extern crate ndarray;

use byteorder::{BigEndian, LittleEndian, NativeEndian, WriteBytesExt, ByteOrder};
use std::io;
use ndarray::prelude::*;
use ndarray::Data;

static MAGIC_VALUE : &[u8] = b"\x93NUMPY";
static NPY_VERSION : &[u8] = b"\x01\x00";

/// Types that can be serialized using this crate.
pub trait DType<B> {
    fn dtype() -> &'static str;
    fn write_bytes(self, w: &mut io::Write) -> io::Result<()>;
}

macro_rules! impl_dtype {
    ($type:ty, $dtype:expr, $byteorder_fn:ident) => {
        impl<B: ByteOrder> DType<B> for $type {
            fn dtype() -> &'static str {
                $dtype
            }

            fn write_bytes(self, w: &mut io::Write) -> io::Result<()> {
                w.$byteorder_fn::<B>(self)
            }
        }
    }
}

impl_dtype!(f32, "f4", write_f32);
impl_dtype!(f64, "f8", write_f64);


trait NumpyEndian {
    fn endian_symbol() -> &'static str;
}

impl NumpyEndian for LittleEndian {
    fn endian_symbol() -> &'static str {
        "<"
    }
}

impl NumpyEndian for BigEndian {
    fn endian_symbol() -> &'static str {
        ">"
    }
}

fn get_header<A, B>(shape: &[usize])-> String
    where A: DType<B>, B: NumpyEndian {
        use std::fmt::Write;
        let mut shape_str = String::new();
        for (i, s) in shape.iter().enumerate() {
            if i > 0 {
                shape_str.push_str(",");
            }
            write!(&mut shape_str, "{}", s).unwrap();
        }
    format!("{{'descr': '{endian}{dtype}','fortran_order': False,'shape': ({shape})}}\n",
        endian = B::endian_symbol(),
        dtype = A::dtype(),
        shape = shape_str
        )
}

/// Write an ndarray to a writer in numpy format.
///
/// Can be saved with extension `npy` and loaded using `numpy.load`.
pub fn write<A, S, D>(w: &mut io::Write, array: &ArrayBase<S, D>) -> io::Result<()>
where S: Data<Elem = A>, D: Dimension, A: DType<NativeEndian> + Copy {
    w.write(MAGIC_VALUE)?;
    w.write(NPY_VERSION)?;
    let header = get_header::<A, NativeEndian>(array.shape());
    let header_len = header.len();
    let align = (16 - (MAGIC_VALUE.len() + 4 + header_len) % 16) % 16;
    w.write_u16::<LittleEndian>((header_len + align) as u16)?;
    w.write(header.as_bytes())?;

    for _ in 0..align {
        w.write_u8(b' ')?;
    }

    for x in array.iter().cloned() {
        x.write_bytes(w)?;
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
