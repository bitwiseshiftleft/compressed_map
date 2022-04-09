/*
 * @file size.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Get size of serialized object.  Should be part of bincode, dunno why it isn't.
 */

use bincode::{Encode,config::Config,enc::EncoderImpl,error::EncodeError,enc::write::Writer};

struct SizeOnlyWriter<'a> {
    bytes_written: &'a mut usize
}

impl<'a> Writer for SizeOnlyWriter<'a> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        *self.bytes_written += bytes.len();
        Ok(())
    }
}

/** Return the serialized size of an `Encode` object. */
pub fn serialized_size<T:Encode,C:Config>(obj:&T, config:C) -> Result<usize, EncodeError> {
    let mut size = 0usize;
    let writer = SizeOnlyWriter { bytes_written: &mut size };
    let mut ei = EncoderImpl::new(writer, config);
    obj.encode(&mut ei)?;
    Ok(size)
}
