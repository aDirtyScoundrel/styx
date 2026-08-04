use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Result as IoResult, Seek};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
}

impl Gguf {
    pub fn open(path: &Path) -> IoResult<Gguf> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let magic = read_u32_le(&mut reader)?;
        if magic != 0x46554747 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid GGUF magic",
            ));
        }

        let version = read_u32_le(&mut reader)?;
        if version != 2 && version != 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unsupported GGUF version",
            ));
        }

        let tensor_count = read_u64_le(&mut reader)?;
        let metadata_kv_count = read_u64_le(&mut reader)?;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut reader)?;
            let value_type = read_u32_le(&mut reader)?;
            let value = read_value(&mut reader, value_type)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut total_offset = 0u64;

        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut reader)?;
            let n_dims = read_u32_le(&mut reader)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64_le(&mut reader)?);
            }
            let ggml_type = read_u32_le(&mut reader)?;
            let offset = read_u64_le(&mut reader)?;

            // Compute size_bytes
            let size_bytes = compute_size_bytes(&dims, ggml_type)?;

            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                size_bytes,
            });

            total_offset += size_bytes;
        }

        // Align data start: current file position after tensor infos, aligned up
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                Value::U32(a) => Some(*a as u64),
                _ => None,
            })
            .unwrap_or(32);
        let pos = reader.stream_position()?;
        let data_start = (pos + alignment - 1) & !(alignment - 1);
        let _ = total_offset;

        Ok(Gguf {
            version,
            metadata,
            tensors,
            data_start,
        })
    }
}

fn read_u32_le<R: Read>(reader: &mut R) -> IoResult<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(reader: &mut R) -> IoResult<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(reader: &mut R) -> IoResult<String> {
    let len = read_u64_le(reader)? as usize;
    if len > (1 << 32) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "String too long",
        ));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid UTF-8: {}", e),
        )
    })
}

fn read_value<R: Read>(reader: &mut R, value_type: u32) -> IoResult<Value> {
    match value_type {
        0 => Ok(Value::U8(read_u8(reader)?)),
        1 => Ok(Value::I8(read_i8(reader)?)),
        2 => Ok(Value::U16(read_u16_le(reader)?)),
        3 => Ok(Value::I16(read_i16_le(reader)?)),
        4 => Ok(Value::U32(read_u32_le(reader)?)),
        5 => Ok(Value::I32(read_i32_le(reader)?)),
        6 => Ok(Value::F32(read_f32_le(reader)?)),
        7 => Ok(Value::Bool(read_u8(reader)? != 0)),
        8 => Ok(Value::String(read_gguf_string(reader)?)),
        9 => read_array(reader),
        10 => Ok(Value::U64(read_u64_le(reader)?)),
        11 => Ok(Value::I64(read_i64_le(reader)?)),
        12 => Ok(Value::F64(read_f64_le(reader)?)),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown value type: {}", value_type),
        )),
    }
}

fn read_u8<R: Read>(reader: &mut R) -> IoResult<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> IoResult<i8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
}

fn read_u16_le<R: Read>(reader: &mut R) -> IoResult<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16_le<R: Read>(reader: &mut R) -> IoResult<i16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_i32_le<R: Read>(reader: &mut R) -> IoResult<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32_le<R: Read>(reader: &mut R) -> IoResult<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_i64_le<R: Read>(reader: &mut R) -> IoResult<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f64_le<R: Read>(reader: &mut R) -> IoResult<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_array<R: Read>(reader: &mut R) -> IoResult<Value> {
    let element_type = read_u32_le(reader)?;
    let count = read_u64_le(reader)? as usize;
    if count > (1 << 32) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Array too long",
        ));
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_value(reader, element_type)?);
    }
    Ok(Value::Array(values))
}

fn compute_size_bytes(dims: &[u64], ggml_type: u32) -> IoResult<u64> {
    let block_size_elements = match ggml_type {
        0 => 1,    // F32
        1 => 1,    // F16
        2 => 32,   // Q4_0
        3 => 32,   // Q4_1
        6 => 32,   // Q5_0
        7 => 32,   // Q5_1
        8 => 32,   // Q8_0
        9 => 32,   // Q8_1
        10 => 256, // Q2_K
        11 => 256, // Q3_K
        12 => 256, // Q4_K
        13 => 256, // Q5_K
        14 => 256, // Q6_K
        15 => 256, // Q8_K
        16 => 256, // IQ2_XXS
        17 => 256, // IQ2_XS
        18 => 256, // IQ3_XXS
        19 => 256, // IQ1_S
        20 => 32,  // IQ4_NL
        21 => 256, // IQ3_S
        22 => 256, // IQ2_S
        23 => 256, // IQ4_XS
        24 => 1,   // I8
        25 => 1,   // I16
        26 => 1,   // I32
        27 => 1,   // I64
        28 => 1,   // F64
        29 => 256, // IQ1_M
        30 => 1,   // BF16
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown ggml_type: {}", ggml_type),
            ));
        }
    };

    let type_size_bytes = match ggml_type {
        0 => 4,    // F32
        1 => 2,    // F16
        2 => 18,   // Q4_0
        3 => 20,   // Q4_1
        6 => 22,   // Q5_0
        7 => 24,   // Q5_1
        8 => 34,   // Q8_0
        9 => 36,   // Q8_1
        10 => 84,  // Q2_K
        11 => 110, // Q3_K
        12 => 144, // Q4_K
        13 => 176, // Q5_K
        14 => 210, // Q6_K
        15 => 292, // Q8_K
        16 => 66,  // IQ2_XXS
        17 => 74,  // IQ2_XS
        18 => 98,  // IQ3_XXS
        19 => 50,  // IQ1_S
        20 => 18,  // IQ4_NL
        21 => 110, // IQ3_S
        22 => 82,  // IQ2_S
        23 => 136, // IQ4_XS
        24 => 1,   // I8
        25 => 2,   // I16
        26 => 4,   // I32
        27 => 8,   // I64
        28 => 8,   // F64
        29 => 56,  // IQ1_M
        30 => 2,   // BF16
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown ggml_type: {}", ggml_type),
            ));
        }
    };

    let mut product = 1u64;
    for &dim in dims {
        product *= dim;
    }

    if product % block_size_elements != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "First dimension not multiple of block size",
        ));
    }

    Ok((product / block_size_elements) * type_size_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    fn create_test_gguf() -> std::io::Result<std::path::PathBuf> {
        let path = std::env::temp_dir().join(format!("gguf_rs_test_{}.gguf", std::process::id()));
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&2u64.to_le_bytes());
        // Metadata count
        buf.extend_from_slice(&3u64.to_le_bytes());

        // Metadata key-value pairs
        // "general.name"
        buf.extend_from_slice(&12u64.to_le_bytes());
        buf.extend_from_slice(b"general.name");
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(b"test");

        // "general.alignment"
        buf.extend_from_slice(&17u64.to_le_bytes());
        buf.extend_from_slice(b"general.alignment");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());

        // "tokenizer.chat_template"
        buf.extend_from_slice(&23u64.to_le_bytes());
        buf.extend_from_slice(b"tokenizer.chat_template");
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(b"{{chat}}");

        // Tensor 0: "t0"
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(b"t0");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Tensor 1: "t1"
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(b"t1");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        // Write to file
        let mut file = File::create(&path)?;
        file.write_all(&buf)?;
        file.flush()?;
        Ok(path)
    }

    #[test]
    fn test_gguf_parsing() -> std::io::Result<()> {
        let path = create_test_gguf()?;
        let gguf = Gguf::open(&path)?;

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 2);

        let t0 = &gguf.tensors[0];
        assert_eq!(t0.name, "t0");
        assert_eq!(t0.dims, vec![4, 2]);
        assert_eq!(t0.ggml_type, 0);
        assert_eq!(t0.offset, 0);
        assert_eq!(t0.size_bytes, 32); // 4*2 * 4 bytes per F32

        let t1 = &gguf.tensors[1];
        assert_eq!(t1.name, "t1");
        assert_eq!(t1.dims, vec![256, 3]);
        assert_eq!(t1.ggml_type, 12);
        assert_eq!(t1.offset, 16);
        assert_eq!(t1.size_bytes, 432); // (256*3)/256 * 144 bytes per Q4_K block

        assert_eq!(gguf.data_start, 256); // file pos after tensor infos (228) aligned up to 32

        Ok(())
    }
}
