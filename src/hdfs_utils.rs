use hdfs_rs::hdfs::Hdfs;
use hdfs_rs::hdfs_builder::HdfsBuilder;
use std::path::Path;

pub struct HdfsUtils {
    hdfs: Hdfs,
}

impl HdfsUtils {
    pub fn new(uri: &str) -> Result<Self, hdfs_rs::errors::HdfsErr> {
        let hdfs = HdfsBuilder::new(uri).connect()?;
        Ok(Self { hdfs })
    }

    pub fn upload_file(
        &self,
        local_path: &str,
        hdfs_path: &str,
    ) -> Result<(), hdfs_rs::errors::HdfsErr> {
        let local_file = std::fs::File::open(local_path)?;
        let hdfs_file = self.hdfs.create(hdfs_path)?;

        let mut buf_reader = std::io::BufReader::new(local_file);
        let mut buf_writer = std::io::BufWriter::new(hdfs_file);

        std::io::copy(&mut buf_reader, &mut buf_writer)?;

        Ok(())
    }
}
