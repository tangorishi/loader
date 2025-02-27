use std::io::Read;
use std::{ffi::c_void, sync::Arc};

use cudarc::driver::sys as cu;
use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::task::JoinSet;
use tokio_stream::StreamExt;

fn cuda_init() -> cu::CUcontext {
    unsafe {
        let result = cu::cuInit(0);
        assert!(
            result == cu::cudaError_enum::CUDA_SUCCESS,
            "init failed: {:?}",
            result
        );

        let mut device: cu::CUdevice = 0;
        let device_result = cu::cuDeviceGet(&mut device, 0);
        assert!(
            device_result == cu::cudaError_enum::CUDA_SUCCESS,
            "device get failed: {:?}",
            device_result
        );

        let mut context: cu::CUcontext = std::ptr::null_mut();
        let context_result = cu::cuCtxCreate_v2(&mut context, 0, device);
        assert!(
            context_result == cu::cudaError_enum::CUDA_SUCCESS,
            "context create failed: {:?}",
            context_result
        );

        context
    }
}

fn cuda_malloc_pair(size: usize) -> (*mut c_void, u64) {
    unsafe {
        let mut ptr = std::ptr::null_mut();
        let alloc_result = cu::cuMemHostAlloc(
            &mut ptr,
            size,
            cu::CU_MEMHOSTALLOC_PORTABLE | cu::CU_MEMHOSTALLOC_WRITECOMBINED,
        );
        assert!(
            alloc_result == cu::cudaError_enum::CUDA_SUCCESS,
            "host alloc failed: {:?}",
            alloc_result
        );

        // alloc device memory
        let mut device_ptr = cu::CUdeviceptr::default();
        let device_alloc_result = cu::cuMemAlloc_v2(&mut device_ptr, size);
        assert!(
            device_alloc_result == cu::cudaError_enum::CUDA_SUCCESS,
            "device alloc failed: {:?}",
            device_alloc_result
        );

        (ptr, device_ptr)
    }
}

async fn download_url(
    url: &str,
    num_workers: Option<usize>,
    header_size: usize,
    data_size: usize,
    host_ptr: usize,
) -> Result<(), reqwest::Error> {
    let num_workers = num_workers.unwrap_or(num_cpus::get());

    let bar = ProgressBar::new(data_size as u64);
    let sty = ProgressStyle::with_template("{spinner:.green} ({msg}) [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes:>12}/{total_bytes:>12} ({bytes_per_sec:>15}, {eta:>5})")
        .unwrap()
        .progress_chars("#>-");
    bar.set_style(sty);
    let arc_bar = Arc::new(bar);

    let mut tasks = JoinSet::new();
    let per_worker_size = data_size / num_workers;
    for i in 0..num_workers {
        let arc_bar = arc_bar.clone();
        let url = url.to_owned();
        tasks.spawn(async move {
            // let range_header = format!("bytes={}-", header_size);
            let range_start = header_size + i * per_worker_size;
            let range_header = {
                if i == num_workers - 1 {
                    format!("bytes={}-", range_start)
                } else {
                    format!("bytes={}-{}", range_start, range_start + per_worker_size)
                }
            };

            let resp = reqwest::Client::new()
                .get(url)
                .header("Range", range_header)
                .send()
                .await?;

            let mut stream = resp.bytes_stream();
            let mut host_ptr = host_ptr + i * per_worker_size;
            while let Some(item) = stream.next().await {
                let bytes = item?;

                arc_bar.inc(bytes.len() as u64);

                unsafe {
                    std::ptr::copy(bytes.as_ptr(), host_ptr as *mut u8, bytes.len());
                    host_ptr += bytes.len();
                }
            }

            Ok(())
        });
    }

    while let Some(res) = tasks.join_next().await {
        let res = res.unwrap();
        if res.is_err() {
            return res;
        }
    }

    Ok(())
}

async fn alloc_and_download(
    cu_ctx_addr: usize,
    url: &str,
    num_workers: Option<usize>,
    header_size: usize,
    data_size: usize,
) -> Result<(u64, u64), reqwest::Error> {
    // set context for current thread
    unsafe {
        let cu_ctx = cu_ctx_addr as cu::CUcontext;
        cu::cuCtxSetCurrent(cu_ctx);
    }

    let (host_ptr, device_ptr) = cuda_malloc_pair(data_size);
    let host_ptr = host_ptr as usize; // casting so it is cheaply copyable

    download_url(url, num_workers, header_size, data_size, host_ptr).await?;

    // copy from host to device
    // there's about 100ms per 1GB of data overhead here. This can entirely go away by
    // using async copy and run it along side with the download. We probably need it for
    // 7B model.
    unsafe {
        let cu_ctx = cu_ctx_addr as cu::CUcontext;
        cu::cuCtxSetCurrent(cu_ctx);
    }
    let start = std::time::Instant::now();
    unsafe {
        let copy_result = cu::cuMemcpyHtoD_v2(device_ptr, host_ptr as *mut c_void, data_size);
        assert!(
            copy_result == cu::cudaError_enum::CUDA_SUCCESS,
            "copy failed: {:?}",
            copy_result
        );
    }
    println!("copy took {:?}", start.elapsed());

    Ok((host_ptr as u64, device_ptr))
}

#[pyfunction]
fn download_to_device(
    urls: Vec<&str>,
    header_size: Vec<usize>,
    data_size: Vec<usize>,
    num_workers: Option<usize>,
) -> PyResult<Vec<(u64, u64)>> {
    let cu_ctx = cuda_init();
    let cu_ctx_addr = cu_ctx as usize;

    assert!(urls.len() == header_size.len());
    assert!(urls.len() == data_size.len());

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let mut tasks = JoinSet::new();
        for ((url, header_size), data_size) in urls.into_iter().zip(header_size).zip(data_size) {
            let url = url.to_owned();
            tasks.spawn(async move {
                alloc_and_download(cu_ctx_addr, &url, num_workers, header_size, data_size).await
            });
        }

        let mut pair_ptrs = Vec::new();
        while let Some(res) = tasks.join_next().await {
            let res = res.unwrap().unwrap();
            pair_ptrs.push(res);
        }

        Ok(pair_ptrs)
    })
}

#[pyfunction]
fn zip_extract(py: Python, host_ptr: u64, file_size: usize, file_name: &str) -> PyObject {
    let vec = unsafe { Vec::from_raw_parts(host_ptr as *mut u8, file_size, 0) };
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(vec)).unwrap();
    let mut file = archive.by_name(file_name).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    PyBytes::new(py, &buf).into()
}

#[pyfunction]
fn zip_list_range(host_ptr: u64, file_size: usize) -> PyResult<Vec<(String, u64, u64)>> {
    let vec = unsafe { Vec::from_raw_parts(host_ptr as *mut u8, file_size, 0) };
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(vec)).unwrap();
    let mut name_ranges = Vec::new();
    for i in 0..archive.len() {
        let file = archive.by_index(i).unwrap();
        assert!(
            file.compression() == zip::CompressionMethod::Stored,
            "file {} is compressed, which is not supported",
            file.name()
        );
        name_ranges.push((file.name().to_owned(), file.data_start(), file.size()));
    }
    Ok(name_ranges)
}

#[pyfunction]
fn free_host_ptr(host_ptr: u64) {
    unsafe {
        let free_result = cu::cuMemFreeHost(host_ptr as *mut c_void);
        assert!(
            free_result == cu::cudaError_enum::CUDA_SUCCESS,
            "free failed: {:?}",
            free_result
        );
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn model_loader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download_to_device, m)?)?;
    m.add_function(wrap_pyfunction!(zip_extract, m)?)?;
    m.add_function(wrap_pyfunction!(zip_list_range, m)?)?;
    m.add_function(wrap_pyfunction!(free_host_ptr, m)?)?;
    Ok(())
}
