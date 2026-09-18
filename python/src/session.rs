// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, future::BoxFuture};
use object_store::{Error as ObjectStoreError, path::Path};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyString};
use pyo3::{Bound, Py, PyAny, PyErr, PyResult, Python, pyclass, pymethods};

use lance::session::{CacheSpec, ExternalBlobFetcher, Session as LanceSession};
use lance_core::cache::{BackendConfig, build_from_config, build_from_uri};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_io::traits::{ByteStream, Reader};

use crate::object_store::PyObjectStoreRegistry;
use crate::rt;

/// The Session holds stateful information for a dataset.
///
/// The session contains caches for opened indices and file metadata.
///
/// Parameters
/// ----------
/// index_cache_size_bytes : int, optional
///     Capacity of the default index cache in bytes.
/// metadata_cache_size_bytes : int, optional
///     Capacity of the default metadata cache in bytes.
/// index_cache_backend : str or dict, optional
///     Custom index cache backend. Strings are backend URIs such as
///     ``"moka://?capacity=1048576"``. Dicts must contain ``"kind"`` and may
///     contain ``"options"``, for example
///     ``{"kind": "moka", "options": {"capacity": "1048576"}}``.
/// metadata_cache_backend : str or dict, optional
///     Custom metadata cache backend with the same format as
///     ``index_cache_backend``.
/// external_blob_fetcher : callable, optional
///     Called with an absolute external blob URI and must return a seekable
///     binary file-like object. Lance reads the returned object directly.
///
/// ``index_cache_backend`` is mutually exclusive with
/// ``index_cache_size_bytes``. ``metadata_cache_backend`` is mutually
/// exclusive with ``metadata_cache_size_bytes``.
#[pyclass(name = "_Session", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct Session {
    pub inner: Arc<LanceSession>,
}

impl Session {
    pub fn new(inner: Arc<LanceSession>) -> Self {
        Self { inner }
    }
}

struct PyExternalBlobFetcher {
    fetcher: Py<PyAny>,
}

#[async_trait]
impl ExternalBlobFetcher for PyExternalBlobFetcher {
    async fn fetch(&self, uri: &str) -> lance::Result<Box<dyn Reader>> {
        let reader = Python::attach(|py| self.fetcher.call1(py, (uri,)))
            .map_err(|error| lance::Error::external(Box::new(error)))?;
        Ok(Box::new(PyExternalBlobReader {
            reader: Arc::new(reader),
            path: Path::from(uri),
        }))
    }
}

struct PyExternalBlobReader {
    reader: Arc<Py<PyAny>>,
    path: Path,
}

impl fmt::Debug for PyExternalBlobReader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PyExternalBlobReader")
            .field("path", &self.path)
            .finish()
    }
}

impl DeepSizeOf for PyExternalBlobReader {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

fn object_store_error(error: PyErr) -> ObjectStoreError {
    ObjectStoreError::Generic {
        store: "python external blob fetcher",
        source: Box::new(error),
    }
}

fn read_python_range(reader: &Py<PyAny>, range: Range<usize>) -> object_store::Result<Bytes> {
    Python::attach(|py| {
        reader.call_method1(py, "seek", (range.start, 0))?;
        let data: Vec<u8> = reader
            .call_method1(py, "read", (range.len(),))?
            .extract(py)?;
        if data.len() != range.len() {
            return Err(PyIOError::new_err(format!(
                "external blob reader returned {} bytes for range {}..{}; expected {}",
                data.len(),
                range.start,
                range.end,
                range.len()
            )));
        }
        Ok(Bytes::from(data))
    })
    .map_err(object_store_error)
}

fn python_range_stream(reader: Arc<Py<PyAny>>, range: Range<usize>) -> ByteStream {
    const CHUNK_SIZE: usize = 8 * 1024 * 1024;
    let range_end = range.end;

    futures::stream::try_unfold((reader, range.start), move |(reader, start)| async move {
        if start >= range_end {
            return Ok(None);
        }
        let end = start.saturating_add(CHUNK_SIZE).min(range_end);
        let bytes = read_python_range(reader.as_ref(), start..end)?;
        Ok(Some((bytes, (reader, end))))
    })
    .boxed()
}

impl Reader for PyExternalBlobReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        8 * 1024 * 1024
    }

    fn io_parallelism(&self) -> usize {
        1
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        Box::pin(async move {
            Python::attach(|py| {
                let current: usize = self.reader.call_method0(py, "tell")?.extract(py)?;
                self.reader.call_method1(py, "seek", (0, 2))?;
                let size = self.reader.call_method0(py, "tell")?.extract(py)?;
                self.reader.call_method1(py, "seek", (current, 0))?;
                Ok(size)
            })
            .map_err(object_store_error)
        })
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let reader = self.reader.clone();
        Box::pin(async move { read_python_range(reader.as_ref(), range) })
    }

    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        Box::pin(async move {
            Python::attach(|py| {
                self.reader.call_method1(py, "seek", (0, 0))?;
                let data: Vec<u8> = self.reader.call_method0(py, "read")?.extract(py)?;
                Ok(Bytes::from(data))
            })
            .map_err(object_store_error)
        })
    }

    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        let reader = self.reader.clone();
        Box::pin(async move { Ok(python_range_stream(reader, range)) })
    }
}

/// Turn a Python-supplied backend descriptor into an `Arc<dyn CacheBackend>`,
/// or return `Ok(None)` when the caller did not pass one.
///
/// Accepts:
///   * `str` — treated as a URI (`moka://?capacity=...`) and passed to
///     [`build_from_uri`].
///   * `dict` — must have string keys `kind` (required) and `options`
///     (optional `dict[str, str]`) matching [`BackendConfig`]; passed to
///     [`build_from_config`].
///
/// Any other Python type is rejected with a clear `TypeError`-style
/// `PyValueError`.
///
/// If `size_field_set` is `true` and `backend` is `Some`, both a size and a
/// backend were provided for the same cache. Rather than silently letting
/// one override the other (Proposal §7), this is rejected up-front so the
/// operator gets an actionable error.
fn resolve_cache_spec(
    backend_field: &str,
    backend: Option<&Bound<'_, PyAny>>,
    size_field: &str,
    size: Option<usize>,
) -> PyResult<CacheSpec> {
    if backend.is_some() && size.is_some() {
        return Err(PyValueError::new_err(format!(
            "{} and {} are mutually exclusive; set one or the other",
            size_field, backend_field,
        )));
    }

    let Some(value) = backend else {
        return Ok(size.map(CacheSpec::Size).unwrap_or(CacheSpec::Default));
    };

    if value.cast::<PyString>().is_ok() {
        let uri: String = value.extract()?;
        return build_from_uri(&uri)
            .map(CacheSpec::Backend)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", backend_field, e)));
    }

    if let Ok(dict) = value.cast::<PyDict>() {
        let cfg = backend_config_from_dict(backend_field, dict)?;
        return build_from_config(&cfg)
            .map(CacheSpec::Backend)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", backend_field, e)));
    }

    let type_name: String = value.get_type().getattr("__name__")?.extract()?;
    Err(PyValueError::new_err(format!(
        "{}: expected str (URI) or dict with 'kind'/'options' keys, got {}",
        backend_field, type_name,
    )))
}

fn backend_config_from_dict(field: &str, dict: &Bound<'_, PyDict>) -> PyResult<BackendConfig> {
    for (key, _) in dict.iter() {
        if key.cast::<PyString>().is_err() {
            return Err(PyValueError::new_err(format!(
                "{}: dict keys must be strings",
                field
            )));
        }
        let key: String = key.extract()?;
        if key != "kind" && key != "options" {
            return Err(PyValueError::new_err(format!(
                "{}: unknown dict key {:?}; expected 'kind' or 'options'",
                field, key
            )));
        }
    }

    let kind_obj = dict.get_item("kind")?.ok_or_else(|| {
        PyValueError::new_err(format!("{}: dict must contain a 'kind' key", field))
    })?;
    if kind_obj.cast::<PyString>().is_err() {
        return Err(PyValueError::new_err(format!(
            "{}: 'kind' must be a string",
            field
        )));
    }
    let kind: String = kind_obj.extract()?;

    let mut options: HashMap<String, String> = HashMap::new();
    if let Some(options_obj) = dict.get_item("options")? {
        let options_dict = options_obj.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("{}: 'options' must be a dict[str, str]", field))
        })?;
        for (k, v) in options_dict.iter() {
            if k.cast::<PyString>().is_err() {
                return Err(PyValueError::new_err(format!(
                    "{}: 'options' keys must be strings",
                    field
                )));
            }
            if v.cast::<PyString>().is_err() {
                return Err(PyValueError::new_err(format!(
                    "{}: 'options' values must be strings",
                    field
                )));
            }
            let key: String = k.extract()?;
            let value: String = v.extract()?;
            options.insert(key, value);
        }
    }

    let mut config = BackendConfig::new(&kind)
        .map_err(|e| PyValueError::new_err(format!("{}: {}", field, e)))?;
    config.options = options;
    Ok(config)
}

#[pymethods]
impl Session {
    #[new]
    #[pyo3(signature=(
        index_cache_size_bytes=None,
        metadata_cache_size_bytes=None,
        index_cache_backend=None,
        metadata_cache_backend=None,
        store_registry=None,
        external_blob_fetcher=None,
    ))]
    fn create(
        index_cache_size_bytes: Option<usize>,
        metadata_cache_size_bytes: Option<usize>,
        index_cache_backend: Option<Bound<'_, PyAny>>,
        metadata_cache_backend: Option<Bound<'_, PyAny>>,
        store_registry: Option<PyObjectStoreRegistry>,
        external_blob_fetcher: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let index_cache = resolve_cache_spec(
            "index_cache_backend",
            index_cache_backend.as_ref(),
            "index_cache_size_bytes",
            index_cache_size_bytes,
        )?;
        let metadata_cache = resolve_cache_spec(
            "metadata_cache_backend",
            metadata_cache_backend.as_ref(),
            "metadata_cache_size_bytes",
            metadata_cache_size_bytes,
        )?;
        let store_registry = store_registry.map(|r| r.inner).unwrap_or_default();
        let mut session =
            LanceSession::with_cache_backends(index_cache, metadata_cache, store_registry);
        if let Some(fetcher) = external_blob_fetcher {
            if !fetcher.is_callable() {
                return Err(PyValueError::new_err(
                    "external_blob_fetcher must be callable",
                ));
            }
            session = session.with_external_blob_fetcher(Arc::new(PyExternalBlobFetcher {
                fetcher: fetcher.unbind(),
            }));
        }
        Ok(Self {
            inner: Arc::new(session),
        })
    }

    fn __repr__(&self) -> String {
        let (index_cache_size, meta_cache_size) = rt()
            .block_on(None, async move {
                (
                    self.inner.index_cache_stats().await.size_bytes,
                    self.inner.metadata_cache_stats().await.size_bytes,
                )
            })
            .unwrap_or((0, 0));
        format!(
            "Session(index_cache_size_bytes={}, metadata_cache_size_bytes={})",
            index_cache_size, meta_cache_size
        )
    }

    /// Return the current size of the session in bytes
    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes()
    }

    /// Return the current size of the index cache in bytes.
    pub fn index_cache_size_bytes(&self) -> PyResult<u64> {
        rt().block_on(None, async move {
            self.inner.index_cache_stats().await.size_bytes as u64
        })
    }

    /// Return whether the other session is the same as this one.
    pub fn is_same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
