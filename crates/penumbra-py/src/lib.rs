//! PyO3 bindings exposing the Penumbra-FHE runtime to Python in-process.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;

mod session;

/// Named crypto-parameter profiles — the single override knob per backend (`PROJECT.md` §12).
#[derive(Clone)]
enum ProfileInner {
    Tfhe(penumbra_tfhe::keys::TfheProfile),
    Ckks { max_poly_degree: usize },
}

#[pyclass(frozen, from_py_object, module = "penumbra._penumbra")]
#[derive(Clone)]
pub struct CryptoProfile {
    inner: ProfileInner,
}

#[pymethods]
impl CryptoProfile {
    /// The TFHE knob: a named parameter profile.
    #[staticmethod]
    #[pyo3(signature = (name="default"))]
    fn tfhe(name: &str) -> PyResult<Self> {
        let prof =
            penumbra_tfhe::keys::TfheProfile::from_name(name).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: ProfileInner::Tfhe(prof),
        })
    }

    /// The CKKS knob: the maximum polynomial degree any op may fit.
    #[staticmethod]
    fn ckks(max_poly_degree: usize) -> PyResult<Self> {
        if max_poly_degree < 1 {
            return Err(PyValueError::new_err(format!(
                "max_poly_degree must be >= 1, got {max_poly_degree}"
            )));
        }
        Ok(Self {
            inner: ProfileInner::Ckks { max_poly_degree },
        })
    }

    #[getter]
    fn backend(&self) -> &'static str {
        match &self.inner {
            ProfileInner::Tfhe(_) => "tfhe",
            ProfileInner::Ckks { .. } => "ckks",
        }
    }

    #[getter]
    fn name(&self) -> String {
        match &self.inner {
            ProfileInner::Tfhe(p) => p.name().to_string(),
            ProfileInner::Ckks { max_poly_degree } => format!("max_poly_degree={max_poly_degree}"),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CryptoProfile(backend='{}', name='{}')",
            self.backend(),
            self.name()
        )
    }
}

impl CryptoProfile {
    fn as_tfhe(&self) -> PyResult<penumbra_tfhe::keys::TfheProfile> {
        match &self.inner {
            ProfileInner::Tfhe(p) => Ok(*p),
            ProfileInner::Ckks { .. } => Err(PyValueError::new_err(
                "profile/backend mismatch: profile is for backend 'ckks', but this run uses 'tfhe'",
            )),
        }
    }

    #[cfg(feature = "ckks")]
    fn as_ckks_poly(&self) -> PyResult<usize> {
        match &self.inner {
            ProfileInner::Ckks { max_poly_degree } => Ok(*max_poly_degree),
            ProfileInner::Tfhe(_) => Err(PyValueError::new_err(
                "profile/backend mismatch: profile is for backend 'tfhe', but this run uses 'ckks'",
            )),
        }
    }
}

/// Key metadata, inspectable without a sidecar file.
#[pyclass(frozen, get_all, module = "penumbra._penumbra")]
pub struct KeyInfo {
    pub backend: String,
    pub num_blocks: Option<usize>,
    pub profile: String,
}

#[pymethods]
impl KeyInfo {
    fn __repr__(&self) -> String {
        format!(
            "KeyInfo(backend='{}', num_blocks={:?}, profile='{}')",
            self.backend, self.num_blocks, self.profile
        )
    }
}

#[derive(Deserialize)]
struct SchemeHeader {
    scheme: String,
}

#[derive(Deserialize)]
struct TfheKeyHeader {
    #[allow(dead_code)]
    scheme: String,
    num_blocks: usize,
    profile: String,
}

#[derive(Deserialize)]
struct CkksKeyHeader {
    #[allow(dead_code)]
    scheme: String,
    payload: CkksPayloadHeader,
}

#[derive(Deserialize)]
struct CkksPayloadHeader {
    params: CkksParamsHeader,
}

#[derive(Deserialize)]
struct CkksParamsHeader {
    #[allow(dead_code)]
    n: usize,
    #[allow(dead_code)]
    base2k: usize,
    #[allow(dead_code)]
    k: usize,
    #[allow(dead_code)]
    log_delta: usize,
    #[allow(dead_code)]
    dsize: usize,
    #[allow(dead_code)]
    rank: usize,
    #[allow(dead_code)]
    lt_slots: usize,
    #[allow(dead_code)]
    giant_step: usize,
    #[allow(dead_code)]
    secret_ternary_prob: f64,
    max_poly_degree: usize,
}

/// Extract metadata from a serialized key envelope.
#[pyfunction]
fn key_info(key: &[u8]) -> PyResult<KeyInfo> {
    let header: SchemeHeader = bincode::deserialize(key).map_err(|e| {
        PyValueError::new_err(format!(
            "cannot deserialize key (is it a Penumbra key file?): {e}"
        ))
    })?;

    match header.scheme.as_str() {
        "tfhe" => {
            let tfhe_header: TfheKeyHeader = bincode::deserialize(key).map_err(|e| {
                PyValueError::new_err(format!(
                    "cannot deserialize client key (is it a Penumbra client key file? keys generated before the crypto-profile field was added must be regenerated): {e}"
                ))
            })?;
            let prof = penumbra_tfhe::keys::TfheProfile::from_name(&tfhe_header.profile)
                .map_err(PyValueError::new_err)?;
            Ok(KeyInfo {
                backend: "tfhe".to_string(),
                num_blocks: Some(tfhe_header.num_blocks),
                profile: prof.name().to_string(),
            })
        }
        "ckks" => {
            let ckks_header: CkksKeyHeader = bincode::deserialize(key).map_err(|e| {
                PyValueError::new_err(format!(
                    "cannot deserialize CKKS key (is it a Penumbra CKKS key file?): {e}"
                ))
            })?;
            Ok(KeyInfo {
                backend: "ckks".to_string(),
                num_blocks: None,
                profile: format!("max_poly_degree={}", ckks_header.payload.params.max_poly_degree),
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown key scheme '{other}' (key material is not portable across backends; see docs/BACKENDS.md)"
        ))),
    }
}

/// Backend names compiled into this build.
#[pyfunction]
fn available_backends() -> Vec<&'static str> {
    #[cfg(feature = "ckks")]
    {
        vec!["tfhe", "ckks"]
    }
    #[cfg(not(feature = "ckks"))]
    {
        vec!["tfhe"]
    }
}

fn unknown_backend_err(backend: &str) -> PyErr {
    if backend == "ckks" {
        PyValueError::new_err(
            "backend 'ckks' is not available in this build of penumbra-fhe: the CKKS backend needs a nightly Rust toolchain and is compiled out of published wheels. Build it from source with `maturin develop --release --features ckks` on nightly (see docs/BACKENDS.md). Available backends: tfhe",
        )
    } else {
        let backends = available_backends().join(", ");
        PyValueError::new_err(format!(
            "unknown backend '{backend}'; available backends: {backends}"
        ))
    }
}

fn parse_graph(graph_json: &str) -> Result<penumbra_core::ir::Graph, String> {
    let value: serde_json::Value =
        serde_json::from_str(graph_json).map_err(|e| format!("graph is not valid JSON: {e}"))?;
    let resolved = if value.get("graph").is_some() {
        value["graph"].to_string()
    } else {
        graph_json.to_string()
    };
    penumbra_core::ir::Graph::from_json(&resolved)
}

/// Generate client and server keys.
#[pyfunction]
#[pyo3(signature = (backend, num_blocks, profile=None))]
fn keygen(
    py: Python<'_>,
    backend: &str,
    num_blocks: usize,
    profile: Option<CryptoProfile>,
) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
    if let Some(p) = &profile {
        if p.backend() != backend {
            return Err(PyValueError::new_err(format!(
                "profile/backend mismatch: profile is for backend '{}', but this run uses '{backend}'",
                p.backend()
            )));
        }
    }

    match backend {
        "tfhe" => {
            let tfhe_prof = match profile {
                Some(p) => p.as_tfhe()?,
                None => penumbra_tfhe::keys::TfheProfile::Default,
            };
            let (ck_bytes, sk_bytes) = py
                .detach(|| -> Result<(Vec<u8>, Vec<u8>), String> {
                    let (ck, sk) = penumbra_tfhe::keys::keygen_with_profile(num_blocks, tfhe_prof);
                    let ck_b = penumbra_tfhe::keys::client_key_bytes(&ck, num_blocks, tfhe_prof)?;
                    let sk_b = penumbra_tfhe::keys::server_key_bytes(&sk, num_blocks, tfhe_prof)?;
                    Ok((ck_b, sk_b))
                })
                .map_err(PyValueError::new_err)?;
            Ok((
                PyBytes::new(py, &ck_bytes).unbind(),
                PyBytes::new(py, &sk_bytes).unbind(),
            ))
        }
        #[cfg(feature = "ckks")]
        "ckks" => {
            let max_poly = match profile {
                Some(p) => p.as_ckks_poly()?,
                None => penumbra_ckks::DEFAULT_PARAMS.max_poly_degree,
            };
            let params = penumbra_ckks::DEFAULT_PARAMS
                .with_max_poly_degree(max_poly)
                .map_err(PyValueError::new_err)?;
            let (ck_bytes, sk_bytes) = py
                .detach(|| -> Result<(Vec<u8>, Vec<u8>), String> {
                    let (ck, sk) = penumbra_ckks::keys::keygen(&params)?;
                    let ck_b = penumbra_ckks::keys::client_key_bytes(&ck)?;
                    let sk_b = penumbra_ckks::keys::server_key_bytes(&sk)?;
                    Ok((ck_b, sk_b))
                })
                .map_err(PyValueError::new_err)?;
            Ok((
                PyBytes::new(py, &ck_bytes).unbind(),
                PyBytes::new(py, &sk_bytes).unbind(),
            ))
        }
        _ => Err(unknown_backend_err(backend)),
    }
}

/// Encrypt a batch of integer rows.
#[pyfunction]
fn encrypt(
    py: Python<'_>,
    backend: &str,
    client_key: &[u8],
    rows: Vec<Vec<i64>>,
) -> PyResult<Py<PyBytes>> {
    match backend {
        "tfhe" => {
            let bytes = py
                .detach(|| -> Result<Vec<u8>, String> {
                    let (ck, _nb, _prof) = penumbra_tfhe::keys::client_key_from_bytes(client_key)?;
                    let batch: Vec<penumbra_tfhe::encrypt::CtVec> = rows
                        .iter()
                        .map(|row| penumbra_tfhe::encrypt::encrypt(&ck, row))
                        .collect();
                    penumbra_tfhe::encrypt::serialize_cts_batch(&batch)
                })
                .map_err(PyValueError::new_err)?;
            Ok(PyBytes::new(py, &bytes).unbind())
        }
        #[cfg(feature = "ckks")]
        "ckks" => {
            let bytes = py
                .detach(|| -> Result<Vec<u8>, String> {
                    let ck = penumbra_ckks::keys::client_key_from_bytes(client_key)?;
                    let batch: Vec<penumbra_ckks::encrypt::CtVec> = rows
                        .iter()
                        .map(|row| penumbra_ckks::encrypt::encrypt(&ck, row))
                        .collect();
                    penumbra_ckks::encrypt::serialize_cts_batch(&batch)
                })
                .map_err(PyValueError::new_err)?;
            Ok(PyBytes::new(py, &bytes).unbind())
        }
        _ => Err(unknown_backend_err(backend)),
    }
}

/// Evaluate an encrypted batch against an IR graph using the public server key.
#[pyfunction]
fn evaluate(
    py: Python<'_>,
    backend: &str,
    graph_json: &str,
    server_key: &[u8],
    cts: &[u8],
) -> PyResult<Py<PyBytes>> {
    let graph = parse_graph(graph_json).map_err(PyValueError::new_err)?;
    match backend {
        "tfhe" => {
            let bytes = py
                .detach(|| -> Result<Vec<u8>, String> {
                    let (sk, key_num_blocks, profile) =
                        penumbra_tfhe::keys::server_key_from_bytes(server_key)?;
                    penumbra_tfhe::keys::ensure_num_blocks_match(key_num_blocks, graph.num_blocks)?;
                    let backend = penumbra_tfhe::TfheBackend::new(profile);
                    let batch = penumbra_tfhe::encrypt::deserialize_cts_batch(cts)?;
                    let out_batch = session::run_evaluate_batch(&backend, &graph, &sk, batch)?;
                    penumbra_tfhe::encrypt::serialize_cts_batch(&out_batch)
                })
                .map_err(PyValueError::new_err)?;
            Ok(PyBytes::new(py, &bytes).unbind())
        }
        #[cfg(feature = "ckks")]
        "ckks" => {
            let bytes = py
                .detach(|| -> Result<Vec<u8>, String> {
                    let sk = penumbra_ckks::keys::server_key_from_bytes(server_key)?;
                    let backend = penumbra_ckks::CkksBackend::new(sk.params());
                    let batch = penumbra_ckks::encrypt::deserialize_cts_batch(cts)?;
                    let out_batch = session::run_evaluate_batch(&backend, &graph, &sk, batch)?;
                    penumbra_ckks::encrypt::serialize_cts_batch(&out_batch)
                })
                .map_err(PyValueError::new_err)?;
            Ok(PyBytes::new(py, &bytes).unbind())
        }
        _ => Err(unknown_backend_err(backend)),
    }
}

/// Decrypt an encrypted batch using the secret client key.
#[pyfunction]
fn decrypt(
    py: Python<'_>,
    backend: &str,
    client_key: &[u8],
    cts: &[u8],
) -> PyResult<Vec<Vec<i64>>> {
    match backend {
        "tfhe" => py
            .detach(|| -> Result<Vec<Vec<i64>>, String> {
                let (ck, _nb, _prof) = penumbra_tfhe::keys::client_key_from_bytes(client_key)?;
                let batch = penumbra_tfhe::encrypt::deserialize_cts_batch(cts)?;
                let decrypted: Vec<Vec<i64>> = batch
                    .iter()
                    .map(|ct| penumbra_tfhe::encrypt::decrypt_vec(&ck, ct))
                    .collect();
                Ok(decrypted)
            })
            .map_err(PyValueError::new_err),
        #[cfg(feature = "ckks")]
        "ckks" => py
            .detach(|| -> Result<Vec<Vec<i64>>, String> {
                let ck = penumbra_ckks::keys::client_key_from_bytes(client_key)?;
                let batch = penumbra_ckks::encrypt::deserialize_cts_batch(cts)?;
                let decrypted: Vec<Vec<i64>> = batch
                    .iter()
                    .map(|ct| penumbra_ckks::encrypt::decrypt_vec(&ck, ct))
                    .collect();
                Ok(decrypted)
            })
            .map_err(PyValueError::new_err),
        _ => Err(unknown_backend_err(backend)),
    }
}

/// Run an all-in-one forward pass on cleartext integer rows.
#[pyfunction]
#[pyo3(signature = (backend, graph_json, rows, profile=None))]
fn predict(
    py: Python<'_>,
    backend: &str,
    graph_json: &str,
    rows: Vec<Vec<i64>>,
    profile: Option<CryptoProfile>,
) -> PyResult<Vec<Vec<i64>>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(p) = &profile {
        if p.backend() != backend {
            return Err(PyValueError::new_err(format!(
                "profile/backend mismatch: profile is for backend '{}', but this run uses '{backend}'",
                p.backend()
            )));
        }
    }

    let graph = parse_graph(graph_json).map_err(PyValueError::new_err)?;
    match backend {
        "tfhe" => {
            let tfhe_prof = match profile {
                Some(p) => p.as_tfhe()?,
                None => penumbra_tfhe::keys::TfheProfile::Default,
            };
            let backend = penumbra_tfhe::TfheBackend::new(tfhe_prof);
            py.detach(|| session::run_predict(&backend, &graph, &rows))
                .map_err(PyValueError::new_err)
        }
        #[cfg(feature = "ckks")]
        "ckks" => {
            let max_poly = match profile {
                Some(p) => p.as_ckks_poly()?,
                None => penumbra_ckks::DEFAULT_PARAMS.max_poly_degree,
            };
            let params = penumbra_ckks::DEFAULT_PARAMS
                .with_max_poly_degree(max_poly)
                .map_err(PyValueError::new_err)?;
            let backend = penumbra_ckks::CkksBackend::new(params);
            py.detach(|| session::run_predict(&backend, &graph, &rows))
                .map_err(PyValueError::new_err)
        }
        _ => Err(unknown_backend_err(backend)),
    }
}

#[pymodule]
fn _penumbra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CryptoProfile>()?;
    m.add_class::<KeyInfo>()?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(keygen, m)?)?;
    m.add_function(wrap_pyfunction!(key_info, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    Ok(())
}
