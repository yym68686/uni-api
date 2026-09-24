//! The uni-api gateway. Process assembly is the only external entry point.

pub(crate) mod api;
pub(crate) mod app;
pub(crate) mod config;
pub(crate) mod control;
pub(crate) mod observability;
pub(crate) mod protocols;
pub(crate) mod providers;
pub(crate) mod routing;
pub(crate) mod runtime;
pub(crate) mod storage;
pub(crate) mod transport;
pub(crate) mod upstream;

pub use app::run;
