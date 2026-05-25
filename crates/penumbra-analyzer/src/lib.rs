//! Depth cost analysis and bootstrapping placement for Penumbra.
//!
//! This crate analyzes IR graphs to compute the multiplicative depth cost of
//! operations and decides where to place bootstrapping operations to prevent
//! noise overflow.
