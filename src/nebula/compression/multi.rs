//! Compression for an arbitrary number of independent Nebula IVC chains.
//!
//! [`super::CompressedSNARK`] is hard-wired to zkEngine's three chains (F / ops / scan) and their
//! memory-checking relations. [`MultiCompressedSNARK`] instead compresses `k >= 1` chains, each
//! with its own step circuit (and therefore its own R1CS shape), into one proof.
//!
//! The verifier checks everything that is generic to Nebula, per chain:
//!
//! * the final IVC instance hashes bind `(pp digest, num_steps, z0, zi, U, prev_IC, r_i)`,
//! * the claimed final incremental commitment is `IC_n = H(prev_IC, comm_omega_prev)`,
//! * the folded primary instances (batched) and the folded CycleFold instance are satisfiable.
//!
//! It then returns `(num_steps, zi)` for every chain. Any relation *between* chains (equal ICs
//! across chains, challenges derived from ICs and placed in some chain's `z0`, final accumulator
//! checks on `zi`, ...) is left to the caller, who knows what the chains mean.

use super::{ProverKey, VerifierKey};
use crate::{
  constants::{BN_N_LIMBS, NIO_CYCLE_FOLD, NUM_FE_IN_EMULATED_POINT, NUM_HASH_BITS},
  cyclefold::util::absorb_primary_relaxed_r1cs,
  errors::NovaError,
  gadgets::scalar_as_base,
  nebula::{
    ic::IC,
    nifs::{CycleFoldRelaxedNIFS, PrimaryNIFS, PrimaryRelaxedNIFS},
    rs::{PublicParams, RecursiveSNARK},
    traits::RecursiveSNARKFieldsTrait,
  },
  r1cs::{R1CSInstance, R1CSShape, RelaxedR1CSInstance},
  traits::{
    commitment::{CommitmentEngineTrait, Len},
    snark::{BatchedRelaxedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    AbsorbInROTrait, CurveCycleEquipped, Dual, Engine, ROTrait,
  },
  Commitment, CommitmentKey,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Public statement of one chain: its initial input and its final incremental commitment.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ChainStatement<E>
where
  E: CurveCycleEquipped,
{
  /// Initial input `z0` of the chain
  pub z0: Vec<E::Scalar>,
  /// Final incremental commitment `IC_n` of the chain
  pub IC: E::Scalar,
}

/// What the verifier learns about one chain after a successful verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainOutput<F> {
  /// Number of IVC steps the chain proved
  pub num_steps: usize,
  /// Final output `zi` of the chain
  pub zi: Vec<F>,
}

/// Per-chain data carried by [`MultiCompressedSNARK`]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct ChainProof<E>
where
  E: CurveCycleEquipped,
{
  r_U: RelaxedR1CSInstance<E>,
  l_u: R1CSInstance<E>,
  r_U_secondary: RelaxedR1CSInstance<Dual<E>>,
  zi: Vec<E::Scalar>,
  num_steps: usize,
  prev_IC: E::Scalar,
  comm_omega_prev: Commitment<E>,
  r_i: E::Scalar,
  nifs: PrimaryNIFS<E>,
  nifs_random: PrimaryRelaxedNIFS<E>,
  U_random: RelaxedR1CSInstance<E>,
  wit_blind: E::Scalar,
  err_blind: E::Scalar,
}

/// A SNARK that proves knowledge of `k` valid Nebula IVC proofs (one per chain).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MultiCompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  snark_primary: S1,
  snark_secondary: S2,
  chains: Vec<ChainProof<E>>,

  // CycleFold data: chain 0's running instance is folded with chains 1..k in order, then with a
  // random instance, then derandomized.
  nifs_secondary: Vec<CycleFoldRelaxedNIFS<E>>,
  nifs_final_secondary: CycleFoldRelaxedNIFS<E>,
  U_random_secondary: RelaxedR1CSInstance<Dual<E>>,
  wit_blind_secondary: <Dual<E> as Engine>::Scalar,
  err_blind_secondary: <Dual<E> as Engine>::Scalar,
}

/// The largest primary commitment key across `pps`. Keys are all derived from the same label, so
/// shorter keys are prefixes of the largest one.
fn biggest_ck<'a, E: CurveCycleEquipped>(pps: &[&'a PublicParams<E>]) -> &'a Arc<CommitmentKey<E>> {
  let mut ck = &pps[0].ck_primary;
  for pp in &pps[1..] {
    if pp.ck_primary.length() > ck.length() {
      ck = &pp.ck_primary;
    }
  }
  ck
}

fn primary_shapes<'a, E: CurveCycleEquipped>(pps: &[&'a PublicParams<E>]) -> Vec<&'a R1CSShape<E>> {
  pps
    .iter()
    .map(|pp| &pp.circuit_shape_primary.r1cs_shape)
    .collect()
}

/// All chains must agree on the CycleFold circuit (it is the same circuit for every step circuit)
/// so that their running CycleFold instances can be folded together.
fn check_cyclefold_compatible<E: CurveCycleEquipped>(
  pps: &[&PublicParams<E>],
) -> Result<(), NovaError> {
  if pps.is_empty() {
    return Err(NovaError::InvalidInitialInputLength);
  }
  let s0 = &pps[0].circuit_shape_cyclefold.r1cs_shape;
  for pp in &pps[1..] {
    if pp.circuit_shape_cyclefold.r1cs_shape != *s0 {
      return Err(NovaError::ProofVerifyError);
    }
  }
  Ok(())
}

impl<E, S1, S2> MultiCompressedSNARK<E, S1, S2>
where
  E: CurveCycleEquipped,
  S1: BatchedRelaxedR1CSSNARKTrait<E>,
  S2: RelaxedR1CSSNARKTrait<Dual<E>>,
{
  /// Creates prover and verifier keys for the chains described by `pps` (in this order).
  pub fn setup(
    pps: &[&PublicParams<E>],
  ) -> Result<(ProverKey<E, S1, S2>, VerifierKey<E, S1, S2>), NovaError> {
    check_cyclefold_compatible(pps)?;
    let ck = biggest_ck(pps);
    let (pk_primary, vk_primary) = S1::setup(ck.clone(), primary_shapes(pps))?;
    let (pk_secondary, vk_secondary) = S2::setup(
      pps[0].ck_cyclefold.clone(),
      &pps[0].circuit_shape_cyclefold.r1cs_shape,
    )?;
    let prover_key = ProverKey {
      primary: pk_primary,
      secondary: pk_secondary,
    };
    let verifier_key = VerifierKey {
      primary: vk_primary,
      secondary: vk_secondary,
      dk_primary: E::CE::derand_key(ck),
      dk_secondary: <Dual<E> as Engine>::CE::derand_key(&pps[0].ck_cyclefold),
    };
    Ok((prover_key, verifier_key))
  }

  /// Compress the chains `rss[j]`, each proven under `pps[j]`.
  pub fn prove(
    pps: &[&PublicParams<E>],
    pk: &ProverKey<E, S1, S2>,
    rss: &[&RecursiveSNARK<E>],
  ) -> Result<Self, NovaError> {
    check_cyclefold_compatible(pps)?;
    if pps.len() != rss.len() {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // Primary SNARK
    //
    // Fold each chain's (U, W, u, w) into (U', W'), fold in a random instance, derandomize, and
    // run all of them through one batched Spartan.
    let mut chains = Vec::with_capacity(rss.len());
    let mut U = Vec::with_capacity(rss.len());
    let mut W = Vec::with_capacity(rss.len());
    for (pp, rs) in pps.iter().zip(rss.iter()) {
      let (U_j, W_j, nifs, nifs_random, wit_blind, err_blind, U_random) =
        rs.fold_ivc_compression_step(pp)?;
      U.push(U_j);
      W.push(W_j);
      chains.push(ChainProof {
        r_U: rs.r_U_primary.clone(),
        l_u: rs.l_u_primary.clone(),
        r_U_secondary: rs.r_U_cyclefold.clone(),
        zi: rs.zi.clone(),
        num_steps: rs.num_steps(),
        prev_IC: rs.prev_IC,
        comm_omega_prev: rs.comm_omega_prev,
        r_i: rs.r_i(),
        nifs,
        nifs_random,
        U_random,
        wit_blind,
        err_blind,
      });
    }
    let snark_primary = S1::prove(biggest_ck(pps), &pk.primary, primary_shapes(pps), &U, &W)?;

    // Secondary SNARK
    //
    // Fold every chain's running CycleFold instance into one, fold in a random instance,
    // derandomize, and run it through Spartan.
    let ck_c = &pps[0].ck_cyclefold;
    let ro_consts = &pps[0].ro_consts;
    let S_c = &pps[0].circuit_shape_cyclefold.r1cs_shape;
    let (U0, W0) = rss[0].r_cyclefold();
    let (mut U_acc, mut W_acc) = (U0.clone(), W0.clone());
    let mut nifs_secondary = Vec::with_capacity(rss.len() - 1);
    for rs in &rss[1..] {
      let (U_j, W_j) = rs.r_cyclefold();
      let (nifs, (U_next, W_next), _) =
        CycleFoldRelaxedNIFS::<E>::prove(ck_c, ro_consts, S_c, &U_acc, &W_acc, U_j, W_j)?;
      nifs_secondary.push(nifs);
      U_acc = U_next;
      W_acc = W_next;
    }
    let (U_random_secondary, W_random_secondary) = S_c.sample_random_instance_witness(ck_c)?;
    let (nifs_final_secondary, (U_fin, W_fin), _) = CycleFoldRelaxedNIFS::<E>::prove(
      ck_c,
      ro_consts,
      S_c,
      &U_acc,
      &W_acc,
      &U_random_secondary,
      &W_random_secondary,
    )?;
    let (derandom_W, wit_blind_secondary, err_blind_secondary) = W_fin.derandomize();
    let derandom_U = U_fin.derandomize(
      &<Dual<E> as Engine>::CE::derand_key(ck_c),
      &wit_blind_secondary,
      &err_blind_secondary,
    );
    let snark_secondary = S2::prove(ck_c, &pk.secondary, S_c, &derandom_U, &derandom_W)?;

    Ok(Self {
      snark_primary,
      snark_secondary,
      chains,
      nifs_secondary,
      nifs_final_secondary,
      U_random_secondary,
      wit_blind_secondary,
      err_blind_secondary,
    })
  }

  /// Verify the compressed proof against the per-chain public statements, returning each chain's
  /// `(num_steps, zi)`. Cross-chain relations are the caller's responsibility.
  pub fn verify(
    &self,
    pps: &[&PublicParams<E>],
    vk: &VerifierKey<E, S1, S2>,
    statements: &[ChainStatement<E>],
  ) -> Result<Vec<ChainOutput<E::Scalar>>, NovaError> {
    check_cyclefold_compatible(pps)?;
    let k = pps.len();
    if self.chains.len() != k || statements.len() != k || self.nifs_secondary.len() + 1 != k {
      return Err(NovaError::ProofVerifyError);
    }

    // Per-chain IVC checks
    for ((pp, c), st) in pps.iter().zip(&self.chains).zip(statements) {
      if c.num_steps == 0
        || st.z0.len() != pp.F_arity_primary
        || c.zi.len() != pp.F_arity_primary
        || c.l_u.X.len() != 2
        || c.r_U.X.len() != 2
      {
        return Err(NovaError::ProofVerifyError);
      }

      // Calculate the hashes of the primary running instance and cyclefold running instance
      let mut hasher_p = <Dual<E> as Engine>::RO::new(
        pp.ro_consts.clone(),
        4 + 2 * pp.F_arity_primary + 2 * NUM_FE_IN_EMULATED_POINT + 3, // (digest, num_steps, prev_IC, r_i) + 2 * arity "(z0, zi)" + U
      );
      hasher_p.absorb(pp.digest());
      hasher_p.absorb(E::Scalar::from(c.num_steps as u64));
      for e in &st.z0 {
        hasher_p.absorb(*e);
      }
      for e in &c.zi {
        hasher_p.absorb(*e);
      }
      absorb_primary_relaxed_r1cs::<E, Dual<E>>(&c.r_U, &mut hasher_p);
      hasher_p.absorb(c.prev_IC);
      hasher_p.absorb(c.r_i);
      let hash_primary = hasher_p.squeeze(NUM_HASH_BITS);

      let mut hasher_c = <Dual<E> as Engine>::RO::new(
        pp.ro_consts.clone(),
        1 + 1 + 1 + 3 + 3 + 1 + NIO_CYCLE_FOLD * BN_N_LIMBS,
      );
      hasher_c.absorb(pp.digest());
      hasher_c.absorb(E::Scalar::from(c.num_steps as u64));
      c.r_U_secondary.absorb_in_ro(&mut hasher_c);
      hasher_c.absorb(c.r_i);
      let hash_cyclefold = hasher_c.squeeze(NUM_HASH_BITS);

      // Verify the hashes equal the public IO for the final primary instance
      if scalar_as_base::<Dual<E>>(hash_primary) != c.l_u.X[0]
        || scalar_as_base::<Dual<E>>(hash_cyclefold) != c.l_u.X[1]
      {
        return Err(NovaError::ProofVerifyError);
      }

      // Abort if C_n != hash(C_{n-1}, C_ω_{n-1})
      if st.IC != IC::<E>::increment_comm_w(&pp.ro_consts, c.prev_IC, c.comm_omega_prev) {
        return Err(NovaError::InvalidIC);
      }
    }

    // Verify primary SNARK over all chains' folded, randomized, derandomized instances
    let dk = &vk.dk_primary;
    let U = pps
      .iter()
      .zip(&self.chains)
      .map(|(pp, c)| {
        let U_f = c.nifs.verify(&pp.ro_consts, &pp.digest(), &c.r_U, &c.l_u);
        let U_r = c
          .nifs_random
          .verify(&pp.ro_consts, &pp.digest(), &U_f, &c.U_random);
        U_r.derandomize(dk, &c.wit_blind, &c.err_blind)
      })
      .collect::<Vec<_>>();
    self.snark_primary.verify(&vk.primary, &U)?;

    // Verify secondary SNARK
    let ro_consts = &pps[0].ro_consts;
    let mut U_acc = self.chains[0].r_U_secondary.clone();
    for (nifs, c) in self.nifs_secondary.iter().zip(&self.chains[1..]) {
      U_acc = nifs.verify(ro_consts, &U_acc, &c.r_U_secondary);
    }
    let U_fin = self
      .nifs_final_secondary
      .verify(ro_consts, &U_acc, &self.U_random_secondary);
    let derandom_U = U_fin.derandomize(
      &vk.dk_secondary,
      &self.wit_blind_secondary,
      &self.err_blind_secondary,
    );
    self.snark_secondary.verify(&vk.secondary, &derandom_U)?;

    Ok(
      self
        .chains
        .iter()
        .map(|c| ChainOutput {
          num_steps: c.num_steps,
          zi: c.zi.clone(),
        })
        .collect(),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    nebula::rs::StepCircuit,
    provider::{ipa_pc, Bn256EngineIPA},
    spartan,
    traits::snark::default_ck_hint,
  };
  use ff::Field;

  type E1 = Bn256EngineIPA;
  type F = <E1 as Engine>::Scalar;
  type EE1 = ipa_pc::EvaluationEngine<E1>;
  type EE2 = ipa_pc::EvaluationEngine<Dual<E1>>;
  type S1 = spartan::batched::BatchedRelaxedR1CSSNARK<E1, EE1>;
  type S2 = spartan::snark::RelaxedR1CSSNARK<Dual<E1>, EE2>;

  /// z -> z^2, with some per-step advice
  #[derive(Clone)]
  struct SquareCircuit {
    advice: Vec<F>,
  }

  impl StepCircuit<F> for SquareCircuit {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      Ok(vec![z[0].square(cs.namespace(|| "sq"))?])
    }

    fn non_deterministic_advice(&self) -> Vec<F> {
      self.advice.clone()
    }
  }

  /// (a, b) -> (a + b, a * b): a different shape and arity from `SquareCircuit`
  #[derive(Clone)]
  struct AddMulCircuit {
    advice: Vec<F>,
  }

  impl StepCircuit<F> for AddMulCircuit {
    fn arity(&self) -> usize {
      2
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let s = z[0].add(cs.namespace(|| "add"), &z[1])?;
      let p = z[0].mul(cs.namespace(|| "mul"), &z[1])?;
      let mut x = p;
      for i in 0..100 {
        x = x.square(cs.namespace(|| format!("pad_{i}")))?;
      }
      Ok(vec![s, x])
    }

    fn non_deterministic_advice(&self) -> Vec<F> {
      self.advice.clone()
    }
  }

  /// Run a chain for `n` steps; returns the chain and its final IC.
  fn run_chain<C: StepCircuit<F>>(
    pp: &PublicParams<E1>,
    circuits: &[C],
    z0: &[F],
  ) -> (RecursiveSNARK<E1>, F) {
    let mut rs = RecursiveSNARK::new(pp, &circuits[0], z0).unwrap();
    let mut IC_i = F::ZERO;
    for c in circuits {
      rs.prove_step(pp, c, IC_i).unwrap();
      IC_i = rs.increment_commitment(pp, c);
    }
    rs.verify(pp, circuits.len(), z0, IC_i).unwrap();
    (rs, IC_i)
  }

  struct Fixture {
    pp_a: PublicParams<E1>,
    pp_b: PublicParams<E1>,
    rs_a: RecursiveSNARK<E1>,
    rs_b: RecursiveSNARK<E1>,
    st: Vec<ChainStatement<E1>>,
  }

  fn fixture() -> Fixture {
    let sq: Vec<_> = (0..3)
      .map(|i| SquareCircuit {
        advice: vec![F::from(i as u64), F::from(7)],
      })
      .collect();
    let am: Vec<_> = (0..5)
      .map(|i| AddMulCircuit {
        advice: vec![F::from(10 + i as u64)],
      })
      .collect();
    let pp_a = PublicParams::<E1>::setup(&sq[0], &*default_ck_hint(), &*default_ck_hint());
    let pp_b = PublicParams::<E1>::setup(&am[0], &*default_ck_hint(), &*default_ck_hint());
    let z0_a = vec![F::from(2)];
    let z0_b = vec![F::from(3), F::from(5)];
    let (rs_a, ic_a) = run_chain(&pp_a, &sq, &z0_a);
    let (rs_b, ic_b) = run_chain(&pp_b, &am, &z0_b);
    let st = vec![
      ChainStatement { z0: z0_a, IC: ic_a },
      ChainStatement { z0: z0_b, IC: ic_b },
    ];
    Fixture {
      pp_a,
      pp_b,
      rs_a,
      rs_b,
      st,
    }
  }

  #[test]
  fn test_multi_compression_two_chains() {
    let f = fixture();
    let pps = [&f.pp_a, &f.pp_b];
    let (pk, vk) = MultiCompressedSNARK::<E1, S1, S2>::setup(&pps).unwrap();
    let snark = MultiCompressedSNARK::<E1, S1, S2>::prove(&pps, &pk, &[&f.rs_a, &f.rs_b]).unwrap();
    let out = snark.verify(&pps, &vk, &f.st).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].num_steps, 3);
    assert_eq!(out[1].num_steps, 5);
    // 2 -> 4 -> 16 -> 256
    assert_eq!(out[0].zi, vec![F::from(256)]);
    assert_eq!(out[1].zi, f.rs_b.zi);
  }

  #[test]
  fn test_multi_compression_single_chain() {
    let f = fixture();
    let pps = [&f.pp_b];
    let (pk, vk) = MultiCompressedSNARK::<E1, S1, S2>::setup(&pps).unwrap();
    let snark = MultiCompressedSNARK::<E1, S1, S2>::prove(&pps, &pk, &[&f.rs_b]).unwrap();
    snark.verify(&pps, &vk, &f.st[1..]).unwrap();
  }

  #[test]
  fn test_multi_compression_rejects_bad_statement() {
    let f = fixture();
    let pps = [&f.pp_a, &f.pp_b];
    let (pk, vk) = MultiCompressedSNARK::<E1, S1, S2>::setup(&pps).unwrap();
    let snark = MultiCompressedSNARK::<E1, S1, S2>::prove(&pps, &pk, &[&f.rs_a, &f.rs_b]).unwrap();

    // wrong z0
    let mut st = f.st.clone();
    st[1].z0[0] += F::ONE;
    assert!(snark.verify(&pps, &vk, &st).is_err());

    // wrong IC
    let mut st = f.st.clone();
    st[0].IC += F::ONE;
    assert!(snark.verify(&pps, &vk, &st).is_err());

    // chains swapped
    let st = vec![f.st[1].clone(), f.st[0].clone()];
    assert!(snark.verify(&[&f.pp_b, &f.pp_a], &vk, &st).is_err());
  }
}
