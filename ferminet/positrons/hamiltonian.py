import chex
import jax.numpy as jnp

from typing import Optional, Sequence, Tuple

from ferminet import networks
from ferminet.hamiltonian import LocalEnergy, local_kinetic_energy

def potential_energy(r_af, r_ff, atoms, charges, nspins, fermion_charges):
  """Returns the potential energy for this fermion configuration.

  Args:
    r_af: Shape (nfermions, natoms). r_af[i, j] gives the distance between
      fermion i and atom j.
    r_ff: Shape (nfermions, nfermions, :). r_ee[i,j,0] gives the distance
      between fermions i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin species.
    fermion_charges: Shape (nspecies, ). Charge of each spin species.
  """
  fc = jnp.concatenate(
    [q*jnp.ones((n,)) for n, q in zip(nspins, fermion_charges)]
  )  # iterable of fermion charges
  v_ff = jnp.sum(jnp.triu(fc[None, ...]*fc[..., None] / r_ff[..., 0], k=1))
  v_af = jnp.sum(charges[None, ...]*fc[..., None] / r_af[..., 0])  # pylint: disable=invalid-unary-operand-type
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  v_aa = jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
  return v_ff + v_af + v_aa

def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    fermion_charges: Tuple[int] = (),
    states: int = 0,
    pp_type: None = None,
    pp_symbols: None = None
) -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian
    states: unused, excited states not implemented
    pp_type: unused, pseudopotentials not implemented
    pp_symbols: unused, pseudopotentials not implemented

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  if states:
    raise NotImplementedError(
      'Excited states not implemented with positrons'
    )


  ke = local_kinetic_energy(f,
                              use_scan=use_scan,
                              complex_output=complex_output,
                              laplacian_method=laplacian_method)


  if pp_symbols or pp_type:
    raise NotImplementedError(
      'Pseudopotentials not implemented with positrons'
    )

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    
    _, _, r_ae, r_ee = networks.construct_input_features(
        data.positions, data.atoms
    )
    potential = potential_energy(
      r_ae, r_ee, data.atoms, charges, nspins, fermion_charges)
    kinetic = ke(params, data)
    total_energy = potential + kinetic
    return total_energy, None

  return _e_l