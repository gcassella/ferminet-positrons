"""Positronic complex with a dilithium molecule."""

from ferminet import base_config
from ferminet.utils import system

def get_config():
    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.electrons = (3, 3, 1)
    cfg.system.molecule = [
        system.Atom("Li", [0., 0., 0.], units='bohr'), 
        system.Atom("Li", [5.051, 0., 0.], units='bohr')
    ]
    # Set up positronic Hamiltonian. Indices of `fermion_charges` correspond to
    # indices of `cfg.system.electrons`
    cfg.system.make_local_energy_fn = \
      'ferminet.positrons.hamiltonian.local_energy'
    cfg.system.make_local_energy_kwargs = {
        "fermion_charges": [-1, -1, 1],
    }
    # Use different move widths per particle species
    cfg.mcmc.separate_spin_moves = True 
    cfg.mcmc.move_width = 0.2
    cfg.mcmc.burn_in = 100
    cfg.mcmc.steps = 20
    # Pretraining is not currently implemented for positronic systems
    cfg.pretrain.method = 'none'

    return cfg
