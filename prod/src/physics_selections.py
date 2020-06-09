"""
This module defines the physics code for the RPV multi-jet analysis with numpy.
That includes the jet object selection, baseline event selection, and signal
region event selection. It also provides the functionality to calculate the
summed jet mass physics variable.
"""

from __future__ import print_function
import numpy as np

class units():
    GeV = 1e3

class cuts():
    # Object selection
    HT_min = 0*units.GeV
    fatjet_pt_min = 30*units.GeV
    jet_pt_min = 30*units.GeV
    jet_eta_max = 2.4
    ## Baseline event selection
    baseline_num_bjet_min = 1
    baseline_num_jet_min = 4
    baseline_MJ_min = 500*units.GeV
    baseline_HT_min = 1500*units.GeV
    #baseline_num_fatjet_min = 3
    #baseline_fatjet_pt_min = 440*units.GeV
    # Signal region event selection
    num_jet_min = 8
    num_bjet_min = 3
    sr_mass_min = 800*units.GeV
    sr_HT_min = 1500*units.GeV

def _apply_indices(a, indices):
    """Helper function for applying index array if it exists"""
    if indices is not None:
        return a[indices]
    else:
        return a

def filter_objects(obj_idx, *obj_arrays):
    """Applies an object filter to a set of object arrays."""
    filtered_arrays = []
    def filt(x, idx):
        return x[idx]
    vec_filter = np.vectorize(filt, otypes=[np.ndarray])
    for obj_array in obj_arrays:
        filtered_arrays.append(vec_filter(obj_array, obj_idx))
    return filtered_arrays

def filter_events(event_idx, *arrays):
    """Applies an event filter to a set of arrays."""
    return map(lambda x: x[event_idx], arrays)

#def select_fatjets(fatjet_pts, fatjet_etas):
def select_fatjets(fatjet_pts):
    """
    Selects the analysis fat jets for one event.

    Input params
      fatjet_pts: array of fat jet pt
      fatjet_etas: array of fat jet eta

    Returns a boolean index-array of the selected jets
    """
    return np.logical_and(
            fatjet_pts > cuts.fatjet_pt_min,
            fatjet_pts > cuts.fatjet_pt_min)

def select_HT(scalar_HT):
    """
    Selects the analysis fat jets for one event.

    Input params
      scalar_HT: array of HT

    Returns a boolean index-array of the selected HT
    """
    return np.logical_and(
            scalar_HT > cuts.HT_min,
            scalar_HT > cuts.HT_min)

def select_jets(jet_pts, jet_etas):
    """
    Selects the analysis jets for one event.

    Input params
      jet_pts: array of jet pt
      jet_etas: array of jet eta

    Returns a boolean index-array of the selected jets
    """
    return np.logical_and(
            jet_pts > cuts.jet_pt_min,
            np.fabs(jet_etas) < cuts.jet_eta_max)

def numbjet(jet_btag):
    """
    Counting the b-tagged jet in the one event.

    Inputs
      btag : b-jet tagging

    Returns a float
    """
    return np.sum(jet_btag)

def sum_fatjet_mass(fatjet_ms, selected_fatjets=None):
    """
    Calculates the summed fat jet mass.
    Uses the 4 leading selected fat jets.

    Inputs
      fatjet_ms: array of fat jet masses
      selected_fatjets: boolean index-array of selected fatjets in the array

    Returns a float
    """
    masses = _apply_indices(fatjet_ms, selected_fatjets)
    #return np.sum(masses[:4])
    return np.sum(masses)


def is_baseline_event(fatjet_ms, jet_btag, jet_pts, selected_jets=None):
    """
    Applies baseline event selection to one event.

    Inputs
      jet_pts: array of jet pt
      selected_jets: boolean index-array of selected jets in the array

    Returns a bool
    """
    pts = _apply_indices(jet_pts, selected_jets)
    #Nb = numbjet(jet_btag)
    #MJ = sum_fatjet_mass(fatjet_ms)
    # number of b-tagged Jet requirement
    # Sum of FatJet mass requirement
    if np.sum(fatjet_ms) < cuts.baseline_MJ_min:
        return False
    if np.sum(jet_btag) < cuts.baseline_num_bjet_min:
        return False
    # HT (sum of all jet pts) requirement
    if np.sum(pts) < cuts.baseline_HT_min:
        return False
    # Jet multiplicity requirement
    if pts.size < cuts.baseline_num_jet_min:
        return False
    return True

#def is_baseline_event(fatjet_pts, selected_fatjets=None):
#    """
#    Applies baseline event selection to one event.
#
#    Inputs
#      fatjet_pts: array of fat jet pt
#      selected_fatjets: boolean index-array of selected fatjets in the array
#
#    Returns a bool
#    """
#    pts = _apply_indices(fatjet_pts, selected_fatjets)
#    ## Fat-jet multiplicity requirement
#    #if pts.size < cuts.baseline_num_fatjet_min:
#    #    return False
#    ## Fat-jet trigger plateau efficiency requirement
#    #if np.max(pts) < cuts.baseline_fatjet_pt_min:
#    #    return False
#    return True



#def pass_srj(num_jet, btag, summed_mass, HT):
def pass_srj(num_jet, num_bjet, summed_mass, HT):
    """
    Applies the jet signal region selection to one event.
    Inputs
      num_jet: number of selected jets
      btag : b-jet tagging
      summed_mass: summed fatjet mass as calculated by sum_fatjet_mass function
      HT : scalar summed all of parton level particles
    Returns a bool
    """
    #num_bjet = np.sum(btag)
    #return all([num_jet >= cuts.num_jet_min,
    #            summed_mass > cuts.sr_mass_min])
    return all([num_jet >= cuts.num_jet_min,
                num_bjet >= cuts.num_bjet_min,
                summed_mass > cuts.sr_mass_min,
                HT > cuts.sr_HT_min])

#def is_signal_region_event(summed_mass, fatjet_pts, jet_etas,
#                           selected_jets, HT, is_baseline=None):
def is_signal_region_event(summed_mass, jet_pts, HT,
                           selected_jets, is_baseline=None):
    """
    Applies signal region selection to one event.

    Inputs
      summed_mass: summed fatjet mass as calculated by sum_fatjet_mass function
      fatjet_pts: array of fat jet pt
      jet_etas: array of jet eta
      selected_fatjets: boolean index-array of selected fatjets in array
      is_baseline: whether the event passes baseline event selection

    Returns a bool
    """
    # Baseline event selection
    if is_baseline == False:
        return False
    if (is_baseline is None and
        not is_baseline_event(jet_pts, selected_jets)):
        return False
    # Jet multiplicity
    num_jets = (np.sum(selected_jets)
                   if selected_jets is not None
                   else jet_etas.size)
    if num_jets < 8:
        return False
    # Summed fat jet mass cut
    if summed_mass < cuts.sr_mass_min:
        return False
    # HT cut
    if HT < cuts.sr_HT_min:
        return False
    # Passes all requirements
    return True
